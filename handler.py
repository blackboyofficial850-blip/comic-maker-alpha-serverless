from __future__ import annotations

import base64
import io
import os
from pathlib import Path

# RunPod Cached Models are already present before worker billing starts.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")

import runpod
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

VOLUME_ROOT = Path(os.environ.get("CMA_VOLUME_ROOT", "/runpod-volume"))
MODEL_ID = os.environ.get("MODEL_NAME", "black-forest-labs/FLUX.2-klein-base-4B")
HF_CACHE_ROOT = Path(os.environ.get("CMA_HF_CACHE_ROOT", str(VOLUME_ROOT / "huggingface-cache" / "hub")))
LORA_DIR = Path(os.environ.get("CMA_LORA_DIR", str(VOLUME_ROOT / "house_style")))
LORA_FILE = os.environ.get("CMA_LORA_FILE", "pytorch_lora_weights.safetensors")
STYLE_DIR = Path(os.environ.get("CMA_STYLE_DIR", str(VOLUME_ROOT / "references" / "style")))
DEFAULT_TRIGGER_TOKEN = os.environ.get("CMA_TRIGGER_TOKEN", "c0m1cHouse")

_PIPE = None
_PIPE_DEVICE_MODE = None
_GPU_NAME = None
_VRAM_GB = None
_MODEL_DIR = None


def progress(job, message: str) -> None:
    print(message, flush=True)
    try:
        updater = getattr(runpod.serverless, "progress_update", None)
        if updater is not None:
            updater(job, message)
    except Exception as exc:
        print(f"Progress update warning (render continues): {exc}", flush=True)


def resolve_cached_snapshot(model_id: str) -> Path:
    if "/" not in model_id:
        raise RuntimeError(f"MODEL_NAME must be org/name, got: {model_id!r}")
    org, name = model_id.split("/", 1)
    model_root = HF_CACHE_ROOT / f"models--{org}--{name}"
    snapshots = model_root / "snapshots"

    ref_main = model_root / "refs" / "main"
    if ref_main.exists():
        commit = ref_main.read_text(encoding="utf-8").strip()
        candidate = snapshots / commit
        if (candidate / "model_index.json").exists():
            return candidate

    if snapshots.exists():
        candidates = [
            p for p in snapshots.iterdir()
            if p.is_dir() and (p / "model_index.json").exists()
        ]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)

    raise RuntimeError(
        f"RunPod cached model {model_id} was not found under {HF_CACHE_ROOT}. "
        "Set the endpoint Model field to black-forest-labs/FLUX.2-klein-base-4B."
    )


def validate_storage() -> Path:
    model_dir = resolve_cached_snapshot(MODEL_ID)
    lora = LORA_DIR / LORA_FILE
    if not lora.exists():
        raise RuntimeError(f"House-style LoRA is missing at {lora}")
    required = ["transformer", "text_encoder", "vae", "tokenizer", "scheduler"]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise RuntimeError(
            f"Cached FLUX snapshot is incomplete at {model_dir}; missing: {', '.join(missing)}"
        )
    return model_dir


def load_pipeline(job):
    global _PIPE, _PIPE_DEVICE_MODE, _GPU_NAME, _VRAM_GB, _MODEL_DIR
    if _PIPE is not None:
        progress(job, f"FLUX already warm on {_GPU_NAME}; reusing loaded pipeline.")
        return _PIPE

    model_dir = validate_storage()
    _MODEL_DIR = str(model_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("RunPod Serverless worker started without a CUDA GPU.")

    _GPU_NAME = torch.cuda.get_device_name(0)
    _VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    progress(job, f"GPU ready: {_GPU_NAME} ({_VRAM_GB:.1f} GiB VRAM)")
    progress(job, f"Loading FLUX.2 Klein from RunPod Cached Models: {model_dir}")

    pipe = Flux2KleinPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    progress(job, "Loading trained Comic Maker Alpha house-style LoRA...")
    pipe.load_lora_weights(
        str(LORA_DIR),
        weight_name=LORA_FILE,
        adapter_name="house_style",
    )

    if _VRAM_GB >= 30:
        pipe.to("cuda")
        _PIPE_DEVICE_MODE = "full GPU"
    else:
        pipe.enable_model_cpu_offload()
        _PIPE_DEVICE_MODE = "CPU offload"

    _PIPE = pipe
    progress(job, f"Pipeline ready ({_PIPE_DEVICE_MODE}).")
    return _PIPE


def style_references(count: int) -> tuple[list[Image.Image], list[str]]:
    if count <= 0 or not STYLE_DIR.exists():
        return [], []
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = [
        p for p in sorted(STYLE_DIR.rglob("*"))
        if p.is_file() and p.suffix.lower() in allowed
    ][:count]
    return [Image.open(p).convert("RGB") for p in paths], [p.name for p in paths]


def handler(job):
    data = job.get("input") or {}
    prompt_text = str(data.get("prompt") or "").strip()
    if not prompt_text:
        raise ValueError("input.prompt is required")

    seed = int(data.get("seed", 12345))
    width = int(data.get("width", 896))
    height = int(data.get("height", 1152))
    steps = int(data.get("steps", 36))
    guidance = float(data.get("guidance", 4.0))
    lora_scale = float(data.get("lora_scale", 1.0))
    reference_count = max(0, min(4, int(data.get("reference_count", 3))))
    trigger_token = str(data.get("trigger_token") or DEFAULT_TRIGGER_TOKEN).strip()

    if width < 256 or height < 256 or width > 1536 or height > 1536:
        raise ValueError("width/height must be between 256 and 1536")
    if steps < 1 or steps > 100:
        raise ValueError("steps must be between 1 and 100")

    pipe = load_pipeline(job)
    try:
        pipe.set_adapters("house_style", adapter_weights=[lora_scale])
    except Exception as exc:
        progress(job, f"Adapter scale setter skipped: {exc}")

    refs, ref_names = style_references(reference_count)
    progress(job, f"Style references selected: {len(refs)}" + (f" ({', '.join(ref_names)})" if ref_names else ""))

    style_instruction = (
        "Create a completely new illustration. "
        "Use the attached images only as visual style references. "
        "Match their drawing language, line quality, shape design, color handling, "
        "lighting, facial rendering, shading, material treatment, and polished "
        "illustration finish. Do not copy their characters, text, poses, or composition. "
    )
    token_prefix = f"{trigger_token}. " if trigger_token else ""
    final_prompt = token_prefix + (style_instruction if refs else "") + prompt_text

    def callback(_pipe, step_index, timestep, callback_kwargs):
        done = step_index + 1
        if done == 1 or done % 4 == 0 or done == steps:
            progress(job, f"Rendering step {done}/{steps}")
        return callback_kwargs

    progress(job, "Starting FLUX generation...")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    kwargs = {
        "prompt": final_prompt,
        "height": height,
        "width": width,
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "generator": generator,
        "callback_on_step_end": callback,
    }
    if refs:
        kwargs["image"] = refs

    image = pipe(**kwargs).images[0]
    output = io.BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    progress(job, "Render complete; returning PNG to Comic Maker Alpha.")

    return {
        "image_base64": encoded,
        "mime_type": "image/png",
        "seed": seed,
        "gpu": _GPU_NAME,
        "vram_gb": round(float(_VRAM_GB or 0), 1),
        "device_mode": _PIPE_DEVICE_MODE,
        "model_dir": _MODEL_DIR,
        "reference_count": len(refs),
        "reference_files": ref_names,
        "lora_scale": lora_scale,
        "trigger_token": trigger_token,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
