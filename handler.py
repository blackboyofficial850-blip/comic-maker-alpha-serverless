from __future__ import annotations

import base64
import io
import os
from pathlib import Path

# RunPod Cached Models are already present before worker billing starts.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")

import boto3
from botocore.config import Config
import runpod
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline


MODEL_ID = os.environ.get(
    "MODEL_NAME",
    "black-forest-labs/FLUX.2-klein-base-4B",
)

HF_CACHE_ROOT = Path(
    os.environ.get(
        "CMA_HF_CACHE_ROOT",
        "/runpod-volume/huggingface-cache/hub",
    )
)

ASSET_ROOT = Path(
    os.environ.get(
        "CMA_ASSET_ROOT",
        "/tmp/comic-maker-alpha-assets",
    )
)

LORA_DIR = ASSET_ROOT / "house_style"
LORA_FILE = os.environ.get(
    "CMA_LORA_FILE",
    "pytorch_lora_weights.safetensors",
)

STYLE_DIR = ASSET_ROOT / "references" / "style"

DEFAULT_TRIGGER_TOKEN = os.environ.get(
    "CMA_TRIGGER_TOKEN",
    "c0m1cHouse",
)

ASSET_BUCKET = os.environ.get("CMA_ASSET_BUCKET", "").strip()
ASSET_REGION = os.environ.get("CMA_ASSET_REGION", "US-CA-2").strip()
ASSET_ENDPOINT = os.environ.get(
    "CMA_ASSET_ENDPOINT",
    "https://s3api-us-ca-2.runpod.io/",
).strip()

_PIPE = None
_PIPE_DEVICE_MODE = None
_GPU_NAME = None
_VRAM_GB = None
_MODEL_DIR = None
_ASSETS_READY = False


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
            path
            for path in snapshots.iterdir()
            if path.is_dir() and (path / "model_index.json").exists()
        ]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)

    raise RuntimeError(
        f"RunPod cached model {model_id} was not found under {HF_CACHE_ROOT}. "
        "Keep Cached model set to black-forest-labs/FLUX.2-klein-base-4B."
    )


def _s3_client():
    access = os.environ.get("CMA_S3_ACCESS_KEY", "").strip()
    secret = os.environ.get("CMA_S3_SECRET_KEY", "").strip()

    if not ASSET_BUCKET:
        raise RuntimeError("CMA_ASSET_BUCKET is not set.")
    if not access or not secret:
        raise RuntimeError(
            "CMA_S3_ACCESS_KEY / CMA_S3_SECRET_KEY are not set. "
            "Use the RunPod S3 API credentials for the asset volume."
        )

    return boto3.client(
        "s3",
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=ASSET_REGION,
        endpoint_url=ASSET_ENDPOINT,
        config=Config(
            connect_timeout=20,
            read_timeout=120,
            retries={"max_attempts": 8, "mode": "standard"},
            max_pool_connections=4,
        ),
    )


def _download_object(client, key: str, local_path: Path) -> None:
    """Download with GetObject directly.

    RunPod's S3-compatible layer supports GetObject. Avoid boto3.download_file()
    here because its transfer manager performs an automatic HeadObject first,
    which is the operation that returned 403 on the first Serverless proof.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(local_path.suffix + ".part")

    response = client.get_object(Bucket=ASSET_BUCKET, Key=key)
    body = response["Body"]

    with tmp.open("wb") as out:
        while True:
            chunk = body.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    body.close()
    tmp.replace(local_path)


def ensure_assets(job) -> None:
    global _ASSETS_READY

    if _ASSETS_READY:
        return

    lora_path = LORA_DIR / LORA_FILE
    existing_refs = [
        p
        for p in STYLE_DIR.glob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    ]
    if lora_path.exists() and existing_refs:
        _ASSETS_READY = True
        progress(job, f"House-style assets already local ({len(existing_refs)} references).")
        return

    progress(job, "Fetching small house-style assets from RunPod S3 storage...")
    client = _s3_client()

    lora_key = f"house_style/{LORA_FILE}"
    progress(job, f"Downloading LoRA: {lora_key}")
    _download_object(client, lora_key, lora_path)

    paginator = client.get_paginator("list_objects_v2")
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    ref_keys: list[str] = []

    for page in paginator.paginate(Bucket=ASSET_BUCKET, Prefix="references/style/"):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if not key or key.endswith("/"):
                continue
            if Path(key).suffix.lower() in allowed:
                ref_keys.append(key)

    ref_keys = sorted(set(ref_keys))
    if not ref_keys:
        raise RuntimeError(
            "No style reference images were found in "
            f"s3://{ASSET_BUCKET}/references/style/"
        )

    for index, key in enumerate(ref_keys, 1):
        destination = STYLE_DIR / Path(key).name
        progress(job, f"Downloading style reference {index}/{len(ref_keys)}: {destination.name}")
        _download_object(client, key, destination)

    _ASSETS_READY = True
    progress(
        job,
        f"House-style assets ready: LoRA + {len(ref_keys)} references. "
        "Network Volume is NOT attached to this worker.",
    )


def validate_storage() -> Path:
    model_dir = resolve_cached_snapshot(MODEL_ID)

    lora_path = LORA_DIR / LORA_FILE
    if not lora_path.exists():
        raise RuntimeError(f"House-style LoRA is missing at {lora_path}")

    required_components = [
        "transformer",
        "text_encoder",
        "vae",
        "tokenizer",
        "scheduler",
    ]
    missing = [
        component
        for component in required_components
        if not (model_dir / component).exists()
    ]
    if missing:
        raise RuntimeError(
            f"Cached FLUX snapshot is incomplete at {model_dir}; "
            f"missing: {', '.join(missing)}"
        )

    return model_dir


def load_pipeline(job):
    global _PIPE, _PIPE_DEVICE_MODE, _GPU_NAME, _VRAM_GB, _MODEL_DIR

    if _PIPE is not None:
        progress(job, f"FLUX already warm on {_GPU_NAME}; reusing loaded pipeline.")
        return _PIPE

    ensure_assets(job)
    model_dir = validate_storage()
    _MODEL_DIR = str(model_dir)

    if not torch.cuda.is_available():
        raise RuntimeError("RunPod Serverless worker started without a CUDA GPU.")

    _GPU_NAME = torch.cuda.get_device_name(0)
    _VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

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

    allowed_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = [
        path
        for path in sorted(STYLE_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ][:count]

    images = [Image.open(path).convert("RGB") for path in paths]
    names = [path.name for path in paths]
    return images, names


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
    trigger_token = str(
        data.get("trigger_token") or DEFAULT_TRIGGER_TOKEN
    ).strip()

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

    reference_message = f"Style references selected: {len(refs)}"
    if ref_names:
        reference_message += f" ({', '.join(ref_names)})"
    progress(job, reference_message)

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


runpod.serverless.start({"handler": handler})
