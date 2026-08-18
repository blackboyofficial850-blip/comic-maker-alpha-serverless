# Comic Maker Alpha — RunPod Serverless cached-model proof worker

This worker is intentionally limited to the first FLUX.2 Klein house-style proof.

Endpoint requirements:

- Queue endpoint.
- Model / cached model: `black-forest-labs/FLUX.2-klein-base-4B`.
- If RunPod asks for a Hugging Face token, provide a token from the account that has access to the model.
- Attach the small Comic Maker Alpha Network Volume containing only:
  - `/runpod-volume/house_style/pytorch_lora_weights.safetensors`
  - `/runpod-volume/references/style`
- Active workers: 0.
- Max workers: 1 for the proof.
- Execution timeout: at least 1200 seconds.
- Prefer 30+ GiB GPUs; 24 GiB-class GPUs use CPU offload.

The worker resolves FLUX from RunPod Cached Models under
`/runpod-volume/huggingface-cache/hub/`. It never downloads FLUX from Hugging Face inside the billed handler.
