# Comic Maker Alpha — RunPod Serverless proof worker 1.2.8.5-alpha

This worker removes the attached-Network-Volume datacenter restriction.

FLUX:
- Keep RunPod Cached model set to `black-forest-labs/FLUX.2-klein-base-4B`.
- The worker resolves it from `/runpod-volume/huggingface-cache/hub/`.

House-style assets:
- Do NOT attach the Comic Maker Alpha Network Volume to the endpoint.
- The worker fetches only the small LoRA + reference images from that volume through RunPod's S3-compatible API on cold start.
- Expected S3 objects:
  - `house_style/pytorch_lora_weights.safetensors`
  - `references/style/*`

Required endpoint environment variables:
- `CMA_ASSET_BUCKET` = your Network Volume ID
- `CMA_ASSET_REGION` = `US-CA-2`
- `CMA_ASSET_ENDPOINT` = `https://s3api-us-ca-2.runpod.io/`
- `CMA_S3_ACCESS_KEY` = your RunPod S3 access/user ID
- `CMA_S3_SECRET_KEY` = your RunPod S3 secret

For the proof:
- Queue endpoint
- Active workers: 0
- Max workers: 1
- GPU count: 1
- Idle timeout: 300 seconds
- Execution timeout: 1200 seconds
- Prefer multiple 24 GB / 48 GB GPU pools
- Allow all datacenters after detaching the Network Volume


## 1.2.8.5 change
- Uses direct S3 `GetObject` downloads instead of `boto3.download_file()`.
- Avoids the automatic `HeadObject` request that returned HTTP 403 on the first live proof worker.
- Uses CMA-specific S3 credential environment variables to avoid collisions with generic AWS variables.
