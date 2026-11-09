"""Generate stills with the Windows x64 GPU venv. No network listen, safetensors only.

Run by the Studio queue as:
  gpu-venv\\Scripts\\python.exe -X utf8 app\\windows_x64_generate.py --request request.json
Torch and diffusers are imported here, never by the UI process.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import windows_x64_stills as stills


def load_request(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("the request is not an object")
    model = data.get("model")
    if model not in stills.MODELS:
        raise ValueError(f"unknown model {model}")
    images = data.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("the request has no images")
    root = Path(data["model_dir"])
    bad = stills.refused_weight(root)
    if bad:
        raise RuntimeError(f"refusing to load {bad}: this backend accepts safetensors only")
    if not stills.model_ready(model):
        # model_ready uses stills.model_dir, which follows STUDIO_X64_HOME. The request names the same folder.
        missing = [name for name in stills.MODELS[model]["markers"] if not (root / name).is_file()]
        if missing:
            raise RuntimeError("model files missing: " + ", ".join(missing))
    return data


def sheet(paths: list[Path], dest: Path) -> None:
    from PIL import Image

    frames = [Image.open(path).convert("RGB") for path in paths]
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    cols = 1 if len(frames) == 1 else 2 if len(frames) <= 4 else 3
    rows = (len(frames) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * width, rows * height), "white")
    for index, frame in enumerate(frames):
        canvas.paste(frame, ((index % cols) * width, (index // cols) * height))
    canvas.save(dest, quality=90)
    for frame in frames:
        frame.close()


def generate(request: dict) -> list[Path]:
    import torch
    from diffusers import StableDiffusionPipeline, ZImagePipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Windows x64 GPU environment. "
                           "The NVIDIA driver did not expose the GPU to this PyTorch wheel.")
    model = request["model"]
    spec = stills.MODELS[model]
    root = Path(request["model_dir"])
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"[tensorfold] engine: cuda", flush=True)
    print(f"[tensorfold] stage 1/3 load {name}", flush=True)
    if spec["kind"] == "zimage":
        pipe = ZImagePipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True,
                                              low_cpu_mem_usage=True)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            str(root), torch_dtype=torch.float16, variant="fp16", local_files_only=True, low_cpu_mem_usage=True,
            safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    offload = spec["kind"] == "zimage" and total < stills.FULL_GPU_BYTES
    if offload:
        pipe.enable_model_cpu_offload()
        print("[tensorfold] stage 2/3 offload text encoder stays in system RAM", flush=True)
    else:
        pipe.to("cuda")
        print("[tensorfold] stage 2/3 gpu", flush=True)
    target = Path(request["output"])
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    images = request["images"]
    for index, item in enumerate(images, start=1):
        print(f"[tensorfold] stage 3/3 image {index}/{len(images)}", flush=True)
        result = pipe(
            item["prompt"], width=int(request["width"]), height=int(request["height"]),
            num_inference_steps=spec["steps"], guidance_scale=spec["guidance"],
            generator=torch.Generator(device="cuda").manual_seed(int(item["seed"])),
            callback_on_step_end=lambda _pipe, step, _timestep, kwargs, total_steps=spec["steps"]: (
                print(f"[tensorfold] step {step + 1}/{total_steps}", flush=True) or kwargs),
        )
        path = target / item["name"]
        result.images[0].save(path)
        written.append(path)
        print(f"[tensorfold] wrote {path.name}", flush=True)
    sheet(written, target / "sheet.jpg")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        written = generate(load_request(args.request))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(str(error), file=sys.stderr)
        return 1
    if not written:
        print("no image was written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
