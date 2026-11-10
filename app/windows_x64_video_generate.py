"""Render one short mp4 with the Windows x64 GPU venv. No network listen.

The text encoder is too large to sit on a 16 GB card with the transformer, so a
short GPU keeps it in system RAM and moves one block at a time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import windows_x64_stills as stills
import windows_x64_video as video


def load_request(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("video_model") not in video.MODELS:
        raise ValueError("the video request is not a known Windows x64 model")
    root = Path(data["model_dir"])
    bad = stills.refused_weight(root)
    if bad:
        raise RuntimeError(f"refusing to load {bad}: this backend accepts safetensors only")
    missing = [name for name in video.MODELS[data["video_model"]]["markers"] if not (root / name).is_file()]
    if missing:
        raise RuntimeError("model files missing: " + ", ".join(missing))
    return data


def place(pipe, free_bytes: int):
    """Keep the text encoder off a 16 GB card. A larger free pool can hold the pipeline."""
    import torch
    if free_bytes >= stills.FULL_GPU_BYTES:
        pipe.to("cuda")
        return "gpu"
    onload, offload = torch.device("cuda"), torch.device("cpu")
    try:
        from diffusers.hooks import apply_group_offloading
        pipe.transformer.enable_group_offload(onload_device=onload, offload_device=offload,
                                              offload_type="leaf_level", use_stream=True)
        apply_group_offloading(pipe.text_encoder, onload_device=onload, offload_device=offload,
                               offload_type="block_level", num_blocks_per_group=2)
        apply_group_offloading(pipe.vae, onload_device=onload, offload_device=offload, offload_type="leaf_level")
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        return "group-offload"
    except (AttributeError, ImportError, TypeError, RuntimeError):
        pipe.enable_model_cpu_offload()
        return "cpu-offload"


def write_mp4(frames, path: Path, fps: int) -> None:
    import numpy as np
    import imageio.v2 as imageio
    arrays = [np.asarray(frame) for frame in frames]
    imageio.mimsave(str(path), arrays, fps=fps, codec="libx264", quality=8)


def generate(request: dict) -> Path:
    import torch
    from diffusers import AutoModel, LTXImageToVideoPipeline, LTXPipeline, WanPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Windows x64 GPU environment.")
    model = request["video_model"]
    spec = video.MODELS[model]
    root = Path(request["model_dir"])
    free_bytes, _total = torch.cuda.mem_get_info()
    print("[tensorfold] engine: cuda", flush=True)
    print(f"[tensorfold] stage 1/3 load {torch.cuda.get_device_name(0)}", flush=True)
    need = video.VRAM_NEED_MB[model] * 1024 * 1024
    if free_bytes < need:
        said = video.memory_fit(model, host_free_mb=video.HOST_NEED_MB[model],
                                gpu_free_mb=free_bytes // (1024 * 1024), holders=stills.gpu_holders())
        raise RuntimeError(said["reason"] or "not enough free GPU memory")
    dtype = torch.bfloat16
    if spec["kind"] == "ltx":
        distilled = root / "ltxv-2b-0.9.6-distilled-04-25.safetensors"
        transformer = AutoModel.from_single_file(str(distilled), torch_dtype=dtype, local_files_only=True)
        cls = LTXImageToVideoPipeline if request.get("image") else LTXPipeline
        pipe = cls.from_pretrained(str(root), transformer=transformer, torch_dtype=dtype, local_files_only=True)
    else:
        pipe = WanPipeline.from_pretrained(str(root), torch_dtype=dtype, local_files_only=True)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    how = place(pipe, free_bytes)
    print(f"[tensorfold] stage 2/3 {how}", flush=True)
    generator = torch.Generator(device="cuda").manual_seed(int(request["seed"]))
    call = {"prompt": request["prompt"], "width": int(request["width"]), "height": int(request["height"]),
            "num_frames": int(request["frames"]), "num_inference_steps": spec["steps"],
            "guidance_scale": spec["guidance"], "generator": generator,
            "callback_on_step_end": lambda _pipe, step, _timestep, kwargs, total=spec["steps"]: (
                print(f"[tensorfold] step {step + 1}/{total}", flush=True) or kwargs)}
    if spec["kind"] == "ltx":
        call.update(decode_timestep=0.05, decode_noise_scale=0.025)
        if request.get("image"):
            from PIL import Image
            call["image"] = Image.open(request["image"]).convert("RGB")
    print("[tensorfold] stage 3/3 frames", flush=True)
    result = pipe(**call)
    frames = result.frames[0]
    target = Path(request["output"])
    target.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(frames, target, int(request["fps"]))
    print(f"[tensorfold] wrote {target.name}", flush=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        generate(load_request(args.request))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
