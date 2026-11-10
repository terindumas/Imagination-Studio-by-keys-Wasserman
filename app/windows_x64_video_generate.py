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


def place(pipe, free_bytes: int, *, tile: bool = True):
    """Keep the text encoder off a 16 GB card. A larger free pool can hold the pipeline.

    LTX decode is not tiled: a 768×512 latent is small, and tiling draws a grid.
    """
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
        if tile and hasattr(pipe.vae, "enable_tiling"):
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


def diffusers_module():
    """diffusers 0.36 is the first release this loader is written against."""
    import diffusers
    parts = []
    for piece in diffusers.__version__.split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    if tuple(parts[:2]) < (0, 36):
        raise RuntimeError(f"diffusers {diffusers.__version__} is too old for LTX-Video. "
                           "The x64 GPU venv needs diffusers 0.36 or newer.")
    return diffusers


# The distilled checkpoint's own sampler. Linear-quadratic, eight steps, no resolution shift.
# Rounded in the file metadata as 1, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725, 0.4219.
LTX_DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875)
LTX_DECODE_TIMESTEP = 0.05
LTX_DECODE_NOISE_SCALE = 0.025
# Diffusers config for the timestep-conditioned 0.9.5/0.9.6 VAE. The weights are inside the
# distilled safetensors. The vae/ folder of Lightricks/LTX-Video is the 0.9.0 decoder.
LTX_VAE_CONFIG = Path(__file__).resolve().parent / "ltx" / "vae-0.9.6"


def ltx_transformer_cls(diffusers):
    """The class that reads ltxv-2b-0.9.6-distilled-04-25.safetensors.

    The generic auto class does not provide this single-file loader on every
    diffusers build. The LTX transformer class does, from 0.36 onward.
    """
    cls = getattr(diffusers, "LTXVideoTransformer3DModel", None)
    if cls is None or not callable(getattr(cls, "from_single_file", None)):
        version = getattr(diffusers, "__version__", "unknown")
        raise RuntimeError(f"diffusers {version} has no LTXVideoTransformer3DModel.from_single_file. "
                           "Install diffusers>=0.36 in the x64 GPU venv.")
    return cls


def missing_tokenizer_deps(import_module=None) -> list[str]:
    """Packages the T5 tokenizer import needs. sentencepiece reads spiece.model and needs protobuf.
    If that fails, transformers tries tiktoken."""
    if import_module is None:
        import importlib
        import_module = importlib.import_module
    missing = []
    for module, label in (("sentencepiece", "sentencepiece"), ("google.protobuf", "protobuf"), ("tiktoken", "tiktoken")):
        try:
            import_module(module)
        except ImportError:
            missing.append(label)
    return missing


def require_tokenizer_deps(import_module=None) -> None:
    missing = missing_tokenizer_deps(import_module)
    if missing:
        raise RuntimeError("The LTX tokenizer could not import " + ", ".join(missing)
                           + ". Install them in the x64 GPU venv with "
                           "scripts\\setup-windows-x64-video.ps1 -Model ltx. The weights stay put.")


def ltx_vae_cls(diffusers):
    """The class that reads the VAE stored in the distilled safetensors."""
    cls = getattr(diffusers, "AutoencoderKLLTXVideo", None)
    if cls is None or not callable(getattr(cls, "from_single_file", None)):
        version = getattr(diffusers, "__version__", "unknown")
        raise RuntimeError(f"diffusers {version} has no AutoencoderKLLTXVideo.from_single_file. "
                           "Install diffusers>=0.36 in the x64 GPU venv.")
    return cls


def distilled_scheduler(scheduler_cls):
    """Eight unshifted sigmas, with the stochastic step the distilled config asks for.

    LTXPipeline always passes a uniform linspace and, with the repo scheduler, a resolution
    shift that ends near 0.1. This checkpoint only denoises the list in its metadata.
    """
    scheduler = scheduler_cls(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False,
                              stochastic_sampling=True)
    original = scheduler.set_timesteps

    def set_timesteps(num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
        # The name `sigmas` has to be a real parameter. LTXPipeline checks the signature
        # and rejects the scheduler when it is only present as **kwargs. The linspace
        # it passes is not the distilled schedule, so it is not forwarded.
        del num_inference_steps, sigmas, mu, timesteps
        return original(device=device, sigmas=list(LTX_DISTILLED_SIGMAS))

    scheduler.set_timesteps = set_timesteps
    return scheduler


def bind_ltx_generator(pipe, generator) -> None:
    """The distilled step adds noise. Use the job seed for that draw too."""
    original = pipe.scheduler.step

    def step(*args, **kwargs):
        kwargs.setdefault("generator", generator)
        return original(*args, **kwargs)

    pipe.scheduler.step = step


def ltx_single_file_args(root: Path) -> dict:
    """Where the distilled checkpoint and the transformer config actually sit.

    model_index.json is at the model root. config.json is at transformer/config.json.
    Passing the root alone makes diffusers look for config.json beside model_index.json.
    The subfolder is required: when config is set, this diffusers build does not add it.
    The VAE is not the vae/ folder. That folder is the 0.9.0 decoder. The matching
    timestep-conditioned VAE is inside the distilled safetensors.
    """
    root = Path(root)
    checkpoint = root / "ltxv-2b-0.9.6-distilled-04-25.safetensors"
    config_dir = root / "transformer"
    missing = [str(path) for path in (checkpoint, root / "model_index.json", config_dir / "config.json",
                                      root / "vae" / "config.json", root / "vae" / "diffusion_pytorch_model.safetensors",
                                      root / "text_encoder" / "config.json",
                                      root / "text_encoder" / "model.safetensors.index.json",
                                      root / "tokenizer" / "tokenizer_config.json",
                                      root / "scheduler" / "scheduler_config.json") if not path.is_file()]
    if missing:
        raise RuntimeError("LTX-Video is missing files the loader reads: " + ", ".join(missing))
    return {"checkpoint": str(checkpoint), "config": str(root), "subfolder": "transformer"}


def wan_folder_args(root: Path) -> dict:
    """Wan is a full diffusers folder: every component config lives in its subfolder."""
    root = Path(root)
    needed = (root / "model_index.json", root / "scheduler" / "scheduler_config.json",
              root / "text_encoder" / "config.json", root / "text_encoder" / "model.safetensors.index.json",
              root / "tokenizer" / "tokenizer_config.json", root / "transformer" / "config.json",
              root / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
              root / "vae" / "config.json", root / "vae" / "diffusion_pytorch_model.safetensors")
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise RuntimeError("Wan 2.1 is missing files the loader reads: " + ", ".join(missing))
    return {"pretrained": str(root)}


def load_ltx(root: Path, dtype, image: str | None):
    require_tokenizer_deps()
    diffusers = diffusers_module()
    args = ltx_single_file_args(root)
    transformer = ltx_transformer_cls(diffusers).from_single_file(
        args["checkpoint"], config=args["config"], subfolder=args["subfolder"],
        torch_dtype=dtype, local_files_only=True)
    # When config is set, this diffusers build does not apply the VAE's default subfolder.
    # config.json lives at vae-0.9.6/vae/config.json, so subfolder is required.
    vae = ltx_vae_cls(diffusers).from_single_file(
        args["checkpoint"], config=str(LTX_VAE_CONFIG), subfolder="vae",
        torch_dtype=dtype, local_files_only=True)
    name = "LTXImageToVideoPipeline" if image else "LTXPipeline"
    pipe_cls = getattr(diffusers, name, None)
    if pipe_cls is None:
        raise RuntimeError(f"diffusers {diffusers.__version__} has no {name}.")
    scheduler_cls = getattr(diffusers, "FlowMatchEulerDiscreteScheduler", None)
    if scheduler_cls is None:
        raise RuntimeError(f"diffusers {diffusers.__version__} has no FlowMatchEulerDiscreteScheduler.")
    pipe = pipe_cls.from_pretrained(str(root), transformer=transformer, vae=vae, torch_dtype=dtype,
                                    local_files_only=True)
    pipe.scheduler = distilled_scheduler(scheduler_cls)
    return pipe


def load_wan(root: Path, dtype):
    require_tokenizer_deps()
    diffusers = diffusers_module()
    pipe_cls = getattr(diffusers, "WanPipeline", None)
    if pipe_cls is None:
        raise RuntimeError(f"diffusers {diffusers.__version__} has no WanPipeline.")
    wan_folder_args(root)
    return pipe_cls.from_pretrained(str(root), torch_dtype=dtype, local_files_only=True)


def generate(request: dict) -> Path:
    import torch

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
        pipe = load_ltx(root, dtype, request.get("image"))
    else:
        pipe = load_wan(root, dtype)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    how = place(pipe, free_bytes, tile=spec["kind"] != "ltx")
    print(f"[tensorfold] stage 2/3 {how}", flush=True)
    generator = torch.Generator(device="cuda").manual_seed(int(request["seed"]))
    call = {"prompt": request["prompt"], "width": int(request["width"]), "height": int(request["height"]),
            "num_frames": int(request["frames"]), "num_inference_steps": spec["steps"],
            "guidance_scale": spec["guidance"], "generator": generator,
            "callback_on_step_end": lambda _pipe, step, _timestep, kwargs, total=spec["steps"]: (
                print(f"[tensorfold] step {step + 1}/{total}", flush=True) or kwargs)}
    if spec["kind"] == "ltx":
        bind_ltx_generator(pipe, generator)
        call.update(decode_timestep=LTX_DECODE_TIMESTEP, decode_noise_scale=LTX_DECODE_NOISE_SCALE,
                    frame_rate=int(request["fps"]))
        print("[tensorfold] ltx vae single-file, sigmas "
              + ",".join(f"{value:.4f}" for value in LTX_DISTILLED_SIGMAS), flush=True)
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
