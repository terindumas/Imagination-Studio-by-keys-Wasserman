"""Short clips on Windows x64 with a 16 GB NVIDIA GPU.

MiniMax H3 and FastH3 need about 58–90 GB and are not this path. LTX-Video 2B
distilled (Lightricks) is the default. Wan 2.1 T2V 1.3B is the other choice.
The text encoder stays in system RAM. One block at a time moves onto the GPU.
The generator does not open a port.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import windows_x64_stills as stills

TIMEOUT_S = 1800
AUTO = "ltx"
SMALL_VRAM_MB = 24 * 1024
H3_NOTE = ("MiniMax H3 and FastH3 are not supported on this GPU. They need about 58–90 GB. "
           "This PC uses LTX-Video 2B (or Wan 2.1 1.3B) instead.")
# Shown in /api/state qualities on a card of 24 GB or less. Not the FastH3 pass labels.
QUALITIES = {
    "ltx": "LTX-Video 2B, 8 steps, 97 frames",
    "ltx121": "LTX-Video 2B, 8 steps, 121 frames",
    "wan13": "Wan 2.1 1.3B, 30 steps, 81 frames",
}
SETUP = r"powershell -ExecutionPolicy Bypass -File scripts\setup-windows-x64-video.ps1 -Model ltx"

# Host is system RAM for the text encoder. VRAM is the free memory one block needs.
HOST_NEED_MB = {"ltx": 24576, "wan13": 24576}
VRAM_NEED_MB = {"ltx": 8192, "wan13": 8192}

MODELS: dict[str, dict] = {
    "ltx": {
        "title": "LTX-Video 2B",
        "note": "Distilled 2B, 8 steps, 768×512, 97 frames at 24 fps. Text to video, or from one picture. LTXV Open Weights License.",
        "repo": "Lightricks/LTX-Video",
        "kind": "ltx",
        "license": "LTXV Open Weights License 0.X",
        "author": "Lightricks",
        "disk_gb": 27,
        "width": 768,
        "height": 512,
        "frames": 97,
        "fps": 24,
        "steps": 8,
        "guidance": 1.0,
        "frame_mod": 8,
        "min_frames": 25,
        "max_frames": 121,
        "max_pixels": 768 * 512,
        "image": True,
        "markers": (
            "model_index.json",
            "text_encoder/model.safetensors.index.json",
            "vae/diffusion_pytorch_model.safetensors",
            "transformer/config.json",
            "ltxv-2b-0.9.6-distilled-04-25.safetensors",
        ),
        "allow": (
            "model_index.json",
            "LTX-Video-Open-Weights-License-0.X.txt",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "vae/*",
            "transformer/config.json",
            "ltxv-2b-0.9.6-distilled-04-25.safetensors",
        ),
    },
    "wan13": {
        "title": "Wan 2.1 1.3B",
        "note": "Text to video, 30 steps, 768×512, 81 frames at 24 fps. Apache-2.0. No picture input on this build.",
        "repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "kind": "wan",
        "license": "Apache-2.0",
        "author": "Wan-AI",
        "disk_gb": 29,
        "width": 768,
        "height": 512,
        "frames": 81,
        "fps": 24,
        "steps": 30,
        "guidance": 5.0,
        "frame_mod": 4,
        "min_frames": 17,
        "max_frames": 81,
        "max_pixels": 768 * 512,
        "image": False,
        "markers": (
            "model_index.json",
            "text_encoder/model.safetensors.index.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
        "allow": (
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "transformer/*",
            "vae/*",
        ),
    },
}


def is_model(name: object) -> bool:
    return isinstance(name, str) and name in MODELS


def gpu_total_mb() -> int | None:
    forced = os.environ.get("STUDIO_X64_VRAM_MB", "").strip()
    if forced.isdigit():
        return int(forced)
    try:
        spark = str(Path(__file__).resolve().parent.parent / "spark")
        if spark not in sys.path:
            sys.path.insert(0, spark)
        import memory_guard
        card = memory_guard.device(fresh=True)
    except (OSError, RuntimeError, ValueError, ImportError):
        return None
    if not card:
        return None
    return int(card.get("card_total_mb", card.get("total_mb", 0)) or 0) or None


def small_gpu() -> bool:
    """Windows x64 cards of 24 GB and under do not run FastH3 or MiniMax H3."""
    if not stills.is_x64_host():
        return False
    forced = os.environ.get("STUDIO_X64_SMALL_GPU", "").strip().lower()
    if forced in {"0", "false", "no"}:
        return False
    if forced in {"1", "true", "yes"}:
        return True
    total = gpu_total_mb()
    if total is None:
        return True
    return total <= SMALL_VRAM_MB


def model_dir(model: str) -> Path:
    return stills.home() / "models" / model


def model_ready(model: str) -> bool:
    root = model_dir(model)
    return all((root / name).is_file() and (root / name).stat().st_size > 0 for name in MODELS[model]["markers"])


def model_why(model: str) -> str:
    if not is_model(model):
        return f"Unknown Windows x64 video model {model}. Choose {', '.join(MODELS)}."
    if not stills.is_x64_host():
        return "Windows x64 video runs on native Windows AMD64 with an NVIDIA GPU."
    if small_gpu():
        prefix = H3_NOTE + " "
    else:
        prefix = ""
    if not stills.python_path().is_file():
        return prefix + "The Windows x64 GPU environment is not installed. Run scripts\\setup-windows-x64-images.ps1 -NoDownload first."
    bad = stills.refused_weight(model_dir(model))
    if bad:
        return prefix + f"{bad} is not a safetensors weight. Remove it from {model_dir(model)}."
    if not model_ready(model):
        return prefix + (f"{MODELS[model]['title']} weights are not installed. "
                         rf"powershell -ExecutionPolicy Bypass -File scripts\setup-windows-x64-video.ps1 -Model {model}")
    return ""


def ready_models() -> list[str]:
    if not stills.is_x64_host() or not stills.python_path().is_file():
        return []
    return [name for name in MODELS if model_ready(name) and not stills.refused_weight(model_dir(name))]


def default_model() -> str:
    ready = ready_models()
    return ready[0] if ready else AUTO


def x64_video_job(job: dict) -> bool:
    params = job.get("params") or {}
    return job.get("kind") == "video" and stills.is_x64_host() and is_model(params.get("video_model"))


def status() -> dict:
    ready = ready_models()
    chosen = ready[0] if ready else AUTO
    spec = MODELS[chosen]
    available = bool(ready)
    why = "" if available else model_why(chosen)
    return {"available": available, "why": why, "installed": available, "qualified": False,
            "models": ready, "model": chosen, "small": small_gpu(), "h3": False, "h3_note": H3_NOTE if small_gpu() else "",
            "width": spec["width"], "height": spec["height"], "frames": spec["frames"], "fps": spec["fps"],
            "engine": f"CUDA · {spec['title']} (Windows x64)" if available else "CUDA · Windows x64 video (not installed)",
            "setup": SETUP, "backend": "x64-cuda"}


def parameters(params: dict) -> dict:
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    allowed = {"prompt", "seed", "width", "height", "frames", "video_model", "image", "fps"}
    extra = set(params) - allowed
    if extra:
        raise ValueError("This GPU does not run FastH3 or MiniMax H3. A small-video job takes prompt, video_model, "
                         "width, height, frames, seed, fps, and an optional image. Extra settings: "
                         + ", ".join(sorted(extra)))
    model = params.get("video_model") or AUTO
    if model not in MODELS:
        raise ValueError(f"unknown Windows x64 video model {model}: one of {', '.join(MODELS)}")
    spec = MODELS[model]
    prompt = params.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20000:
        raise ValueError("prompt is 1 to 20,000 characters")
    width = params.get("width", spec["width"])
    height = params.get("height", spec["height"])
    frames = params.get("frames", spec["frames"])
    seed = params.get("seed", 1)
    fps = params.get("fps", spec["fps"])
    if any(type(value) is not int for value in (width, height, frames, seed, fps)):
        raise ValueError("width, height, frames, seed and fps must be whole numbers")
    if width % 32 or height % 32:
        raise ValueError("video sides are multiples of 32")
    if not 256 <= width <= 768 or not 256 <= height <= 768:
        raise ValueError(f"{spec['title']} sides are 256 to 768")
    if width * height > spec["max_pixels"]:
        raise ValueError(f"{spec['title']} keeps the frame at or under {spec['max_pixels']} pixels "
                         f"({width}×{height} is {width * height})")
    if not spec["min_frames"] <= frames <= spec["max_frames"] or (frames - 1) % spec["frame_mod"]:
        raise ValueError(f"{spec['title']} takes {spec['min_frames']} to {spec['max_frames']} frames, "
                         f"one more than a multiple of {spec['frame_mod']}")
    if not 0 <= seed <= 2 ** 32 - 1:
        raise ValueError("seed is an unsigned 32-bit whole number")
    if fps != spec["fps"]:
        raise ValueError(f"{spec['title']} exports at {spec['fps']} fps")
    image = params.get("image")
    if image in ("", None):
        image = None
    elif not isinstance(image, str) or len(image) > 400:
        raise ValueError("image is a path inside the studio folder")
    if image and not spec["image"]:
        raise ValueError(f"{spec['title']} is text to video. LTX-Video can start from a picture.")
    out = {"prompt": prompt.strip(), "video_model": model, "width": width, "height": height,
           "frames": frames, "seed": seed, "fps": fps}
    if image:
        out["image"] = image.replace("\\", "/")
    return out


def memory_fit(model: str, *, host_free_mb: int | None, gpu_free_mb: int | None,
               holders: list[str] | None = None) -> dict:
    if model not in MODELS:
        raise ValueError(f"unknown Windows x64 video model {model}")
    title = MODELS[model]["title"]
    host_need, vram_need = HOST_NEED_MB[model], VRAM_NEED_MB[model]
    host_ok = host_free_mb is not None and host_free_mb >= host_need
    vram_known = gpu_free_mb is not None
    vram_ok = (not vram_known) or gpu_free_mb >= vram_need
    parts = []
    if not (host_ok and vram_ok):
        if host_free_mb is None:
            parts.append(f"{title} needs about {host_need / 1024:.0f} GB of free system RAM and that figure could not be read.")
        else:
            parts.append(f"{title} needs about {host_need / 1024:.0f} GB of free system RAM and this PC has {host_free_mb / 1024:.0f} GB.")
    if vram_known and not vram_ok:
        parts.append(f"It needs about {vram_need / 1024:.0f} GB of free GPU memory and this GPU has {gpu_free_mb / 1024:.1f} GB free. "
                     "The text encoder stays in system RAM.")
        named = holders or []
        if named:
            parts.append("Holding that GPU memory: " + "; ".join(named) + ". The Studio does not stop that program.")
        else:
            parts.append("No GPU process of at least 1 GB was named. Check Task Manager's Dedicated GPU memory column.")
    return {"fits": host_ok and vram_ok, "need_mb": host_need,
            "available_mb": 0 if host_free_mb is None else host_free_mb, "floor_mb": 0,
            "vram_need_mb": vram_need, "vram_free_mb": gpu_free_mb, "reason": " ".join(parts)}


def admission(params: dict) -> dict:
    model = (params or {}).get("video_model") or AUTO
    gpu = stills.gpu_free_mb()
    return memory_fit(model, host_free_mb=stills.host_free_mb(), gpu_free_mb=gpu,
                      holders=stills.gpu_holders() if gpu is not None else [])


def build(params: dict, home_dir: Path, job_id: str) -> tuple[list[str], dict, list[str]]:
    params = parameters(params)
    why = model_why(params["video_model"])
    if why:
        raise RuntimeError(why)
    target = Path(home_dir) / "renders" / f"{job_id}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    request = {**params, "model_dir": str(model_dir(params["video_model"])), "output": str(target)}
    if params.get("image"):
        request["image"] = str(Path(home_dir) / params["image"])
    path = target.with_suffix(".request.json")
    path.write_text(json.dumps(request), encoding="utf-8")
    script = Path(__file__).with_name("windows_x64_video_generate.py")
    return [str(stills.python_path()), "-X", "utf8", str(script), "--request", str(path)], {}, [str(target)]


def plan(model: str = "auto") -> dict:
    if model != "auto" and model not in MODELS:
        raise ValueError(f"Unknown video model {model}. Choose auto, {', '.join(MODELS)}.")
    selected = AUTO if model == "auto" else model
    spec = MODELS[selected]
    return {"requested_model": model, "selected_model": selected, "title": spec["title"], "repo": spec["repo"],
            "license": spec["license"], "author": spec["author"], "disk_gb": spec["disk_gb"], "token": False,
            "gated": False, "format": "safetensors only", "host_required": "native Windows AMD64 (x64)",
            "install_allowed_on_this_process": stills.real_x64_host(), "listen": "127.0.0.1:7870",
            "ports_opened": [], "h3": False, "h3_note": H3_NOTE,
            "fit": spec["note"], "home": str(model_dir(selected)),
            "why": "Auto installs LTX-Video 2B distilled. Wan 2.1 1.3B is -Model wan13." if model == "auto"
            else "Explicit model retained."}


def download(model: str) -> dict:
    if not stills.real_x64_host() and os.environ.get("STUDIO_X64_ALLOW_HOST", "").strip() != "1":
        raise RuntimeError("This video installer requires native Windows AMD64. Nothing was downloaded.")
    selected = AUTO if model == "auto" else model
    if selected not in MODELS:
        raise ValueError(f"Unknown video model {model}.")
    spec = MODELS[selected]
    dest = model_dir(selected)
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(spec["repo"], local_dir=str(dest), allow_patterns=list(spec["allow"]),
                      ignore_patterns=["*.ckpt", "*.pt", "*.pth", "*.bin", "assets/*", "examples/*", "media/*"],
                      token=False)
    bad = stills.refused_weight(dest)
    if bad:
        raise RuntimeError(f"Download left a non-safetensors weight ({bad}). It was not loaded.")
    if not model_ready(selected):
        raise RuntimeError(f"{spec['title']} did not leave a complete safetensors pipeline in {dest}.")
    return {"model": selected, "dir": str(dest)}


def rollback(model: str = "auto", *, dry: bool = False) -> dict:
    selected = AUTO if model == "auto" else model
    if selected not in MODELS:
        raise ValueError(f"Unknown video model {model}.")
    root = model_dir(selected).resolve()
    if root.name != selected or root.parent.name != "models" or root.parent.parent.name != "x64":
        raise RuntimeError(f"Refusing to remove {root}. Video weights must be ImaginationStudio\\x64\\models\\{selected}.")
    present = root.is_dir()
    if present and not dry:
        import shutil
        shutil.rmtree(root)
    return {"path": str(root), "existed": present, "removed": present and not dry, "dry": dry,
            "kept": "the x64 GPU venv and stills weights"}


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--rollback-plan", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.rollback or args.rollback_plan:
            print(json.dumps(rollback(args.model, dry=args.rollback_plan or not args.rollback)))
            return 0
        if args.download:
            print(json.dumps(download(args.model)))
            return 0
        print(json.dumps(plan(args.model)))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
