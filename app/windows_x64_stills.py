"""Windows x64 CUDA stills for a discrete NVIDIA GPU.

This is not the Windows on Arm / RTX Spark path. That installer and its SD-Turbo
storyboard stay as they are. Here a separate GPU venv runs PyTorch's Windows x64
CUDA wheel and Hugging Face diffusers, and writes PNG files. It does not open a
port. The Studio page on 127.0.0.1:7870 queues the job.

Weights are public safetensors. Nothing in this module downloads them and nothing
here reads a Hugging Face token.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys

MAX_COUNT = 6
TIMEOUT_S = 1800
AUTO = "z-image-turbo"
# A 16 GB laptop cannot hold Z-Image's text encoder and transformer on the GPU together.
# At or above this much *free* VRAM the generator keeps the whole pipeline on the GPU.
FULL_GPU_BYTES = 22 * (1024 ** 3)
# Admission is two pools. Host is system RAM (the offloaded text encoder lives there).
# VRAM is what one part needs on the card. These are not the Mac scout peak (28 GB).
HOST_NEED_MB = {"z-image-turbo": 16384, "sd15": 8192}
VRAM_NEED_MB = {"z-image-turbo": 10240, "sd15": 5120}

MODELS: dict[str, dict] = {
    "z-image-turbo": {
        "title": "Z-Image-Turbo",
        "note": "8 steps, 1024-class stills on a 16 GB NVIDIA GPU; Apache-2.0, no Hugging Face token",
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "kind": "zimage",
        "license": "Apache-2.0",
        "author": "Tongyi-MAI (Z-Image team)",
        "disk_gb": 33,
        "weights_bytes": 32_850_000_000,
        "default_width": 1024,
        "default_height": 1024,
        "min_side": 512,
        "max_side": 1344,
        "max_pixels": 1024 * 1024,
        "steps": 9,
        "guidance": 0.0,
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
    "sd15": {
        "title": "DreamShaper 8",
        "note": "SD1.5-class stills, 512–768; CreativeML OpenRAIL-M, no Hugging Face token",
        "repo": "Lykon/dreamshaper-8",
        "kind": "sd15",
        "license": "CreativeML OpenRAIL-M",
        "author": "Lykon",
        "disk_gb": 2.5,
        "weights_bytes": 2_500_000_000,
        "default_width": 512,
        "default_height": 512,
        "min_side": 512,
        "max_side": 768,
        "max_pixels": 768 * 768,
        "steps": 25,
        "guidance": 7.0,
        "markers": (
            "model_index.json",
            "text_encoder/model.fp16.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ),
        "allow": (
            "model_index.json",
            "scheduler/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "tokenizer/*",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ),
    },
}

SETUP = r"powershell -ExecutionPolicy Bypass -File scripts\setup-windows-x64-images.ps1 -Model auto"
VENV_GB = 4
REFUSED_SUFFIXES = {".ckpt", ".pt", ".pth", ".bin", ".pickle", ".pkl"}


def is_model(name: object) -> bool:
    return isinstance(name, str) and name in MODELS


def real_x64_host() -> bool:
    """The machine this process is actually running on, ignoring the test switch."""
    return sys.platform == "win32" and platform.machine().upper() in {"AMD64", "X86_64"}


def is_x64_host() -> bool:
    """Native Windows AMD64. STUDIO_X64_HOST is only a test switch."""
    forced = os.environ.get("STUDIO_X64_HOST", "").strip().lower()
    if forced in {"1", "true", "yes"}:
        return True
    if forced in {"0", "false", "no"}:
        return False
    return real_x64_host()


def home() -> Path:
    override = os.environ.get("STUDIO_X64_HOME", "").strip()
    if override:
        return Path(override)
    local = os.environ.get("LOCALAPPDATA", "").strip()
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    return base / "ImaginationStudio" / "x64"


def python_path() -> Path:
    return home() / "gpu-venv" / "Scripts" / "python.exe"


def model_dir(model: str) -> Path:
    return home() / "models" / model


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def model_ready(model: str) -> bool:
    root = model_dir(model)
    return all(_nonempty(root / name) for name in MODELS[model]["markers"])


def refused_weight(root: Path) -> str:
    """A pickle checkpoint beside the safetensors, which this backend will not load."""
    if not root.is_dir():
        return ""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in REFUSED_SUFFIXES:
            return path.name
    return ""


def model_why(model: str) -> str:
    if not is_model(model):
        return f"Unknown Windows x64 picture model {model}. Choose {', '.join(MODELS)}."
    if not is_x64_host():
        return ("Windows x64 CUDA stills run on native Windows AMD64 with an NVIDIA GPU. "
                "Windows on Arm keeps scripts\\setup-windows-images.ps1.")
    if not python_path().is_file():
        return f"The Windows x64 GPU environment is not installed. {SETUP}"
    bad = refused_weight(model_dir(model))
    if bad:
        return f"{bad} is not a safetensors weight. Remove it from {model_dir(model)} and install again."
    if not model_ready(model):
        flag = "auto" if model == AUTO else model
        return (f"{MODELS[model]['title']} weights are not installed. "
                rf"powershell -ExecutionPolicy Bypass -File scripts\setup-windows-x64-images.ps1 -Model {flag}")
    return ""


def option_rows() -> list[dict]:
    rows = []
    for model, spec in MODELS.items():
        why = model_why(model)
        ready = is_x64_host() and not why
        rows.append({
            "id": model,
            "title": spec["title"],
            "label": spec["title"],
            "note": spec["note"],
            "installed": ready,
            "qualified": False,
            "available": ready,
            "qualification_required": False,
            "state": "available" if ready else "not_installed",
            "why": "" if ready else why,
            "backend": "x64-cuda",
            "quality": "stills",
            "scope": (f"{spec['min_side']}–{spec['max_side']} px, multiples of 16, "
                      f"at most {spec['max_pixels']} pixels, 1–{MAX_COUNT} images."),
            "requirements": [
                {"pool": "device", "required_mb": VRAM_NEED_MB[model], "reserve_mb": 0, "basis": "estimated"},
                {"pool": "host", "required_mb": HOST_NEED_MB[model], "reserve_mb": 0, "basis": "estimated"},
            ],
            "setup": SETUP if model == AUTO else SETUP.replace("-Model auto", f"-Model {model}"),
            "check_available": False,
            "measurement_source": "Author guidance and weight size. Not a measured peak on this GPU.",
            "license": spec["license"],
            "repo": spec["repo"],
            "width": spec["default_width"],
            "height": spec["default_height"],
            "max_count": MAX_COUNT,
            "model": model,
        })
    return rows


def ready_models() -> list[str]:
    return [row["id"] for row in option_rows() if row["available"]]


def status() -> dict:
    ready = ready_models()
    chosen = ready[0] if ready else AUTO
    spec = MODELS[chosen]
    available = bool(ready)
    why = "" if available else model_why(chosen)
    return {
        "available": available,
        "why": why,
        "installed": available,
        "qualified": False,
        "qualification_required": False,
        "runtime_updated": False,
        "models": ready,
        "model": chosen,
        "width": spec["default_width"],
        "height": spec["default_height"],
        "max_count": MAX_COUNT,
        "backend": {"running": False, "loaded": False, "native": False, "kind": "x64-cuda",
                    "address": "", "why": why, "busy": False},
        "engine": f"CUDA · {spec['title']} (Windows x64)" if available else "CUDA · Windows x64 stills (not installed)",
        "setup": SETUP,
        "note": ("Windows x64 CUDA stills. MiniMax H3, FastH3, Qwen-Image, FLUX3-K, Fish Audio Drama 3 "
                 "and MiniMax Music 3 do not fit a 16 GB GPU."),
    }


def overlay(found: dict) -> dict:
    """The stills card. An installed x64 backend is that card. Otherwise, on Windows x64, say how to install it."""
    if not is_x64_host():
        return found
    ours = status()
    if ours["available"]:
        return ours
    if not found.get("available"):
        return {**found, "why": ours["why"], "setup": ours["setup"], "backend_kind": "x64-cuda"}
    return found


def parameters(params: dict) -> dict:
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    allowed = {"prompt", "prompts", "seeds", "count", "seed", "width", "height", "image_model", "preset",
               "suite_import", "suite_shot", "suite_shots"}
    extra = set(params) - allowed
    if extra:
        raise ValueError("unsupported Windows x64 still settings: " + ", ".join(sorted(extra)))
    model = params.get("image_model") or AUTO
    if model not in MODELS:
        raise ValueError(f"unknown Windows x64 picture model {model}: one of {', '.join(MODELS)}")
    spec = MODELS[model]
    batch = "prompts" in params
    prompts = params["prompts"] if batch else [params.get("prompt")]
    if (not isinstance(prompts, list) or not 1 <= len(prompts) <= MAX_COUNT
            or any(not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20000 for prompt in prompts)):
        raise ValueError(f"provide 1 to {MAX_COUNT} prompts, each of 1 to 20,000 characters")
    if batch and "prompt" in params:
        raise ValueError("use prompt or prompts, not both")
    out = {"prompts" if batch else "prompt": prompts if batch else prompts[0].strip(),
           "count": params.get("count", len(prompts) if batch else 1),
           "seed": params.get("seed", 1),
           "width": params.get("width", spec["default_width"]),
           "height": params.get("height", spec["default_height"]),
           "image_model": model}
    if any(type(out[key]) is not int for key in ("count", "seed", "width", "height")):
        raise ValueError("count, seed, width and height must be whole numbers")
    if not 1 <= out["count"] <= MAX_COUNT or not 0 <= out["seed"] <= 2 ** 32 - MAX_COUNT:
        raise ValueError(f"Windows x64 stills take 1 to {MAX_COUNT} images and a seed from 0 to {2 ** 32 - MAX_COUNT}")
    if batch and out["count"] != len(prompts):
        raise ValueError("count must match the number of prompts")
    if "seeds" in params:
        seeds = params["seeds"]
        if (not batch or not isinstance(seeds, list) or len(seeds) != len(prompts)
                or any(type(seed) is not int or not 0 <= seed <= 2 ** 32 - 1 for seed in seeds)):
            raise ValueError("seeds must give one unsigned 32-bit whole number per batched prompt")
        out["seeds"] = list(seeds)
    width, height = out["width"], out["height"]
    if width % 16 or height % 16:
        raise ValueError("image sides are multiples of 16")
    if not spec["min_side"] <= width <= spec["max_side"] or not spec["min_side"] <= height <= spec["max_side"]:
        raise ValueError(f"{spec['title']} sides are {spec['min_side']} to {spec['max_side']}")
    if width * height > spec["max_pixels"]:
        raise ValueError(f"{spec['title']} keeps the picture at or under {spec['max_pixels']} pixels "
                         f"so it fits a 16 GB GPU ({width}×{height} is {width * height})")
    for key in ("suite_import", "suite_shot"):
        if key in params:
            if not isinstance(params[key], str) or not params[key] or len(params[key]) > 64:
                raise ValueError(f"{key} must be an import/shot id")
            out[key] = params[key]
    if "suite_shots" in params:
        ids = params["suite_shots"]
        if (not batch or "suite_shot" in out or not isinstance(ids, list) or len(ids) != len(prompts)
                or any(not isinstance(one, str) or not one or len(one) > 64 for one in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("suite_shots must give one distinct import shot id per batched prompt")
        out["suite_shots"] = list(ids)
    if ("suite_import" in out) != ("suite_shot" in out or "suite_shots" in out):
        raise ValueError("suite_import and its shot id(s) are supplied together")
    if batch and "suite_shot" in out:
        raise ValueError("batched prompts use suite_shots, not suite_shot")
    return out


def image_plan(params: dict) -> list[tuple[str, int, str]]:
    batch = "prompts" in params
    prompts = params["prompts"] if batch else [params["prompt"]] * params["count"]
    seeds = params.get("seeds", [params["seed"] + i for i in range(params["count"])])
    return [(prompt, seed, f"scout_s{seed}.png") for prompt, seed in zip(prompts, seeds, strict=True)]


def x64_still_job(job: dict) -> bool:
    """A queued still on this Windows x64 CUDA path, not a Mac scout and not SD-Turbo."""
    params = job.get("params") or {}
    return job.get("kind") == "scout" and is_x64_host() and is_model(params.get("image_model"))


def should_offload(kind: str, free_bytes: int) -> bool:
    """Z-Image moves one part onto the GPU when free VRAM is under a full resident load."""
    return kind == "zimage" and free_bytes < FULL_GPU_BYTES


def parse_compute_apps(text: str) -> list[tuple[int, str, int]]:
    """nvidia-smi compute-app rows: pid, process name, used MiB."""
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit() or not parts[2].isdigit():
            continue
        rows.append((int(parts[0]), parts[1], int(parts[2])))
    return rows


def describe_holders(rows: list[tuple[int, str, int]]) -> list[str]:
    large = sorted((row for row in rows if row[2] >= 1024), key=lambda row: -row[2])
    return [f"{name or 'unknown'} (PID {pid}, {mb / 1024:.1f} GB)" for pid, name, mb in large[:3]]


def host_free_mb() -> int | None:
    """Free system RAM in MiB. Not GPU memory."""
    if sys.platform == "win32":
        try:
            import windows_memory
            return int(windows_memory.host_memory()[1])
        except (OSError, RuntimeError, ValueError):
            return None
    return None


def gpu_free_mb() -> int | None:
    """Free VRAM in MiB on the studio card, or None when nvidia-smi has no discrete card."""
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
    return int(card.get("head_free_mb", card["free_mb"]))


def gpu_holders() -> list[str]:
    """Names of compute apps using at least 1 GB of VRAM. Empty when nvidia-smi does not answer."""
    try:
        said = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=0x08000000 if sys.platform == "win32" else 0)
    except (OSError, subprocess.SubprocessError):
        return []
    if said.returncode != 0:
        return []
    return describe_holders(parse_compute_apps(said.stdout))


def memory_fit(model: str, *, host_free_mb: int | None, gpu_free_mb: int | None,
               holders: list[str] | None = None) -> dict:
    """Host RAM and VRAM checked apart. A low GPU does not pretend the PC has 0 GB of RAM, and the
    Mac scout peak is not the requirement."""
    if model not in MODELS:
        raise ValueError(f"unknown Windows x64 picture model {model}")
    title = MODELS[model]["title"]
    host_need = HOST_NEED_MB[model]
    vram_need = VRAM_NEED_MB[model]
    host_ok = host_free_mb is not None and host_free_mb >= host_need
    vram_known = gpu_free_mb is not None
    offload = MODELS[model]["kind"] == "zimage" and (not vram_known or gpu_free_mb * 1024 * 1024 < FULL_GPU_BYTES)
    vram_ok = (not vram_known) or gpu_free_mb >= vram_need
    parts = []
    if not (host_ok and vram_ok):
        if host_free_mb is None:
            parts.append(f"{title} needs about {host_need / 1024:.0f} GB of free system RAM and that figure could not be read.")
        else:
            parts.append(f"{title} needs about {host_need / 1024:.0f} GB of free system RAM and this PC has {host_free_mb / 1024:.0f} GB.")
    if vram_known and not vram_ok:
        how = "The text encoder stays in system RAM and one part moves onto the GPU." if offload else "The model stays on the GPU."
        parts.append(f"It needs about {vram_need / 1024:.0f} GB of free GPU memory and this GPU has {gpu_free_mb / 1024:.1f} GB free. {how}")
        named = holders or []
        if named:
            parts.append("Holding that GPU memory: " + "; ".join(named) + ". The Studio does not stop that program.")
        else:
            parts.append("No GPU process of at least 1 GB was named. Check Task Manager's Dedicated GPU memory column.")
    return {"fits": host_ok and vram_ok, "need_mb": host_need,
            "available_mb": 0 if host_free_mb is None else host_free_mb, "floor_mb": 0,
            "vram_need_mb": vram_need, "vram_free_mb": gpu_free_mb, "offload": offload and vram_ok,
            "reason": " ".join(parts)}


def admission(params: dict) -> dict:
    model = (params or {}).get("image_model") or AUTO
    gpu = gpu_free_mb()
    return memory_fit(model, host_free_mb=host_free_mb(), gpu_free_mb=gpu,
                      holders=gpu_holders() if gpu is not None else [])


def build(params: dict, home_dir: Path, job_id: str) -> tuple[list[str], dict, list[str]]:
    params = parameters(params)
    why = model_why(params["image_model"])
    if why:
        raise RuntimeError(why)
    target = Path(home_dir) / "scouts" / job_id
    target.mkdir(parents=True, exist_ok=True)
    request = {
        "model": params["image_model"],
        "model_dir": str(model_dir(params["image_model"])),
        "images": [{"prompt": prompt, "seed": seed, "name": name} for prompt, seed, name in image_plan(params)],
        "width": params["width"],
        "height": params["height"],
        "output": str(target),
    }
    path = target / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    script = Path(__file__).with_name("windows_x64_generate.py")
    command = [str(python_path()), "-X", "utf8", str(script), "--request", str(path)]
    outputs = [str(target / name) for _, _, name in image_plan(params)] + [str(target / "sheet.jpg")]
    return command, {}, outputs
