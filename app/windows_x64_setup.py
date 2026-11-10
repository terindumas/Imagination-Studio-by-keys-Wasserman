"""Plan, download and remove the Windows x64 CUDA stills install.

Planning does not touch the network or the GPU. Downloading refuses every host
that is not native Windows AMD64, and it refuses pickle checkpoints. The files
land under %LOCALAPPDATA%\\ImaginationStudio\\x64, which is removed only by the
rollback below.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import windows_x64_stills as stills

IGNORE = ("*.ckpt", "*.pt", "*.pth", "*.bin", "*.pickle", "*.pkl", "assets/*",
          "safety_checker/*", "feature_extractor/*")


def plan(model: str = "auto") -> dict:
    if model != "auto" and model not in stills.MODELS:
        raise ValueError(f"Unknown image model {model}. Choose auto, {', '.join(stills.MODELS)}.")
    selected = stills.AUTO if model == "auto" else model
    spec = stills.MODELS[selected]
    others = [name for name in stills.MODELS if name != selected]
    return {
        "requested_model": model,
        "selected_model": selected,
        "title": spec["title"],
        "repo": spec["repo"],
        "license": spec["license"],
        "author": spec["author"],
        "kind": spec["kind"],
        "why": ("Auto installs Z-Image-Turbo: an Apache-2.0 6B stills model that its authors run on a 16 GB GPU. "
                "DreamShaper 8 is the smaller SD1.5-class alternative (-Model sd15)."
                if model == "auto" else "Explicit model retained."),
        "disk_gb": spec["disk_gb"] + stills.VENV_GB,
        "weights_gb": spec["disk_gb"],
        "venv_gb": stills.VENV_GB,
        "fit": spec["note"],
        "format": "safetensors only",
        "token": False,
        "gated": False,
        "host_required": "native Windows AMD64 (x64)",
        "install_allowed_on_this_process": stills.real_x64_host(),
        "listen": "127.0.0.1:7870",
        "ports_opened": [],
        "home": str(stills.home()),
        "alternatives": others,
        "notices": notices(selected),
        "rollback": r"powershell -ExecutionPolicy Bypass -File scripts\setup-windows-x64-images.ps1 -Rollback",
    }


def notices(model: str) -> list[str]:
    spec = stills.MODELS[model]
    shared = [
        "The GPU venv is %LOCALAPPDATA%\\ImaginationStudio\\x64\\gpu-venv (about 4 GB: PyTorch cu128, diffusers).",
        "The installer does not open a port. Studio stays on 127.0.0.1:7870. Ports 8080 and 1420 are not used.",
        "Rollback removes only %LOCALAPPDATA%\\ImaginationStudio\\x64.",
        "MiniMax H3, FastH3, Qwen-Image, FLUX3-K, Fish Audio Drama 3 and MiniMax Music 3 do not fit 16 GB.",
    ]
    if model == "z-image-turbo":
        own = [f"Z-Image-Turbo from {spec['repo']}: about {spec['disk_gb']} GB of safetensors, Apache-2.0. "
               "No Hugging Face account. On a 16 GB card the text encoder stays in system RAM and one part is on the GPU at a time."]
    else:
        own = [f"DreamShaper 8 from {spec['repo']}: about {spec['disk_gb']} GB of fp16 safetensors, "
               "CreativeML OpenRAIL-M. No safety checker is downloaded or loaded. 512 to 768 on a side."]
    return own + shared


def assert_install_host() -> None:
    if os.environ.get("STUDIO_X64_ALLOW_HOST", "").strip() == "1":
        return
    if not stills.real_x64_host():
        raise RuntimeError("This CUDA stills installer requires native Windows AMD64 (x64). "
                           "Windows on Arm uses scripts\\setup-windows-images.ps1. Nothing was downloaded.")


def safe_root(path: Path | None = None) -> Path:
    root = (path or stills.home()).resolve()
    if root.name != "x64" or root.parent.name != "ImaginationStudio":
        raise RuntimeError(f"Refusing to change {root}. The x64 install must be ImaginationStudio\\x64.")
    if root == Path(root.anchor) or len(root.parts) < 4:
        raise RuntimeError(f"Refusing to change {root}.")
    return root


def download(model: str) -> dict:
    assert_install_host()
    selected = stills.AUTO if model == "auto" else model
    if selected not in stills.MODELS:
        raise ValueError(f"Unknown image model {model}.")
    spec = stills.MODELS[selected]
    dest = stills.model_dir(selected)
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(spec["repo"], local_dir=str(dest), allow_patterns=list(spec["allow"]),
                      ignore_patterns=list(IGNORE), token=False)
    bad = stills.refused_weight(dest)
    if bad:
        raise RuntimeError(f"Download left a non-safetensors weight ({bad}). It was not loaded.")
    if not stills.model_ready(selected):
        raise RuntimeError(f"{spec['title']} did not leave a complete safetensors pipeline in {dest}.")
    receipt = safe_root() / "install.json"
    previous = {}
    if receipt.is_file():
        try:
            found = json.loads(receipt.read_text(encoding="utf-8"))
            previous = found if isinstance(found, dict) else {}
        except (OSError, ValueError):
            previous = {}
    models = sorted(set(previous.get("models") or []) | {selected})
    payload = {"version": 1, "backend": "diffusers", "arch": "AMD64", "python": str(stills.python_path()),
               "models": models, "listen": "127.0.0.1", "ports_opened": [], "license": {selected: spec["license"]}}
    receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"model": selected, "dir": str(dest), "receipt": str(receipt)}


def rollback(*, dry: bool = False) -> dict:
    root = safe_root()
    present = root.is_dir()
    if present and not dry:
        shutil.rmtree(root)
    return {"path": str(root), "existed": present, "removed": present and not dry, "dry": dry}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--rollback-plan", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.rollback or args.rollback_plan:
            print(json.dumps(rollback(dry=args.rollback_plan or not args.rollback)))
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
