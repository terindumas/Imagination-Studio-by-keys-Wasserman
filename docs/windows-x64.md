# Windows x64 stills (discrete NVIDIA, 16 GB)

This path is for **native Windows 11 AMD64** with a discrete NVIDIA GPU. The machine it was written for is a
GeForce RTX 4090 Laptop (16 GB), 64 GB of system RAM, and an i9-13900HX. It does not replace the
[Windows on Arm / RTX Spark](windows.md) installer. `scripts\setup-windows-images.ps1` still refuses anything
that is not native Windows ARM64, and `oneshot-setup-rtx-spark-n1x.ps1` stays the Arm path.

Stills go through the Studio queue (`POST /api/jobs`, kind `scout`). The generator is a subprocess of the
x64 GPU venv. It does not listen on a port. The page stays on **127.0.0.1:7870**. This installer does not
bind 8080 or 1420 and does not touch another drive.

## What fits this GPU

| Feature | On a 16 GB RTX 4090 Laptop |
|---|---|
| **Z-Image-Turbo** stills (default) | Yes. Apache-2.0, public safetensors, no Hugging Face account. About 33 GB on disk. The transformer is about 12 GB in bf16, so a 16 GB card keeps the text encoder in system RAM and runs one part at a time. 1024×1024, and 1344×768, fit the pixel cap. |
| **DreamShaper 8** stills (`-Model sd15`) | Yes. CreativeML OpenRAIL-M, fp16 safetensors, about 2.5 GB. 512 to 768 on a side. No safety checker is downloaded or loaded. |
| Qwen-Image-2.1, FLUX3-K, MOSS-Image | No. Those weights and the Arm ComfyUI recipe are far past 16 GB. |
| MiniMax H3 / FastH3 video | No. The int8 engine's measured pool requirement is 90 GB; a short 480p clip already peaks near 58 GB. |
| Fish Audio Drama 3, MiniMax Music 3 | No. Drama 3 is about 42 GB on a Spark. They are not installed by this script. |
| TensorFold H3 | Not used here. The page can still report it unavailable. That is accurate on this GPU. |

Weights are safetensors only. A `.ckpt`, `.pt`, `.pth`, `.bin` or pickle file in the model folder makes that
model unavailable.

## Install

From the checkout (for example `C:\ImaginationStudio`), after the UI venv already exists:

```powershell
cd C:\ImaginationStudio
powershell -ExecutionPolicy Bypass -File scripts\setup-windows-x64-images.ps1 -Model auto -PlanOnly
powershell -ExecutionPolicy Bypass -File scripts\setup-windows-x64-images.ps1 -Model auto
```

`-PlanOnly` prints the model, licence, and disk figure and does not download. `-Model auto` installs
Z-Image-Turbo. `-Model sd15` installs DreamShaper 8 instead. `-NoDownload` builds the GPU venv only.

The venv is `%LOCALAPPDATA%\ImaginationStudio\x64\gpu-venv`. It is Python 3.13, PyTorch
`2.9.1+cu128` from the official Windows AMD64 CUDA 12.8 index, and diffusers 0.36 or newer (the first
release with `ZImagePipeline`). An RTX 4090 (sm_89) runs that wheel. The installer stops before any weight
download if `torch.cuda.is_available()` is false.

Disk, roughly:

- GPU venv: 4 GB
- Z-Image-Turbo: 33 GB
- DreamShaper 8: 2.5 GB
- Auto (venv + Z-Image-Turbo): about 37 GB
- Both models: about 40 GB

## Launch and a generation test

Start the page the way you already do (tray, or the UI script). It listens on loopback only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\app.ps1
```

Open http://127.0.0.1:7870. Image only should list **Z-Image-Turbo**. A prompt such as
`a red apple on a white table` at 1024×1024 is the check.

The same job from PowerShell, with Studio already running:

```powershell
$body = @{
  kind = "scout"
  title = "apple"
  params = @{
    prompt = "a red apple on a white table"
    count = 1
    seed = 1
    width = 1024
    height = 1024
    image_model = "z-image-turbo"
  }
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:7870/api/jobs -ContentType "application/json" -Body $body
```

Admission checks two pools. Z-Image-Turbo needs about **16 GB of free system RAM** and about **10 GB of free GPU memory** (the transformer, with the text encoder left in RAM). DreamShaper 8 needs about 8 GB of RAM and 5 GB of VRAM. A low GPU does not make the PC look like it has 0 GB of RAM, and the Mac scout figure of 28 GB is not used. When the GPU is short, the refusal names the compute app holding it. Z-Image offloads whenever free VRAM is under 22 GB.

`/api/state` then reports `windows_stills.available: true`, the model in `image_models`, and `ui_only: false`.
The PNG lands under the Studio home's `scouts\<job id>\`. The first run spends most of its time loading
weights. Later runs reuse the process only for that job; the venv is not a resident server.

Use `-Model sd15` and `"image_model": "sd15"` with 512×512 for the smaller model.

## Rollback

This deletes the x64 venv and the weights that installer downloaded. It does not remove the UI venv, the
checkout, or anything outside `%LOCALAPPDATA%\ImaginationStudio\x64`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-windows-x64-images.ps1 -Rollback -PlanOnly
powershell -ExecutionPolicy Bypass -File scripts\setup-windows-x64-images.ps1 -Rollback
```

## What was tested where

Unit tests cover AMD64 detection, the plan (no network), rollback refusing any directory that is not
`ImaginationStudio\x64`, pickle refusal, and `/api/state` plus `POST /api/jobs` when a fake weight tree is
present. They do not load PyTorch and do not talk to Hugging Face.

A real image still has to be run on the Windows PC with the NVIDIA driver: the wheel install, CUDA
visibility, the download, and the apple prompt above.
