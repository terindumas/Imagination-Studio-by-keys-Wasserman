"""Windows x64 short video: LTX-Video and Wan, not FastH3. No GPU and no downloads."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import server
import windows_x64_stills as stills
import windows_x64_video as video
import windows_x64_video_generate as windows_x64_video_generate
from test_windows_ui import client  # noqa: F401


def _markers(root: Path, model: str) -> None:
    folder = root / "models" / model
    for name in video.MODELS[model]["markers"]:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}" if name.endswith(".json") else b"safetensors")
    python = root / "gpu-venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"")


def test_small_cards_are_told_h3_does_not_fit(monkeypatch):
    monkeypatch.delenv("STUDIO_X64_HOST", raising=False)
    monkeypatch.setenv("STUDIO_X64_HOST", "0")
    assert video.small_gpu() is False
    monkeypatch.setenv("STUDIO_X64_HOST", "1")
    monkeypatch.setenv("STUDIO_X64_VRAM_MB", "16384")
    assert video.small_gpu() is True
    monkeypatch.setenv("STUDIO_X64_VRAM_MB", "48000")
    assert video.small_gpu() is False
    assert "not supported" in video.H3_NOTE
    assert "gpu-venv" not in video.H3_NOTE
    assert "58" in video.H3_NOTE


def test_plan_and_frame_limits_and_rollback(tmp_path, monkeypatch):
    found = video.plan("auto")
    assert found["selected_model"] == "ltx"
    assert found["repo"] == "Lightricks/LTX-Video"
    assert found["disk_gb"] == 27 and found["token"] is False and found["gated"] is False
    assert "13b" not in " ".join(video.MODELS["ltx"]["allow"])
    assert video.plan("wan13")["disk_gb"] == 29
    assert video.plan("wan13")["repo"] == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    clip = video.parameters({"prompt": "a red apple on a white table", "video_model": "ltx"})
    assert (clip["width"], clip["height"], clip["frames"], clip["fps"]) == (768, 512, 97, 24)
    video.parameters({"prompt": "from a picture", "video_model": "ltx", "image": "uploads/apple.png", "frames": 121})
    with pytest.raises(ValueError, match="256 to 768"):
        video.parameters({"prompt": "too big", "video_model": "ltx", "width": 1024, "height": 512})
    with pytest.raises(ValueError, match="frames"):
        video.parameters({"prompt": "odd", "video_model": "ltx", "frames": 96})
    with pytest.raises(ValueError, match="text to video"):
        video.parameters({"prompt": "picture", "video_model": "wan13", "image": "uploads/apple.png"})
    with pytest.raises(ValueError, match="FastH3"):
        video.parameters({"prompt": "old", "video_model": "ltx", "quality": "fh8", "preset": "small"})
    short = video.memory_fit("ltx", host_free_mb=40 * 1024, gpu_free_mb=200,
                             holders=["sd-cuda.exe (PID 4321, 14.0 GB)"])
    assert short["fits"] is False and short["need_mb"] == 24576
    assert "sd-cuda.exe" in short["reason"] and "24 GB" in short["reason"] and "28" not in short["reason"]
    assert "gpu-venv" not in short["reason"]
    assert video.memory_fit("ltx", host_free_mb=40 * 1024, gpu_free_mb=10 * 1024, holders=[])["fits"] is True
    root = tmp_path / "ImaginationStudio" / "x64"
    stills_dir = root / "models" / "z-image-turbo"
    stills_dir.mkdir(parents=True)
    (stills_dir / "keep.txt").write_text("stills", encoding="utf-8")
    monkeypatch.setenv("STUDIO_X64_HOME", str(root))
    (video.model_dir("ltx")).mkdir(parents=True)
    (video.model_dir("ltx") / "weight.safetensors").write_bytes(b"x")
    assert video.rollback("ltx", dry=True)["existed"] is True
    assert video.rollback("ltx", dry=False)["removed"] is True
    assert not video.model_dir("ltx").exists()
    assert (stills_dir / "keep.txt").read_text(encoding="utf-8") == "stills"


def test_api_hides_h3_and_queues_ltx_when_the_weights_exist(client, monkeypatch, tmp_path):
    root = tmp_path / "ImaginationStudio" / "x64"
    _markers(root, "ltx")
    monkeypatch.setenv("STUDIO_X64_HOST", "1")
    monkeypatch.setenv("STUDIO_X64_HOME", str(root))
    monkeypatch.setenv("STUDIO_X64_VRAM_MB", "16384")
    monkeypatch.setattr(server, "jobs", {})
    monkeypatch.setattr(server, "order", [])
    monkeypatch.setattr(server, "wake", type("Wake", (), {"set": staticmethod(lambda: None)})())
    state = client.get("/api/state").json()
    assert state["x64_video"]["small"] is True and state["x64_video"]["h3"] is False
    assert "not supported" in state["x64_video"]["h3_note"]
    assert "gpu-venv" not in state["capabilities"]["video"].get("why", "")
    assert state["capabilities"]["video"]["available"] is True
    assert state["capabilities"]["video"]["model"] == "ltx"
    assert set(state["qualities"]) == {"ltx", "ltx121", "wan13"}
    assert all("FastH3" not in label for label in state["qualities"].values())
    assert "LTX-Video" in state["engine"]
    response = client.post("/api/jobs", json={"kind": "video", "title": "apple", "params": {
        "prompt": "a red apple on a white table", "video_model": "ltx"}})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["params"]["frames"] == 97 and body["params"]["video_model"] == "ltx"
    command, _env, outputs = video.build(body["params"], server.HOME, body["id"])
    assert command[0].endswith("python.exe") and outputs[0].endswith(".mp4")
    request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
    assert request["prompt"] == "a red apple on a white table"
    monkeypatch.setattr(stills, "host_free_mb", lambda: 40 * 1024)
    monkeypatch.setattr(stills, "gpu_free_mb", lambda: 200)
    monkeypatch.setattr(stills, "gpu_holders", lambda: ["sd-cuda.exe (PID 9, 15.0 GB)"])
    monkeypatch.setattr(server, "SPARK", False)
    refused = server.job_fit({"kind": "video", "params": body["params"]})
    assert refused["fits"] is False and "sd-cuda.exe" in refused["reason"] and "28" not in refused["reason"]


def _touch(root: Path, names: tuple[str, ...]) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")


def test_load_paths_match_the_downloaded_layout(tmp_path, monkeypatch):
    ltx = tmp_path / "ltx"
    _touch(ltx, ("model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
                 "text_encoder/model.safetensors.index.json", "tokenizer/tokenizer_config.json",
                 "tokenizer/spiece.model", "transformer/config.json", "vae/config.json",
                 "vae/diffusion_pytorch_model.safetensors", "ltxv-2b-0.9.6-distilled-04-25.safetensors"))
    wan = tmp_path / "wan13"
    _touch(wan, ("model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
                 "text_encoder/model.safetensors.index.json", "tokenizer/tokenizer_config.json",
                 "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors.index.json",
                 "vae/config.json", "vae/diffusion_pytorch_model.safetensors"))
    calls = []

    class Transformer:
        @classmethod
        def from_single_file(cls, path, **kwargs):
            calls.append(("single", path, kwargs))
            return "transformer"

    class TextPipe:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("ltx", path, kwargs))
            return "ltx-pipe"

    class ImagePipe:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("i2v", path, kwargs))
            return "i2v-pipe"

    class WanPipe:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("wan", path, kwargs))
            return "wan-pipe"

    class Diffusers:
        __version__ = "0.39.0"
        LTXVideoTransformer3DModel = Transformer
        LTXPipeline = TextPipe
        LTXImageToVideoPipeline = ImagePipe
        WanPipeline = WanPipe

    monkeypatch.setattr(windows_x64_video_generate, "diffusers_module", lambda: Diffusers)
    monkeypatch.setattr(windows_x64_video_generate, "require_tokenizer_deps", lambda import_module=None: None)
    assert windows_x64_video_generate.load_ltx(ltx, "bf16", None) == "ltx-pipe"
    kind, path, kwargs = calls[0]
    assert kind == "single"
    assert Path(path) == ltx / "ltxv-2b-0.9.6-distilled-04-25.safetensors"
    assert Path(kwargs["config"]) == ltx and kwargs["subfolder"] == "transformer"
    assert (Path(kwargs["config"]) / kwargs["subfolder"] / "config.json").is_file()
    assert kwargs["local_files_only"] is True
    pipe_path, pipe_kwargs = calls[1][1], calls[1][2]
    assert Path(pipe_path) == ltx and pipe_kwargs["transformer"] == "transformer"
    assert pipe_kwargs["local_files_only"] is True
    assert windows_x64_video_generate.load_ltx(ltx, "bf16", "frame.png") == "i2v-pipe"
    assert calls[-1][0] == "i2v" and calls[-1][2]["transformer"] == "transformer"
    assert windows_x64_video_generate.load_wan(wan, "bf16") == "wan-pipe"
    assert Path(calls[-1][1]) == wan and calls[-1][2]["local_files_only"] is True
    assert "transformer" not in calls[-1][2]


def test_ltx_loader_uses_the_transformer_class_not_automodel():
    source = Path(windows_x64_video_generate.__file__).read_text(encoding="utf-8")
    assert "AutoModel.from_single_file" not in source
    assert "LTXVideoTransformer3DModel.from_single_file" in source
    assert "LTXImageToVideoPipeline" in source
    diffusers = pytest.importorskip("diffusers")
    root = Path(diffusers.__file__).resolve().parent
    parts = [int(piece) for piece in diffusers.__version__.split(".") if piece.isdigit()]
    assert tuple(parts[:2]) >= (0, 36)
    single = (root / "loaders" / "single_file_model.py").read_text(encoding="utf-8")
    assert "LTXVideoTransformer3DModel" in single and "def from_single_file" in single
    assert "class LTXPipeline" in (root / "pipelines" / "ltx" / "pipeline_ltx.py").read_text(encoding="utf-8")
    assert "class LTXImageToVideoPipeline" in (root / "pipelines" / "ltx" / "pipeline_ltx_image2video.py").read_text(encoding="utf-8")
    assert "class WanPipeline" in (root / "pipelines" / "wan" / "pipeline_wan.py").read_text(encoding="utf-8")


def test_tokenizer_preflight_names_protobuf_and_tiktoken():
    def refuse(name):
        raise ImportError(name)
    assert windows_x64_video_generate.missing_tokenizer_deps(refuse) == ["sentencepiece", "protobuf", "tiktoken"]
    requirements = (Path(__file__).resolve().parents[1] / "app" / "requirements-windows-x64.txt").read_text(encoding="utf-8")
    assert "protobuf>=" in requirements and "tiktoken>=" in requirements and "sentencepiece>=" in requirements
    script = (Path(__file__).resolve().parents[1] / "scripts" / "setup-windows-x64-video.ps1").read_text(encoding="utf-8")
    assert "Invoke-Python @($GpuPython) @('-m', 'pip', 'install', '-r', $Requirements)" in script
    assert "& $GpuPython -m pip install" not in script
    assert "--check-tokenizer" in script


def test_missing_video_weights_name_the_installer_not_the_arm_venv(client, monkeypatch, tmp_path):
    root = tmp_path / "ImaginationStudio" / "x64"
    python = root / "gpu-venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    monkeypatch.setenv("STUDIO_X64_HOST", "1")
    monkeypatch.setenv("STUDIO_X64_HOME", str(root))
    monkeypatch.setenv("STUDIO_X64_VRAM_MB", "16384")
    response = client.post("/api/jobs", json={"kind": "video", "params": {"prompt": "a red apple on a white table"}})
    assert response.status_code == 501
    assert "setup-windows-x64-video.ps1" in response.json()["error"]
    assert "not supported" in response.json()["error"]
    assert "gpu-venv" not in response.json()["error"]
