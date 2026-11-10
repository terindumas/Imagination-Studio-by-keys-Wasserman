"""Windows x64 CUDA stills: detection, admission and rollback. No GPU, no downloads."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import memory_manager
import server
import windows_runtime
import windows_x64_generate as generate
import windows_x64_setup as setup
import windows_x64_stills as stills
from test_windows_ui import client  # noqa: F401


def _markers(root: Path, model: str, *, pickle: bool = False) -> None:
    folder = root / "models" / model
    for name in stills.MODELS[model]["markers"]:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}" if name.endswith(".json") else b"safetensors")
    python = root / "gpu-venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"")
    if pickle:
        (folder / "weights.ckpt").write_bytes(b"no")


def test_amd64_is_the_only_windows_host_this_backend_claims(monkeypatch):
    monkeypatch.delenv("STUDIO_X64_HOST", raising=False)
    assert stills.real_x64_host() is False or sys.platform == "win32"
    monkeypatch.setattr(stills.sys, "platform", "win32")
    monkeypatch.setattr(stills.platform, "machine", lambda: "AMD64")
    assert stills.real_x64_host()
    monkeypatch.setattr(stills.platform, "machine", lambda: "x86_64")
    assert stills.real_x64_host()
    monkeypatch.setattr(stills.platform, "machine", lambda: "ARM64")
    assert not stills.real_x64_host()
    monkeypatch.setattr(stills.sys, "platform", "linux")
    monkeypatch.setattr(stills.platform, "machine", lambda: "AMD64")
    assert not stills.real_x64_host()
    monkeypatch.setenv("STUDIO_X64_HOST", "1")
    assert stills.is_x64_host()
    monkeypatch.setenv("STUDIO_X64_HOST", "0")
    assert not stills.is_x64_host()


def test_auto_plan_selects_z_image_without_a_token_or_a_download():
    found = setup.plan("auto")
    assert found["selected_model"] == "z-image-turbo"
    assert found["repo"] == "Tongyi-MAI/Z-Image-Turbo"
    assert found["license"] == "Apache-2.0"
    assert found["token"] is False and found["gated"] is False
    assert found["disk_gb"] == 37
    assert found["ports_opened"] == []
    assert found["listen"] == "127.0.0.1:7870"
    assert "sd15" in found["alternatives"]
    small = setup.plan("sd15")
    assert small["repo"] == "Lykon/dreamshaper-8"
    assert small["disk_gb"] == 6.5
    assert small["license"] == "CreativeML OpenRAIL-M"
    for model in stills.MODELS.values():
        assert all(not pattern.endswith((".ckpt", ".pt", ".pth", ".bin")) for pattern in model["allow"])
    assert any(pattern.startswith("*.ckpt") for pattern in setup.IGNORE)
    with pytest.raises(RuntimeError, match="AMD64"):
        setup.assert_install_host()


def test_canvas_limits_keep_a_16_gb_card_and_refuse_pickle(tmp_path, monkeypatch):
    root = tmp_path / "ImaginationStudio" / "x64"
    monkeypatch.setenv("STUDIO_X64_HOST", "1")
    monkeypatch.setenv("STUDIO_X64_HOME", str(root))
    _markers(root, "z-image-turbo")
    params = stills.parameters({"prompt": "a red apple on a white table", "width": 1344, "height": 768})
    assert params["image_model"] == "z-image-turbo"
    assert params["width"] == 1344 and params["height"] == 768
    with pytest.raises(ValueError, match="1344"):
        stills.parameters({"prompt": "too big", "image_model": "z-image-turbo", "width": 1664, "height": 928})
    with pytest.raises(ValueError, match="pixels"):
        stills.parameters({"prompt": "too big", "image_model": "z-image-turbo", "width": 1344, "height": 1024})
    with pytest.raises(ValueError, match="768"):
        stills.parameters({"prompt": "too big", "image_model": "sd15", "width": 1024, "height": 1024})
    with pytest.raises(ValueError, match="multiples of 16"):
        stills.parameters({"prompt": "odd", "width": 1000, "height": 1000})
    _markers(root, "z-image-turbo", pickle=True)
    assert not stills.status()["available"]
    assert "safetensors" in stills.model_why("z-image-turbo")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "model": "z-image-turbo", "model_dir": str(stills.model_dir("z-image-turbo")),
        "images": [{"prompt": "a red apple on a white table", "seed": 1, "name": "scout_s1.png"}],
        "width": 1024, "height": 1024,
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="safetensors"):
        generate.load_request(request)


def test_rollback_removes_only_the_x64_install_directory(tmp_path, monkeypatch):
    root = tmp_path / "ImaginationStudio" / "x64"
    kept = tmp_path / "ImaginationStudio" / "ui-venv"
    root.mkdir(parents=True)
    kept.mkdir()
    (root / "gpu-venv").mkdir()
    (root / "weights.bin").write_bytes(b"no")
    monkeypatch.setenv("STUDIO_X64_HOME", str(root))
    preview = setup.rollback(dry=True)
    assert preview["dry"] and preview["existed"] and root.is_dir()
    removed = setup.rollback(dry=False)
    assert removed["removed"] and not root.exists() and kept.is_dir()
    with pytest.raises(RuntimeError, match="ImaginationStudio"):
        setup.safe_root(tmp_path / "elsewhere" / "x64")


def test_missing_backend_names_the_x64_installer_and_keeps_the_ui_only(client, monkeypatch, tmp_path):
    monkeypatch.setenv("STUDIO_X64_HOST", "1")
    monkeypatch.setenv("STUDIO_X64_HOME", str(tmp_path / "ImaginationStudio" / "x64"))
    state = client.get("/api/state").json()
    assert state["ui_only"] is True
    assert state["image_models"] == []
    assert state["windows_stills"]["available"] is False
    assert "setup-windows-x64-images.ps1" in state["windows_stills"]["why"]
    response = client.post("/api/jobs", json={"kind": "scout", "params": {
        "prompt": "a red apple on a white table", "image_model": "z-image-turbo"}})
    assert response.status_code == 501
    assert "setup-windows-x64-images.ps1" in response.json()["error"]
    assert client.post("/api/jobs", json={"kind": "video", "params": {"prompt": "a clip"}}).status_code == 501


def test_installed_backend_reports_stills_and_queues_the_apple(client, monkeypatch, tmp_path):
    root = tmp_path / "ImaginationStudio" / "x64"
    _markers(root, "z-image-turbo")
    monkeypatch.setenv("STUDIO_X64_HOST", "1")
    monkeypatch.setenv("STUDIO_X64_HOME", str(root))
    monkeypatch.setattr(server, "jobs", {})
    monkeypatch.setattr(server, "order", [])
    monkeypatch.setattr(server, "wake", type("Wake", (), {"set": staticmethod(lambda: None), "wait": staticmethod(lambda: None)})())
    state = client.get("/api/state").json()
    assert state["windows_stills"]["available"] is True
    assert state["image_models"][0]["id"] == "z-image-turbo"
    assert state["ui_only"] is False
    assert "16 GB" in state["platform_note"]
    assert "CUDA" in state["engine"]
    assert windows_runtime.bounded("scout", {"image_model": "z-image-turbo"})
    assert windows_runtime.bounded("scout", {"image_model": "sd-turbo"})
    assert not windows_runtime.bounded("scout", {"image_model": "qwen"})
    response = client.post("/api/jobs", json={"kind": "scout", "title": "apple", "params": {
        "prompt": "a red apple on a white table", "count": 1, "seed": 1, "width": 1024, "height": 1024}})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["params"]["image_model"] == "z-image-turbo"
    assert body["params"]["prompt"] == "a red apple on a white table"
    assert body["params"]["width"] == 1024 and body["params"]["height"] == 1024
    command, env, outputs = stills.build(body["params"], server.HOME, body["id"])
    assert command[0].endswith("python.exe") and command[-1].endswith("request.json")
    assert env == {}
    request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
    assert request["images"][0]["prompt"] == "a red apple on a white table"
    assert [Path(path).name for path in outputs] == ["scout_s1.png", "sheet.jpg"]
    assert client.post("/api/jobs", json={"kind": "video", "params": {"prompt": "a clip"}}).status_code == 501


def test_x64_admission_uses_host_ram_and_names_the_gpu_holder(monkeypatch):
    rows = stills.parse_compute_apps("4321, sd-cuda.exe, 14336\n9, [N/A], 100\n")
    assert stills.describe_holders(rows) == ["sd-cuda.exe (PID 4321, 14.0 GB)"]
    short = stills.memory_fit("z-image-turbo", host_free_mb=30 * 1024, gpu_free_mb=200, holders=stills.describe_holders(rows))
    assert short["fits"] is False
    assert short["need_mb"] == 16384
    assert short["available_mb"] == 30 * 1024
    assert "16 GB" in short["reason"] and "sd-cuda.exe" in short["reason"]
    assert "28" not in short["reason"] and "0.2 GB" in short["reason"]
    fits = stills.memory_fit("z-image-turbo", host_free_mb=30 * 1024, gpu_free_mb=14 * 1024, holders=[])
    assert fits["fits"] is True and fits["offload"] is True and fits["need_mb"] == 16384
    small = stills.memory_fit("sd15", host_free_mb=10 * 1024, gpu_free_mb=6 * 1024, holders=[])
    assert small["fits"] is True and small["offload"] is False
    low_ram = stills.memory_fit("z-image-turbo", host_free_mb=12 * 1024, gpu_free_mb=14 * 1024, holders=[])
    assert low_ram["fits"] is False and "16 GB" in low_ram["reason"] and "28" not in low_ram["reason"]
    assert stills.should_offload("zimage", 12 * (1024 ** 3))
    assert not stills.should_offload("zimage", 23 * (1024 ** 3))
    assert not stills.should_offload("sd15", 4 * (1024 ** 3))
    assert server.mac_peak_gb({"kind": "scout", "params": {}}) == 28.0

    monkeypatch.setattr(server, "SPARK", False)
    monkeypatch.setattr(server, "WINDOWS", True)
    monkeypatch.setenv("STUDIO_X64_HOST", "1")
    monkeypatch.setattr(stills, "host_free_mb", lambda: 30 * 1024)
    monkeypatch.setattr(stills, "gpu_free_mb", lambda: 200)
    monkeypatch.setattr(stills, "gpu_holders", lambda: ["sd-cuda.exe (PID 4321, 14.0 GB)"])
    job = {"kind": "scout", "params": {"image_model": "z-image-turbo", "prompt": "a red apple on a white table"}}
    refused = server.job_fit(job)
    assert refused["fits"] is False and refused["need_mb"] == 16384
    assert "sd-cuda.exe" in refused["reason"] and "28" not in refused["reason"]
    fits_bolt, why = server.bolt_need(job, 8)
    assert fits_bolt is False and "sd-cuda.exe" in why and "28" not in why
    monkeypatch.setattr(stills, "gpu_free_mb", lambda: 14 * 1024)
    monkeypatch.setattr(stills, "gpu_holders", lambda: [])
    assert server.job_fit(job)["fits"] is True
    assert server.bolt_need(job, 8)[0] is True

    monkeypatch.setenv("STUDIO_MEMORY_MANAGER", "on")
    monkeypatch.setattr(memory_manager, "foreign", lambda *args, **kwargs: [])
    monkeypatch.setattr(memory_manager.time, "sleep", lambda _seconds: None)
    queued = {"id": "j", "kind": "scout", "status": "queued", "params": job["params"]}
    with pytest.raises(RuntimeError, match="sd-cuda.exe") as error:
        memory_manager.admit(queued, lambda: short, wait_s=0)
    assert "28" not in str(error.value)
    assert "16 GB" in str(error.value)


def test_generator_does_not_listen_and_the_arm_installer_stays_arm_only():
    source = Path(generate.__file__).read_text(encoding="utf-8")
    installer = Path(setup.__file__).read_text(encoding="utf-8")
    script = (Path(__file__).resolve().parents[1] / "scripts" / "setup-windows-x64-images.ps1").read_text(encoding="utf-8")
    arm = (Path(__file__).resolve().parents[1] / "scripts" / "setup-windows-images.ps1").read_text(encoding="utf-8")
    for text in (source, installer, script):
        assert "0.0.0.0" not in text
        assert "socket" not in text
    assert "8080" not in source and "1420" not in source
    assert "native Windows ARM64" in arm
    assert "AMD64" in script and "setup-windows-images.ps1" in script
