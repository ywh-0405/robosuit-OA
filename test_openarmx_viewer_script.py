import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("run_openarmx_twoarmlift_viewer.py")


def load_script_module():
    spec = importlib.util.spec_from_file_location("openarmx_viewer_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_env_kwargs_targets_openarmx_viewer():
    module = load_script_module()

    kwargs = module.build_env_kwargs()

    assert kwargs["env_name"] == "TwoArmLift"
    assert kwargs["robots"] == "OpenArmX"
    assert kwargs["has_renderer"] is True
    assert kwargs["has_offscreen_renderer"] is False
    assert kwargs["use_camera_obs"] is False
