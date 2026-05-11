import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("run_openarmx_standalone_viewer.py")


def load_script_module():
    spec = importlib.util.spec_from_file_location("openarmx_standalone_viewer_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_standalone_sim_contains_openarmx_tabletop_cube_scene():
    module = load_script_module()

    sim, robot_model = module.build_standalone_sim()

    assert robot_model.__class__.__name__ == "OpenArmX"
    assert "robot0_openarmx_base" in sim.model.body_names
    assert "table" in sim.model.body_names
    assert "cube_main" in sim.model.body_names
    assert len(robot_model.joints) == 18


def test_zero_pose_preset_keeps_arms_at_zero_and_grippers_open():
    module = load_script_module()

    qpos = module.build_initial_qpos("zero")

    assert len(qpos) == 18
    assert all(value == 0.0 for value in qpos[:7])
    assert qpos[7] == 0.044
    assert qpos[8] == 0.044
    assert all(value == 0.0 for value in qpos[9:16])
    assert qpos[16] == 0.044
    assert qpos[17] == 0.044

def test_standalone_sim_places_robot_farther_from_table_center():
    module = load_script_module()

    sim, _ = module.build_standalone_sim()

    base_id = sim.model.body_name2id("robot0_openarmx_base")
    assert sim.data.body_xpos[base_id][0] <= -0.6
