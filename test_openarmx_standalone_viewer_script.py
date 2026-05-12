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


def test_standalone_sim_adds_right_gripper_fingerpad_collision_geoms():
    module = load_script_module()

    sim, robot_model = module.build_standalone_sim()

    finger_names = module.find_gripper_finger_geom_names(robot_model, "right")
    pad_names = module.find_gripper_fingerpad_geom_names(robot_model, "right")
    assert all(name in sim.model.geom_names for name in finger_names)
    assert all(name in sim.model.geom_names for name in pad_names)
    assert not any("pinch_pad" in name for name in sim.model.geom_names)


def test_fingerpads_are_near_inner_contact_point_of_original_finger_meshes():
    module = load_script_module()
    sim, robot_model = module.build_standalone_sim()

    original_inner_point = module.gripper_inner_contact_point(sim, robot_model, "right")
    pad_center = module.gripper_fingerpad_center(sim, robot_model, "right")

    assert module.np.linalg.norm(pad_center - original_inner_point) < 0.025


def test_control_and_assisted_lift_use_centered_fingerpads():
    module = load_script_module()
    sim, robot_model = module.build_standalone_sim()

    control_point = module.gripper_control_point(sim, robot_model, "right")
    hold_pos = module.cube_hold_pos_from_tcp(sim, robot_model, "right")
    pad_center = module.gripper_fingerpad_center(sim, robot_model, "right")

    assert module.np.linalg.norm(control_point - pad_center) < 1e-9
    assert module.np.linalg.norm(hold_pos - pad_center) < 1e-9


def test_assisted_lift_quality_gate_accepts_centered_cube_between_fingerpads():
    module = load_script_module()
    sim, robot_model = module.build_standalone_sim()
    cube_joint_name = module.find_free_joint_name(sim, "cube")
    pad_center = module.gripper_fingerpad_center(sim, robot_model, "right")
    module.set_free_joint_pose(sim, cube_joint_name, pad_center)
    sim.forward()

    assert module.grasp_quality_ok_for_assisted_lift(sim, robot_model, "right", cube_joint_name)


def test_assisted_lift_quality_gate_rejects_side_offset_cube():
    module = load_script_module()
    sim, robot_model = module.build_standalone_sim()
    cube_joint_name = module.find_free_joint_name(sim, "cube")
    pad_center = module.gripper_fingerpad_center(sim, robot_model, "right")
    side_offset = pad_center + module.np.array([0.0, 0.05, 0.0])
    module.set_free_joint_pose(sim, cube_joint_name, side_offset)
    sim.forward()

    assert not module.grasp_quality_ok_for_assisted_lift(sim, robot_model, "right", cube_joint_name)


def test_contact_hold_freezes_cube_when_contact_pop_velocity_is_too_high():
    module = load_script_module()
    sim, _ = module.build_standalone_sim()
    cube_joint_name = module.find_free_joint_name(sim, "cube")
    hold_pos = module.free_joint_pos(sim, cube_joint_name)
    qvel_addr = sim.model.get_joint_qvel_addr(cube_joint_name)
    sim.data.qvel[qvel_addr[0] : qvel_addr[0] + 3] = module.np.array([0.0, 0.0, 1.0])

    _, held = module.apply_contact_hold_if_cube_pops(sim, cube_joint_name, hold_pos, max_speed=0.35)

    assert held is True
    assert module.np.linalg.norm(module.free_joint_pos(sim, cube_joint_name) - hold_pos) < 1e-9
    assert module.np.linalg.norm(module.free_joint_vel(sim, cube_joint_name)) < 1e-9


def test_contact_hold_leaves_slow_cube_dynamic():
    module = load_script_module()
    sim, _ = module.build_standalone_sim()
    cube_joint_name = module.find_free_joint_name(sim, "cube")
    hold_pos = module.free_joint_pos(sim, cube_joint_name)
    qvel_addr = sim.model.get_joint_qvel_addr(cube_joint_name)
    sim.data.qvel[qvel_addr[0] : qvel_addr[0] + 3] = module.np.array([0.0, 0.0, 0.1])

    _, held = module.apply_contact_hold_if_cube_pops(sim, cube_joint_name, hold_pos, max_speed=0.35)

    assert held is False
    assert module.np.linalg.norm(module.free_joint_vel(sim, cube_joint_name)) > 0.0


def test_visual_contact_hold_keeps_cube_at_recorded_contact_pose():
    module = load_script_module()
    sim, _ = module.build_standalone_sim()
    cube_joint_name = module.find_free_joint_name(sim, "cube")
    hold_pos = module.free_joint_pos(sim, cube_joint_name)
    displaced = hold_pos + module.np.array([0.0, 0.0, 0.05])
    module.set_free_joint_pose(sim, cube_joint_name, displaced)

    returned = module.apply_visual_contact_hold(sim, cube_joint_name, hold_pos)

    assert returned is True
    assert module.np.linalg.norm(module.free_joint_pos(sim, cube_joint_name) - hold_pos) < 1e-9
    assert module.np.linalg.norm(module.free_joint_vel(sim, cube_joint_name)) < 1e-9


def test_visual_center_grasp_puts_cube_center_at_fingerpad_center():
    module = load_script_module()
    sim, robot_model = module.build_standalone_sim()
    cube_joint_name = module.find_free_joint_name(sim, "cube")
    pad_center = module.gripper_fingerpad_center(sim, robot_model, "right")
    module.set_free_joint_pose(sim, cube_joint_name, pad_center + module.np.array([0.02, 0.0, 0.0]))

    moved = module.apply_visual_center_grasp(sim, robot_model, "right", cube_joint_name)

    assert moved is True
    assert module.np.linalg.norm(module.free_joint_pos(sim, cube_joint_name) - pad_center) < 1e-9
    assert module.np.linalg.norm(module.free_joint_vel(sim, cube_joint_name)) < 1e-9


def test_demo_cube_gripper_close_qpos_maps_total_inner_gap_to_single_finger_qpos():
    module = load_script_module()

    qpos = module.demo_cube_gripper_close_qpos(cube_size=(0.015, 0.015, 0.015))

    expected_inner_gap = 0.030 - module.GRASP_GRIPPER_GAP_MARGIN
    assert module.np.isclose(2.0 * qpos, expected_inner_gap)


def test_visual_grasp_mode_keeps_cube_centered_during_close_and_lift_when_enabled():
    module = load_script_module()
    sim, robot_model = module.build_standalone_sim()
    cube_joint_name = module.find_free_joint_name(sim, "cube")
    pad_center = module.gripper_fingerpad_center(sim, robot_model, "right")
    module.set_free_joint_pose(sim, cube_joint_name, pad_center + module.np.array([0.02, 0.0, 0.0]))

    hold_pos = module.apply_visual_grasp_mode(
        sim,
        robot_model,
        "right",
        cube_joint_name,
        state="close",
        assisted_lift=True,
        contact_hold_pos=None,
    )

    assert module.np.linalg.norm(module.free_joint_pos(sim, cube_joint_name) - pad_center) < 1e-9

    stale_contact_hold = pad_center + module.np.array([0.0, 0.0, -0.05])
    module.set_free_joint_pose(sim, cube_joint_name, stale_contact_hold)
    hold_pos = module.apply_visual_grasp_mode(
        sim,
        robot_model,
        "right",
        cube_joint_name,
        state="lift",
        assisted_lift=True,
        contact_hold_pos=stale_contact_hold,
    )

    assert module.np.linalg.norm(module.free_joint_pos(sim, cube_joint_name) - pad_center) < 1e-9
    assert module.np.linalg.norm(hold_pos - pad_center) < 1e-9


def test_pure_physics_visual_grasp_mode_leaves_cube_dynamic():
    module = load_script_module()
    sim, robot_model = module.build_standalone_sim()
    cube_joint_name = module.find_free_joint_name(sim, "cube")
    start_pos = module.free_joint_pos(sim, cube_joint_name)

    hold_pos = module.apply_visual_grasp_mode(
        sim,
        robot_model,
        "right",
        cube_joint_name,
        state="close",
        assisted_lift=False,
        contact_hold_pos=None,
    )

    assert hold_pos is None
    assert module.np.linalg.norm(module.free_joint_pos(sim, cube_joint_name) - start_pos) < 1e-9


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


def make_demo_for_state_tests(module, state, control_point):
    demo = module.IKGraspDemo.__new__(module.IKGraspDemo)
    demo.state = state
    demo.state_steps = 999
    demo.cube_start_pos = module.np.array([-0.30, -0.14, 0.815])
    demo.xy_target = demo.cube_start_pos.copy()
    demo.approach_height = module.DEFAULT_APPROACH_HEIGHT
    demo.grasp_height_offset = module.DEFAULT_GRASP_HEIGHT_OFFSET
    demo.lift_height = module.DEFAULT_LIFT_HEIGHT
    demo.lift_steps = module.DEFAULT_LIFT_STEPS
    demo.settle_steps = module.DEFAULT_SETTLE_STEPS
    demo.table_clearance = module.DEFAULT_TABLE_CLEARANCE
    demo.target_xmat = module.top_down_grasp_xmat()
    demo._test_control_point = module.np.array(control_point)
    return demo


def test_default_parser_uses_fast_pure_physics_small_object_grasp_settings():
    module = load_script_module()

    args = module.build_arg_parser().parse_args([])

    assert args.approach_height <= 0.10
    assert args.ik_iters >= 10
    assert args.ik_max_step >= 0.03
    assert args.orientation_weight == 0.0
    assert args.assisted_lift is True
    assert args.settle_steps >= 40


def test_no_assisted_lift_legacy_flag_keeps_visual_grasp_enabled_for_old_commands():
    module = load_script_module()

    args = module.build_arg_parser().parse_args(["--no-assisted-lift"])

    assert args.assisted_lift is True
    assert args.legacy_no_assisted_lift is True


def test_parser_can_disable_visual_grasp_for_pure_physics_debugging():
    module = load_script_module()

    args = module.build_arg_parser().parse_args(["--pure-physics"])

    assert args.assisted_lift is False


def test_ik_close_transitions_to_settle_before_lift():
    module = load_script_module()
    demo = make_demo_for_state_tests(module, "close", [-0.304, -0.143, 0.823])
    demo.state_steps = 0
    demo.close_steps = 3
    demo.settle_steps = 2

    demo.update_state(position_error=0.0, rotation_error=0.0, control_point=demo._test_control_point)
    assert demo.state == "close"

    demo.update_state(position_error=0.0, rotation_error=0.0, control_point=demo._test_control_point)
    assert demo.state == "close"

    demo.update_state(position_error=0.0, rotation_error=0.0, control_point=demo._test_control_point)
    assert demo.state == "close"

    demo.update_state(position_error=0.0, rotation_error=0.0, control_point=demo._test_control_point)
    assert demo.state == "settle"


def test_ik_settle_holds_grasp_before_lift():
    module = load_script_module()
    demo = make_demo_for_state_tests(module, "settle", [-0.304, -0.143, 0.823])
    demo.state_steps = 0
    demo.settle_steps = 2

    target, target_xmat, gripper = demo.target_and_gripper()

    assert module.np.isclose(target[2], demo.cube_start_pos[2] + demo.grasp_height_offset)
    assert target_xmat is demo.target_xmat
    assert gripper == module.demo_cube_gripper_close_qpos()

    demo.update_state(position_error=0.0, rotation_error=0.0, control_point=demo._test_control_point)
    assert demo.state == "settle"

    demo.update_state(position_error=0.0, rotation_error=0.0, control_point=demo._test_control_point)
    assert demo.state == "settle"

    demo.update_state(position_error=0.0, rotation_error=0.0, control_point=demo._test_control_point)
    assert demo.state == "lift"


def test_lift_target_ramps_up_from_grasp_height_instead_of_jumping():
    module = load_script_module()
    demo = make_demo_for_state_tests(module, "lift", [-0.304, -0.143, 0.823])
    demo.state_steps = 0

    first_target, _, first_gripper = demo.target_and_gripper()

    demo.state_steps = demo.lift_steps // 2
    mid_target, _, _ = demo.target_and_gripper()

    demo.state_steps = demo.lift_steps
    final_target, _, final_gripper = demo.target_and_gripper()

    grasp_z = demo.cube_start_pos[2] + demo.grasp_height_offset
    final_z = demo.cube_start_pos[2] + demo.lift_height
    assert module.np.isclose(first_target[2], grasp_z)
    assert grasp_z < mid_target[2] < final_z
    assert module.np.isclose(final_target[2], final_z)
    assert first_gripper == final_gripper == module.demo_cube_gripper_close_qpos()
