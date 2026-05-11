import argparse
import time

import mujoco
import numpy as np
from mujoco import viewer

from robosuite.models.arenas import TableArena
from robosuite.models.bases import robot_base_factory
from robosuite.models.grippers import gripper_factory
from robosuite.models.objects import BoxObject
from robosuite.models.robots import create_robot
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.binding_utils import MjSim
from robosuite.utils.mjcf_utils import array_to_string


DEFAULT_FREE_CAM = {
    "lookat": [0.0, 0.0, 0.85],
    "distance": 1.8,
    "azimuth": 180,
    "elevation": -20,
}
TABLE_FULL_SIZE = (0.8, 0.8, 0.05)
TABLE_OFFSET = (0.0, 0.0, 0.8)
DEFAULT_BASE_X = -0.62
DEFAULT_BASE_Z = 0.40
DEFAULT_CUBE_SIZE = (0.015, 0.015, 0.015)
DEFAULT_CUBE_POS = (-0.30, -0.14, TABLE_OFFSET[2] + DEFAULT_CUBE_SIZE[2])
OPEN_GRIPPER_QPOS = 0.044
CLOSED_GRIPPER_QPOS = 0.0
DEFAULT_CLOSE_AFTER = 1.5
DEFAULT_CLOSE_SECONDS = 2.0
DEFAULT_CUBE_FORWARD_OFFSET = 0.0
DEFAULT_CUBE_Z_OFFSET = 0.0
DEFAULT_APPROACH_HEIGHT = 0.16
DEFAULT_GRASP_HEIGHT_OFFSET = 0.008
DEFAULT_LIFT_HEIGHT = 0.18
DEFAULT_IK_ITERS = 3
DEFAULT_IK_DAMPING = 0.06
DEFAULT_IK_GAIN = 0.28
DEFAULT_IK_MAX_STEP = 0.008
DEFAULT_ORIENTATION_WEIGHT = 0.22
DEFAULT_APPROACH_MAX_STEPS = 180
DEFAULT_DESCEND_MAX_STEPS = 160
DEFAULT_CLOSE_STEPS = 120
DEFAULT_TABLE_CLEARANCE = 0.006
DEFAULT_TCP_TO_FINGER_Z = 0.08
DEFAULT_FINGERTIP_BACKOFF = 0.012
DEFAULT_INNER_SURFACE_MARGIN = 0.002

LEFT_PARK_QPOS = [-2.05223, 0.03461, 0.01063, 0.0, 0.00201, -0.00173, 0.11128]
RIGHT_GRASP_QPOS = [1.82583, -0.00426, -0.11421, 0.79909, 0.06046, 0.15543, -1.08681]


def build_initial_qpos(pose):
    if pose == "task":
        return np.array(
            [
                *LEFT_PARK_QPOS,
                OPEN_GRIPPER_QPOS,
                OPEN_GRIPPER_QPOS,
                *RIGHT_GRASP_QPOS,
                OPEN_GRIPPER_QPOS,
                OPEN_GRIPPER_QPOS,
            ]
        )
    if pose == "right_grasp":
        return np.array(
            [
                *LEFT_PARK_QPOS,
                OPEN_GRIPPER_QPOS,
                OPEN_GRIPPER_QPOS,
                *RIGHT_GRASP_QPOS,
                OPEN_GRIPPER_QPOS,
                OPEN_GRIPPER_QPOS,
            ]
        )
    if pose == "zero":
        qpos = np.zeros(18)
        qpos[7] = qpos[8] = qpos[16] = qpos[17] = OPEN_GRIPPER_QPOS
        return qpos
    raise ValueError(f"Unsupported pose preset: {pose}")


def find_gripper_joint_names(robot_model, arm):
    return [robot_model.correct_naming(f"openarmx_{arm}_finger_joint{i}") for i in (1, 2)]


def find_arm_joint_names(robot_model, arm):
    return [robot_model.correct_naming(f"openarmx_{arm}_joint{i}") for i in range(1, 8)]


def find_gripper_finger_body_names(robot_model, arm):
    return [
        robot_model.correct_naming(f"openarmx_{arm}_right_finger"),
        robot_model.correct_naming(f"openarmx_{arm}_left_finger"),
    ]


def find_gripper_finger_geom_names(robot_model, arm):
    return [
        robot_model.correct_naming(f"openarmx_{arm}_right_finger_collision"),
        robot_model.correct_naming(f"openarmx_{arm}_left_finger_collision"),
    ]


def actuator_names(sim):
    if hasattr(sim.model, "actuator_names"):
        return sim.model.actuator_names
    return [sim.model.actuator_id2name(actuator_id) for actuator_id in range(sim.model.nu)]


def find_gripper_actuator_ids(sim, arm):
    name_fragment = f"gripper0_{arm}_finger_joint"
    return [
        actuator_id
        for actuator_id, actuator_name in enumerate(actuator_names(sim))
        if name_fragment in actuator_name
    ]


def set_gripper_qpos(sim, joint_names, qpos):
    for joint_name in joint_names:
        qpos_idx = sim.model.get_joint_qpos_addr(joint_name)
        sim.data.qpos[qpos_idx] = qpos


def set_gripper_ctrl(sim, actuator_ids, ctrl):
    if not actuator_ids:
        return
    sim.data.ctrl[actuator_ids] = ctrl


def get_joint_qpos_vector(sim, joint_names):
    return np.array([sim.data.qpos[sim.model.get_joint_qpos_addr(joint_name)] for joint_name in joint_names])


def set_joint_qpos_vector(sim, joint_names, qpos):
    for joint_name, joint_value in zip(joint_names, qpos):
        qpos_idx = sim.model.get_joint_qpos_addr(joint_name)
        sim.data.qpos[qpos_idx] = joint_value


def zero_joint_qvel(sim, joint_names):
    for joint_name in joint_names:
        qvel_idx = sim.model.get_joint_qvel_addr(joint_name)
        sim.data.qvel[qvel_idx] = 0.0


def hold_joint_qpos(sim, joint_names, qpos):
    set_joint_qpos_vector(sim, joint_names, qpos)
    zero_joint_qvel(sim, joint_names)


def joint_qvel_ids(sim, joint_names):
    return [sim.model.get_joint_qvel_addr(joint_name) for joint_name in joint_names]


def joint_ranges(sim, joint_names):
    ranges = []
    for joint_name in joint_names:
        joint_id = sim.model.joint_name2id(joint_name)
        ranges.append(sim.model.jnt_range[joint_id].copy())
    return np.array(ranges)


def body_xpos(sim, body_name):
    return sim.data.body_xpos[sim.model.body_name2id(body_name)].copy()


def geom_xpos(sim, geom_name):
    return sim.data.geom_xpos[sim.model.geom_name2id(geom_name)].copy()


def site_xpos(sim, site_name):
    return sim.data.site_xpos[sim.model.site_name2id(site_name)].copy()


def find_free_joint_name(sim, name_fragment):
    for joint_name in sim.model.joint_names:
        joint_id = sim.model.joint_name2id(joint_name)
        if name_fragment in joint_name and sim.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return joint_name
    raise ValueError(f"No free joint containing {name_fragment!r} found")


def set_free_joint_pose(sim, joint_name, pos, quat=(1.0, 0.0, 0.0, 0.0)):
    qpos_addr = sim.model.get_joint_qpos_addr(joint_name)
    if not isinstance(qpos_addr, tuple):
        raise ValueError(f"{joint_name} is not a free joint")
    sim.data.qpos[qpos_addr[0] : qpos_addr[1]] = np.array([*pos, *quat])

    qvel_addr = sim.model.get_joint_qvel_addr(joint_name)
    sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = 0.0


def free_joint_pos(sim, joint_name):
    qpos_addr = sim.model.get_joint_qpos_addr(joint_name)
    if not isinstance(qpos_addr, tuple):
        raise ValueError(f"{joint_name} is not a free joint")
    return sim.data.qpos[qpos_addr[0] : qpos_addr[0] + 3].copy()


def grasp_center(sim, robot_model, arm, forward_offset=0.0, z_offset=0.0):
    finger_positions = [body_xpos(sim, name) for name in find_gripper_finger_body_names(robot_model, arm)]
    center = np.mean(finger_positions, axis=0)

    tcp_pos = body_xpos(sim, robot_model.eef_name[arm])
    forward = center - tcp_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm > 1e-6:
        center = center + forward_offset * forward / forward_norm

    center[2] += z_offset
    return center


def place_cube_at_gripper(sim, robot_model, arm, cube_joint_name, forward_offset=0.0, z_offset=0.0):
    cube_pos = grasp_center(
        sim,
        robot_model,
        arm,
        forward_offset=forward_offset,
        z_offset=z_offset,
    )
    set_free_joint_pose(sim, cube_joint_name, cube_pos)
    return cube_pos


def mujoco_model(sim):
    return getattr(sim.model, "_model", sim.model)


def mujoco_data(sim):
    return getattr(sim.data, "_data", sim.data)


def skew(vec):
    x, y, z = vec
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def cube_hold_pos_from_tcp(sim, robot_model, arm, tcp_to_finger_z=DEFAULT_TCP_TO_FINGER_Z):
    return gripper_control_point(sim, robot_model, arm)


def gripper_tcp_site_name(robot_model, arm):
    return robot_model.correct_naming(f"openarmx_{arm}_tcp_site")


def gripper_tcp_point(sim, robot_model, arm):
    return site_xpos(sim, gripper_tcp_site_name(robot_model, arm))


def gripper_tcp_xmat(sim, robot_model, arm):
    site_id = sim.model.site_name2id(gripper_tcp_site_name(robot_model, arm))
    return mujoco_data(sim).site_xmat[site_id].reshape(3, 3).copy()


def geom_world_vertices(sim, geom_name):
    geom_id = sim.model.geom_name2id(geom_name)
    model = mujoco_model(sim)
    data = mujoco_data(sim)
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return np.array([data.geom_xpos[geom_id].copy()])

    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    local_vertices = model.mesh_vert[vert_adr : vert_adr + vert_num]
    geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)
    geom_xpos = data.geom_xpos[geom_id]
    return geom_xpos + local_vertices @ geom_xmat.T


def gripper_inner_contact_point(
    sim,
    robot_model,
    arm,
    fingertip_backoff=DEFAULT_FINGERTIP_BACKOFF,
    inner_surface_margin=DEFAULT_INNER_SURFACE_MARGIN,
):
    tcp_pos = gripper_tcp_point(sim, robot_model, arm)
    tcp_xmat = gripper_tcp_xmat(sim, robot_model, arm)
    contact_points = []

    for geom_name in find_gripper_finger_geom_names(robot_model, arm):
        vertices = geom_world_vertices(sim, geom_name)
        if len(vertices) == 0:
            continue

        local_vertices = (vertices - tcp_pos) @ tcp_xmat
        z_max = float(np.max(local_vertices[:, 2]))
        tip_band = max(2.0 * fingertip_backoff, 0.024)
        candidates = local_vertices[local_vertices[:, 2] >= z_max - tip_band]
        if len(candidates) == 0:
            candidates = local_vertices

        if float(np.mean(local_vertices[:, 1])) < 0.0:
            inner_y = float(np.max(candidates[:, 1]))
            surface = candidates[candidates[:, 1] >= inner_y - inner_surface_margin]
        else:
            inner_y = float(np.min(candidates[:, 1]))
            surface = candidates[candidates[:, 1] <= inner_y + inner_surface_margin]

        if len(surface) == 0:
            surface = candidates

        contact_local = np.mean(surface, axis=0)
        contact_local[2] = z_max - fingertip_backoff
        contact_points.append(tcp_pos + tcp_xmat @ contact_local)

    if contact_points:
        return np.mean(contact_points, axis=0)

    return np.mean([geom_xpos(sim, name) for name in find_gripper_finger_geom_names(robot_model, arm)], axis=0)


def gripper_control_point(
    sim,
    robot_model,
    arm,
    fingertip_backoff=DEFAULT_FINGERTIP_BACKOFF,
    inner_surface_margin=DEFAULT_INNER_SURFACE_MARGIN,
):
    return gripper_inner_contact_point(
        sim,
        robot_model,
        arm,
        fingertip_backoff=fingertip_backoff,
        inner_surface_margin=inner_surface_margin,
    )


def gripper_tcp_jacobians(sim, robot_model, arm):
    site_id = sim.model.site_name2id(gripper_tcp_site_name(robot_model, arm))
    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    mujoco.mj_jacSite(mujoco_model(sim), mujoco_data(sim), jacp, jacr, site_id)
    return jacp, jacr


def gripper_control_jacobians(sim, robot_model, arm, control_point):
    jacp, jacr = gripper_tcp_jacobians(sim, robot_model, arm)
    tcp_pos = gripper_tcp_point(sim, robot_model, arm)
    offset = np.asarray(control_point) - tcp_pos
    return jacp - skew(offset) @ jacr, jacr


def clamp_target_above_table(target_pos, table_z=TABLE_OFFSET[2], clearance=DEFAULT_TABLE_CLEARANCE):
    target = np.array(target_pos, dtype=float)
    target[2] = max(target[2], table_z + clearance)
    return target


def top_down_grasp_xmat(yaw=0.0):
    width_axis = np.array([np.sin(yaw), np.cos(yaw), 0.0])
    down_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.cross(width_axis, down_axis)
    x_axis /= np.linalg.norm(x_axis)
    return np.column_stack((x_axis, width_axis, down_axis))


def orientation_error(current_xmat, target_xmat):
    return 0.5 * (
        np.cross(current_xmat[:, 0], target_xmat[:, 0])
        + np.cross(current_xmat[:, 1], target_xmat[:, 1])
        + np.cross(current_xmat[:, 2], target_xmat[:, 2])
    )


def ik_step_to_target(
    sim,
    robot_model,
    arm,
    target_pos,
    target_xmat=None,
    damping=DEFAULT_IK_DAMPING,
    gain=DEFAULT_IK_GAIN,
    max_step=DEFAULT_IK_MAX_STEP,
    orientation_weight=DEFAULT_ORIENTATION_WEIGHT,
    fingertip_backoff=DEFAULT_FINGERTIP_BACKOFF,
    inner_surface_margin=DEFAULT_INNER_SURFACE_MARGIN,
):
    sim.forward()
    joint_names = find_arm_joint_names(robot_model, arm)
    qvel_ids = joint_qvel_ids(sim, joint_names)
    current_pos = gripper_control_point(
        sim,
        robot_model,
        arm,
        fingertip_backoff=fingertip_backoff,
        inner_surface_margin=inner_surface_margin,
    )
    position_error = np.asarray(target_pos) - current_pos
    jacp, jacr = gripper_control_jacobians(sim, robot_model, arm, current_pos)

    if target_xmat is None or orientation_weight <= 0.0:
        error = position_error
        jac = jacp[:, qvel_ids]
        rot_error = np.zeros(3)
    else:
        rot_error = orientation_error(gripper_tcp_xmat(sim, robot_model, arm), target_xmat)
        error = np.concatenate((position_error, orientation_weight * rot_error))
        jac = np.vstack((jacp, orientation_weight * jacr))[:, qvel_ids]

    lhs = jac @ jac.T + (damping**2) * np.eye(jac.shape[0])
    try:
        delta_q = jac.T @ np.linalg.solve(lhs, gain * error)
    except np.linalg.LinAlgError:
        delta_q = jac.T @ np.linalg.pinv(lhs) @ (gain * error)

    delta_norm = np.linalg.norm(delta_q)
    if delta_norm > max_step:
        delta_q = delta_q * (max_step / delta_norm)

    qpos = get_joint_qpos_vector(sim, joint_names)
    limits = joint_ranges(sim, joint_names)
    qpos = np.clip(qpos + delta_q, limits[:, 0], limits[:, 1])
    hold_joint_qpos(sim, joint_names, qpos)
    sim.forward()
    return float(np.linalg.norm(position_error)), float(np.linalg.norm(rot_error))


class IKGraspDemo:
    def __init__(
        self,
        sim,
        robot_model,
        arm,
        cube_joint_name,
        approach_height=DEFAULT_APPROACH_HEIGHT,
        grasp_height_offset=DEFAULT_GRASP_HEIGHT_OFFSET,
        lift_height=DEFAULT_LIFT_HEIGHT,
        table_clearance=DEFAULT_TABLE_CLEARANCE,
        fingertip_backoff=DEFAULT_FINGERTIP_BACKOFF,
        inner_surface_margin=DEFAULT_INNER_SURFACE_MARGIN,
        grasp_yaw=0.0,
    ):
        self.sim = sim
        self.robot_model = robot_model
        self.arm = arm
        self.cube_joint_name = cube_joint_name
        self.approach_height = approach_height
        self.grasp_height_offset = grasp_height_offset
        self.lift_height = lift_height
        self.table_clearance = table_clearance
        self.fingertip_backoff = fingertip_backoff
        self.inner_surface_margin = inner_surface_margin
        self.target_xmat = top_down_grasp_xmat(grasp_yaw)
        self.state = "center_xy"
        self.state_steps = 0
        self.cube_start_pos = free_joint_pos(sim, cube_joint_name)
        self.xy_target = self.cube_start_pos.copy()
        self.xy_target[2] = gripper_control_point(
            sim,
            robot_model,
            arm,
            fingertip_backoff=fingertip_backoff,
            inner_surface_margin=inner_surface_margin,
        )[2]

    def safe_grasp_point_target(self, grasp_point_target):
        target = np.array(grasp_point_target, dtype=float)
        return clamp_target_above_table(target, clearance=self.table_clearance)

    def closing_gripper_qpos(self):
        fraction = min(1.0, self.state_steps / max(1, DEFAULT_CLOSE_STEPS))
        return OPEN_GRIPPER_QPOS + fraction * (CLOSED_GRIPPER_QPOS - OPEN_GRIPPER_QPOS)

    def target_and_gripper(self):
        if self.state == "approach":
            return self.safe_grasp_point_target(
                self.cube_start_pos + np.array([0.0, 0.0, self.approach_height])
            ), self.target_xmat, OPEN_GRIPPER_QPOS
        if self.state == "center_xy":
            return self.xy_target, self.target_xmat, OPEN_GRIPPER_QPOS
        if self.state == "descend":
            return self.safe_grasp_point_target(
                self.cube_start_pos + np.array([0.0, 0.0, self.grasp_height_offset])
            ), self.target_xmat, OPEN_GRIPPER_QPOS
        if self.state == "close":
            return self.safe_grasp_point_target(
                self.cube_start_pos + np.array([0.0, 0.0, self.grasp_height_offset])
            ), self.target_xmat, self.closing_gripper_qpos()
        return (
            self.safe_grasp_point_target(self.cube_start_pos + np.array([0.0, 0.0, self.lift_height])),
            self.target_xmat,
            CLOSED_GRIPPER_QPOS,
        )

    def update_state(self, position_error, rotation_error):
        self.state_steps += 1
        if self.state == "center_xy" and (position_error < 0.012 or self.state_steps > DEFAULT_APPROACH_MAX_STEPS):
            self.state = "approach"
            self.state_steps = 0
        elif self.state == "approach" and (position_error < 0.018 or self.state_steps > DEFAULT_APPROACH_MAX_STEPS):
            self.state = "descend"
            self.state_steps = 0
        elif self.state == "descend" and (position_error < 0.012 or self.state_steps > DEFAULT_DESCEND_MAX_STEPS):
            self.state = "close"
            self.state_steps = 0
        elif self.state == "close" and self.state_steps > DEFAULT_CLOSE_STEPS:
            self.state = "lift"
            self.state_steps = 0


def grasp_fraction(elapsed, close_after, close_seconds):
    if elapsed <= close_after:
        return 0.0
    if close_seconds <= 0:
        return 1.0
    return min(1.0, (elapsed - close_after) / close_seconds)


def apply_visual_grasp(sim, joint_names, actuator_ids, elapsed, close_after, close_seconds):
    fraction = grasp_fraction(elapsed, close_after, close_seconds)
    target = OPEN_GRIPPER_QPOS + fraction * (CLOSED_GRIPPER_QPOS - OPEN_GRIPPER_QPOS)
    set_gripper_qpos(sim, joint_names, target)
    set_gripper_ctrl(sim, actuator_ids, target)


def lock_robot_qpos(sim, robot_model, initial_qpos):
    for joint_name, joint_qpos in zip(robot_model.joints, initial_qpos):
        if "finger" in joint_name:
            continue
        qpos_idx = sim.model.get_joint_qpos_addr(joint_name)
        sim.data.qpos[qpos_idx] = joint_qpos
        qvel_idx = sim.model.get_joint_qvel_addr(joint_name)
        sim.data.qvel[qvel_idx] = 0.0


def build_standalone_sim(
    pose="right_grasp",
    base_x=DEFAULT_BASE_X,
    base_z=DEFAULT_BASE_Z,
    cube_pos=DEFAULT_CUBE_POS,
):
    robot_model = create_robot("OpenArmX")
    robot_model.add_base(robot_base_factory(robot_model.default_base))
    robot_model.update_joints()
    robot_model.update_actuators()

    for arm, gripper_name in robot_model.default_gripper.items():
        gripper = gripper_factory(gripper_name, idn=f"0_{arm}")
        robot_model.add_gripper(gripper, robot_model.eef_name[arm])

    robot_model.set_base_xpos(np.array([base_x, 0.0, base_z]))

    arena = TableArena(table_full_size=TABLE_FULL_SIZE, table_offset=TABLE_OFFSET, has_legs=True)
    arena.set_origin([0, 0, 0])

    cube = BoxObject(
        name="cube",
        size=DEFAULT_CUBE_SIZE,
        rgba=[1, 0, 0, 1],
        friction=[2.0, 0.01, 0.0001],
        density=300,
        joints="default",
    )
    cube.get_obj().set("pos", array_to_string(np.array(cube_pos)))

    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[robot_model], mujoco_objects=[cube])
    sim = MjSim(task.get_model())

    initial_qpos = build_initial_qpos(pose)

    for joint_name, joint_qpos in zip(robot_model.joints, initial_qpos):
        qpos_idx = sim.model.get_joint_qpos_addr(joint_name)
        sim.data.qpos[qpos_idx] = joint_qpos

    sim.forward()
    return sim, robot_model


def configure_camera(vwr, camera):
    vwr.cam.lookat = DEFAULT_FREE_CAM["lookat"]
    vwr.cam.distance = DEFAULT_FREE_CAM["distance"]
    vwr.cam.azimuth = DEFAULT_FREE_CAM["azimuth"]
    vwr.cam.elevation = DEFAULT_FREE_CAM["elevation"]
    if camera == "free":
        vwr.cam.type = mujoco.mjtCamera.mjCAMERA_FREE


def main():
    parser = argparse.ArgumentParser(description="Visualize OpenArmX in a tabletop block grasping scene.")
    parser.add_argument("--camera", default="free", help="Viewer camera mode. Default: free")
    parser.add_argument("--max-fr", type=float, default=30.0, help="Maximum viewer frame rate.")
    parser.add_argument(
        "--pose",
        choices=["zero", "task", "right_grasp"],
        default="right_grasp",
        help="Robot pose preset. 'right_grasp' uses one gripper near the cube; 'zero' is for inspection.",
    )
    parser.add_argument(
        "--base-x",
        type=float,
        default=DEFAULT_BASE_X,
        help="Robot base x position. More negative means farther from the table center.",
    )
    parser.add_argument(
        "--base-z",
        type=float,
        default=DEFAULT_BASE_Z,
        help="Robot base z position.",
    )
    parser.add_argument(
        "--cube-x",
        type=float,
        default=DEFAULT_CUBE_POS[0],
        help="Cube x position.",
    )
    parser.add_argument(
        "--cube-y",
        type=float,
        default=DEFAULT_CUBE_POS[1],
        help="Cube y position.",
    )
    parser.add_argument(
        "--cube-z",
        type=float,
        default=DEFAULT_CUBE_POS[2],
        help="Cube z position.",
    )
    parser.add_argument(
        "--active-gripper",
        choices=["right", "left"],
        default="right",
        help="Which gripper should close in the visual grasp demo.",
    )
    parser.add_argument(
        "--no-auto-close",
        action="store_true",
        help="Keep the active gripper open instead of closing it after startup.",
    )
    parser.add_argument(
        "--close-after",
        type=float,
        default=DEFAULT_CLOSE_AFTER,
        help="Seconds to wait before closing the active gripper.",
    )
    parser.add_argument(
        "--close-seconds",
        type=float,
        default=DEFAULT_CLOSE_SECONDS,
        help="Seconds used to close the active gripper.",
    )
    parser.add_argument(
        "--step-physics",
        dest="step_physics",
        action="store_true",
        default=True,
        help="Advance MuJoCo physics after each scripted control update. Enabled by default.",
    )
    parser.add_argument(
        "--no-step-physics",
        dest="step_physics",
        action="store_false",
        help="Disable MuJoCo physics stepping and only call sim.forward(); useful for static inspection.",
    )
    parser.add_argument(
        "--manual-cube-pos",
        action="store_true",
        help="Use --cube-x / --cube-y / --cube-z instead of centering the cube in --scripted-grasp mode.",
    )
    parser.add_argument(
        "--scripted-grasp",
        action="store_true",
        help="Old visual demo mode: keep the cube centered in the active gripper instead of doing tabletop IK grasp.",
    )
    parser.add_argument(
        "--cube-forward-offset",
        type=float,
        default=DEFAULT_CUBE_FORWARD_OFFSET,
        help="Move the auto-placed cube farther along the gripper direction, in meters.",
    )
    parser.add_argument(
        "--cube-z-offset",
        type=float,
        default=DEFAULT_CUBE_Z_OFFSET,
        help="Move the auto-placed cube up / down after centering it in the gripper in --scripted-grasp mode.",
    )
    parser.add_argument(
        "--no-ik-grasp",
        action="store_true",
        help="Disable the IK tabletop grasp state machine.",
    )
    parser.add_argument(
        "--approach-height",
        type=float,
        default=DEFAULT_APPROACH_HEIGHT,
        help="Height above the cube used by the IK approach stage.",
    )
    parser.add_argument(
        "--grasp-height-offset",
        type=float,
        default=DEFAULT_GRASP_HEIGHT_OFFSET,
        help="Vertical offset from cube center used by the IK grasp stage.",
    )
    parser.add_argument(
        "--tcp-to-finger-z",
        type=float,
        default=DEFAULT_TCP_TO_FINGER_Z,
        help="Legacy assisted-lift offset. IK grasp now uses actual finger collision geometry instead.",
    )
    parser.add_argument(
        "--fingertip-backoff",
        type=float,
        default=DEFAULT_FINGERTIP_BACKOFF,
        help="Meters behind the fingertip used as the grasp contact point.",
    )
    parser.add_argument(
        "--inner-surface-margin",
        type=float,
        default=DEFAULT_INNER_SURFACE_MARGIN,
        help="Meters of finger inner-surface vertices used to estimate the grasp contact point.",
    )
    parser.add_argument(
        "--grasp-yaw",
        type=float,
        default=0.0,
        help="Top-down grasp yaw in radians; 0 keeps the gripper width axis along world +Y.",
    )
    parser.add_argument(
        "--table-clearance",
        type=float,
        default=DEFAULT_TABLE_CLEARANCE,
        help="Minimum TCP target clearance above the tabletop.",
    )
    parser.add_argument(
        "--lift-height",
        type=float,
        default=DEFAULT_LIFT_HEIGHT,
        help="Height above the starting cube position used by the IK lift stage.",
    )
    parser.add_argument(
        "--ik-iters",
        type=int,
        default=DEFAULT_IK_ITERS,
        help="IK iterations per viewer frame.",
    )
    parser.add_argument(
        "--ik-gain",
        type=float,
        default=DEFAULT_IK_GAIN,
        help="IK position gain. Lower is slower and gentler.",
    )
    parser.add_argument(
        "--ik-max-step",
        type=float,
        default=DEFAULT_IK_MAX_STEP,
        help="Maximum joint-space IK step per iteration.",
    )
    parser.add_argument(
        "--ik-damping",
        type=float,
        default=DEFAULT_IK_DAMPING,
        help="Damped least-squares IK damping. Higher is smoother but less aggressive.",
    )
    parser.add_argument(
        "--orientation-weight",
        type=float,
        default=DEFAULT_ORIENTATION_WEIGHT,
        help="Weight for top-down TCP orientation IK. Set 0 to disable orientation control.",
    )
    parser.add_argument(
        "--assisted-lift",
        action="store_true",
        help="Attach the cube to the gripper during IK lift stage. Off by default so misses stay visible.",
    )
    args = parser.parse_args()

    cube_pos = (args.cube_x, args.cube_y, args.cube_z)
    initial_qpos = build_initial_qpos(args.pose)
    sim, robot_model = build_standalone_sim(
        pose=args.pose,
        base_x=args.base_x,
        base_z=args.base_z,
        cube_pos=cube_pos,
    )
    gripper_joint_names = find_gripper_joint_names(robot_model, args.active_gripper)
    gripper_actuator_ids = find_gripper_actuator_ids(sim, args.active_gripper)
    cube_joint_name = find_free_joint_name(sim, "cube")
    inactive_gripper = "left" if args.active_gripper == "right" else "right"
    set_gripper_qpos(sim, find_gripper_joint_names(robot_model, inactive_gripper), OPEN_GRIPPER_QPOS)
    set_gripper_ctrl(sim, find_gripper_actuator_ids(sim, inactive_gripper), OPEN_GRIPPER_QPOS)

    if args.scripted_grasp and not args.manual_cube_pos:
        cube_pos = place_cube_at_gripper(
            sim,
            robot_model,
            args.active_gripper,
            cube_joint_name,
            forward_offset=args.cube_forward_offset,
            z_offset=args.cube_z_offset,
        )
        sim.forward()

    ik_demo = None
    if not args.no_ik_grasp and not args.scripted_grasp:
        ik_demo = IKGraspDemo(
            sim,
            robot_model,
            args.active_gripper,
            cube_joint_name,
            approach_height=args.approach_height,
            grasp_height_offset=args.grasp_height_offset,
            lift_height=args.lift_height,
            table_clearance=args.table_clearance,
            fingertip_backoff=args.fingertip_backoff,
            inner_surface_margin=args.inner_surface_margin,
            grasp_yaw=args.grasp_yaw,
        )

    print("robot_name =", robot_model.__class__.__name__)
    print("robot_base =", robot_model.root_body)
    print("joint_count =", len(robot_model.joints))
    print("eef_names =", robot_model.eef_name)
    print("pose_preset =", args.pose)
    print("base_x =", args.base_x)
    print("base_z =", args.base_z)
    print("cube_pos =", cube_pos)
    print("active_gripper =", args.active_gripper)
    print("auto_close =", not args.no_auto_close)
    print("step_physics =", args.step_physics)
    print("manual_cube_pos =", args.manual_cube_pos)
    print("scripted_grasp =", args.scripted_grasp)
    print("ik_grasp =", ik_demo is not None)
    print("assisted_lift =", args.assisted_lift)
    print("tcp_to_finger_z =", args.tcp_to_finger_z)
    print("table_clearance =", args.table_clearance)
    print("fingertip_backoff =", args.fingertip_backoff)
    print("inner_surface_margin =", args.inner_surface_margin)
    print("orientation_weight =", args.orientation_weight)
    print("grasp_yaw =", args.grasp_yaw)
    print("cube_joint_name =", cube_joint_name)
    print("gripper_actuator_ids =", gripper_actuator_ids)

    with viewer.launch_passive(sim.model._model, sim.data._data, show_left_ui=False, show_right_ui=False) as vwr:
        configure_camera(vwr, args.camera)
        start_time = time.monotonic()
        while vwr.is_running():
            if args.scripted_grasp and not args.no_auto_close:
                elapsed = time.monotonic() - start_time
                apply_visual_grasp(
                    sim,
                    gripper_joint_names,
                    gripper_actuator_ids,
                    elapsed,
                    args.close_after,
                    args.close_seconds,
                )

            if args.scripted_grasp:
                place_cube_at_gripper(
                    sim,
                    robot_model,
                    args.active_gripper,
                    cube_joint_name,
                    forward_offset=args.cube_forward_offset,
                    z_offset=args.cube_z_offset,
                )
            elif ik_demo is not None:
                target_pos, target_xmat, gripper_qpos = ik_demo.target_and_gripper()
                position_error = 0.0
                rotation_error = 0.0
                for _ in range(max(1, args.ik_iters)):
                    position_error, rotation_error = ik_step_to_target(
                        sim,
                        robot_model,
                        args.active_gripper,
                        target_pos,
                        target_xmat=target_xmat,
                        damping=args.ik_damping,
                        gain=args.ik_gain,
                        max_step=args.ik_max_step,
                        orientation_weight=args.orientation_weight,
                        fingertip_backoff=args.fingertip_backoff,
                        inner_surface_margin=args.inner_surface_margin,
                    )
                set_gripper_qpos(sim, gripper_joint_names, gripper_qpos)
                set_gripper_ctrl(sim, gripper_actuator_ids, gripper_qpos)
                if ik_demo.state == "lift" and args.assisted_lift:
                    set_free_joint_pose(
                        sim,
                        cube_joint_name,
                        cube_hold_pos_from_tcp(
                            sim,
                            robot_model,
                            args.active_gripper,
                            tcp_to_finger_z=args.tcp_to_finger_z,
                        ),
                    )
                ik_demo.update_state(position_error, rotation_error)

            if args.step_physics and ik_demo is None:
                lock_robot_qpos(sim, robot_model, initial_qpos)
                sim.step()
            elif args.step_physics:
                sim.step()
            else:
                sim.forward()
            vwr.sync()
            time.sleep(max(0.0, 1.0 / args.max_fr))


if __name__ == "__main__":
    main()
