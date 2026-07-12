"""Habitat-GS rendering helpers."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys


def render_gaussian_snapshot(
    scene: Path,
    output: Path,
    width: int,
    height: int,
    position: list[float],
    yaw_deg: float,
    pitch_deg: float,
) -> None:
    flags = sys.getdlopenflags()
    sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

    import magnum as mn
    import numpy as np
    from PIL import Image

    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis

    color = habitat_sim.CameraSensorSpec()
    color.uuid = "color_sensor"
    color.sensor_type = habitat_sim.SensorType.COLOR
    color.resolution = [height, width]
    color.position = [0.0, 0.0, 0.0]
    color.hfov = 90.0

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = False
    sim_cfg.create_renderer = True
    sim_cfg.gpu_device_id = 0
    if hasattr(sim_cfg, "enable_hbao"):
        sim_cfg.enable_hbao = True

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [color]
    agent_cfg.height = 1.5
    agent_cfg.radius = 0.1

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        helper = habitat_sim.RenderInstanceHelper(sim, use_xyzw_orientations=False)
        helper.add_instance(asset_filepath=str(scene), semantic_id=0, scale=mn.Vector3(1.0, 1.0, 1.0))
        helper.set_world_poses(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )

        agent = sim.get_agent(0)
        state = habitat_sim.AgentState()
        state.position = np.array(position, dtype=np.float32)
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        state.rotation = quat_from_angle_axis(float(yaw), np.array([0, 1, 0])) * quat_from_angle_axis(
            float(pitch), np.array([1, 0, 0])
        )
        agent.set_state(state, reset_sensors=True)

        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][:, :, :3]
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(output)
    finally:
        sim.close()
