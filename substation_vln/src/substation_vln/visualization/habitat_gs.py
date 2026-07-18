"""Habitat-GS rendering helpers."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys


class HabitatGSTrajectoryRenderer:
    """Keep one Habitat-GS simulator alive while rendering camera poses."""

    def __init__(
        self,
        scene: Path,
        *,
        width: int,
        height: int,
        hfov_deg: float,
        instance_translation: list[float],
        instance_rotation_wxyz: list[float],
        instance_scale: float,
        gpu_device_id: int = 0,
    ) -> None:
        flags = sys.getdlopenflags()
        sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

        import habitat_sim
        import magnum as mn
        import numpy as np
        from habitat_sim.utils.common import quat_from_angle_axis

        self._np = np
        self._habitat_sim = habitat_sim
        self._quat_from_angle_axis = quat_from_angle_axis

        color = habitat_sim.CameraSensorSpec()
        color.uuid = "color_sensor"
        color.sensor_type = habitat_sim.SensorType.COLOR
        color.resolution = [int(height), int(width)]
        color.position = [0.0, 0.0, 0.0]
        color.hfov = float(hfov_deg)

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = "NONE"
        sim_cfg.enable_physics = False
        sim_cfg.create_renderer = True
        sim_cfg.gpu_device_id = int(gpu_device_id)
        if hasattr(sim_cfg, "enable_hbao"):
            sim_cfg.enable_hbao = True

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [color]
        agent_cfg.height = 0.0
        agent_cfg.radius = 0.1

        self.sim = habitat_sim.Simulator(
            habitat_sim.Configuration(sim_cfg, [agent_cfg])
        )
        self.render_helper = habitat_sim.RenderInstanceHelper(
            self.sim, use_xyzw_orientations=False
        )
        self.render_helper.add_instance(
            asset_filepath=str(scene),
            semantic_id=0,
            scale=mn.Vector3(float(instance_scale)),
        )
        self.render_helper.set_world_poses(
            np.asarray([instance_translation], dtype=np.float32),
            np.asarray([instance_rotation_wxyz], dtype=np.float32),
        )
        self.agent = self.sim.get_agent(0)

    def render(
        self,
        position: list[float],
        yaw_deg: float | None = None,
        pitch_deg: float | None = None,
        rotation_wxyz: list[float] | None = None,
    ):
        state = self._habitat_sim.AgentState()
        state.position = self._np.asarray(position, dtype=self._np.float32)
        if rotation_wxyz is not None:
            from habitat_sim.utils.common import quat_from_coeffs

            quat = self._np.asarray(rotation_wxyz, dtype=self._np.float64)
            if quat.shape != (4,):
                raise ValueError("rotation_wxyz must contain [w, x, y, z]")
            # quat_from_coeffs accepts [x, y, z, w].
            state.rotation = quat_from_coeffs(
                self._np.asarray([quat[1], quat[2], quat[3], quat[0]])
            )
        else:
            if yaw_deg is None or pitch_deg is None:
                raise ValueError("yaw_deg and pitch_deg are required without rotation_wxyz")
            yaw = self._np.deg2rad(float(yaw_deg))
            pitch = self._np.deg2rad(float(pitch_deg))
            state.rotation = self._quat_from_angle_axis(
                float(yaw), self._np.asarray([0.0, 1.0, 0.0])
            ) * self._quat_from_angle_axis(
                float(pitch), self._np.asarray([1.0, 0.0, 0.0])
            )
        self.agent.set_state(state, reset_sensors=True)
        return self.sim.get_sensor_observations()["color_sensor"][:, :, :3]

    def close(self) -> None:
        if getattr(self, "sim", None) is not None:
            self.sim.close()
            self.sim = None

    def __enter__(self) -> "HabitatGSTrajectoryRenderer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


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
