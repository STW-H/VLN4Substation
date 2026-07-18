"""Constant-speed camera trajectory construction for planned routes."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Any


def wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class CameraPose:
    x: float
    y: float
    z: float
    yaw_rad: float
    pitch_rad: float = 0.0


class ConstantSpeedCameraTrajectory:
    def __init__(
        self,
        keyframes: list[CameraPose],
        segment_durations_s: list[float],
    ) -> None:
        if not keyframes:
            raise ValueError("Trajectory needs at least one keyframe")
        if len(segment_durations_s) != len(keyframes) - 1:
            raise ValueError("Each keyframe pair needs one segment duration")
        self.keyframes = keyframes
        self.cumulative_times_s = [0.0]
        for duration in segment_durations_s:
            if duration < 0.0 or not math.isfinite(duration):
                raise ValueError(f"Invalid trajectory duration: {duration}")
            self.cumulative_times_s.append(self.cumulative_times_s[-1] + duration)

    @property
    def duration_s(self) -> float:
        return self.cumulative_times_s[-1]

    def pose_at(self, time_s: float) -> CameraPose:
        if len(self.keyframes) == 1 or time_s <= 0.0:
            return self.keyframes[0]
        if time_s >= self.duration_s:
            return self.keyframes[-1]
        index = min(
            bisect_right(self.cumulative_times_s, float(time_s)) - 1,
            len(self.keyframes) - 2,
        )
        start_time = self.cumulative_times_s[index]
        end_time = self.cumulative_times_s[index + 1]
        ratio = 1.0 if end_time <= start_time else (time_s - start_time) / (end_time - start_time)
        start = self.keyframes[index]
        end = self.keyframes[index + 1]
        yaw_delta = wrap_angle(end.yaw_rad - start.yaw_rad)
        pitch_delta = wrap_angle(end.pitch_rad - start.pitch_rad)
        return CameraPose(
            x=start.x + ratio * (end.x - start.x),
            y=start.y + ratio * (end.y - start.y),
            z=start.z + ratio * (end.z - start.z),
            yaw_rad=wrap_angle(start.yaw_rad + ratio * yaw_delta),
            pitch_rad=wrap_angle(start.pitch_rad + ratio * pitch_delta),
        )


def trajectory_from_route_payload(
    payload: dict[str, Any],
    *,
    camera_height_m: float,
    linear_speed_mps: float,
    angular_speed_deg_s: float,
    apply_terminal_camera_pose: bool,
    terminal_hold_s: float,
    travel_yaw_source: str = "body",
) -> ConstantSpeedCameraTrajectory:
    if linear_speed_mps <= 0.0:
        raise ValueError("linear_speed_mps must be positive")
    if angular_speed_deg_s <= 0.0:
        raise ValueError("angular_speed_deg_s must be positive")
    if terminal_hold_s < 0.0:
        raise ValueError("terminal_hold_s cannot be negative")
    states = payload.get("route", {}).get("states", [])
    if not states:
        raise ValueError("Route JSON has no route.states")

    if travel_yaw_source not in {"body", "motion_tangent"}:
        raise ValueError("travel_yaw_source must be 'body' or 'motion_tangent'")
    travel_yaws = [float(state["yaw_rad"]) for state in states]
    if travel_yaw_source == "motion_tangent" and len(states) > 1:
        last_motion_yaw = travel_yaws[0]
        for index, state in enumerate(states):
            current_xy = state["xy"]
            motion_yaw = None
            for next_state in states[index + 1 :]:
                dx = float(next_state["xy"][0]) - float(current_xy[0])
                dy = float(next_state["xy"][1]) - float(current_xy[1])
                if math.hypot(dx, dy) > 1.0e-9:
                    motion_yaw = math.atan2(dy, dx)
                    break
            if motion_yaw is not None:
                last_motion_yaw = motion_yaw
            travel_yaws[index] = last_motion_yaw

    keyframes = [
        CameraPose(
            x=float(state["xy"][0]),
            y=float(state["xy"][1]),
            z=float(camera_height_m),
            yaw_rad=travel_yaws[index],
        )
        for index, state in enumerate(states)
    ]
    angular_speed_rad_s = math.radians(angular_speed_deg_s)

    if apply_terminal_camera_pose:
        camera = payload.get("target", {}).get("route_segment", {}).get("camera")
        if camera:
            last = keyframes[-1]
            final_body_yaw = float(states[-1]["yaw_rad"])
            keyframes.append(
                CameraPose(
                    x=last.x,
                    y=last.y,
                    z=last.z,
                    yaw_rad=wrap_angle(final_body_yaw + float(camera["pan_rad"])),
                    # Z-up positive elevation remains positive pitch after
                    # (x, y, z) -> (x, z, -y) conversion to Habitat Y-up.
                    pitch_rad=float(camera["tilt_rad"]),
                )
            )
    if terminal_hold_s > 0.0:
        keyframes.append(keyframes[-1])

    durations: list[float] = []
    for index, (start, end) in enumerate(
        zip(keyframes[:-1], keyframes[1:], strict=True)
    ):
        is_hold = index == len(keyframes) - 2 and start == end
        if is_hold:
            durations.append(float(terminal_hold_s))
            continue
        distance = math.hypot(end.x - start.x, end.y - start.y)
        yaw_change = abs(wrap_angle(end.yaw_rad - start.yaw_rad))
        pitch_change = abs(wrap_angle(end.pitch_rad - start.pitch_rad))
        durations.append(
            max(
                distance / float(linear_speed_mps),
                yaw_change / angular_speed_rad_s,
                pitch_change / angular_speed_rad_s,
            )
        )
    return ConstantSpeedCameraTrajectory(keyframes, durations)
