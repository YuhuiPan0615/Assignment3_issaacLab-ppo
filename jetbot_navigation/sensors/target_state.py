"""Relative target state; global position and yaw never enter policy observations."""
import numpy as np


def goal_state(position, yaw, goal):
    delta = goal - position
    distance = np.linalg.norm(delta, axis=1)
    bearing = np.arctan2(delta[:, 1], delta[:, 0]) - yaw
    return distance, np.sin(bearing), np.cos(bearing)
