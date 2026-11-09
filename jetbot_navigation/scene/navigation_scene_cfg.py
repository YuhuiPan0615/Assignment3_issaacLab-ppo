"""Fixed physical geometry. None of this geometry is included in observations."""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import numpy as np


@dataclass
class Box:
    name: str
    center: tuple[float, float]
    size: tuple[float, float]
    height: float = 0.50

    def clearance(self, point):
        delta = np.maximum(np.abs(np.asarray(point) - self.center) - np.asarray(self.size) / 2, 0)
        return float(np.linalg.norm(delta))


@dataclass
class FixedScene:
    width: float = 4.0
    length: float = 6.0
    wall_thickness: float = 0.12
    start: tuple[float, float] = (0.0, -2.55)
    start_yaw: float = math.pi / 2
    goal: tuple[float, float] = (-1.40, 2.45)
    # Bounding circle is provisional, not a measurement of the USD collision shape.
    robot_radius: float = 0.15
    clearance_margin: float = 0.04

    @property
    def boxes(self):
        w, h, t = self.width, self.length, self.wall_thickness
        return (
            Box("outer_left", (-w / 2 - t / 2, 0), (t, h + 2 * t)),
            Box("outer_right", (w / 2 + t / 2, 0), (t, h + 2 * t)),
            Box("outer_bottom", (0, -h / 2 - t / 2), (w, t)),
            Box("outer_top", (0, h / 2 + t / 2), (w, t)),
            Box("start_left", (-1.35, -2.45), (0.80, 0.45)),
            Box("start_right", (1.35, -2.45), (0.80, 0.45)),
            Box("lower_block", (0, -1.45), (1.60, 0.45)),
            Box("middle_left", (-1.35, -0.55), (1.30, t)),
            Box("middle_right", (1.35, -0.55), (1.30, t)),
            Box("upper_block", (0.60, 0.55), (2.80, 0.65)),
            Box("top_partition", (-0.75, 1.55), (2.50, t)),
            Box("top_stub", (1.80, 1.55), (0.40, t)),
            Box("top_vertical", (0.50, 2.65), (t, 0.70)),
        )

    def valid_position(self, point, extra_margin=0.0):
        point = np.asarray(point)
        if point.shape != (2,) or not np.isfinite(point).all():
            return False
        radius = self.robot_radius + self.clearance_margin + extra_margin
        if abs(point[0]) + radius > self.width / 2 or abs(point[1]) + radius > self.length / 2:
            return False
        return all(box.clearance(point) >= radius for box in self.boxes)

    def validate(self, lidar_height=0.20, goal_radius=0.20):
        if not 0 < lidar_height < min(box.height for box in self.boxes):
            raise ValueError("LiDAR plane must be above ground and below all obstacle tops")
        if not self.valid_position(self.start):
            raise ValueError("start overlaps geometry or lacks turning clearance")
        if not self.valid_position(self.goal, extra_margin=goal_radius):
            raise ValueError("goal region overlaps geometry or lacks turning clearance")
        return self

    def sample_reset(self, count, rng, position_jitter=0.0, yaw_jitter=0.0, attempts=100):
        if count < 1 or not isinstance(count, int):
            raise ValueError("count must be a positive integer")
        if not np.isfinite([position_jitter, yaw_jitter]).all() or min(position_jitter, yaw_jitter) < 0:
            raise ValueError("jitter must be finite and nonnegative")
        positions = []
        for _ in range(count):
            for _ in range(attempts):
                candidate = np.asarray(self.start) + rng.uniform(-position_jitter, position_jitter, 2)
                if self.valid_position(candidate):
                    positions.append(candidate)
                    break
            else:
                raise RuntimeError("cannot sample a collision-free reset pose")
        return np.asarray(positions), self.start_yaw + rng.uniform(-yaw_jitter, yaw_jitter, count)


def combined_mesh(scene, origins=None):
    """Real cuboid surfaces, optionally baked at many environment origins.

    RayCaster accepts one static mesh. A vectorized
    scene therefore cannot point it at one cloned mesh per environment. The
    collision rooms remain cloned normally; this function builds one scan-only
    mesh containing every translated room.
    """
    if origins is None:
        origins = np.zeros((1, 3), dtype=np.float64)
    origins = np.asarray(origins, dtype=np.float64)
    if origins.ndim != 2 or origins.shape[1] not in (2, 3) or not np.isfinite(origins).all():
        raise ValueError("origins must have finite shape (N, 2) or (N, 3)")
    if len(origins) < 1:
        raise ValueError("at least one environment origin is required")
    if origins.shape[1] == 2:
        origins = np.column_stack((origins, np.zeros(len(origins))))
    points, faces = [], []
    corner_signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                             [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    triangles = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                          [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                          [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    mesh_index = 0
    for origin in origins:
        for box in scene.boxes:
            points.extend(corner_signs * np.array([*box.size, box.height]) / 2
                          + np.array([*box.center, box.height / 2]) + origin)
            faces.extend(triangles + 8 * mesh_index)
            mesh_index += 1
    return np.asarray(points), np.asarray(faces, dtype=np.int32)


def define_parallel_scan_mesh(stage, prim_path, origins, scene=None):
    """Define the one world-space scan mesh required by RayCaster.

    This setup-time operation is the only device-to-host transfer performed by
    the parallel environment. Stepping, observations and rewards stay on GPU.
    """
    from pxr import Gf, UsdGeom

    scene = (scene or FixedScene()).validate()
    if hasattr(origins, "torch"):
        origins = origins.torch
    if hasattr(origins, "detach"):
        origins = origins.detach().cpu().numpy()
    points, faces = combined_mesh(scene, origins)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateVisibilityAttr("invisible")
    return prim_path


def write_usda(path, scene=None):
    """Emit inspectable USD source without needing Isaac Sim installed.

    Collision cubes are the physical obstacles. ScanMesh is a duplicate surface
    used ONLY for ray casting, without collision, so there is no duplicate
    physical contact. For vectorized scenes its vertices must be baked at each
    environment origin; exporting one room does not configure cloning.
    """
    scene = (scene or FixedScene()).validate()
    points, faces = combined_mesh(scene)
    lines = ['#usda 1.0', '(', '    defaultPrim = "Navigation"', '    metersPerUnit = 1',
             '    upAxis = "Z"', ')', 'def Xform "Navigation"', '{',
             '    def Xform "Obstacles"', '    {']
    for box in scene.boxes:
        lines += [f'        def Cube "{box.name}" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsRigidBodyAPI"])',
                  '        {', '            double size = 1',
                  f'            double3 xformOp:scale = ({box.size[0]}, {box.size[1]}, {box.height})',
                  f'            double3 xformOp:translate = ({box.center[0]}, {box.center[1]}, {box.height / 2})',
                  '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                  '            bool physics:collisionEnabled = true',
                  '            bool physics:kinematicEnabled = true', '        }']
    lines += ['    }', '    def Mesh "ScanMesh"', '    {',
              '        uniform token subdivisionScheme = "none"',
              '        token visibility = "invisible"',
              '        point3f[] points = [' + ', '.join(f'({x}, {y}, {z})' for x, y, z in points) + ']',
              '        int[] faceVertexCounts = [' + ', '.join('3' for _ in faces) + ']',
              '        int[] faceVertexIndices = [' + ', '.join(str(int(i)) for i in faces.ravel()) + ']',
              '    }', '}']
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return Path(path)


def write_manifest(path, scene=None):
    from dataclasses import asdict
    scene = (scene or FixedScene()).validate()
    manifest = asdict(scene)
    manifest['boxes'] = [asdict(box) for box in scene.boxes]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
