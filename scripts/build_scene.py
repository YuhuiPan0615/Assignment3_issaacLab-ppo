"""Run from the repository root: python -m scripts.build_scene."""
from pathlib import Path
from jetbot_navigation.scene import FixedScene, write_usda, write_manifest


def main():
    root = Path(__file__).resolve().parents[1]
    scene = FixedScene().validate()
    write_usda(root / "assets" / "navigation_scene.usda", scene)
    write_manifest(root / "assets" / "scene_manifest.json", scene)
    print("Exported assets/navigation_scene.usda and assets/scene_manifest.json")


if __name__ == "__main__":
    main()
