"""Measure lighting invariance of saved static-3DGS predictions at fixed poses."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image


def pixels(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def evaluate_static_response(run):
    config = json.loads((run / "training_config.json").read_text())
    if not config["model"]["eval"] or config["model"]["train_test_exp"]:
        raise ValueError("Static-lighting checks require --eval without per-image exposure")
    source = Path(config["model"]["source_path"])
    frames = json.loads((source / "transforms_test.json").read_text())["frames"]
    expected = {frame["file_path"]: frame for frame in frames}
    results = {}
    for method in sorted((run / "test").iterdir()):
        if not method.is_dir():
            continue
        manifest = json.loads((method / "manifest.json").read_text())
        if len(manifest) != len(frames) or {row["file_path"] for row in manifest} != set(expected):
            raise ValueError("Render manifest does not match the test split")
        groups = defaultdict(list)
        for row in manifest:
            frame = expected[row["file_path"]]
            if row["time_index"] != frame["time_index"]:
                raise ValueError("Manifest sun ID does not match source data")
            pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
            # Group by the full pose, rather than trusting a camera ID alone.
            groups[tuple(pose.flatten())].append(row)
        repeated = [rows for rows in groups.values() if len({row["time_index"] for row in rows}) > 1]
        if not repeated:
            raise ValueError("No fixed-pose, different-sun test samples")
        checked = []
        for rows in repeated:
            reference = pixels(method / "renders" / rows[0]["image"])
            maximum = 0
            for row in rows[1:]:
                other = pixels(method / "renders" / row["image"])
                maximum = max(maximum, int(np.abs(other.astype(np.int16) - reference).max()))
            checked.append({"camera_index": rows[0]["camera_index"], "count": len(rows),
                            "sun_ids": sorted({row["time_index"] for row in rows}),
                            "prediction_max_abs_difference_uint8": maximum})
        # Deterministic representative: first camera, then greatest sun-angle
        # separation. Selection uses input metadata, not reconstruction error.
        representative = min(repeated, key=lambda rows: rows[0]["file_path"])
        def angle_pair():
            pairs = []
            for index, left in enumerate(representative):
                for right in representative[index + 1:]:
                    if left["time_index"] == right["time_index"]:
                        continue
                    directions = [np.asarray(row["sun_direction"], dtype=np.float64) for row in (left, right)]
                    cosine = float(np.dot(*directions) / (np.linalg.norm(directions[0]) * np.linalg.norm(directions[1])))
                    pairs.append((cosine, left, right))
            return min(pairs, key=lambda pair: pair[0])
        cosine, left, right = angle_pair()
        gt = [pixels(method / "gt" / row["image"]).astype(np.float32) / 255 for row in (left, right)]
        predictions = [pixels(method / "renders" / row["image"]).astype(np.float32) / 255 for row in (left, right)]
        results[method.name] = {
            "same_pose_different_sun_groups": len(repeated),
            "groups_with_identical_predictions": sum(row["prediction_max_abs_difference_uint8"] == 0 for row in checked),
            "groups": checked,
            "representative": {
                "camera_index": left["camera_index"], "frames": [left, right],
                "sun_angle_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
                "gt_mae_between_lights": float(np.abs(gt[0] - gt[1]).mean()),
                "prediction_mae_between_lights": float(np.abs(predictions[0] - predictions[1]).mean()),
                "gt_mse_between_lights": float(np.square(gt[0] - gt[1]).mean()),
            },
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    run = parser.parse_args().model_path.resolve()
    result = evaluate_static_response(run)
    (run / "lighting_response.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
