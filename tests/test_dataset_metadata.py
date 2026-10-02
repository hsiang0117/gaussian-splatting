"""Keep frame identities and lighting metadata without conditioning the model."""

from argparse import ArgumentParser
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from arguments import PipelineParams
from gaussian_renderer import render
from scene.cameras import Camera
from scene.dataset_readers import readCamerasFromTransforms
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from tools.eval_static_lighting import evaluate_static_response
from tools.eval_test_groups import group_metrics


class DatasetMetadataTests(unittest.TestCase):
    def test_header_only_reader_preserves_repeated_stems_and_sun_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for index in (0, 1):
                path = root / f"cam{index:02d}/images/0000.png"
                path.parent.mkdir(parents=True)
                Image.new("RGBA", (23, 19)).save(path)
                frames.append({"file_path": path.relative_to(root).as_posix(),
                               "transform_matrix": np.eye(4).tolist(),
                               "sun_direction": [0, 1, 0], "camera_index": index, "time_index": 0})
            (root / "transforms_train.json").write_text(json.dumps({"camera_angle_x": 1., "frames": frames}))
            with patch.object(Image.Image, "load", side_effect=AssertionError("eager decode")):
                cameras = readCamerasFromTransforms(str(root), "transforms_train.json", "", False, False)
            self.assertEqual([camera.image_name for camera in cameras], [frame["file_path"] for frame in frames])
            self.assertEqual(len({camera.image_name for camera in cameras}), 2)
            self.assertEqual([camera.camera_index for camera in cameras], [0, 1])
            self.assertEqual(cameras[0].sun_direction, [0, 1, 0])
            self.assertEqual((cameras[0].width, cameras[0].height), (23, 19))

    def make_run(self, root):
        dataset = root / "dataset"
        dataset.mkdir()
        frames = [{"file_path": "cam00/images/0007.png", "camera_index": 0, "time_index": 7,
                   "sun_direction": [0, 1, 0], "transform_matrix": np.eye(4).tolist()},
                  {"file_path": "cam00/images/0001.png", "camera_index": 0, "time_index": 1,
                   "sun_direction": [1, 0, 0], "transform_matrix": np.eye(4).tolist()}]
        (dataset / "transforms_test.json").write_text(json.dumps({"frames": frames}))
        (dataset / "transforms_train.json").write_text(json.dumps({"frames": [frames[1]]}))
        (root / "training_config.json").write_text(json.dumps({"model": {
            "source_path": str(dataset), "eval": True, "train_test_exp": False}}))
        method = root / "test/ours_1000"
        (method / "renders").mkdir(parents=True)
        (method / "gt").mkdir()
        manifest = [{"image": f"{index:05d}.png", **frame} for index, frame in enumerate(frames)]
        (method / "manifest.json").write_text(json.dumps(manifest))
        for index, row in enumerate(manifest):
            Image.new("RGB", (8, 8), (100, 100, 100)).save(method / "renders" / row["image"])
            Image.new("RGB", (8, 8), (index * 200, 0, 0)).save(method / "gt" / row["image"])
        (root / "per_view.json").write_text(json.dumps({"ours_1000": {"PSNR": {
            "00000.png": 20., "00001.png": 30.}}}))
        return method

    def test_grouping_checks_test_coverage_and_actual_sun_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method = self.make_run(root)
            result = group_metrics(root)
            self.assertEqual(result["heldout_sun_ids"], [7])
            self.assertEqual(result["results"]["ours_1000"]["heldout_sun"], {"n": 1, "PSNR": 20.})
            (method / "manifest.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "full test split"):
                group_metrics(root)

    def test_response_reports_gt_changes_and_constant_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method = self.make_run(root)
            result = evaluate_static_response(root)["ours_1000"]
            self.assertEqual(result["same_pose_different_sun_groups"], 1)
            self.assertEqual(result["groups_with_identical_predictions"], 1)
            self.assertGreater(result["representative"]["gt_mae_between_lights"], 0)
            self.assertEqual(result["representative"]["prediction_mae_between_lights"], 0)
            Image.new("RGB", (8, 8), (101, 101, 101)).save(method / "renders/00001.png")
            changed = evaluate_static_response(root)["ours_1000"]
            self.assertEqual(changed["groups_with_identical_predictions"], 0)


@unittest.skipUnless(torch.cuda.is_available(), "Renderer test requires CUDA")
class StaticRendererTests(unittest.TestCase):
    def test_recorded_sun_does_not_change_pixels_or_geometry_gradients(self):
        torch.eye(4, device="cuda").inverse()
        cameras = [Camera((64, 64), index, np.eye(3), np.array([0., 0., 4.]),
                          1., 1., None, Image.new("RGB", (64, 64)), None,
                          f"cam00/images/{index:04d}.png", index, data_device="cpu",
                          sun_direction=direction, camera_index=0, time_index=index)
                   for index, direction in enumerate(([0, 1, 0], [1, 0, 0]))]
        rng = np.random.default_rng(4)
        cloud = BasicPointCloud(rng.uniform(-.4, .4, (24, 3)).astype(np.float32),
                                np.full((24, 3), .65, np.float32), np.zeros((24, 3), np.float32))
        model = GaussianModel(3)
        model.create_from_pcd(cloud, cameras, 1.)
        parser = ArgumentParser()
        pipe = PipelineParams(parser).extract(parser.parse_args([]))
        images, gradients = [], []
        for camera in cameras:
            model._xyz.grad = None
            image = render(camera, model, pipe, torch.zeros(3, device="cuda"))["render"]
            image.mean().backward()
            images.append(image.detach())
            gradients.append(model._xyz.grad.detach().clone())
        torch.testing.assert_close(images[0], images[1], rtol=0, atol=0)
        # Rasterizer backward sums contributions with CUDA atomics; even two
        # identical renders can differ in the last gradient bits.
        torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-4, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
