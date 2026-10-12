"""Functional installation checks, run automatically by scripts/setup.sh.

Run directly with the environment's Python. CUDA builds exercise the selected GPU; CPU-only builds exercise the CPU. No checkpoints or dataset are needed.
"""

import importlib
import importlib.metadata
import os
import sys
from pathlib import Path
import unittest


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]


class EnvironmentTests(unittest.TestCase):
    def test_python_version(self):
        self.assertGreaterEqual(sys.version_info[:2], (3, 10))

    def test_required_package_versions(self):
        from packaging.requirements import Requirement

        requirements_file = PROJECT_DIRECTORY / 'environments/requirements.txt'

        for line in requirements_file.read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue

            requirement = Requirement(line)

            if requirement.marker and not requirement.marker.evaluate():
                continue

            with self.subTest(package=requirement.name):
                installed_version = importlib.metadata.version(requirement.name)
                self.assertIn(installed_version, requirement.specifier)

    def test_declared_dependencies(self):
        from packaging.requirements import Requirement

        requirements_file = PROJECT_DIRECTORY / 'environments/requirements.txt'

        for line in requirements_file.read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue

            package = Requirement(line)
            if package.marker and not package.marker.evaluate():
                continue

            distribution = importlib.metadata.distribution(package.name)
            for declaration in distribution.requires or []:
                dependency = Requirement(declaration)
                if dependency.marker and not dependency.marker.evaluate({'extra': ''}):
                    continue

                with self.subTest(package=package.name, dependency=dependency.name):
                    version = importlib.metadata.version(dependency.name)
                    self.assertIn(version, dependency.specifier)

    def test_project_and_library_imports(self):
        modules = (
            'adaptive_video',
            'scipy.linalg',
            'pandas',
            'onnx',
            'transformers.models.rt_detr_v2.modeling_rt_detr_v2',
            'sklearn',
            'pycocotools.mask',
            'trackeval',
            'ultralytics',
            'matplotlib',
        )

        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_numpy_and_opencv(self):
        import cv2
        import numpy as np

        from packaging.version import Version

        self.assertGreaterEqual(Version(np.__version__), Version('1.26'))
        self.assertGreaterEqual(Version(cv2.__version__), Version('4.8'))

        image = np.zeros((2, 2, 3), dtype=np.uint8)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.assertEqual(grayscale.shape, (2, 2))

    def test_scientific_and_export_operations(self):
        import numpy as np
        import pandas as pd
        import onnx
        from onnx import numpy_helper
        from scipy.linalg import solve

        matrix = np.array([[3., 1.], [1., 2.]])
        target = np.array([9., 8.])
        np.testing.assert_allclose(solve(matrix, target), [2., 3.])
        self.assertEqual(pd.DataFrame({'value': target})['value'].sum(), 17.)

        values = matrix.astype(np.float32)
        tensor = numpy_helper.from_array(values)
        np.testing.assert_array_equal(numpy_helper.to_array(tensor), values)

        node = onnx.helper.make_node('Identity', ['input'], ['output'])
        inputs = [onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [2, 2])]
        outputs = [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [2, 2])]
        graph = onnx.helper.make_graph([node], 'environment_check', inputs, outputs)
        onnx.checker.check_model(onnx.helper.make_model(graph))

    def test_torch_and_torchvision_operations(self):
        import numpy as np
        from packaging.version import Version
        import torch
        import torchvision

        self.assertGreaterEqual(Version(torch.__version__), Version('2.4'))
        self.assertGreaterEqual(Version(torchvision.__version__), Version('0.16'))

        device = os.environ.get('FRAMEVIGIL_DEVICE', 'auto')

        if device == 'auto':
            device = 'cuda' if torch.version.cuda is not None else 'cpu'

        self.assertIn(device, {'cpu', 'cuda'})

        if device == 'cuda':
            self.assertIsNotNone(torch.version.cuda, 'Installed Torch is a CPU-only build')
            self.assertTrue(torch.cuda.is_available(), 'CUDA is unavailable')

        values = torch.from_numpy(np.ones((2, 2), dtype=np.float32)).to(device)
        product = values @ values
        self.assertTrue(torch.all(product == 2).item())

        convolution = torch.nn.Conv2d(3, 4, kernel_size=3).to(device)
        image = torch.ones((1, 3, 16, 16), device=device)
        features = convolution(image)
        self.assertEqual(tuple(features.shape), (1, 4, 14, 14))
        self.assertTrue(torch.isfinite(features).all().item())

        boxes = torch.tensor([[0., 0., 1., 1.]], device=device)
        scores = torch.tensor([0.9], device=device)
        kept_indices = torchvision.ops.nms(boxes, scores, 0.5)

        self.assertEqual(kept_indices.tolist(), [0])

        if device == 'cuda':
            torch.cuda.synchronize()


if __name__ == '__main__':
    unittest.main(verbosity=2)
