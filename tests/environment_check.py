"""Functional installation checks, run automatically by scripts/setup.sh.

Run directly with the environment's Python. CUDA builds exercise the selected GPU; CPU-only builds exercise the CPU. No checkpoints or dataset are needed.
"""

import importlib
import importlib.metadata
from pathlib import Path
import unittest


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]


class EnvironmentTests(unittest.TestCase):
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

    def test_project_and_library_imports(self):
        modules = (
            'adaptive_video',
            'scipy',
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

        self.assertEqual(np.__version__, '1.26.4')
        self.assertEqual(cv2.__version__, '4.8.0')

        image = np.zeros((2, 2, 3), dtype=np.uint8)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.assertEqual(grayscale.shape, (2, 2))

    def test_torch_and_torchvision_operations(self):
        import numpy as np
        from packaging.version import Version
        import torch
        import torchvision

        self.assertGreaterEqual(Version(torch.__version__), Version('2.1'))
        self.assertGreaterEqual(Version(torchvision.__version__), Version('0.16'))

        device = 'cuda' if torch.version.cuda is not None else 'cpu'

        if device == 'cuda':
            self.assertTrue(torch.cuda.is_available(), 'CUDA is unavailable')

        values = torch.from_numpy(np.ones((2, 2), dtype=np.float32)).to(device)
        product = values @ values
        self.assertTrue(torch.all(product == 2).item())

        boxes = torch.tensor([[0., 0., 1., 1.]], device=device)
        scores = torch.tensor([0.9], device=device)
        kept_indices = torchvision.ops.nms(boxes, scores, 0.5)

        self.assertEqual(kept_indices.tolist(), [0])

        if device == 'cuda':
            torch.cuda.synchronize()


if __name__ == '__main__':
    unittest.main(verbosity=2)
