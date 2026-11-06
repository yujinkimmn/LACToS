import os
from pathlib import Path

import pytest
import torch

from wafer_diffusion.security import enable_offline_guard

torch.set_num_threads(2)
enable_offline_guard()


@pytest.fixture
def upstream():
    root = os.environ.get("LACTOS_REVIEWED_SOURCE")
    if not root:
        pytest.fail("Set LACTOS_REVIEWED_SOURCE to the pinned local source; integration tests must not silently skip.")
    return Path(root)
