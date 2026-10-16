"""VerifiableRL needs PyTorch (the ``[rl]`` extra); without it, these are skipped."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_*.py"]
