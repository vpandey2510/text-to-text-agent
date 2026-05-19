import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_PATH = PROJECT_ROOT / "text-agent.py"


@pytest.fixture
def text_agent():
    """Load text-agent.py as a module (hyphenated filename is not importable normally)."""
    spec = importlib.util.spec_from_file_location("text_agent", AGENT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["text_agent"] = module
    spec.loader.exec_module(module)
    return module
