import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def load_sample():
    def _load(name: str):
        return json.loads((SAMPLES_DIR / name).read_text())
    return _load
