# -------------------------
# Utilities
# -------------------------
from datetime import datetime
import uuid
import yaml


def now_iso():
    return datetime.now().isoformat()


def make_id(prefix="A"):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def load_params(param_path: str) -> dict:
    try:
        with open(param_path, "r") as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        return {}
