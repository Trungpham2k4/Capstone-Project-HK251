# -------------------------
# Utilities
# -------------------------
from datetime import datetime
import uuid


def now_iso():
    return datetime.now().isoformat()

def make_id(prefix="A"):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"