"""Test configuration helpers."""
from __future__ import annotations

import site
from pathlib import Path

# Ensure the repository root is treated as a site directory so the local ``pbai``
# package can be imported without requiring an editable install in ad-hoc test
# runs (e.g., CI calling ``pytest`` directly from the checkout).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
site.addsitedir(str(PROJECT_ROOT))
