#!/usr/bin/env python3
"""Build Day-3 feature matrix from the Bloomberg clean Corn/Wheat panel.

Reproducible and deterministic: re-running on the same clean input rewrites
identical feature values under data/features/.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_utils import (  # noqa: E402
    FEATURE_COLUMNS,
    build_corn_wheat_features,
    load_clean_pair,
    to_rel,
    write_features,
)

SOURCE_PANEL = "corn_wheat_panel_bloomberg.parquet"
DATA_SOURCE = "bloomberg"


def main() -> None:
    panel = load_clean_pair(SOURCE_PANEL)
    features = build_corn_wheat_features(panel)
    manifest = write_features(
        features,
        source_panel=f"data/clean/{SOURCE_PANEL}",
        data_source=DATA_SOURCE,
        stem="features",
    )
    z = features["ratio_zscore_50"]
    print(f"Wrote {manifest['outputs']['parquet']}")
    print(f"Wrote {manifest['outputs']['csv']}")
    print(f"Wrote {manifest['outputs']['manifest']}")
    print(f"Rows: {manifest['rows']}  |  {manifest['date_start']} → {manifest['date_end']}")
    print(
        f"Paper feature ratio_zscore_50 non-null: {int(z.notna().sum())} "
        f"(window={50}, first valid={z.first_valid_index().date() if z.first_valid_index() is not None else None})"
    )
    print(f"Feature columns: {list(FEATURE_COLUMNS)}")
    print(f"Source panel: {to_rel(ROOT / 'data' / 'clean' / SOURCE_PANEL)}")


if __name__ == "__main__":
    main()
