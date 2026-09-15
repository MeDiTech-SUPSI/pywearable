import pandas as pd
import pytest

from pywearable.loader.labfront.loader import (
    LabfrontLoader,
    _LABFRONT_SOURCE_FILE_COL,
)


def _summary():
    return pd.DataFrame(
        {
            "sleepSummaryId": ["night-1"],
            "validation": ["ENHANCED_FINAL"],
            "durationInMs": [100 * 60_000],
            "lightSleepDurationInMs": [100 * 60_000],
            "deepSleepDurationInMs": [0],
            "remSleepDurationInMs": [0],
            "awakeDurationInMs": [98 * 60_000],
            "unmeasurableSleepDurationInMs": [0],
        }
    )


def _snapshot(file_name, awake_minutes=98, awake_start="2024-01-01 02:00"):
    return pd.DataFrame(
        {
            "sleepSummaryId": ["night-1", "night-1"],
            "isoDate": pd.to_datetime(["2024-01-01 00:00", awake_start]),
            "durationInMs": [100 * 60_000, awake_minutes * 60_000],
            "type": ["light", "awake"],
            _LABFRONT_SOURCE_FILE_COL: [file_name, file_name],
        }
    )


def test_repeated_and_revised_exports_select_one_coherent_snapshot():
    repeated = [_snapshot(f"export-{i}") for i in range(4)]
    rows = pd.concat([*repeated, _snapshot("revised", awake_minutes=90)])
    result = LabfrontLoader._reconcile_sleep_stage_snapshots(rows, _summary())
    assert len(result) == 2
    assert result.loc[result["type"] == "awake", "durationInMs"].sum() == 98 * 60_000
    assert _LABFRONT_SOURCE_FILE_COL not in result


def test_conflicting_matching_snapshots_are_not_silently_combined():
    rows = pd.concat(
        [_snapshot("first"), _snapshot("second", awake_start="2024-01-01 03:00")]
    )
    with pytest.raises(ValueError, match="Conflicting"):
        LabfrontLoader._reconcile_sleep_stage_snapshots(rows, _summary())


def test_missing_summary_accepts_identical_repeats_but_rejects_revisions():
    repeated = pd.concat([_snapshot("first"), _snapshot("second")])
    empty_summary = pd.DataFrame()
    result = LabfrontLoader._reconcile_sleep_stage_snapshots(repeated, empty_summary)
    assert len(result) == 2
    revised = pd.concat([repeated, _snapshot("revised", awake_minutes=90)])
    with pytest.raises(ValueError, match="without summary"):
        LabfrontLoader._reconcile_sleep_stage_snapshots(revised, empty_summary)
