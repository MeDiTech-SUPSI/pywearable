import pandas as pd
import pytest

from pywearable.loader.labfront.loader import (
    LabfrontLoader,
    _LABFRONT_SOURCE_FILE_COL,
)


def _summary(summary_id="night-1"):
    return pd.DataFrame(
        {
            "sleepSummaryId": [summary_id],
            "validation": ["ENHANCED_FINAL"],
            "durationInMs": [100 * 60_000],
            "lightSleepDurationInMs": [100 * 60_000],
            "deepSleepDurationInMs": [0],
            "remSleepDurationInMs": [0],
            "awakeDurationInMs": [98 * 60_000],
            "unmeasurableSleepDurationInMs": [0],
        }
    )


def _snapshot(
    file_name,
    awake_minutes=98,
    awake_start="2024-01-01 02:00",
    summary_id="night-1",
):
    return pd.DataFrame(
        {
            "sleepSummaryId": [summary_id, summary_id],
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
        LabfrontLoader._reconcile_sleep_stage_snapshots(
            rows, _summary(), on_conflict="raise"
        )


def test_missing_summary_accepts_identical_repeats_in_default_policy():
    repeated = pd.concat([_snapshot("first"), _snapshot("second")])
    empty_summary = pd.DataFrame()
    result = LabfrontLoader._reconcile_sleep_stage_snapshots(repeated, empty_summary)
    assert len(result) == 2


def test_default_policy_drops_only_ambiguous_stage_rows():
    repeated = pd.concat([_snapshot("first"), _snapshot("second")])
    revised = pd.concat([repeated, _snapshot("revised", awake_minutes=90)])
    unambiguous = _snapshot("only", summary_id="night-2")

    with pytest.warns(UserWarning, match="night-1"):
        result = LabfrontLoader._reconcile_sleep_stage_snapshots(
            pd.concat([revised, unambiguous]), _summary(summary_id="night-2")
        )

    assert set(result["sleepSummaryId"]) == {"night-2"}


def test_raise_policy_rejects_revisions_without_a_summary():
    revised = pd.concat(
        [_snapshot("first"), _snapshot("revised", awake_minutes=90)]
    )
    with pytest.raises(ValueError, match="without summary"):
        LabfrontLoader._reconcile_sleep_stage_snapshots(
            revised, pd.DataFrame(), on_conflict="raise"
        )


def test_last_policy_uses_last_loaded_snapshot():
    revised = pd.concat(
        [_snapshot("first"), _snapshot("last", awake_minutes=90)]
    )
    result = LabfrontLoader._reconcile_sleep_stage_snapshots(
        revised, pd.DataFrame(), on_conflict="last"
    )
    assert len(result) == 2
    assert result.loc[result["type"] == "awake", "durationInMs"].sum() == 90 * 60_000


def test_invalid_conflict_policy_is_rejected():
    with pytest.raises(ValueError, match="on_conflict must be one of"):
        LabfrontLoader._reconcile_sleep_stage_snapshots(
            _snapshot("first"), pd.DataFrame(), on_conflict="unknown"
        )
