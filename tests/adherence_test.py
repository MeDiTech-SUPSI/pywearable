from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pywearable import adherence


class AdherenceLoader:
    heart_rate_time_columns = ("utc_time", "local_time")
    hrv_source = "bbi"
    hrv_time_columns = ("utc_time", "local_time")

    def __init__(self, heart_rate=None, bbi=None, sleep_summary=None, sessions=None):
        self.heart_rate = (
            heart_rate if heart_rate is not None else self._empty_heart_rate()
        )
        self.bbi = bbi if bbi is not None else self._empty_bbi()
        self.sleep_summary = (
            sleep_summary if sleep_summary is not None else self._empty_sleep_summary()
        )
        self.sessions = sessions if sessions is not None else pd.DataFrame()
        self.sleep_session_loads = 0

    @staticmethod
    def _empty_heart_rate():
        return pd.DataFrame(
            columns=["utc_time", "local_time", "timezone", "heart_rate"]
        )

    @staticmethod
    def _empty_bbi():
        return pd.DataFrame(columns=["utc_time", "local_time", "timezone", "bbi"])

    @staticmethod
    def _empty_sleep_summary():
        return pd.DataFrame(
            columns=["sleep_summary_id", "calendar_date", "local_time"]
        )

    def get_user_ids(self):
        return ["user"]

    def load_heart_rate(
        self, user_id, start_date=None, end_date=None, time_col="local_time"
    ):
        return self.heart_rate.copy()

    def load_bbi(self, user_id, start_date=None, end_date=None):
        return self.bbi.copy()

    def load_sleep_summary(
        self, user_id, start_date=None, end_date=None, drop_duplicates=True
    ):
        return self.sleep_summary.copy()

    def load_sleep_sessions(self, user_id, start_date=None, end_date=None):
        self.sleep_session_loads += 1
        return self.sessions.copy()


def _heart_rate(times):
    local = pd.DatetimeIndex(times)
    return pd.DataFrame(
        {
            "local_time": local,
            "utc_time": local - pd.Timedelta(hours=1),
            "timezone": pd.Timedelta(hours=1),
            "heart_rate": 60,
        }
    )


def _bbi(start, count, value=1000.0):
    local = pd.date_range(start, periods=count, freq="s")
    return pd.DataFrame(
        {
            "local_time": local,
            "utc_time": local - pd.Timedelta(hours=1),
            "timezone": pd.Timedelta(hours=1),
            "bbi": value,
        }
    )


def _sleep_summary(dates, validations=None):
    dates = pd.DatetimeIndex(dates)
    frame = pd.DataFrame(
        {
            "sleep_summary_id": [
                f"night-{index}-{day:%Y%m%d}"
                for index, day in enumerate(dates)
            ],
            "calendar_date": dates,
            "local_time": dates - pd.Timedelta(hours=1),
        }
    )
    if validations is not None:
        frame["validation"] = validations
    return frame


def test_five_minute_adherence_uses_minute_occupancy_and_bbi_duration():
    loader = AdherenceLoader(
        heart_rate=_heart_rate(
            [
                "2024-01-01 00:00:10",
                "2024-01-01 00:01:10",
                "2024-01-01 00:01:40",
                "2024-01-01 00:03:10",
                "2024-01-01 00:02:10",
            ]
        ),
        bbi=_bbi("2024-01-01 00:00:00", 150),
    )
    loader.heart_rate.loc[
        loader.heart_rate["local_time"].eq(pd.Timestamp("2024-01-01 00:02:10")),
        "heart_rate",
    ] = 0

    result = adherence.get_adherence(
        loader,
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 00:05",
        resolution="5min",
        time_reference="local",
    )

    row = result.iloc[0]
    assert row["heart_rate_adherence"] == pytest.approx(3 / 5)
    assert row["bbi_adherence"] == pytest.approx(0.5)
    assert pd.isna(row["sleep_adherence"])


def test_daily_sleep_adherence_is_binary_on_expected_local_dates():
    result = adherence.get_adherence(
        AdherenceLoader(sleep_summary=_sleep_summary(["2024-01-01", "2024-01-03"])),
        metric="sleep",
        start_date="2024-01-01",
        end_date="2024-01-04",
        resolution="daily",
    )

    assert result["sleep_adherence"].tolist() == [1.0, 0.0, 1.0]


def test_sleep_adherence_at_an_upper_calendar_level_is_available_nights_over_dates():
    result = adherence.get_adherence(
        AdherenceLoader(sleep_summary=_sleep_summary(["2024-01-01", "2024-01-03"])),
        metric="sleep",
        start_date="2024-01-01",
        end_date="2024-01-04",
        resolution=["daily", "weekly"],
    )
    assert result.iloc[0]["sleep_adherence"] == pytest.approx(2 / 3)


def test_sleep_skips_fixed_levels_and_enters_at_its_native_daily_grain():
    result = adherence.get_adherence(
        AdherenceLoader(
            sleep_summary=_sleep_summary(["2024-01-01", "2024-01-03"])
        ),
        metric="sleep",
        start_date="2024-01-01",
        end_date="2024-01-08",
        resolution=["5min", "weekly"],
        min_coverage=[0.7, None],
    )
    assert result.iloc[0]["sleep_adherence"] == pytest.approx(2 / 7)


def test_sleep_hierarchy_applies_thresholds_after_its_native_grain():
    dates = [
        *pd.date_range("2024-01-01", periods=7, freq="D"),
        *pd.date_range("2024-01-08", periods=4, freq="D"),
    ]
    result = adherence.get_adherence(
        AdherenceLoader(sleep_summary=_sleep_summary(dates)),
        metric="sleep",
        start_date="2024-01-01",
        end_date="2024-01-15",
        resolution=["5min", "weekly", "yearly"],
        min_coverage=[0.7, 0.7, None],
    )
    assert result.iloc[0]["sleep_adherence"] == pytest.approx(0.5)


def test_sleep_validation_excludes_user_entered_summaries_by_default():
    summaries = _sleep_summary(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03"],
        ["MANUAL", "AUTO_MANUAL", "AUTO_FINAL", "MANUAL"],
    )
    loader = AdherenceLoader(sleep_summary=summaries)

    filtered = adherence.get_adherence(
        loader,
        metric="sleep",
        start_date="2024-01-01",
        end_date="2024-01-04",
        resolution="weekly",
    )
    unfiltered = adherence.get_adherence(
        loader,
        metric="sleep",
        start_date="2024-01-01",
        end_date="2024-01-04",
        resolution="weekly",
        check_sleep_validation=False,
    )

    assert filtered.iloc[0]["sleep_adherence"] == pytest.approx(1 / 3)
    assert unfiltered.iloc[0]["sleep_adherence"] == 1


def test_sleep_validation_filter_does_not_change_heart_rate_or_bbi():
    loader = AdherenceLoader(
        heart_rate=_heart_rate(["2024-01-01 00:00:10"]),
        bbi=_bbi("2024-01-01 00:00:00", 60),
        sleep_summary=_sleep_summary(["2024-01-01"], ["MANUAL"]),
    )
    filtered = adherence.get_adherence(
        loader,
        start_date="2024-01-01",
        end_date="2024-01-02",
        resolution="daily",
    )
    unfiltered = adherence.get_adherence(
        loader,
        start_date="2024-01-01",
        end_date="2024-01-02",
        resolution="daily",
        check_sleep_validation=False,
    )

    assert filtered.iloc[0]["sleep_adherence"] == 0
    assert unfiltered.iloc[0]["sleep_adherence"] == 1
    assert filtered.iloc[0]["heart_rate_adherence"] == unfiltered.iloc[0][
        "heart_rate_adherence"
    ]
    assert filtered.iloc[0]["bbi_adherence"] == unfiltered.iloc[0][
        "bbi_adherence"
    ]


@pytest.mark.parametrize(
    "sleep_resolution",
    ["nightly", "cycle_night", "cycle_physiological", "daily"],
)
def test_sleep_calendar_adherence_is_independent_of_non_awake_domain(
    sleep_resolution,
):
    result = adherence.get_adherence(
        AdherenceLoader(
            sleep_summary=_sleep_summary(["2024-01-01", "2024-01-03"])
        ),
        metric="sleep",
        start_date="2024-01-01",
        end_date="2024-01-08",
        resolution=[sleep_resolution, "weekly"],
        min_coverage=[0.7, None],
    )
    assert result.iloc[0]["sleep_adherence"] == pytest.approx(2 / 7)


def test_sleep_adherence_is_zero_in_awake_cycle_hierarchies():
    result = adherence.get_adherence(
        AdherenceLoader(
            sleep_summary=_sleep_summary(
                pd.date_range("2024-01-01", periods=7, freq="D")
            )
        ),
        metric="sleep",
        start_date="2024-01-01",
        end_date="2024-01-08",
        resolution=["cycle_awake", "weekly"],
        min_coverage=[0.7, None],
    )
    assert result.iloc[0]["sleep_adherence"] == 0


def test_supported_empty_streams_are_zero_but_provider_bbi_is_unavailable():
    empty = adherence.get_adherence(
        AdherenceLoader(),
        metric=["heart_rate", "bbi"],
        start_date="2024-01-01",
        end_date="2024-01-01 00:05",
        resolution="5min",
        time_reference="local",
    )
    assert empty.iloc[0]["heart_rate_adherence"] == 0
    assert empty.iloc[0]["bbi_adherence"] == 0

    provider = AdherenceLoader()
    provider.hrv_source = "provider_samples"
    provider.load_bbi = SimpleNamespace(called=False)
    unavailable = adherence.get_adherence(
        provider,
        metric="bbi",
        start_date="2024-01-01",
        end_date="2024-01-01 00:05",
        resolution="5min",
        time_reference="local",
    )
    assert pd.isna(unavailable.iloc[0]["bbi_adherence"])


def test_cycle_hierarchy_applies_each_intermediate_coverage_threshold():
    sessions = pd.DataFrame(
        {
            "id": ["short", "long"],
            "calendar_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "type": "main_sleep",
            "main_sleep": True,
            "local_time": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-02 00:00"]
            ),
            "local_end_time": pd.to_datetime(
                ["2024-01-01 01:00", "2024-01-02 03:00"]
            ),
            "time_in_bed_seconds": [3600, 10800],
        }
    )
    heart_rate = pd.concat(
        [
            _heart_rate(pd.date_range("2024-01-01 00:00", periods=60, freq="min")),
            _heart_rate(pd.date_range("2024-01-02 00:00", periods=120, freq="min")),
        ],
        ignore_index=True,
    )

    loader = AdherenceLoader(heart_rate=heart_rate, sessions=sessions)
    result = adherence.get_adherence(
        loader,
        metric="heart_rate",
        start_date="2024-01-01",
        end_date="2024-01-03",
        resolution=["1min", "cycle_night", "monthly"],
        min_coverage=[0.7, 0.7, None],
    )

    assert result.iloc[0]["heart_rate_adherence"] == pytest.approx(0.5)
    assert list(result.columns) == ["heart_rate_adherence"]
    assert loader.sleep_session_loads == 1


def test_invalid_cycle_is_a_failed_child_of_an_upper_period():
    sessions = pd.DataFrame(
        {
            "id": ["night"],
            "calendar_date": pd.to_datetime(["2024-01-01"]),
            "type": ["main_sleep"],
            "main_sleep": [True],
            "local_time": pd.to_datetime(["2024-01-01 00:00"]),
            "local_end_time": pd.to_datetime(["2024-01-01 01:00"]),
            "time_in_bed_seconds": [3600],
        }
    )
    result = adherence.get_adherence(
        AdherenceLoader(
            heart_rate=_heart_rate(
                pd.date_range("2024-01-01 00:00", periods=60, freq="min")
            ),
            sessions=sessions,
        ),
        metric="heart_rate",
        start_date="2024-01-01",
        end_date="2024-01-03",
        resolution=["1min", "cycle_night", "monthly"],
        min_coverage=[0.7, 0.7, None],
    )
    assert result.iloc[0]["heart_rate_adherence"] == pytest.approx(0.5)


def test_invalid_cycle_is_missing_when_it_is_the_final_resolution():
    result = adherence.get_adherence(
        AdherenceLoader(),
        metric="heart_rate",
        start_date="2024-01-01",
        end_date="2024-01-02",
        resolution="cycle_night",
    )
    assert pd.isna(result.iloc[0]["heart_rate_adherence"])


def test_bbi_coverage_is_clipped_to_one():
    result = adherence.get_adherence(
        AdherenceLoader(bbi=_bbi("2024-01-01 00:00", 400)),
        metric="bbi",
        start_date="2024-01-01",
        end_date="2024-01-01 00:05",
        resolution="5min",
        time_reference="local",
    )
    assert result.iloc[0]["bbi_adherence"] == 1


def test_bounds_are_required_and_sleep_calendar_adherence_is_local():
    with pytest.raises(ValueError, match="start_date and end_date are required"):
        adherence.get_adherence(AdherenceLoader(), end_date="2024-01-02")

    with pytest.raises(ValueError, match="sleep adherence uses local calendar dates"):
        adherence.get_adherence(
            AdherenceLoader(),
            metric="sleep",
            start_date="2024-01-01",
            end_date="2024-01-02",
            resolution="daily",
            time_reference="utc",
        )


def test_an_empty_participant_selection_preserves_the_index_contract():
    loader = AdherenceLoader()
    loader.get_user_ids = lambda: []
    result = adherence.get_adherence(
        loader,
        start_date="2024-01-01",
        end_date="2024-01-02",
        resolution="daily",
    )
    assert result.empty
    assert result.index.names == ["user_id", "calendar_date"]


def test_daily_wearing_time_remains_an_hours_compatibility_wrapper():
    result = adherence.get_daily_wearing_time(
        AdherenceLoader(
            heart_rate=_heart_rate(
                [
                    "2024-01-01 00:00:10",
                    "2024-01-01 00:01:10",
                    "2024-01-01 00:03:10",
                ]
            )
        ),
        user_id="user",
        start_date="2024-01-01",
        end_date="2024-01-02",
    )
    assert result.iloc[0] == pytest.approx(3 / 60)
    assert result.name == "daily-hr"


def test_daily_wearing_time_uses_actual_duration_for_partial_days():
    result = adherence.get_daily_wearing_time(
        AdherenceLoader(
            heart_rate=_heart_rate(
                pd.date_range(
                    "2024-01-01 12:00", periods=60, freq="min"
                )
            )
        ),
        user_id="user",
        start_date="2024-01-01 12:00",
        end_date="2024-01-01 13:00",
    )
    assert result.iloc[0] == 1
