"""Read Oura API v2 JSON exports without changing the source files."""

import json
import warnings
from pathlib import Path

import pandas as pd

from ... import heart_rate_schema
from ..base import BaseLoader


class OuraLoader(BaseLoader):
    """Load a local directory of participant-level Oura API v2 responses.

    The root directory must contain one directory per participant. Each
    participant directory contains one directory per Oura endpoint, and each
    endpoint directory contains one or more JSON response pages::

        data_dir/
            participant_id/
                sleep/
                    data.json
                    data_2.json
                daily_sleep/
                    data.json
                heartrate/
                    data.json

    Every JSON file must contain an object with a ``data`` list, matching the
    Oura API v2 response envelope. Pagination files are merged in filename
    order. Identical repeated records are collapsed; records with the same
    provider identifier but different content raise an error rather than being
    overwritten silently. Missing endpoint directories produce empty tables.

    Daily and nightly loaders use Oura's ``day`` as ``calendar_date``.
    Subdaily loaders expose offset-aware instants as timezone-naive UTC in
    ``utc_time``. Sleep loaders additionally expose participant-local clock
    times derived from the offsets included in each sleep record.
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(self.data_dir)
        self.ids = sorted(path.name for path in self.data_dir.iterdir() if path.is_dir())

    def get_user_ids(self) -> list:
        return self.ids.copy()

    def _read_records(self, user_id: str, metric: str) -> list[dict]:
        if user_id not in self.ids:
            raise ValueError(f"Unknown Oura user_id: {user_id}")
        metric_dir = self.data_dir / user_id / metric
        if not metric_dir.is_dir():
            return []

        records = {}
        for file in sorted(metric_dir.glob("*.json"), key=lambda path: path.name):
            with file.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            # Some copied exports contain numbered placeholder pages such as "0".
            if isinstance(payload, str) and payload.isdigit() and file.stem.startswith("data_"):
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise ValueError(f"Invalid Oura export structure: {file}")
            for record in payload["data"]:
                if not isinstance(record, dict):
                    raise ValueError(f"Invalid Oura record in {file}")
                if "id" in record:
                    key = ("id", record["id"])
                elif metric == "heartrate" and "timestamp" in record:
                    key = ("timestamp_source", record["timestamp"], record.get("source"))
                else:
                    key = ("record", json.dumps(record, sort_keys=True))
                previous = records.get(key)
                if previous is not None and previous != record:
                    raise ValueError(f"Conflicting Oura {metric} export records for {key}")
                records[key] = record
        return list(records.values())

    @staticmethod
    def _day_filter(records: list[dict], start_date=None, end_date=None) -> list[dict]:
        start = pd.Timestamp(start_date).date() if start_date is not None else None
        end = pd.Timestamp(end_date).date() if end_date is not None else None
        return [
            record
            for record in records
            if "day" in record
            and (start is None or pd.Timestamp(record["day"]).date() >= start)
            and (end is None or pd.Timestamp(record["day"]).date() < end)
        ]

    @staticmethod
    def _utc_time(value) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            raise ValueError(f"Oura subdaily timestamp lacks an offset: {value}")
        return timestamp.tz_convert("UTC").tz_localize(None)

    @staticmethod
    def _local_time(value) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            raise ValueError(f"Oura sleep timestamp lacks an offset: {value}")
        return timestamp.tz_localize(None)

    @staticmethod
    def _timestamp_offset(value):
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return None
        return pd.Timedelta(timestamp.utcoffset())

    @staticmethod
    def _day_offsets(records, timestamp_field):
        offsets = {}
        for record in records:
            if "day" not in record:
                continue
            offset = OuraLoader._timestamp_offset(record.get(timestamp_field))
            if offset is None:
                continue
            day = pd.Timestamp(record["day"]).date()
            offsets.setdefault(day, set()).add(offset)
        return {
            day: next(iter(values))
            for day, values in offsets.items()
            if len(values) == 1
        }

    def _heart_rate_local_context(self, user_id, utc_time):
        """Infer local offsets from offset-bearing Oura records without interpolation."""
        offsets = pd.Series(pd.NaT, index=utc_time.index, dtype="timedelta64[ns]")
        offset_source = pd.Series(pd.NA, index=utc_time.index, dtype="object")
        blocked = pd.Series(False, index=utc_time.index)

        # An offset carried by an event is the strongest evidence while that
        # event is in progress. If its endpoints disagree, the transition time
        # is unknowable without an IANA timezone and the interval stays local-unknown.
        intervals = (
            ("sleep", "bedtime_start", "bedtime_end"),
            ("workout", "start_datetime", "end_datetime"),
            ("session", "start_datetime", "end_datetime"),
        )
        for metric, start_field, end_field in intervals:
            for record in self._read_records(user_id, metric):
                if start_field not in record or end_field not in record:
                    continue
                start = pd.Timestamp(record[start_field])
                end = pd.Timestamp(record[end_field])
                if start.tzinfo is None or end.tzinfo is None:
                    continue
                start_utc = start.tz_convert("UTC").tz_localize(None)
                end_utc = end.tz_convert("UTC").tz_localize(None)
                in_interval = (utc_time >= start_utc) & (utc_time < end_utc)
                if start.utcoffset() != end.utcoffset():
                    blocked |= in_interval
                    offsets.loc[in_interval] = pd.NaT
                    offset_source.loc[in_interval] = pd.NA
                    continue

                candidate = pd.Timedelta(start.utcoffset())
                conflict = in_interval & offsets.notna() & offsets.ne(candidate)
                blocked |= conflict
                offsets.loc[conflict] = pd.NaT
                offset_source.loc[conflict] = pd.NA
                assign = in_interval & offsets.isna() & ~blocked
                offsets.loc[assign] = candidate
                offset_source.loc[assign] = metric

        def assign_days(day_offsets, source_name):
            candidates = pd.Series(
                pd.NaT, index=utc_time.index, dtype="timedelta64[ns]"
            )
            ambiguous = pd.Series(False, index=utc_time.index)
            eligible = offsets.isna() & ~blocked
            for day, candidate in day_offsets.items():
                matches_day = (utc_time + candidate).dt.date == day
                matches_day &= eligible
                conflict = matches_day & candidates.notna() & candidates.ne(candidate)
                ambiguous |= conflict
                candidates.loc[matches_day & candidates.isna()] = candidate
            assign = candidates.notna() & ~ambiguous & eligible
            offsets.loc[assign] = candidates.loc[assign]
            offset_source.loc[assign] = source_name
            blocked.loc[ambiguous] = True

        activity_records = self._read_records(user_id, "daily_activity")
        assign_days(self._day_offsets(activity_records, "timestamp"), "daily_activity")

        # Sleep is a fallback day anchor where daily activity is absent. Its
        # explicit interval above still takes precedence for overnight samples.
        sleep_records = [
            record
            for record in self._read_records(user_id, "sleep")
            if record.get("type") == "long_sleep"
        ]
        assign_days(self._day_offsets(sleep_records, "bedtime_end"), "sleep_day")
        return offsets, offset_source

    def load_sleep_sessions(self, user_id, start_date=None, end_date=None, sleep_type=None):
        """Return Oura sessions, including naps, without making them nightly features."""
        records = self._day_filter(self._read_records(user_id, "sleep"), start_date, end_date)
        if sleep_type is not None:
            records = [record for record in records if record.get("type") == sleep_type]
        columns = [
            "id", "calendar_date", "type", "main_sleep", "local_time",
            "local_end_time", "time_in_bed_seconds",
        ]
        rows = [
            {
                "id": record["id"],
                "calendar_date": pd.Timestamp(record["day"]).normalize(),
                "type": record.get("type"),
                "main_sleep": record.get("type") == "long_sleep",
                "local_time": self._local_time(record["bedtime_start"]),
                "local_end_time": self._local_time(record["bedtime_end"]),
                "time_in_bed_seconds": record.get("time_in_bed"),
            }
            for record in records
        ]
        return pd.DataFrame(rows, columns=columns).sort_values("local_time").reset_index(drop=True) if rows else pd.DataFrame(columns=columns)

    def load_sleep_summary(self, user_id, start_date=None, end_date=None, drop_duplicates=True):
        """Return long_sleep nights in the shared sleep-summary schema."""
        records = self._day_filter(self._read_records(user_id, "sleep"), start_date, end_date)
        records = [record for record in records if record.get("type") == "long_sleep"]
        columns = [
            "sleep_summary_id", "calendar_date", "local_time", "local_end_time",
            "utc_time", "utc_end_time", "unix_timestamp_in_ms", "timezone_offset_in_ms",
            "duration_in_ms", "awake_duration_in_ms", "deep_sleep_duration_in_ms",
            "light_sleep_duration_in_ms", "rem_sleep_duration_in_ms",
            "unmeasurable_sleep_duration_in_ms", "overall_sleep_score",
        ]
        if not records:
            return pd.DataFrame(columns=columns)

        scores = {}
        for record in self._day_filter(self._read_records(user_id, "daily_sleep"), start_date, end_date):
            day = record["day"]
            if day in scores and scores[day] != record.get("score"):
                raise ValueError(f"Conflicting Oura daily sleep scores for {day}")
            scores[day] = record.get("score")

        rows = []
        duration_fields = {
            "total_sleep_duration": "duration_in_ms",
            "awake_time": "awake_duration_in_ms",
            "deep_sleep_duration": "deep_sleep_duration_in_ms",
            "light_sleep_duration": "light_sleep_duration_in_ms",
            "rem_sleep_duration": "rem_sleep_duration_in_ms",
        }
        for record in records:
            start = pd.Timestamp(record["bedtime_start"])
            end = pd.Timestamp(record["bedtime_end"])
            if start.tzinfo is None or end.tzinfo is None:
                raise ValueError(f"Oura sleep session {record['id']} lacks a time offset")
            row = {
                "sleep_summary_id": record["id"],
                "calendar_date": pd.Timestamp(record["day"]).normalize(),
                "local_time": start.tz_localize(None),
                "local_end_time": end.tz_localize(None),
                "utc_time": start.tz_convert("UTC").tz_localize(None),
                "utc_end_time": end.tz_convert("UTC").tz_localize(None),
                "unix_timestamp_in_ms": start.value // 1_000_000,
                "timezone_offset_in_ms": int(start.utcoffset().total_seconds() * 1000),
                "unmeasurable_sleep_duration_in_ms": 0,
                "overall_sleep_score": scores.get(record["day"]),
            }
            for source, target in duration_fields.items():
                value = record.get(source)
                row[target] = pd.NA if value is None else value * 1000
            rows.append(row)
        frame = pd.DataFrame(rows, columns=columns).sort_values("calendar_date")
        if drop_duplicates:
            frame = (
                frame.sort_values(["calendar_date", "duration_in_ms", "sleep_summary_id"])
                .drop_duplicates(subset="calendar_date", keep="last")
                .sort_values("calendar_date")
            )
        return frame.reset_index(drop=True)

    def _load_sleep_stages(
        self, user_id, start_date=None, end_date=None, main_sleep_only=True
    ):
        """Decode five-minute phases into contiguous stage runs.

        Run durations follow Oura's five-minute grid and can differ by seconds
        from the session's exact ``time_in_bed`` interval.
        """
        stage_map = {"1": "deep", "2": "light", "3": "rem", "4": "awake"}
        columns = [
            "sleep_summary_id", "utc_time", "local_time", "timezone_offset_in_ms",
            "unix_timestamp_in_ms", "duration_in_ms", "type", "local_time_uncertain",
        ]
        rows = []
        uncertain_sessions_in_window = set()
        requested_start = pd.Timestamp(start_date) if start_date is not None else None
        requested_end = pd.Timestamp(end_date) if end_date is not None else None
        for record in self._read_records(user_id, "sleep"):
            if main_sleep_only and record.get("type") != "long_sleep":
                continue
            phases = record.get("sleep_phase_5_min") or ""
            if not phases:
                continue
            start = pd.Timestamp(record["bedtime_start"])
            end = pd.Timestamp(record["bedtime_end"])
            if start.tzinfo is None:
                raise ValueError(f"Oura sleep session {record['id']} lacks a time offset")
            if end.tzinfo is None:
                raise ValueError(f"Oura sleep session {record['id']} lacks an end time offset")
            uncertain_local_clock = start.utcoffset() != end.utcoffset()
            if uncertain_local_clock and (
                requested_start is None or end.tz_localize(None) > requested_start
            ) and (
                requested_end is None or start.tz_localize(None) < requested_end
            ):
                uncertain_sessions_in_window.add(record["id"])
            run_start = 0
            for index in range(1, len(phases) + 1):
                if index < len(phases) and phases[index] == phases[run_start]:
                    continue
                instant = start + pd.Timedelta(minutes=5 * run_start)
                rows.append(
                    {
                        "sleep_summary_id": record["id"],
                        "utc_time": instant.tz_convert("UTC").tz_localize(None),
                        "local_time": instant.tz_localize(None),
                        "timezone_offset_in_ms": int(start.utcoffset().total_seconds() * 1000),
                        "unix_timestamp_in_ms": instant.value // 1_000_000,
                        "duration_in_ms": (index - run_start) * 5 * 60_000,
                        "type": stage_map.get(phases[run_start], "unmeasurable"),
                        "local_time_uncertain": uncertain_local_clock,
                    }
                )
                run_start = index
        frame = pd.DataFrame(rows, columns=columns)
        within_clock_window = ~frame["local_time_uncertain"]
        if requested_start is not None:
            within_clock_window &= frame["local_time"] >= requested_start
        if requested_end is not None:
            within_clock_window &= frame["local_time"] < requested_end
        frame = frame[
            within_clock_window
            | frame["sleep_summary_id"].isin(uncertain_sessions_in_window)
        ]
        return frame.sort_values("local_time").reset_index(drop=True)

    def load_sleep_stage(self, user_id, start_date=None, end_date=None):
        """Return the stage trajectory for nightly ``long_sleep`` sessions."""
        return self._load_sleep_stages(
            user_id, start_date, end_date, main_sleep_only=True
        )

    def load_sleep_session_stages(self, user_id, start_date=None, end_date=None):
        """Return staged Oura sleep sessions, including naps and rests."""
        return self._load_sleep_stages(
            user_id, start_date, end_date, main_sleep_only=False
        )

    def load_heart_rate(self, user_id, start_date=None, end_date=None, time_col="local_time"):
        """Return Oura HR with UTC instants and conservatively inferred local time.

        ``daily_activity.timestamp`` supplies the primary local-day offset.
        Offset-bearing sleep, workout, and session intervals take precedence,
        and long-sleep offsets provide a fallback when activity is absent.
        Offsets are never interpolated across missing days or offset changes.
        """
        heart_rate_schema.validate_time_col(time_col)
        columns = [
            "utc_time", "local_time", "timezone", "heart_rate", "source",
            "timestamp_conflict", "local_time_source", "local_time_uncertain",
        ]
        rows = [
            {
                "utc_time": self._utc_time(record["timestamp"]),
                "heart_rate": record.get("bpm"),
                "source": record.get("source"),
            }
            for record in self._read_records(user_id, "heartrate")
        ]
        frame = pd.DataFrame(rows, columns=["utc_time", "heart_rate", "source"])
        if frame.empty:
            return pd.DataFrame(columns=columns)

        # Local bounds cannot be converted exactly before offsets are known.
        # A one-day UTC margin safely narrows the records that need enrichment.
        if time_col == "utc_time":
            frame = heart_rate_schema.filter_by_time(
                frame, start_date, end_date, time_col
            )
        else:
            broad_start = heart_rate_schema.normalize_time_bound(
                start_date, time_col
            )
            broad_end = heart_rate_schema.normalize_time_bound(end_date, time_col)
            if broad_start is not None:
                frame = frame[
                    frame["utc_time"] >= broad_start - pd.Timedelta(days=1)
                ]
            if broad_end is not None:
                frame = frame[frame["utc_time"] < broad_end + pd.Timedelta(days=1)]
        if frame.empty:
            return pd.DataFrame(columns=columns)

        frame["timezone"], frame["local_time_source"] = (
            self._heart_rate_local_context(user_id, frame["utc_time"])
        )
        frame["local_time"] = frame["utc_time"] + frame["timezone"]
        frame["local_time_uncertain"] = frame["local_time"].isna()
        frame["timestamp_conflict"] = frame.duplicated("utc_time", keep=False)
        if time_col == "local_time" and frame["local_time_uncertain"].any():
            unresolved = int(frame["local_time_uncertain"].sum())
            warnings.warn(
                f"{unresolved} Oura heart-rate samples have no unambiguous local "
                "offset; they remain available only through utc_time."
            )
        return (
            heart_rate_schema.filter_by_time(
                frame, start_date, end_date, time_col
            )
            .sort_values(["utc_time", "source"])
            .reset_index(drop=True)
        )[columns]

    def load_hrv_summary(self, user_id, start_date=None, end_date=None):
        """Return Oura-provided average nightly rMSSD (milliseconds)."""
        rows = [
            {
                "sleep_summary_id": record["id"],
                "calendar_date": pd.Timestamp(record["day"]).normalize(),
                "oura_average_rmssd_ms": record.get("average_hrv"),
            }
            for record in self._day_filter(self._read_records(user_id, "sleep"), start_date, end_date)
            if record.get("type") == "long_sleep"
        ]
        return pd.DataFrame(rows, columns=["sleep_summary_id", "calendar_date", "oura_average_rmssd_ms"])

    def load_hrv_samples(self, user_id, start_date=None, end_date=None):
        """Return Oura-provided five-minute rMSSD estimates, not BBI-derived HRV."""
        columns = ["sleep_summary_id", "utc_time", "local_time", "oura_rmssd_ms"]
        rows = []
        for record in self._read_records(user_id, "sleep"):
            if record.get("type") != "long_sleep":
                continue
            series = record.get("hrv") or {}
            items = series.get("items") or []
            if not items:
                continue
            anchor = pd.Timestamp(series["timestamp"])
            if anchor.tzinfo is None:
                raise ValueError(f"Oura HRV series for {record['id']} lacks a time offset")
            interval = pd.Timedelta(seconds=float(series["interval"]))
            for index, value in enumerate(items):
                if value is None:
                    continue
                instant = anchor + index * interval
                rows.append(
                    {
                        "sleep_summary_id": record["id"],
                        "utc_time": instant.tz_convert("UTC").tz_localize(None),
                        "local_time": instant.tz_localize(None),
                        "oura_rmssd_ms": value,
                    }
                )
        frame = pd.DataFrame(rows, columns=columns)
        start = pd.Timestamp(start_date) if start_date is not None else None
        end = pd.Timestamp(end_date) if end_date is not None else None
        if start is not None and start.tzinfo is not None:
            start = start.tz_convert("UTC").tz_localize(None)
        if end is not None and end.tzinfo is not None:
            end = end.tz_convert("UTC").tz_localize(None)
        if start is not None:
            frame = frame[frame["utc_time"] >= start]
        if end is not None:
            frame = frame[frame["utc_time"] < end]
        return frame.sort_values("utc_time").reset_index(drop=True)

    def load_sleep_respiration_summary(self, user_id, start_date=None, end_date=None):
        """Return Oura's nightly respiratory-rate estimate, not a continuous signal."""
        rows = [
            {
                "sleep_summary_id": record["id"],
                "calendar_date": pd.Timestamp(record["day"]).normalize(),
                "oura_average_breaths_per_minute": record.get("average_breath"),
            }
            for record in self._day_filter(self._read_records(user_id, "sleep"), start_date, end_date)
            if record.get("type") == "long_sleep"
        ]
        return pd.DataFrame(rows, columns=["sleep_summary_id", "calendar_date", "oura_average_breaths_per_minute"])

    def load_steps_summary(self, user_id, start_date=None, end_date=None):
        """Return Oura daily steps keyed by its local calendar day."""
        rows = [
            {
                "calendar_date": pd.Timestamp(record["day"]).normalize(),
                "steps": (
                    pd.NA if record.get("steps") == 0 else record.get("steps")
                ),
            }
            for record in self._day_filter(
                self._read_records(user_id, "daily_activity"), start_date, end_date
            )
        ]
        frame = pd.DataFrame(rows, columns=["calendar_date", "steps"])
        frame["steps"] = frame["steps"].astype("Int64")
        return frame

    def load_pulse_ox_summary(self, user_id, start_date=None, end_date=None):
        """Return Oura's provider daily SpO2 percentage, not raw pulse-ox samples."""
        rows = [
            {
                "calendar_date": pd.Timestamp(record["day"]).normalize(),
                "oura_daily_spo2_percentage": (
                    pd.NA
                    if (record.get("spo2_percentage") or {}).get("average") == 0
                    else (record.get("spo2_percentage") or {}).get("average")
                ),
            }
            for record in self._day_filter(
                self._read_records(user_id, "daily_spo2"), start_date, end_date
            )
        ]
        frame = pd.DataFrame(
            rows, columns=["calendar_date", "oura_daily_spo2_percentage"]
        )
        frame["oura_daily_spo2_percentage"] = frame[
            "oura_daily_spo2_percentage"
        ].astype("Float64")
        return frame

    def load_readiness(self, user_id, start_date=None, end_date=None):
        """Return Oura's daily readiness score and vendor contributor scores."""
        columns = [
            "calendar_date", "oura_readiness_score", "temperature_deviation",
            "temperature_trend_deviation", "readiness_activity_balance",
            "readiness_body_temperature", "readiness_hrv_balance",
            "readiness_previous_day_activity", "readiness_previous_night",
            "readiness_recovery_index", "readiness_resting_heart_rate",
            "readiness_sleep_balance",
        ]
        rows = []
        for record in self._day_filter(
            self._read_records(user_id, "daily_readiness"), start_date, end_date
        ):
            row = {
                "calendar_date": pd.Timestamp(record["day"]).normalize(),
                "oura_readiness_score": record.get("score"),
                "temperature_deviation": record.get("temperature_deviation"),
                "temperature_trend_deviation": record.get("temperature_trend_deviation"),
            }
            row.update(
                {f"readiness_{key}": value for key, value in (record.get("contributors") or {}).items()}
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def load_resilience(self, user_id, start_date=None, end_date=None):
        """Return Oura's categorical daily resilience and vendor contributors."""
        columns = [
            "calendar_date", "oura_resilience_level", "resilience_daytime_recovery",
            "resilience_sleep_recovery", "resilience_stress",
        ]
        rows = []
        for record in self._day_filter(
            self._read_records(user_id, "daily_resilience"), start_date, end_date
        ):
            row = {
                "calendar_date": pd.Timestamp(record["day"]).normalize(),
                "oura_resilience_level": record.get("level"),
            }
            row.update(
                {f"resilience_{key}": value for key, value in (record.get("contributors") or {}).items()}
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def load_recovery(self, user_id, start_date=None, end_date=None):
        raise NotImplementedError("Oura readiness is vendor-specific; use load_readiness")

    def load_activity_summary(self, user_id, start_date=None, end_date=None):
        # TODO: map Oura activity dimensions to generic pywearable features.
        raise NotImplementedError("Oura activity features are not mapped yet")

    def load_intensity(self, user_id, start_date=None, end_date=None):
        # TODO: decode Oura's five-minute activity class series.
        raise NotImplementedError("Oura activity intensity is not mapped yet")

    def load_pulse_ox(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame(columns=["utc_time", "pulse_ox"])

    def load_respiration(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame(columns=["utc_time", "breaths_per_minute"])

    def load_sleep_pulse_ox(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame(columns=["utc_time", "pulse_ox"])

    def load_sleep_respiration(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame(columns=["utc_time", "breaths_per_minute"])
