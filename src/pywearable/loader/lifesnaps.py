import os
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .. import schema
from .base import BaseLoader


class LifeSnapsMongoDBLoader(BaseLoader):
    pass


class LifeSnapsParquetLoader(BaseLoader):
    """Loader for the LifeSnaps dataset.

    Before using this code, make sure to convert
    the LifeSnaps dataset in Parquet format
    using the tool available as:

    ``
    pywearable convert lifesnaps-to-parquet --help
    ``

    Parameters
    ----------
    root_folder : :class:`os.PathLike`
        Path to the folder containing LifeSnaps data in
        parquet format.
    """

    heart_rate_time_columns = ("local_time",)
    hrv_source = "provider_samples"
    hrv_time_columns = ("local_time",)
    respiratory_rate_source = "provider_summary"
    spo2_source = "provider_summary"
    activity_source = "samples"
    activity_time_columns = (schema.LOCAL_TIME_COL,)
    activity_primary_resolution = "1min"
    provider_features = frozenset(
        {
            "fitbit_stress_management_score",
            "fitbit_stress_sleep_points",
            "fitbit_stress_max_sleep_points",
            "fitbit_stress_responsiveness_points",
            "fitbit_stress_max_responsiveness_points",
            "fitbit_stress_exertion_points",
            "fitbit_stress_max_exertion_points",
        }
    )

    def __init__(self, root_folder: os.PathLike):
        root_folder = Path(root_folder)
        if not root_folder.exists():
            raise FileNotFoundError(f"Folder {root_folder} does not exist.")
        self.root_folder = root_folder
        self.user_list = [x.name for x in root_folder.iterdir() if x.is_dir()]

    def get_user_ids(self) -> list:
        return self.user_list

    def _check_user_id(self, user_id: str):
        if not user_id in self.user_list:
            raise ValueError(
                f"User id {user_id} is not among LifeSnaps valid user ids."
            )

    def load_respiration(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ):
        """Load respiration data.

        .. warning::

            LifeSnaps dataset doesn't have raw respiration data. Therefore
            this function will always return an empty :class:`pd.DataFrame`.
            You can use the function :func:`load_respiration_summary` that
            returns a summary of night respiration data.


        Parameters
        ----------
        user_id : str
            Unique user identifier.
        start_date : :class:`pandas.Timestamp` or ``None``, optional
        end_date : :class:`pandas.Timestamp` or ``None``, optional

        Returns
        -------
        :class:`pandas.DataFrame`
            Empty dataframe.
        """
        self._check_user_id(user_id)
        return schema.empty_respiration_frame()

    def load_sleep_respiration(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ):
        """Load sleep respiration data.

        .. warning::

            LifeSnaps dataset doesn't have raw sleep respiration data. Therefore
            this function will always return an empty :class:`pd.DataFrame`.
            You can use the function :func:`load_respiration_summary` that
            returns a summary of night respiration data.


        Parameters
        ----------
        user_id : str
            Unique user identifier.
        start_date : :class:`pandas.Timestamp` or ``None``, optional
        end_date : :class:`pandas.Timestamp` or ``None``, optional

        Returns
        -------
        :class:`pandas.DataFrame`
            Empty dataframe.
        """
        self._check_user_id(user_id)
        return schema.empty_respiration_frame()

    def load_pulse_ox(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ):
        """Load pulse oximetry data.

        Parameters
        ----------
        user_id : str
            Unique user identifier.
        start_date : :class:`pandas.Timestamp` or ``None``, optional

        end_date : :class:`pandas.Timestamp` or ``None``, optional


        Returns
        -------
        :class:`pandas.DataFrame`
            Empty dataframe.
        """
        self._check_user_id(user_id)
        return schema.empty_spo2_frame()

    def load_sleep_pulse_ox(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ):
        """Load pulse oximetry data during sleep.

        Parameters
        ----------
        user_id : str
            Unique user identifier.
        start_date : :class:`pandas.Timestamp` or ``None``, optional

        end_date : :class:`pandas.Timestamp` or ``None``, optional


        Returns
        -------
        :class:`pandas.DataFrame`
            Empty dataframe.
        """
        self._check_user_id(user_id)
        return schema.empty_spo2_frame()

    def load_bbi(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ):
        """Load Beat-to-Beat Intervals (BBI).

        .. warning::

            LifeSnaps dataset doesn't have raw BBI data. Therefore
            this function will always return an empty :class:`pd.DataFrame`.


        Parameters
        ----------
        user_id : str
            Unique user identifier.
        start_date : :class:`pandas.Timestamp` or ``None``, optional
            Start date for BBI data retrieval, by default ``None``
        end_date : :class:`pandas.Timestamp` or ``None``, optional
            End date for BBI data retrieval, by default ``None``

        Returns
        -------
        :class:`pandas.DataFrame`
            Empty dataframe.
        """
        self._check_user_id(user_id)
        return pd.DataFrame(columns=["utc_time", "local_time", "bbi"])

    def load_heart_rate(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
        time_col: str = "local_time",
    ) -> pd.DataFrame:
        """Load heart rate data.

        Parameters
        ----------
        user_id : str
            Unique user identifier.
        start_date : :class:`pandas.Timestamp` or ``None``, optional
            Start date for heart rate data retrieval, by default ``None``
        end_date : :class:`pandas.Timestamp` or ``None``, optional
            End date for heart rate data retrieval, by default ``None``

        Notes
        -----
        LifeSnaps heart-rate timestamps are local clock values. The public
        dataset does not provide their UTC offsets, so ``utc_time`` and
        ``timezone`` are returned as missing and UTC filtering is unavailable.

        Returns
        -------
        :class:`pandas.DataFrame`
            Heart-rate samples in the shared schema.
        """
        self._check_user_id(user_id)
        schema.validate_heart_rate_time_col(time_col)
        if time_col == schema.UTC_TIME_COL:
            raise ValueError(
                "LifeSnaps heart-rate data does not provide UTC timestamps; "
                "use time_col='local_time'."
            )
        file_path = self.root_folder / user_id / "heart_rate.parquet"
        if file_path.exists():
            heart_rate = pd.read_parquet(file_path).rename(
                columns={"dateTime": "local_time", "bpm": "heart_rate"}
            )
            heart_rate["local_time"] = pd.to_datetime(heart_rate["local_time"])
            heart_rate["utc_time"] = pd.NaT
            heart_rate["timezone"] = pd.NaT
            heart_rate[schema.TIMESTAMP_CONFLICT_COL] = (
                heart_rate.groupby("local_time")["heart_rate"].transform("nunique")
                > 1
            )
            return (
                schema.filter_heart_rate_by_time(
                    heart_rate, start_date, end_date, time_col
                )
                .sort_values("local_time")
                .reset_index(drop=True)
            )
        else:
            return schema.empty_heart_rate_frame()

    def load_respiration_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        """Return Fitbit's provider-native nightly respiratory-rate summary."""
        self._check_user_id(user_id)
        file_path = self.root_folder / user_id / "respiratory_rate_summary.parquet"
        if not file_path.exists():
            return schema.prepare_respiration_summary(pd.DataFrame())

        respiration = (
            pd.read_parquet(file_path).drop_duplicates().reset_index(drop=True)
        )
        for source, target in {
            "mean": schema.RESPIRATORY_RATE_MEAN_COL,
            "standard_deviation": schema.RESPIRATORY_RATE_STD_COL,
            "full_sleep_breathing_rate": schema.RESPIRATORY_RATE_MEAN_COL,
            "full_sleep_standard_deviation": schema.RESPIRATORY_RATE_STD_COL,
        }.items():
            if source in respiration.columns and target not in respiration.columns:
                respiration[target] = respiration[source]
        respiration["timestamp"] = pd.to_datetime(
            respiration["timestamp"], errors="coerce"
        )
        respiration["calendar_date"] = respiration["timestamp"].dt.normalize()
        sleep_summary = self.load_sleep_summary(user_id=user_id)
        if sleep_summary.empty:
            return schema.prepare_respiration_summary(pd.DataFrame())
        sleep_summary = schema.prepare_sleep_summary(sleep_summary)
        candidates = sleep_summary.dropna(subset=[schema.LOCAL_END_TIME_COL])[
            [
                schema.CALENDAR_DATE_COL,
                schema.LOCAL_TIME_COL,
                schema.LOCAL_END_TIME_COL,
                schema.SLEEP_SUMMARY_ID_COL,
            ]
        ].sort_values(schema.LOCAL_END_TIME_COL)
        if candidates.empty:
            return schema.prepare_respiration_summary(pd.DataFrame())
        matched = respiration.merge(
            candidates,
            left_on="timestamp",
            right_on=schema.LOCAL_END_TIME_COL,
            how="left",
            suffixes=("", "_sleep"),
            validate="many_to_one",
        )
        matched = matched[matched[schema.SLEEP_SUMMARY_ID_COL].notna()].copy()
        matched_date = matched[f"{schema.CALENDAR_DATE_COL}_sleep"]
        matched[schema.CALENDAR_DATE_COL] = matched_date.fillna(
            matched[schema.CALENDAR_DATE_COL]
        )
        respiration = matched.drop(columns=[f"{schema.CALENDAR_DATE_COL}_sleep"])
        respiration["source"] = "fitbit_respiratory_rate_summary"
        respiration = schema.prepare_respiration_summary(respiration)
        if start_date is not None:
            respiration = respiration[
                respiration[schema.CALENDAR_DATE_COL]
                >= pd.Timestamp(start_date).normalize()
            ]
        if end_date is not None:
            respiration = respiration[
                respiration[schema.CALENDAR_DATE_COL]
                < pd.Timestamp(end_date).normalize()
            ]
        return respiration.reset_index(drop=True)

    def load_spo2_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        """Return Fitbit's provider-native sleep-derived SpO2 summary."""
        self._check_user_id(user_id)
        file_path = self.root_folder / user_id / "daily_spo2.parquet"
        if not file_path.exists():
            return schema.prepare_spo2_summary(pd.DataFrame())
        spo2 = pd.read_parquet(file_path).drop_duplicates().reset_index(drop=True)
        for source, target in {
            "average_value": schema.SPO2_MEAN_COL,
            "lower_bound": schema.SPO2_LOWER_BOUND_COL,
            "upper_bound": schema.SPO2_UPPER_BOUND_COL,
        }.items():
            if source in spo2.columns and target not in spo2.columns:
                spo2[target] = spo2[source]
        spo2[schema.CALENDAR_DATE_COL] = pd.to_datetime(
            spo2["timestamp"], errors="coerce"
        ).dt.normalize()
        spo2["source"] = "fitbit_daily_spo2"
        spo2 = schema.prepare_spo2_summary(spo2)
        if start_date is not None:
            spo2 = spo2[
                spo2[schema.CALENDAR_DATE_COL]
                >= pd.Timestamp(start_date).normalize()
            ]
        if end_date is not None:
            spo2 = spo2[
                spo2[schema.CALENDAR_DATE_COL]
                < pd.Timestamp(end_date).normalize()
            ]
        return spo2.reset_index(drop=True)

    def load_pulse_ox_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        """Compatibility alias for :meth:`load_spo2_summary`."""
        return self.load_spo2_summary(user_id, start_date, end_date)

    def load_calories(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        file_path = self.root_folder / user_id / "calories.parquet"
        if file_path.exists():
            calories = pd.read_parquet(file_path).rename(
                columns={"dateTime": "isoDate", "value": "calories"}
            )
            calories["isoDate"] = pd.to_datetime(
                calories["isoDate"],
            )
            if not (start_date is None):
                calories = calories[calories["isoDate"] >= pd.Timestamp(start_date)]
            if not (end_date is None):
                calories = calories[calories["isoDate"] < pd.Timestamp(end_date)]
            return calories
        else:
            return pd.DataFrame(columns=["isoDate", "calories"])

    def load_temperature(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        file_path = self.root_folder / user_id / "computed_temperature.parquet"
        if file_path.exists():
            calories = pd.read_parquet(file_path)
            # .rename(
            # columns={"dateTime": "isoDate", "value": "calories"}
            # )
            print(calories)
            raise
            calories["isoDate"] = pd.to_datetime(calories["isoDate"])
            return calories
        else:
            return pd.DataFrame(columns=["isoDate", "calories"])

    def _load_sleep_stages(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
        drop_duplicates: bool = True,
        main_sleep_only: bool = True,
    ) -> pd.DataFrame:
        file_path = self.root_folder / user_id / "sleep_stages.parquet"
        if file_path.exists():
            def read_stages(path):
                stages = pd.read_parquet(path).rename(
                    columns={
                        "logId": "sleep_summary_id",
                        "dateTime": "utc_time",
                        "seconds": "duration_in_ms",
                        "level": "type",
                    }
                )
                stages["duration_in_ms"] = (
                    pd.to_numeric(stages["duration_in_ms"], errors="coerce") * 1000
                )
                stages["utc_time"] = pd.to_datetime(stages["utc_time"])
                return stages.drop_duplicates(
                    subset=["sleep_summary_id", "utc_time", "type"]
                )

            sleep_stages = read_stages(file_path)
            short_stages_path = (
                self.root_folder / user_id / "sleep_short_stages.parquet"
            )
            if short_stages_path.exists():
                sleep_stages = self._overlay_short_sleep_stages(
                    sleep_stages, read_stages(short_stages_path)
                )

            # LifeSnaps timestamps were converted to participant-local time
            # before release, while the corresponding UTC offsets were withheld.
            sleep_stages["local_time"] = sleep_stages["utc_time"]
            sleep_stages["utc_time"] = pd.NaT
            sleep_stages["timezone_offset_in_ms"] = pd.NA
            sleep_stages["type"] = sleep_stages["type"].replace(
                {
                    "wake": "awake",
                    "restless": "awake",
                    "unknown": "unmeasurable",
                }
            )
            if main_sleep_only:
                sessions = self.load_sleep_sessions(user_id)
                main_sleep_ids = sessions.loc[sessions["main_sleep"], "id"]
                sleep_stages = sleep_stages[
                    sleep_stages["sleep_summary_id"].isin(main_sleep_ids)
                ]
            if drop_duplicates:
                sleep_stages = sleep_stages.drop_duplicates(
                    subset=["sleep_summary_id", "local_time", "type"]
                )

            sleep_stages = self._drop_incoherent_sleep_stages(user_id, sleep_stages)

            if not (start_date is None):
                sleep_stages = sleep_stages[
                    sleep_stages["local_time"] >= pd.Timestamp(start_date)
                ]
            if not (end_date is None):
                sleep_stages = sleep_stages[
                    sleep_stages["local_time"] < pd.Timestamp(end_date)
                ]
            from ..schema import (
                coalesce_sleep_stage_runs,
                normalize_sleep_columns,
            )

            return coalesce_sleep_stage_runs(
                normalize_sleep_columns(sleep_stages)
            )
        else:
            return pd.DataFrame(
                columns=["sleep_summary_id", "local_time", "type", "duration_in_ms"]
            )

    def load_sleep_stage(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        """Return main-sleep stages used by nightly feature extraction."""
        return self._load_sleep_stages(
            user_id, start_date, end_date, drop_duplicates, main_sleep_only=True
        )

    def load_sleep_session_stages(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        """Return Fitbit stages for main sleep and non-main sessions."""
        return self._load_sleep_stages(
            user_id, start_date, end_date, main_sleep_only=False
        )

    @staticmethod
    def _overlay_short_sleep_stages(
        sleep_stages: pd.DataFrame, short_sleep_stages: pd.DataFrame
    ) -> pd.DataFrame:
        """Overlay Fitbit short wake intervals and return non-overlapping runs."""
        if short_sleep_stages.empty:
            return sleep_stages

        overlay_groups = {
            summary_id: LifeSnapsParquetLoader._stage_intervals(group)
            for summary_id, group in short_sleep_stages.groupby(
                "sleep_summary_id", sort=False
            )
        }
        rows = []
        for summary_id, group in sleep_stages.groupby("sleep_summary_id", sort=False):
            base_intervals = LifeSnapsParquetLoader._stage_intervals(group)
            overlay_intervals = overlay_groups.get(summary_id, [])
            boundaries = sorted(
                {
                    boundary
                    for start, end, _ in base_intervals + overlay_intervals
                    for boundary in (start, end)
                }
            )
            base_index = 0
            overlay_index = 0
            combined = []
            for segment_start, segment_end in zip(boundaries, boundaries[1:]):
                while (
                    base_index < len(base_intervals)
                    and base_intervals[base_index][1] <= segment_start
                ):
                    base_index += 1
                if base_index == len(base_intervals):
                    continue
                base_start, base_end, stage_type = base_intervals[base_index]
                if not (base_start <= segment_start < base_end):
                    continue

                while (
                    overlay_index < len(overlay_intervals)
                    and overlay_intervals[overlay_index][1] <= segment_start
                ):
                    overlay_index += 1
                if overlay_index < len(overlay_intervals):
                    overlay_start, overlay_end, overlay_type = overlay_intervals[
                        overlay_index
                    ]
                    if overlay_start <= segment_start < overlay_end:
                        stage_type = overlay_type

                if (
                    combined
                    and combined[-1][2] == stage_type
                    and combined[-1][1] == segment_start
                ):
                    combined[-1] = (combined[-1][0], segment_end, stage_type)
                else:
                    combined.append((segment_start, segment_end, stage_type))

            rows.extend(
                {
                    "sleep_summary_id": summary_id,
                    "utc_time": start,
                    "duration_in_ms": (end - start).total_seconds() * 1000,
                    "type": stage_type,
                }
                for start, end, stage_type in combined
            )
        return pd.DataFrame(
            rows,
            columns=["sleep_summary_id", "utc_time", "duration_in_ms", "type"],
        )

    @staticmethod
    def _stage_intervals(stages: pd.DataFrame) -> list:
        stages = stages.sort_values("utc_time", kind="stable")
        return [
            (
                row.utc_time,
                row.utc_time + pd.to_timedelta(row.duration_in_ms, unit="ms"),
                row.type,
            )
            for row in stages.itertuples()
        ]

    def _drop_incoherent_sleep_stages(
        self, user_id: str, sleep_stages: pd.DataFrame
    ) -> pd.DataFrame:
        """Drop stages when known sleep differs from the Fitbit summary by over 5 min."""
        summary_path = self.root_folder / user_id / "sleep.parquet"
        if not summary_path.exists():
            return sleep_stages

        summaries = pd.read_parquet(summary_path).drop_duplicates("logId", keep="last")
        expected_sleep = pd.to_numeric(
            summaries.set_index("logId")["minutesAsleep"], errors="coerce"
        )
        known_sleep = sleep_stages[
            ~sleep_stages["type"].isin(["awake", "unmeasurable"])
        ]
        observed_sleep = (
            known_sleep.groupby("sleep_summary_id")["duration_in_ms"].sum() / 60_000
        )
        difference = observed_sleep.sub(
            expected_sleep.reindex(observed_sleep.index)
        ).abs()
        conflicting_ids = difference[difference > 5].index.tolist()
        if len(conflicting_ids) > 0:
            displayed_ids = ", ".join(map(str, conflicting_ids[:10]))
            if len(conflicting_ids) > 10:
                displayed_ids += ", ..."
            warnings.warn(
                "Dropped LifeSnaps sleep stages for "
                f"{len(conflicting_ids)} session(s) whose stage durations conflict "
                f"with their sleep summaries: {displayed_ids}.",
                UserWarning,
                stacklevel=2,
            )
            sleep_stages = sleep_stages[
                ~sleep_stages["sleep_summary_id"].isin(conflicting_ids)
            ]
        return sleep_stages.reset_index(drop=True)

    def load_hrv_features(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        file_path = (
            self.root_folder / user_id / "heart_rate_variability_details.parquet"
        )
        if file_path.exists():
            hrv_details = pd.read_parquet(file_path).rename(
                columns={
                    "timestamp": "local_time",
                    "rmssd": schema.HRV_RMSSD_COL,
                    "low_frequency": schema.HRV_LF_POWER_COL,
                    "high_frequency": schema.HRV_HF_POWER_COL,
                    "coverage": schema.HRV_COVERAGE_COL,
                }
            )
            hrv_details["local_time"] = pd.to_datetime(hrv_details["local_time"])
            hrv_details["utc_time"] = pd.NaT
            hrv_details["timezone"] = pd.NaT
            hrv_details["source"] = "fitbit_hrv_details"
            if not (start_date is None):
                hrv_details = hrv_details[
                    hrv_details["local_time"] >= pd.Timestamp(start_date)
                ]
            if not (end_date is None):
                hrv_details = hrv_details[
                    hrv_details["local_time"] < pd.Timestamp(end_date)
                ]
            return hrv_details.sort_values(
                "local_time", ascending=True, ignore_index=True
            )
        else:
            return pd.DataFrame(
                columns=[
                    "local_time",
                    "utc_time",
                    "timezone",
                    schema.HRV_RMSSD_COL,
                    schema.HRV_COVERAGE_COL,
                    schema.HRV_LF_POWER_COL,
                    schema.HRV_HF_POWER_COL,
                    "source",
                ]
            )

    def load_hrv_samples(self, user_id, start_date=None, end_date=None):
        """Return Fitbit-provided five-minute HRV estimates in canonical columns."""
        return self.load_hrv_features(user_id, start_date, end_date)

    def load_hrv_summary(self, user_id, start_date=None, end_date=None):
        """Return Fitbit-provided daily HRV summaries without using them as fallback."""
        file_path = (
            self.root_folder / user_id / "daily_heart_rate_variability_summary.parquet"
        )
        columns = [
            "calendar_date",
            schema.HRV_RMSSD_COL,
            "fitbit_nrem_heart_rate",
            "fitbit_entropy",
        ]
        if not file_path.exists():
            return pd.DataFrame(columns=columns)
        summary = pd.read_parquet(file_path).rename(
            columns={
                "timestamp": "calendar_date",
                "rmssd": schema.HRV_RMSSD_COL,
                "nremhr": "fitbit_nrem_heart_rate",
                "entropy": "fitbit_entropy",
            }
        )
        summary["calendar_date"] = pd.to_datetime(
            summary["calendar_date"]
        ).dt.normalize()
        if start_date is not None:
            summary = summary[
                summary["calendar_date"] >= pd.Timestamp(start_date).normalize()
            ]
        if end_date is not None:
            summary = summary[
                summary["calendar_date"] < pd.Timestamp(end_date).normalize()
            ]
        return summary[columns].sort_values("calendar_date").reset_index(drop=True)

    def load_sleep_sessions(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        """Return all Fitbit sleep sessions, including non-main sessions."""
        columns = [
            "id",
            "calendar_date",
            "type",
            "main_sleep",
            "local_time",
            "local_end_time",
            "time_in_bed_seconds",
        ]
        file_path = self.root_folder / user_id / "sleep.parquet"
        if not file_path.exists():
            return pd.DataFrame(columns=columns)

        sessions = pd.read_parquet(file_path)
        sessions["calendar_date"] = pd.to_datetime(
            sessions["dateOfSleep"]
        ).dt.normalize()
        sessions["local_time"] = pd.to_datetime(sessions["startTime"])
        sessions["local_end_time"] = pd.to_datetime(sessions["endTime"])
        sessions["time_in_bed_seconds"] = (
            pd.to_numeric(sessions["timeInBed"], errors="coerce") * 60
        )
        sessions = sessions.rename(
            columns={"logId": "id", "mainSleep": "main_sleep"}
        )
        if start_date is not None:
            sessions = sessions[
                sessions["calendar_date"] >= pd.Timestamp(start_date).normalize()
            ]
        if end_date is not None:
            sessions = sessions[
                sessions["calendar_date"] < pd.Timestamp(end_date).normalize()
            ]
        return (
            sessions[columns]
            .drop_duplicates(subset="id", keep="last")
            .sort_values("local_time", kind="stable")
            .reset_index(drop=True)
        )

    def load_sleep_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        file_path = self.root_folder / user_id / "sleep.parquet"
        if file_path.exists():
            sleep = pd.read_parquet(file_path).rename(
                columns={
                    "logId": "sleep_summary_id",
                    "dateOfSleep": "calendar_date",
                    "startTime": "utc_time",
                    "endTime": "utc_end_time",
                }
            )
            for col in ["calendar_date", "utc_time", "utc_end_time"]:
                sleep[col] = pd.to_datetime(sleep[col])

            sleep["local_time"] = pd.to_datetime(sleep["utc_time"])
            sleep["local_end_time"] = pd.to_datetime(sleep["utc_end_time"])
            sleep["utc_time"] = pd.NaT
            sleep["utc_end_time"] = pd.NaT
            sleep["timezone_offset_in_ms"] = pd.NA

            if "mainSleep" in sleep.columns:
                sleep = sleep[sleep["mainSleep"].eq(True)]

            minutes_asleep = pd.to_numeric(
                sleep.get("minutesAsleep"), errors="coerce"
            )
            time_in_bed = pd.to_numeric(sleep.get("timeInBed"), errors="coerce")
            wake_source = (
                sleep["wake_minutes"]
                if "wake_minutes" in sleep.columns
                else sleep["minutesAwake"]
            )
            wake_minutes = pd.to_numeric(wake_source, errors="coerce")
            staged_sleep = sleep["type"].eq("stages")
            unmeasurable = pd.Series(0.0, index=sleep.index)
            unmeasurable.loc[staged_sleep] = (
                time_in_bed - minutes_asleep - wake_minutes
            ).loc[staged_sleep].clip(lower=0)
            minutes_awake = time_in_bed - minutes_asleep - unmeasurable
            sleep["duration_in_ms"] = (
                minutes_asleep + unmeasurable
            ) * 60_000
            sleep["awake_duration_in_ms"] = minutes_awake * 60_000
            for stage in ("deep", "light", "rem"):
                if f"{stage}_minutes" in sleep.columns:
                    sleep[f"{stage}_sleep_duration_in_ms"] = (
                        pd.to_numeric(
                            sleep[f"{stage}_minutes"], errors="coerce"
                        )
                        * 60_000
                    )
            sleep["unmeasurable_sleep_duration_in_ms"] = unmeasurable * 60_000

            if not (start_date is None):
                sleep = sleep[
                    sleep["calendar_date"] >= pd.Timestamp(start_date).normalize()
                ]
            if not (end_date is None):
                sleep = sleep[
                    sleep["calendar_date"] < pd.Timestamp(end_date).normalize()
                ]
            if drop_duplicates:
                sleep = sleep.drop_duplicates(subset="calendar_date", keep="last")
            from ..schema import normalize_sleep_columns

            return normalize_sleep_columns(sleep)
        return pd.DataFrame(
            columns=[
                "sleep_summary_id",
                "calendar_date",
                "utc_time",
                "local_time",
                "timezone_offset",
                "duration_in_ms",
            ]
        )

    def load_steps(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        """Return Fitbit minute steps in the canonical activity schema."""
        return self.load_activity_samples(user_id, start_date, end_date)

    def load_activity_samples(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        """Return Fitbit minute step counts indexed by local time."""
        file_path = self.root_folder / user_id / "steps.parquet"
        if file_path.exists():
            steps = pd.read_parquet(file_path).rename(
                columns={
                    "dateTime": schema.LOCAL_TIME_COL,
                    "value": schema.ACTIVITY_STEPS_COL,
                }
            )
            steps[schema.LOCAL_TIME_COL] = pd.to_datetime(
                steps[schema.LOCAL_TIME_COL], errors="coerce"
            )
            steps[schema.UTC_TIME_COL] = pd.NaT
            steps[schema.TIMEZONE_COL] = pd.NaT
            steps[schema.DURATION_IN_MS_COL] = 60_000
            steps["source"] = "fitbit_intraday_steps"
            if not (start_date is None):
                steps = steps[steps[schema.LOCAL_TIME_COL] >= pd.Timestamp(start_date)]
            if not (end_date is None):
                steps = steps[steps[schema.LOCAL_TIME_COL] < pd.Timestamp(end_date)]
            return schema.prepare_activity_samples(steps)
        return schema.empty_activity_frame(extra_columns=("source",))

    def load_stress(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        file_path = self.root_folder / user_id / "stress_score.parquet"
        if file_path.exists():
            stress = pd.read_parquet(file_path)
            stress.columns = stress.columns.str.lower()
            stress = stress.rename(columns={"date": "calendarDate"})
            stress["calendarDate"] = pd.to_datetime(stress["calendarDate"])

            if not (start_date is None):
                stress = stress[
                    stress["calendarDate"] >= pd.Timestamp(start_date).normalize()
                ]
            if not (end_date is None):
                stress = stress[
                    stress["calendarDate"] < pd.Timestamp(end_date).normalize()
                ]
            return stress
        else:
            return pd.DataFrame(
                columns=[
                    "calendarDate",
                    "stress_score",
                    "sleep_points",
                    "max_sleep_points",
                ]
            )

    def load_provider_feature(
        self, feature: str, user_id: str, start_date=None, end_date=None, **kwargs
    ) -> pd.DataFrame:
        """Translate one Fitbit Stress Management field to native daily grain."""
        columns = {
            "fitbit_stress_management_score": "stress_score",
            "fitbit_stress_sleep_points": "sleep_points",
            "fitbit_stress_max_sleep_points": "max_sleep_points",
            "fitbit_stress_responsiveness_points": "responsiveness_points",
            "fitbit_stress_max_responsiveness_points": "max_responsiveness_points",
            "fitbit_stress_exertion_points": "exertion_points",
            "fitbit_stress_max_exertion_points": "max_exertion_points",
        }
        if feature not in columns:
            return super().load_provider_feature(
                feature, user_id, start_date, end_date, **kwargs
            )
        stress = self.load_stress(user_id, start_date, end_date).drop_duplicates()
        if stress.empty:
            return schema.prepare_provider_feature_summary(
                pd.DataFrame(columns=[schema.CALENDAR_DATE_COL, feature]), feature
            )
        stress = stress.rename(columns={"calendarDate": schema.CALENDAR_DATE_COL})
        if stress[schema.CALENDAR_DATE_COL].duplicated().any():
            raise ValueError(
                "conflicting Fitbit stress summaries remain for one calendar date"
            )
        # The LifeSnaps Fitbit export uses READY for finalized daily scores.
        # NO_DATA and IN_PROGRESS rows identify dates but do not contain a usable
        # score. Older exports may omit these lifecycle fields.
        calculation_ready = (
            stress["status"].eq("READY")
            if "status" in stress
            else pd.Series(True, index=stress.index)
        )
        calculation_succeeded = (
            ~stress["calculation_failed"].fillna(True)
            if "calculation_failed" in stress
            else pd.Series(True, index=stress.index)
        )
        valid = calculation_ready & calculation_succeeded
        result = stress[[schema.CALENDAR_DATE_COL, columns[feature]]].rename(
            columns={columns[feature]: feature}
        )
        result.loc[~valid, feature] = np.nan
        return schema.prepare_provider_feature_summary(result, feature)

    @staticmethod
    def _filter_survey_rows(
        survey: pd.DataFrame,
        user_id: Union[str, None],
        start_date,
        end_date,
        time_col: str,
    ) -> pd.DataFrame:
        if user_id is not None:
            survey = survey[survey["user_id"] == user_id]
        if start_date is not None:
            survey = survey[survey[time_col] >= pd.Timestamp(start_date)]
        if end_date is not None:
            survey = survey[survey[time_col] < pd.Timestamp(end_date)]
        return survey.reset_index(drop=True)

    def load_survey(
        self,
        survey_name: str,
        user_id: Union[str, None] = None,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """Load native LifeSnaps responses for one named survey."""
        survey = pd.read_parquet(self.root_folder / f"{survey_name}.parquet").copy()
        for column in ("submitdate", "startdate", "datestamp"):
            if column in survey:
                survey[column] = pd.to_datetime(
                    survey[column], errors="coerce", dayfirst=True
                )
        return self._filter_survey_rows(
            survey, user_id, start_date, end_date, "submitdate"
        )

    def load_survey_scored(
        self,
        survey_name: str,
        user_id: Union[str, None] = None,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """Load provider-supplied LifeSnaps scores for one named survey."""
        survey = pd.read_parquet(
            self.root_folder / f"{survey_name}_scored.parquet"
        ).copy()
        survey["submitdate"] = pd.to_datetime(
            survey["submitdate"], errors="coerce"
        )
        return self._filter_survey_rows(
            survey, user_id, start_date, end_date, "submitdate"
        )

    def load_profile(self, user_id: str) -> pd.DataFrame:
        file_path = self.root_folder / user_id / "profile.parquet"
        if file_path.exists():
            return pd.read_parquet(file_path)
        else:
            return pd.DataFrame()

    def load_ema(
        self,
        user_id: Union[str, None] = None,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """Load native LifeSnaps EMA records, optionally filtering creation time."""
        file_path = self.root_folder / "sema.parquet"
        if not (file_path.exists()):
            raise FileNotFoundError(f"Couldn't find file {file_path}")
        sema = pd.read_parquet(file_path)
        for col in [
            "CREATED_TS",
            "SCHEDULED_TS",
            "STARTED_TS",
            "COMPLETED_TS",
            "EXPIRED_TS",
            "UPLOADED_TS",
        ]:
            sema[col] = pd.to_datetime(sema[col])
        return self._filter_survey_rows(
            sema, user_id, start_date, end_date, "CREATED_TS"
        )
