import ast
import os
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd

from ... import schema
from ..base import BaseLoader
from ..garmin import prepare_epoch_activity_samples


class TesseraeParquetLoader(BaseLoader):
    """Load a local Tesserae dataset converted from CSV to Parquet.

    Parameters
    ----------
    root_folder
        Directory containing the converted Tesserae Parquet files.
    """

    hrv_source = "bbi"
    activity_source = "samples"
    activity_time_columns = (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL)
    activity_primary_resolution = "15min"
    provider_features = frozenset({"garmin_stress_level"})
    provider_feature_time_columns = (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL)

    def __init__(self, root_folder: os.PathLike):
        self.root_folder = Path(root_folder)
        if not self.root_folder.exists():
            raise FileNotFoundError(f"Folder {self.root_folder} doesn't exist.")
        self.user_ids = [
            x.name
            for x in self.root_folder.iterdir()
            if not "_global_metrics" in x.name
        ]

        self.demographics = None
        self.dailies = None

    ##################################################
    #               Garmin Health API                #
    ##################################################
    def load_activities(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        if user_id not in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")

        path_to_file = (
            self.root_folder / user_id / "GarminHealthAPI" / "Activities.parquet"
        )
        cols = [
            "active_kilocalories",
            "distance_meters",
            "average_speed_meters_per_second",
            "max_pace_minutes_per_km",
            "steps",
            "duration_seconds",
            "average_hr",
            "max_hr",
            "max_speed_meters_per_second",
            "average_pace_minutes_per_km",
            "average_run_cadence_in_steps_per_minute",
            "max_run_cadence_in_steps_per_minute",
        ]
        if not path_to_file.exists():
            return pd.DataFrame(columns=cols + ["utc_time", "local_time"])

        df = pd.read_parquet(path_to_file).rename(columns={"Timestamp": "utc_time"})
        df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True).dt.tz_localize(None)
        df["local_time"] = pd.to_datetime(df["local_time"])
        df["timezone_offset"] = df["local_time"] - df["utc_time"]
        for col in cols:
            df[col] = pd.to_numeric(df[col].replace("NULL", ""))
        if not (start_date is None):
            df = df[df["local_time"] >= pd.Timestamp(start_date)]
        if not (end_date is None):
            df = df[df["local_time"] < pd.Timestamp(end_date)]
        return df

    def load_activity_series(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        if user_id not in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")

        path_to_file = (
            self.root_folder / user_id / "GarminHealthAPI" / "ActivitySeries.parquet"
        )
        numeric_cols = [
            "sedentary_seconds",
            "active_seconds",
            "highly_active_seconds",
            "total_activity_seconds",
            "steps",
            "num_activities",
        ]
        cat_cols = ["active", "max_intensity"]
        if not path_to_file.exists():
            return pd.DataFrame(
                columns=numeric_cols + cat_cols + ["utc_time", "local_time"]
            )

        df = pd.read_parquet(path_to_file).rename(
            columns={
                "Timestamp": "utc_time",
                "start_time": "local_time",
                "sedentary_secs": "sedentary_seconds",
                "active_secs": "active_seconds",
                "highly_active_secs": "highly_active_seconds",
                "total_activity_secs": "total_activity_seconds",
                "step_count": "steps",
            }
        )

        df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True).dt.tz_localize(None)
        df["local_time"] = pd.to_datetime(df["local_time"])
        df["timezone_offset"] = df["local_time"] - df["utc_time"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].replace("NULL", ""))

        df["duration_in_seconds"] = (
            pd.to_datetime(df["stop_time"]) - df["local_time"]
        ).dt.total_seconds()
        df = df.drop(["stop_time"], axis=1)

        if not (start_date is None):
            df = df[df["local_time"] >= pd.Timestamp(start_date)]
        if not (end_date is None):
            df = df[df["local_time"] < pd.Timestamp(end_date)]
        return df

    def load_heart_rate(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        backend="health_api",
        time_col: str = "local_time",
    ) -> pd.DataFrame:
        if user_id not in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        schema.validate_heart_rate_time_col(time_col)
        if backend == "health_api":
            path_to_file = self.root_folder / user_id / "GarminHealthAPI" / "HR.parquet"
        elif backend == "pa":
            path_to_file = self.root_folder / user_id / "GarminPA" / "HR.parquet"
        else:
            raise ValueError(f"Backend of type {backend} is not supported")

        if not path_to_file.exists():
            return schema.empty_heart_rate_frame(
                ["timezone_offset", "tz_source", "beats_per_minute"]
            )

        hr = pd.read_parquet(path_to_file).rename(
            columns={"Timestamp": "utc_time", "hr": "beats_per_minute"}
        )
        hr["utc_time"] = pd.to_datetime(hr["utc_time"], utc=True).dt.tz_localize(None)
        hr["local_time"] = pd.to_datetime(hr["local_time"])
        hr["timezone"] = hr["local_time"] - hr["utc_time"]
        hr["timezone_offset"] = hr["timezone"]
        hr["heart_rate"] = pd.to_numeric(hr["beats_per_minute"], errors="coerce")
        hr = hr[hr["heart_rate"] > 0]
        # Preserve the loader's established latest-record policy for repeated
        # Tesserae export timestamps.
        hr = hr.drop_duplicates(subset="utc_time", keep="last")
        hr = schema.prepare_heart_rate_samples(hr)
        return (
            schema.filter_heart_rate_by_time(hr, start_date, end_date, time_col)
            .sort_values("utc_time")
            .reset_index(drop=True)
        )

    def load_move_iq(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        if user_id not in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")

        path_to_file = self.root_folder / user_id / "GarminHealthAPI" / "MoveIQ.parquet"
        numeric_cols = [
            "duration_in_seconds",
        ]
        cat_cols = ["activity_type", "activity_subtype"]
        if not path_to_file.exists():
            return pd.DataFrame(
                columns=numeric_cols + cat_cols + ["utc_time", "local_time"]
            )

        df = pd.read_parquet(path_to_file).rename(
            columns={"Timestamp": "utc_time", "duration_seconds": "duration_in_seconds"}
        )

        df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True).dt.tz_localize(None)
        df["local_time"] = pd.to_datetime(df["local_time"])
        df["timezone_offset"] = df["local_time"] - df["utc_time"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].replace("NULL", ""))

        if not (start_date is None):
            df = df[df["local_time"] >= pd.Timestamp(start_date)]
        if not (end_date is None):
            df = df[df["local_time"] < pd.Timestamp(end_date)]
        return df

    def load_pulse_ox(self, user_id: str = None, start_date=None, end_date=None):
        """Load oxygen saturation data.

        ..warning:: Tesserae dataset doesn't have respiration data, so this
        method will always return an empty DataFrame.


        Args:
            user_id (str): Unique user identifier.
            start_date (pd.Timestamp): Start date for data retrieval.
            end_date (pd.Timestamp): End date for data retrieval.

        Returns:
            pd.DataFrame: Empty dataframe.
        """
        return pd.DataFrame(
            columns=["utc_time", "local_time", "timezone_offset", "spo2"]
        )

    def load_respiration(self, user_id: str, start_date: pd.Timestamp, end_date):
        """Load respiration data.

        ..warning:: Tesserae dataset doesn't have respiration data, so this
        method will always return an empty DataFrame.


        Args:
            user_id (str): Unique user identifier.
            start_date (pd.Timestamp): Start date for data retrieval.
            end_date (pd.Timestamp): End date for data retrieval.

        Returns:
            pd.DataFrame: Empty dataframe.
        """
        return pd.DataFrame(
            columns=["utc_time", "local_time", "timezone_offset", "breaths_per_minute"]
        )

    def load_stress(
        self, user_id: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ):
        path_to_file = self.root_folder / user_id / "GarminHealthAPI" / "Stress.parquet"

        if not path_to_file.exists():
            return pd.DataFrame(
                columns=["local_time", "utc_time", "timezone_offset", "stress"]
            )

        filters = []
        if not (start_date is None):
            filters.append((("local_time", ">=", pd.Timestamp(start_date))))
        if not (end_date is None):
            filters.append((("local_time", "<", pd.Timestamp(end_date))))

        read_kwargs = {"filters": [filters]} if filters else {}
        stress = pd.read_parquet(path_to_file, **read_kwargs).rename(
            columns={"Timestamp": "utc_time", "hr": "stress"}
        )
        stress["utc_time"] = (
            stress["utc_time"].dt.tz_convert("UTC").dt.tz_localize(None)
        )
        stress["timezone_offset"] = stress["local_time"] - stress["utc_time"]
        stress = stress.drop_duplicates(subset="utc_time", keep="last")
        return stress

    def load_sleep_stage(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        sleep_summary = self.load_sleep_summary(user_id=user_id)
        if len(sleep_summary) == 0:
            return pd.DataFrame(
                columns=[
                    "sleep_summary_id",
                    "utc_time",
                    "local_time",
                    "duration_in_ms",
                    "type",
                ]
            )

        sleep_summary["timezone_offset"] = (
            sleep_summary["local_time"] - sleep_summary["utc_time"]
        )
        # Let's create the same sleep summary identifier
        sleep_summary["local_end_time"] = sleep_summary["local_time"] + pd.to_timedelta(
            sleep_summary["duration_seconds"].fillna(0)
            + sleep_summary["awake_seconds"].fillna(0),
            unit="seconds",
        )

        sleep_summary["sleep_levels_map"] = (
            sleep_summary["sleep_levels_map"]
            .replace("NULL", "{}")
            .fillna("{}")
            .apply(lambda x: ast.literal_eval(x))
        )

        # We need to store the sleep summary Id in the sleep levels map as well
        def assign_sleep_summary_id(sleep_levels: dict, sleep_summary_id: str):
            for k, v in sleep_levels.items():
                for item in v:
                    item["sleep_summary_id"] = sleep_summary_id
            return sleep_levels

        sleep_summary["sleep_levels_map"] = sleep_summary.apply(
            lambda x: assign_sleep_summary_id(
                x["sleep_levels_map"], x["sleep_summary_id"]
            ),
            axis=1,
        )

        json_norm = pd.json_normalize(sleep_summary["sleep_levels_map"])

        sleep_stage = (
            pd.concat(
                [
                    pd.json_normalize(json_norm[sleep_stage_type].dropna().explode())
                    .assign(type=sleep_stage_type)
                    .dropna(how="any")
                    for sleep_stage_type in json_norm.columns
                ],
                ignore_index=True,
            )
            .sort_values(by="startTime", ignore_index=True)
            .rename(
                columns={
                    "startTime": "local_time",
                    "endTime": "local_end_time",
                }
            )
        )

        sleep_stage = pd.merge(
            left=sleep_stage,
            right=sleep_summary.loc[:, ["sleep_summary_id", "timezone_offset"]],
            on="sleep_summary_id",
            how="left",
        )

        for col in ["local_time", "local_end_time"]:
            sleep_stage[col] = pd.to_datetime(sleep_stage[col])

        sleep_stage["utc_time"] = (
            sleep_stage["local_time"] - sleep_stage["timezone_offset"]
        )
        sleep_stage["utc_end_time"] = (
            sleep_stage["local_end_time"] - sleep_stage["timezone_offset"]
        )

        sleep_stage = sleep_stage.drop(
            [
                "startTimeWithTimezone",
                "endTimeWithTimezone",
                "startTimeInSeconds",
            ],
            axis=1,
        )

        sleep_stage["duration_in_ms"] = (
            sleep_stage["local_end_time"] - sleep_stage["local_time"]
        ).dt.total_seconds() * 1000

        # Filter by isoDate
        if not (start_date is None):
            sleep_stage = sleep_stage.loc[
                (sleep_stage["local_time"] >= start_date),
                :,
            ]
        if not (end_date is None):
            sleep_stage = sleep_stage.loc[
                (sleep_stage["local_time"] < end_date),
                :,
            ]
        return sleep_stage

    def load_sleep_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        path_to_file = self.root_folder / user_id / "GarminHealthAPI" / "Sleep.parquet"

        if not path_to_file.exists():
            return pd.DataFrame(
                columns=[
                    "utc_time",
                    "local_time",
                    "duration_in_seconds",
                    "rem_sleep_duration_in_seconds",
                    "deep_sleep_duration_in_seconds",
                    "awake_duration_in_seconds",
                    "light_sleep_duration_in_seconds",
                ]
            )

        sleep = pd.read_parquet(
            path_to_file,
        )
        # Make sure columns are of float dtype
        for col in [
            "duration_seconds",
            "rem_sleep_seconds",
            "deep_sleep_seconds",
            "awake_seconds",
            "light_sleep_seconds",
        ]:
            sleep[col] = pd.to_numeric(sleep[col], errors="coerce")
        if "unmeasurable_sleep_seconds" in sleep.columns:
            sleep["unmeasurable_sleep_seconds"] = pd.to_numeric(
                sleep["unmeasurable_sleep_seconds"], errors="coerce"
            )
        else:
            sleep["unmeasurable_sleep_seconds"] = 0.0

        classified_sleep_seconds = sleep[
            [
                "deep_sleep_seconds",
                "light_sleep_seconds",
                "rem_sleep_seconds",
            ]
        ].sum(axis=1, min_count=1)
        staged_sleep_seconds = classified_sleep_seconds + sleep[
            "unmeasurable_sleep_seconds"
        ].fillna(0)
        # Older AUTO_TENTATIVE Tesserae rows use duration_seconds for the
        # complete interval, including awake time. Normalize only rows whose
        # detailed stages demonstrate that exact legacy convention.
        duration_includes_awake = (
            classified_sleep_seconds.notna()
            & sleep["awake_seconds"].fillna(0).gt(0)
            & np.isclose(
                sleep["duration_seconds"],
                staged_sleep_seconds + sleep["awake_seconds"].fillna(0),
                rtol=0,
                atol=1e-6,
            )
        )
        sleep.loc[duration_includes_awake, "duration_seconds"] = (
            staged_sleep_seconds.loc[duration_includes_awake]
        )

        sleep = sleep[
            sleep["duration_seconds"].fillna(0) + sleep["awake_seconds"].fillna(0) > 0
        ].copy()

        sleep["calendar_date"] = (
            sleep["local_time"]
            + pd.to_timedelta(
                sleep["duration_seconds"].fillna(0) + sleep["awake_seconds"].fillna(0),
                unit="seconds",
            )
        ).dt.normalize()

        # We need to drop duplicates by day
        sleep["validation_map"] = sleep["validation"].map(
            {
                "MANUAL": 0,
                "DEVICE": 1,
                "OFF_WRIST": 2,
                "AUTO_MANUAL": 3,
                "AUTO_TENTATIVE": 4,
                "AUTO_FINAL": 5,
                "ENHANCED_TENTATIVE": 6,
                "ENHANCED_FINAL": 7,
            }
        )
        sleep = sleep.sort_values(
            by=(
                [
                    "calendar_date",
                    "validation_map",
                    "duration_seconds",
                ]
            ),
            ascending=True,
        )
        if drop_duplicates:
            sleep = sleep.drop_duplicates(subset=["calendar_date"], keep="last")
        # Remove created column
        sleep = sleep.drop(["validation_map"], axis=1)
        sleep["sleep_summary_id"] = user_id + "-" + sleep.index.astype(str)
        sleep["utc_time"] = sleep["Timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        sleep["timezone_offset_in_ms"] = (
            sleep["local_time"] - sleep["utc_time"]
        ) / pd.Timedelta(milliseconds=1)
        sleep["unix_timestamp_in_ms"] = (
            sleep["utc_time"].dt.as_unit("ms").astype("int64")
        )
        sleep["local_end_time"] = sleep["local_time"] + pd.to_timedelta(
            sleep["duration_seconds"].fillna(0) + sleep["awake_seconds"].fillna(0),
            unit="seconds",
        )
        sleep["utc_end_time"] = sleep["utc_time"] + pd.to_timedelta(
            sleep["duration_seconds"].fillna(0) + sleep["awake_seconds"].fillna(0),
            unit="seconds",
        )
        for source, target in (
            ("duration_seconds", "duration_in_ms"),
            ("awake_seconds", "awake_duration_in_ms"),
            ("deep_sleep_seconds", "deep_sleep_duration_in_ms"),
            ("light_sleep_seconds", "light_sleep_duration_in_ms"),
            ("rem_sleep_seconds", "rem_sleep_duration_in_ms"),
        ):
            sleep[target] = sleep[source].fillna(0) * 1000
        sleep["unmeasurable_sleep_duration_in_ms"] = (
            sleep["unmeasurable_sleep_seconds"].fillna(0) * 1000
        )
        # Nightly bounds apply to the local calendar day assigned to the sleep.
        if not (start_date is None):
            sleep = sleep.loc[
                (sleep["calendar_date"] >= pd.Timestamp(start_date).normalize()),
                :,
            ]
        if not (end_date is None):
            sleep = sleep.loc[
                (sleep["calendar_date"] < pd.Timestamp(end_date).normalize()),
                :,
            ]

        return sleep

    def load_provider_feature(
        self, feature: str, user_id: str, start_date=None, end_date=None, **kwargs
    ) -> pd.DataFrame:
        """Translate Tesserae's Garmin stress stream to canonical samples."""
        if feature != "garmin_stress_level":
            return super().load_provider_feature(
                feature, user_id, start_date, end_date, **kwargs
            )
        stress = self.load_stress(user_id, start_date, end_date).rename(
            columns={"stress": feature, "timezone_offset": schema.TIMEZONE_COL}
        )
        if feature not in stress:
            stress[feature] = pd.Series(dtype=float)
        stress[feature] = pd.to_numeric(stress[feature], errors="coerce")
        stress.loc[~stress[feature].between(0, 100), feature] = np.nan
        return schema.prepare_provider_feature_samples(stress, feature)

    ##################################################
    #                    Garmin PA                  #
    ##################################################

    def load_bbi(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        path_to_file = self.root_folder / user_id / "GarminPA" / "BBI.parquet"
        if not path_to_file.exists():
            return schema.empty_bbi_frame(["source"])
        bbi = (
            pd.read_parquet(
                self.root_folder / user_id / "GarminPA" / "BBI.parquet",
            )
            .drop("tz_source", axis=1, errors="ignore")
            .rename(columns={"Timestamp": "utc_time"})
        )

        if len(bbi) == 0:
            return schema.empty_bbi_frame(["source"])
        bbi["utc_time"] = bbi["utc_time"].dt.tz_convert("UTC").dt.tz_localize(None)
        bbi["timezone"] = bbi["local_time"] - bbi["utc_time"]
        # we may have some milliseconds difference between utc and local time
        bbi["timezone"] = bbi["timezone"].dt.round("1min")

        if not (start_date is None):
            bbi = bbi[bbi["local_time"] >= pd.Timestamp(start_date)]
        if not (end_date is None):
            bbi = bbi[bbi["local_time"] < pd.Timestamp(end_date)]
        bbi["source"] = "garmin_pa_bbi"
        return schema.prepare_bbi(bbi)

    def _load_garmin_pa_metric(
        self,
        metric: str,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        numeric_columns: list = [],
        categorical_coluns: list = [],
        date_col: str = "local_time",
    ) -> pd.DataFrame:
        metric_to_file_map = {
            "calories": "Calories.parquet",
            "floors": "Floors.parquet",
            "battery": "GarminBattery.parquet",
            "hrv_series": "HRVSeries.parquet",
            "intensity": "Intensity.parquet",
            "steps": "Steps.parquet",
        }
        path_to_file = (
            self.root_folder / user_id / "GarminPA" / metric_to_file_map[metric]
        )
        if not path_to_file.exists():
            return pd.DataFrame(
                columns=["utc_time", date_col] + numeric_columns + categorical_coluns,
            )
        df = pd.read_parquet(path_to_file).rename(columns={"Timestamp": "utc_time"})

        if len(df) == 0:
            return pd.DataFrame(
                columns=["utc_time", date_col] + numeric_columns + categorical_coluns,
            )
        df["utc_time"] = df["utc_time"].dt.tz_convert("UTC").dt.tz_localize(None)
        if date_col == "date":
            df = df.rename(columns={date_col: "calendar_date"})
            date_col = "calendar_date"
        df[date_col] = pd.to_datetime(df[date_col])

        df["timezone"] = df[date_col] - df["utc_time"]
        # we may have some milliseconds difference between utc and local time
        df["timezone"] = df["timezone"].dt.round("1min")

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col].replace("NULL", ""))
        if not (start_date is None):
            df = df[df[date_col] >= pd.Timestamp(start_date)]
        if not (end_date is None):
            df = df[df[date_col] < pd.Timestamp(end_date)]

        return df

    def load_calories(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        return self._load_garmin_pa_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            numeric_columns=["active_calories", "total_calories"],
            metric="calories",
        )

    def load_floors(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        return self._load_garmin_pa_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            numeric_columns=["floors_climbed", "floors_descended"],
            metric="floors",
        )

    def load_device_battery(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        return self._load_garmin_pa_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            numeric_columns=["level"],
            metric="battery",
        )

    def load_hrv_series(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        return self._load_garmin_pa_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            numeric_columns=[
                "duration",
                "num_points",
                "num_points_locked",
                "time_locked",
                "percent_locked",
                "mean_bbi",
                "median_bbi",
                "sdnn",
                "rmssd",
            ],
            metric="hrv_series",
            date_col="date",
        )

    def load_intensity(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        return self._load_garmin_pa_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            numeric_columns=[
                "intensity_minutes_iphone",
                "daily_vigorous_minutes",
                "weekly_minutes",
            ],
            metric="intensity",
            date_col="local_time",
        )

    def load_steps_from_garmin_pa(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        return self._load_garmin_pa_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            numeric_columns=["step_count"],
            metric="steps",
            date_col="local_time",
        )

    def load_steps(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        backend: Literal["garmin_health_api", "garmin_pa"] = "garmin_health_api",
    ) -> pd.DataFrame:
        """Return Garmin Health API step intervals."""
        if backend == "garmin_health_api":
            return self.load_activity_samples(
                user_id=user_id, start_date=start_date, end_date=end_date
            )
        else:
            return self.load_steps_from_garmin_pa(
                user_id=user_id, start_date=start_date, end_date=end_date
            )

    def load_activity_samples(
        self,
        user_id: str,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """Return Garmin Health API epochs as canonical step intervals."""
        path_to_file = self.root_folder / user_id / "GarminHealthAPI" / "Epochs.parquet"
        if not path_to_file.exists():
            return schema.empty_activity_frame(extra_columns=("source",))
        epochs = pd.read_parquet(path_to_file)
        epochs[schema.LOCAL_TIME_COL] = pd.to_datetime(
            epochs["local_time"], errors="coerce"
        )
        epochs[schema.UTC_TIME_COL] = pd.to_datetime(
            epochs["Timestamp"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        epochs[schema.TIMEZONE_COL] = (
            epochs[schema.LOCAL_TIME_COL] - epochs[schema.UTC_TIME_COL]
        )
        epochs[schema.DURATION_IN_MS_COL] = (
            pd.to_numeric(epochs["duration_seconds"], errors="coerce") * 1000
        )
        epochs[schema.ACTIVITY_STEPS_COL] = epochs["steps"]
        if start_date is not None:
            epochs = epochs[epochs[schema.LOCAL_TIME_COL] >= pd.Timestamp(start_date)]
        if end_date is not None:
            epochs = epochs[epochs[schema.LOCAL_TIME_COL] < pd.Timestamp(end_date)]
        return prepare_epoch_activity_samples(epochs)

    ###################################################
    #                 Summaries
    ###################################################
    def _load_global_metric(
        self,
        metric: str,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
        date_col: str = "date",
    ) -> pd.DataFrame:
        garmin_summaries = [
            "activity_summary",
            "adj_imputed_sleep_summary",
            "daily_summary",
            "hr_summary",
            "hrv_summary",
            "stress_summary",
        ]
        gimbal_summaries = [
            "interaction_summary",
            "beacon_summary",
            "home_office_hours",
        ]
        contextual_summaries = ["noaa_storm_events", "weather_summary"]
        mapping = ["beacons", "participant_info"]
        mixed = [
            "home_office_hours_loc",
            "hrv_summary_baseline",
            "hrv_work_desk_home_summary",
            "pa_summary",
        ]
        metric_map = {
            # Garmin
            "activity_summary": "ActivitySummary.parquet",
            "adj_imputed_sleep_summary": "AdjImputedSleep.parquet",
            "daily_summary": "DailySummaries.parquet",
            "hr_summary": "HRSummary.parquet",
            "hrv_summary": "HRVSummary.parquet",
            "stress_summary": "StressSummary.parquet",
            # Gimbal
            "beacon_summary": "BeaconSummary.parquet",
            "interaction_summary": "InteractionSummary.parquet",
            "home_office_hours": "HomeOfficeHours.parquet",
            # Contextual
            "noaa_storm_events": "NOAAStormEvents.parquet",
            "weather_summary": "WeatherSummary.parquet",
            # Mapping
            "beacons": "Beacons.parquet",
            "participant_info": "ParticipantInfo.parquet",
            # Mixed
            "home_office_hours_loc": "HomeOfficeHoursLoc.parquet",
            "hrv_summary_baseline": "HRVSummaryBaseline.parquet",
            "hrv_work_desk_home_summary": "HRVWorkDeskHomeSummary",
            "pa_summary": "PASummary.parquet",
        }
        if metric in garmin_summaries:
            folder = "GarminSummary"
        elif metric in gimbal_summaries:
            folder = "GimbalSummary"
        elif metric in contextual_summaries:
            folder = "Contextual"
        elif metric in mapping:
            folder = "Mapping"
        elif metric in mixed:
            folder = "Mixed"
        path_to_file = (
            self.root_folder / "_global_metrics" / folder / metric_map[metric]
        )
        if not path_to_file.exists():
            return pd.DataFrame()
        df = (
            pd.read_parquet(path_to_file, filters=[["ParticipantID", "==", user_id]])
            .drop(["ParticipantID", "tz_source", "Timestamp"], axis=1, errors="ignore")
            .rename(columns={"local_time": "calendar_date", "date": "calendar_date"})
        )
        if date_col == "date":
            date_col = "calendar_date"
            df["calendar_date"] = pd.to_datetime(df["calendar_date"])
        elif date_col is None:
            return df
        else:
            df[date_col] = pd.to_datetime(df[date_col])
        if not (start_date is None):
            df = df[df[date_col] >= pd.Timestamp(start_date).dt.normalize()]
        if not (end_date is None):
            df = df[df[date_col] < pd.Timestamp(end_date).dt.normalize()]
        return df

    def load_daily_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        daily_summary = self._load_global_metric(
            metric="daily_summary",
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        if len(daily_summary) == 0:
            return daily_summary
        if not (start_date is None):
            daily_summary = daily_summary.loc[
                (daily_summary["calendar_date"] >= pd.Timestamp(start_date)) :,
            ]
        if not (end_date is None):
            daily_summary = daily_summary.loc[
                (daily_summary["calendar_date"] < pd.Timestamp(end_date)),
                :,
            ]

        if drop_duplicates:
            daily_summary = daily_summary.sort_values(
                by=["calendar_date", "steps"]
            ).drop_duplicates(subset="calendar_date", keep="last", ignore_index=True)
        return daily_summary

    def load_activity_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        activity_summary = self._load_global_metric(
            metric="activity_summary",
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        if len(activity_summary) == 0:
            return activity_summary
        if not (start_date is None):
            activity_summary = activity_summary.loc[
                (activity_summary["calendar_date"] >= pd.Timestamp(start_date)) :,
            ]
        if not (end_date is None):
            activity_summary = activity_summary.loc[
                (activity_summary["calendar_date"] < pd.Timestamp(end_date)),
                :,
            ]

        return activity_summary

    def load_adjusted_imputed_sleep_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        adj_sleep_summary = self._load_global_metric(
            metric="adj_imputed_sleep_summary",
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        if len(adj_sleep_summary) == 0:
            return adj_sleep_summary

        for col in ["adjusted_bed_time", "adjusted_wakeup_time"]:
            adj_sleep_summary[col] = (
                adj_sleep_summary["calendar_date"]
                + pd.to_timedelta(
                    adj_sleep_summary[col].replace("NULL", None).astype(float),
                    unit="h",
                )
            ).dt.round("1min")
        adj_sleep_summary["adjusted_sleep_duration"] = pd.to_timedelta(
            adj_sleep_summary["adjusted_sleep_duration"]
            .replace("NULL", None)
            .astype(float),
            unit="h",
        ).dt.round("1min")

        if not (start_date is None):
            adj_sleep_summary = adj_sleep_summary.loc[
                (adj_sleep_summary["calendar_date"] >= pd.Timestamp(start_date)) :,
            ]
        if not (end_date is None):
            adj_sleep_summary = adj_sleep_summary.loc[
                (adj_sleep_summary["calendar_date"] < pd.Timestamp(end_date)),
                :,
            ]

        return adj_sleep_summary

    def load_heart_rate_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        hr_summary = self._load_global_metric(
            metric="hr_summary",
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        if len(hr_summary) == 0:
            return hr_summary
        hr_summary = hr_summary.rename(
            columns={
                "ave_hr": "average_heart_rate",
                "median_hr": "median_heart_rate",
                "mode_hr": "mode_heart_rate",
                "min_hr": "min_heart_rate",
                "max_hr": "max_heart_rate",
                "range_hr": "range_heart_rate",
            }
        )
        for col in [
            "average_heart_rate",
            "median_heart_rate",
            "mode_heart_rate",
            "min_heart_rate",
            "max_heart_rate",
            "range_heart_rate",
        ]:
            hr_summary[col] = hr_summary[col].replace("NULL", None).astype(float)
        if not (start_date is None):
            hr_summary = hr_summary.loc[
                (hr_summary["calendar_date"] >= pd.Timestamp(start_date)) :,
            ]
        if not (end_date is None):
            hr_summary = hr_summary.loc[
                (hr_summary["calendar_date"] < pd.Timestamp(end_date)),
                :,
            ]

        return hr_summary

    def load_hrv_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        hrv_summary = self._load_global_metric(
            metric="hrv_summary",
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        if len(hrv_summary) == 0:
            return hrv_summary

        for col in [
            "num_windows",
            "ave_lockedp",
            "ave_hrv",
            "min_hrv",
            "max_hrv",
            "median_hrv",
            "sd_hrv",
            "sdann",
            "num_first_windows",
            "ave_hrv_first_hour_work",
            "num_last_windows",
            "ave_hrv_last_hour_work",
            "num_before_windows",
            "ave_hrv_15min_before_work",
            "num_work_windows",
            "ave_hrv_work",
            "min_hrv_work",
            "max_hrv_work",
            "sdann_work",
            "num_not_work_windows",
            "ave_hrv_not_work",
            "min_hrv_not_work",
            "max_hrv_not_work",
            "sdann_not_work",
            "diff_sdann_work_to_not",
            "ratio_sdann_work_to_not",
            "num_8_to_6_windows",
            "ave_hrv_8_to_6",
            "min_hrv_8_to_6",
            "max_hrv_8_to_6",
            "sdann_8_to_6",
            "num_not_8_to_6_windows",
            "ave_hrv_not_8_to_6",
            "min_hrv_not_8_to_6",
            "max_hrv_not_8_to_6",
            "sdann_not_8_to_6",
            "diff_sdann_8_to_6_to_not",
            "ratio_sdann_8_to_6_to_not",
        ]:
            hrv_summary[col] = hrv_summary[col].replace("NULL", None).astype(float)
        if not (start_date is None):
            hrv_summary = hrv_summary.loc[
                (hrv_summary["calendar_date"] >= pd.Timestamp(start_date)) :,
            ]
        if not (end_date is None):
            hrv_summary = hrv_summary.loc[
                (hrv_summary["calendar_date"] < pd.Timestamp(end_date)),
                :,
            ]

        return hrv_summary

    def load_stress_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        stress_summary = self._load_global_metric(
            metric="stress_summary",
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        if len(stress_summary) == 0:
            return stress_summary
        stress_summary = stress_summary.rename(
            columns={
                "ave_stress": "average_stress",
                "median_stress": "median_stress",
                "mode_stress": "mode_stress",
                "min_stress": "min_stress",
                "max_stress": "max_stress",
                "range_stress": "range_stress",
            }
        )
        for col in [
            "average_stress",
            "median_stress",
            "mode_stress",
            "min_stress",
            "max_stress",
            "range_stress",
        ]:
            stress_summary[col] = (
                stress_summary[col].replace("NULL", None).astype(float)
            )
        if not (start_date is None):
            stress_summary = stress_summary.loc[
                (stress_summary["calendar_date"] >= pd.Timestamp(start_date)) :,
            ]
        if not (end_date is None):
            stress_summary = stress_summary.loc[
                (stress_summary["calendar_date"] < pd.Timestamp(end_date)),
                :,
            ]

        return stress_summary

    #############################################################
    #                   Phone
    #############################################################
    def _load_standard_metric(
        self,
        user_id: str,
        folder: str,
        metric: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        date_col: str = "local_time",
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")

        metric_map = {
            "pedometer": "Pedometer.parquet",
            "wifi": "Wifi.parquet",
            "beacon_callback": "BeaconCallback.parquet",
            "beacon_place": "BeaconPlace.parquet",
            "beacon_sighting": "BeaconSighting.parquet",
            # "stress_summary": "StressSummary.parquet",
        }

        path_to_file = self.root_folder / user_id / folder / metric_map[metric.lower()]
        if not path_to_file.exists():
            return pd.DataFrame()

        df = pd.read_parquet(path_to_file)
        df = df.drop(["ParticipantID", "utc_time"], axis=1, errors="ignore")

        df = df.rename(columns={"Timestamp": "utc_time"})
        df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True).dt.tz_localize(None)
        df["timezone_offset"] = df["local_time"] - df["utc_time"]

        if date_col == "date":
            df = df.rename(columns={date_col: "calendar_date"})
            date_col = "calendar_date"
        elif date_col == None:
            df["local_time"] = df["utc_time"] + pd.to_timedelta(
                df["timezone_offset"], unit="ms"
            )
            date_col = "local_time"
        df[date_col] = pd.to_datetime(df[date_col])

        if not (start_date is None):
            df = df.loc[df[date_col] >= pd.Timestamp(start_date)]
        if not (end_date is None):
            df = df.loc[df[date_col] < pd.Timestamp(end_date)]

        return df

    def load_phone_pedometer(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        pedometer = self._load_standard_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Phone",
            metric="Pedometer",
            date_col="local_time",
        )
        if len(pedometer) == 0:
            return pd.DataFrame(
                columns=[
                    "utc_time",
                    "local_time",
                    "duration_seconds",
                    "steps",
                    "distance",
                    "floors_climbed",
                    "floors_descended",
                ]
            )
        return pedometer

    def load_phone_wifi(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        return self._load_standard_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Phone",
            metric="Wifi",
            date_col="local_time",
        )

    #############################################################
    #                   Weather
    #############################################################
    def load_weather_summary(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        return self._load_global_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            metric="weather_summary",
        )

    def load_noaa_storm_events(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        return self._load_global_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            metric="noaa_storm_events",
            date_col="start_time",
        )

    #############################################################
    #                   Beacon
    #############################################################
    def load_beacon_callback(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        return self._load_standard_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Gimbal",
            metric="beacon_callback",
            date_col="local_time",
        )

    def load_beacon_place(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        gimbal_beacon_place = self._load_standard_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Gimbal",
            metric="beacon_place",
            date_col="local_time",
        )
        if len(gimbal_beacon_place) > 0:
            for col in ["departure_time", "arrival_time"]:
                gimbal_beacon_place[col] = pd.to_datetime(
                    gimbal_beacon_place[col].replace("NULL", np.nan)
                )
        return gimbal_beacon_place

    def load_beacon_sighting(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        gimbal_beacon_sighting = self._load_standard_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Gimbal",
            metric="beacon_sighting",
            date_col="local_time",
        )
        return gimbal_beacon_sighting

    def load_beacon_summary(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        gimbal_beacon_summary = self._load_global_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            metric="beacon_summary",
        )
        return gimbal_beacon_summary

    def load_home_office_hours(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        home_office_hours = self._load_global_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            metric="home_office_hours",
        )
        if len(home_office_hours) > 0:
            for col in ["home_hours", "office_hours"]:
                home_office_hours[col] = (
                    home_office_hours[col].replace("NULL", None).astype(float)
                )
        return home_office_hours

    def load_interaction_summary(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        interaction_summary = self._load_global_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            metric="interaction_summary",
        )
        return interaction_summary

    #############################################################
    #                   Mapping
    #############################################################
    def load_mapping_beacons(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        return self._load_global_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            metric="beacons",
            date_col="calendar_date",
        )

    def load_mapping_participant_info(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        return self._load_global_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            metric="participant_info",
            date_col=None,
        )

    #############################################################
    #                   Mixed
    #############################################################
    def load_mixed_home_office_hours_loc(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        home_office_hours = self._load_global_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            metric="home_office_hours_loc",
        )
        if len(home_office_hours) > 0:
            for col in ["home_hours", "office_hours"]:
                home_office_hours[col] = (
                    home_office_hours[col].replace("NULL", None).astype(float)
                )
        return home_office_hours

    def load_mixed_hrv_summary_baseline(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        hrv_summary_baseline = self._load_global_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            metric="hrv_summary_baseline",
        )
        columns = [
            "num_windows",
            "ave_lockedp",
            "ave_hrv",
            "min_hrv",
            "max_hrv",
            "median_hrv",
            "sd_hrv",
            "sdann",
            "num_first_windows",
            "ave_hrv_first_hour_work",
            "num_last_windows",
            "ave_hrv_last_hour_work",
            "num_before_windows",
            "ave_hrv_15min_before_work",
            "num_work_windows",
            "ave_hrv_work",
            "min_hrv_work",
            "max_hrv_work",
            "sdann_work",
            "num_not_work_windows",
            "ave_hrv_not_work",
            "min_hrv_not_work",
            "max_hrv_not_work",
            "sdann_not_work",
            "diff_sdann_work_to_not",
            "ratio_sdann_work_to_not",
            "num_8_to_6_windows",
            "ave_hrv_8_to_6",
            "min_hrv_8_to_6",
            "max_hrv_8_to_6",
            "sdann_8_to_6",
            "num_not_8_to_6_windows",
            "ave_hrv_not_8_to_6",
            "min_hrv_not_8_to_6",
            "max_hrv_not_8_to_6",
            "sdann_not_8_to_6",
            "diff_sdann_8_to_6_to_not",
            "ratio_sdann_8_to_6_to_not",
        ]
        if len(hrv_summary_baseline) > 0:
            hrv_summary_baseline[columns] = (
                hrv_summary_baseline[columns].replace("NULL", None).astype(float)
            )
        return hrv_summary_baseline

    def load_mixed_hrv_work_desk_home_summary(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        hrv = self._load_global_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            metric="hrv_work_desk_home_summary",
        )
        if len(hrv) > 0:
            columns = [
                "ave_sdnn_hrv_at_work",
                "ave_rmssd_hrv_at_work",
                "sdann_at_work",
                "ave_sdnn_hrv_at_desk",
                "ave_rmssd_hrv_at_desk",
                "sdann_at_desk",
                "ave_sdnn_hrv_not_at_work",
                "ave_rmssd_hrv_not_at_work",
                "sdann_not_at_work",
            ]
            hrv[columns] = hrv[columns].replace("NULL", None).astype(float)
        return hrv

    def load_mixed_pa_summary(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        return self._load_global_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            metric="pa_summary",
        )

    ##########################################
    #       Survey information
    ##########################################

    def load_daily_survey(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        """Load Tesserae daily survey scores for one participant."""
        if user_id not in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        path = (
            self.root_folder
            / "_global_metrics"
            / "GroundTruthSurvey"
            / "DailiesScored.parquet"
        )
        if not path.exists():
            return pd.DataFrame()

        daily_survey = pd.read_parquet(path)
        daily_survey = daily_survey[daily_survey["ParticipantID"] == user_id].copy()
        daily_survey["date"] = pd.to_datetime(daily_survey["date"])
        daily_survey = daily_survey.rename(columns={"date": "calendarDate"})
        for column in (
            "survey_sent_datetime",
            "survey_start_datetime",
            "survey_end_datetime",
        ):
            daily_survey[column] = pd.to_datetime(daily_survey[column], errors="coerce")
        utc_time = pd.to_datetime(
            daily_survey["Timestamp"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        daily_survey["timezoneOffsetInMs"] = (
            daily_survey["survey_sent_datetime"] - utc_time
        ).dt.total_seconds() * 1000
        text_columns = daily_survey.select_dtypes(include=["object", "string"]).columns
        for column in text_columns:
            daily_survey[column] = daily_survey[column].mask(
                daily_survey[column].eq("NULL")
            )

        excluded = {
            "ParticipantID",
            "Timestamp",
            "timezone_change",
            "calendarDate",
            "timezoneOffsetInMs",
        }
        response_columns = [
            column
            for column in daily_survey.columns
            if column not in excluded and "survey" not in column
        ]
        for column in response_columns:
            daily_survey[column] = pd.to_numeric(daily_survey[column], errors="coerce")
        daily_survey = daily_survey.rename(
            columns={column: f"survey_{column}" for column in response_columns}
        ).drop(["ParticipantID", "Timestamp", "timezone_change"], axis=1)
        if start_date is not None:
            daily_survey = daily_survey.loc[
                daily_survey["calendarDate"] >= pd.Timestamp(start_date)
            ]
        if end_date is not None:
            daily_survey = daily_survey.loc[
                daily_survey["calendarDate"] < pd.Timestamp(end_date)
            ]
        return daily_survey.reset_index(drop=True)

    def load_demographics_raw(self, user_id: str) -> pd.DataFrame:
        """Load Tesserae demographic responses for one participant."""
        return self._load_demographics_survey(
            user_id,
            "IGTBDemographics.parquet",
            numeric=False,
        )

    def load_demographics_scored(self, user_id: str) -> pd.DataFrame:
        """Load Tesserae scored demographic instruments for one participant."""
        return self._load_demographics_survey(
            user_id,
            "IGTBScored.parquet",
            numeric=True,
        )

    def _load_demographics_survey(
        self,
        user_id: str,
        filename: str,
        *,
        numeric: bool,
    ) -> pd.DataFrame:
        if user_id not in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        path = self.root_folder / "_global_metrics" / "GroundTruthSurvey" / filename
        if not path.exists():
            return pd.DataFrame()

        demographics = pd.read_parquet(path)
        demographics = demographics[demographics["ParticipantID"] == user_id].copy()
        demographics["igtb.datetime"] = pd.to_datetime(
            demographics["igtb.datetime"], errors="coerce"
        )
        demographics = demographics.rename(columns={"igtb.datetime": "isoDate"})
        utc_time = pd.to_datetime(
            demographics["Timestamp"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        demographics["timezoneOffsetInMs"] = (
            demographics["isoDate"] - utc_time
        ).dt.total_seconds() * 1000
        text_columns = demographics.select_dtypes(include=["object", "string"]).columns
        for column in text_columns:
            demographics[column] = demographics[column].mask(
                demographics[column].eq("NULL")
            )

        excluded = {
            "ParticipantID",
            "Timestamp",
            "timezone_change",
            "isoDate",
            "timezoneOffsetInMs",
        }
        response_columns = [
            column for column in demographics.columns if column not in excluded
        ]
        if numeric:
            for column in response_columns:
                demographics[column] = pd.to_numeric(
                    demographics[column], errors="coerce"
                )
        demographics["calendarDate"] = demographics["isoDate"].dt.normalize()
        demographics = demographics.rename(
            columns={column: f"demographics_{column}" for column in response_columns}
        ).drop(
            [
                "ParticipantID",
                "Timestamp",
                "timezone_change",
            ],
            axis=1,
        )
        return demographics.reset_index(drop=True)

    def get_all_user_dailies(self):
        if self.dailies is None:
            path_to_dailies = (
                self.root_folder
                / "_global_metrics"
                / "GroundTruthSurvey"
                / "DailiesScored.parquet"
            )
            if not path_to_dailies.exists():
                raise FileNotFoundError(
                    f"Could not find dailies file at {path_to_dailies}"
                )
            self.dailies = (
                pd.read_parquet(path_to_dailies)
                .drop(["Timestamp", "timezone_change"], axis=1)
                .replace("NULL", np.nan)
                .rename(columns={"ParticipantID": "user_id"})
                # .set_index("user_id")
            )
            self.dailies["date"] = pd.to_datetime(self.dailies["date"])
            self.dailies["stress"] = self.dailies["stress"].astype(float)
            self.dailies["sleep"] = self.dailies["sleep"].astype(float)

        return self.dailies

    def get_all_users_demographics(self):
        if self.demographics is None:
            path_to_demographics_scored = (
                self.root_folder
                / "_global_metrics"
                / "GroundTruthSurvey"
                / "IGTBScored.parquet"
            )
            if not path_to_demographics_scored.exists():
                raise FileNotFoundError(
                    f"Could not find demographics scored file at {path_to_demographics_scored}"
                )
            demographics_scored = (
                pd.read_parquet(path_to_demographics_scored)
                .drop(["Timestamp", "timezone_change"], axis=1)
                .rename(
                    columns={"igtb.datetime": "local_time", "ParticipantID": "user_id"}
                )
                .set_index("user_id")
            )

            path_to_demographics = (
                self.root_folder
                / "_global_metrics"
                / "GroundTruthSurvey"
                / "IGTBDemographics.parquet"
            )
            if not path_to_demographics.exists():
                raise FileNotFoundError(
                    f"Could not find demographics file at {path_to_demographics}"
                )
            demographics = (
                pd.read_parquet(path_to_demographics)
                .drop(["Timestamp", "timezone_change"], axis=1)
                .rename(
                    columns={
                        "igtb.datetime": "local_time",
                        "ParticipantID": "user_id",
                        "gender": "sex",
                        "age_range": "age",
                    }
                )
                .set_index("user_id")
            )

            self.demographics = pd.concat(
                (demographics, demographics_scored.drop("local_time", axis=1)), axis=1
            )
        return self.demographics

    def get_user_ids(self):
        return self.user_ids


__all__ = ["TesseraeParquetLoader"]
