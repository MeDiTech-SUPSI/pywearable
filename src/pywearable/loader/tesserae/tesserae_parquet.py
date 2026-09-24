import ast
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ... import schema
from ..base import BaseLoader


class TesseraeParquetLoader(BaseLoader):
    """
    Loader for Tesserae dataset converted in parqeut format.

    You can use this loader if you have previously converted
    the Tesserae dataset, which is provided in plain CSV files,
    to Parquet using the script available in the CLI
    of the package.
    """

    hrv_source = "bbi"

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
        return (
            schema.filter_heart_rate_by_time(hr, start_date, end_date, time_col)
            .sort_values("utc_time")
            .reset_index(drop=True)
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

        stress = pd.read_parquet(
            path_to_file,
            filters=[filters],
        ).rename(columns={"Timestamp": "utc_time", "hr": "stress"})
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
        from ...schema import normalize_sleep_columns

        return normalize_sleep_columns(sleep_stage)

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
        staged_sleep_seconds = (
            classified_sleep_seconds
            + sleep["unmeasurable_sleep_seconds"].fillna(0)
        )
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
            sleep["duration_seconds"].fillna(0)
            + sleep["awake_seconds"].fillna(0)
            > 0
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
            (sleep["local_time"] - sleep["utc_time"]) / pd.Timedelta(milliseconds=1)
        )
        sleep["unix_timestamp_in_ms"] = sleep["utc_time"].astype("int64") // 1_000_000
        sleep["local_end_time"] = sleep["local_time"] + pd.to_timedelta(
            sleep["duration_seconds"].fillna(0)
            + sleep["awake_seconds"].fillna(0),
            unit="seconds",
        )
        sleep["utc_end_time"] = sleep["utc_time"] + pd.to_timedelta(
            sleep["duration_seconds"].fillna(0)
            + sleep["awake_seconds"].fillna(0),
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

        from ...schema import normalize_sleep_columns

        return normalize_sleep_columns(sleep)

    def load_bbi(
        self, user_id: str, start_date: pd.Timestamp, end_date: pd.Timestamp
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

    def load_steps(
        self,
        user_id: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        backend: str = "health_api",
    ) -> pd.DataFrame:
        if backend == "health_api":
            path_to_file = (
                self.root_folder / user_id / "GarminHealthAPI" / "Epochs.parquet"
            )
        elif backend == "pa":
            path_to_file = self.root_folder / user_id / "GarminPA" / "Steps.parquet"
        else:
            raise ValueError(f"backend must be either health_api or pa, not {backend}")
        if not path_to_file.exists():
            return pd.DataFrame(
                columns=["local_time", "utc_time", "timezone_offset", "steps"]
            )

        steps = (
            pd.read_parquet(
                path_to_file,
            )
            .sort_values(by="Timestamp", ignore_index=True)
            .rename(
                columns={
                    "Timestamp": "utc_time",
                }
            )
            .drop("tz_source", axis=1)
        )
        steps["local_time"] = pd.to_datetime(steps["local_time"])

        steps["utc_time"] = steps["utc_time"].dt.tz_convert("UTC").dt.tz_localize(None)
        steps["timezone_offset"] = steps["local_time"] - steps["utc_time"]

        # Filter by isoDate
        if not (start_date is None):
            steps = steps.loc[
                (steps["local_time"] >= start_date),
                :,
            ]
        if not (end_date is None):
            steps = steps.loc[
                (steps["local_time"] < end_date),
                :,
            ]

        # Let's convert them on a 5-sec resolution
        # starting from 15-minute resolution
        steps = steps.sort_values(
            ["utc_time", "steps"], ascending=True
        ).drop_duplicates(["utc_time", "intensity"], keep="last")

        steps = steps[steps["steps"] > 0]

        steps = (
            steps.set_index("utc_time")
            .resample(rule="5s")["steps"]
            .mean()
            .ffill(limit=179)
            / 180
        ).reset_index()

        steps = steps.drop_duplicates("utc_time")
        return steps

    def load_pulse_ox(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame()

    def load_sleep_pulse_ox(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame()

    def load_sleep_respiration(self, user_id, start_date=None, end_date=None):
        return pd.DataFrame()

    ##########################################
    #       Use information
    ##########################################

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

    def get_all_users_age(self):
        return self.get_all_users_demographics()["age"]

    def get_all_users_sex(self):
        return self.get_all_users_demographics()["sex"]

    def get_user_ids(self):
        return self.user_ids

    def get_user_demographics(self, user_id: str) -> dict:
        if user_id in self.get_all_users_demographics().index:
            return self.get_all_users_demographics().loc[user_id].to_dict()
        else:
            return {}

    def get_user_sex(self, user_id: str) -> str:
        if user_id in self.get_all_users_demographics().index:
            return self.get_all_users_demographics().loc[user_id, "sex"]
        else:
            return ""

    def get_user_age(self, user_id: str) -> str:
        if user_id in self.get_all_users_demographics().index:
            return self.get_all_users_demographics().loc[user_id, "age"]
        else:
            return ""

    def get_user_end_date(self, user_id) -> pd.Timestamp:
        pass

    def get_user_start_date(self, user_id) -> pd.Timestamp:
        if user_id in self.get_all_users_demographics().index:
            return self.get_all_users_demographics().loc[user_id, "local_time"]
        else:
            return pd.NaT
