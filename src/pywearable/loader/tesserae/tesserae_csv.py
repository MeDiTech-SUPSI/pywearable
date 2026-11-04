import ast
import glob
import os
from pathlib import Path
from typing import Literal, Union

import duckdb
import numpy as np
import pandas as pd

from ... import schema
from ..base import BaseLoader
from ..garmin import _normalize_garmin_sleep_columns, prepare_epoch_activity_samples


class TesseraeCSVLoader(BaseLoader):
    """Load a local Tesserae dataset in its published CSV layout.

    Parameters
    ----------
    root_folder
        Directory containing one Tesserae data folder per participant.
    """

    activity_source = "samples"
    activity_time_columns = (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL)
    activity_primary_resolution = "15min"
    provider_features = frozenset({"garmin_stress_level"})
    provider_feature_time_columns = (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL)

    def __init__(self, root_folder: Union[str, os.PathLike]):
        if not Path(root_folder).exists():
            raise ValueError(f"Folder {root_folder} does not exist.")
        self.root_folder = Path(root_folder)
        self.user_ids = [x.name for x in self.root_folder.iterdir()]
        self.con = duckdb.connect()

    def get_user_ids(self):
        return self.user_ids

    ##################################################
    #               Garmin Health API                #
    ##################################################
    def load_activity_series(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        activity_series_folder = self.root_folder / user_id / "GarminHealthAPI"

        pattern = os.path.join(activity_series_folder, "*ActivitySeries*.csv")
        activity_series_files = glob.glob(pattern)
        if len(activity_series_files) > 0:
            query = f"""
                    SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                """
            activity_series = (
                self.con.execute(query)
                .df()
                .drop(
                    ["ParticipantID"],
                    axis=1,
                )
                .rename(columns={"Timestamp": "utc_time", "start_time": "local_time"})
            )

            activity_series["utc_time"] = pd.to_datetime(
                activity_series["utc_time"], utc=True
            ).dt.tz_localize(None)

            activity_series["local_time"] = pd.to_datetime(
                activity_series["local_time"]
            )

            if start_date is not None:
                activity_series = activity_series[
                    activity_series["local_time"] >= pd.Timestamp(start_date)
                ]
            if end_date is not None:
                activity_series = activity_series[
                    activity_series["local_time"] < pd.Timestamp(end_date)
                ]
            return activity_series
        else:
            return pd.DataFrame(
                columns=[
                    "local_time",
                    "utc_time",
                    "steps",
                    "duration_seconds",
                    "active_time_seocnds",
                    "distance_meters",
                ]
            )

    def load_activities(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        activities_folder = self.root_folder / user_id / "GarminHealthAPI"

        pattern = os.path.join(activities_folder, "*Activities*.csv")
        activity_series_files = glob.glob(pattern)
        if len(activity_series_files) > 0:
            query = f"""
                        SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                    """
            activities = (
                self.con.execute(query)
                .df()
                .drop(
                    ["ParticipantID"],
                    axis=1,
                )
                .rename(columns={"Timestamp": "utc_time", "start_time": "local_time"})
            )
            if not (start_date is None):
                activities = activities[activities["local_time"] >= start_date]
            if not (end_date is None):
                activities = activities[activities["local_time"] < end_date]
            return activities
        else:
            return pd.DataFrame(
                columns=[
                    "utc_time",
                    "local_time",
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
            )

    def load_epochs(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        epochs_folder = self.root_folder / user_id / "GarminHealthAPI"

        pattern = os.path.join(epochs_folder, "*Epochs*.csv")
        epochs_files = glob.glob(pattern)
        if len(epochs_files) > 0:
            query = f"""
                    SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                """
            epochs = (
                self.con.execute(query)
                .df()
                .drop(["ParticipantID", "tz_source"], axis=1)
                .rename(columns={"Timestamp": "utc_time"})
            )

            epochs["utc_time"] = pd.to_datetime(
                epochs["utc_time"], utc=True
            ).dt.tz_localize(None)

            epochs["local_time"] = pd.to_datetime(epochs["local_time"])

            if start_date is not None:
                epochs = epochs[epochs["local_time"] >= pd.Timestamp(start_date)]
            if end_date is not None:
                epochs = epochs[epochs["local_time"] < pd.Timestamp(end_date)]
            return epochs
        else:
            return pd.DataFrame(
                columns=[
                    "local_time",
                    "utc_time",
                    "steps",
                    "duration_seconds",
                    "active_time_seocnds",
                    "distance_meters",
                ]
            )

    def load_move_iq(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ):
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        moveiq_folder = self.root_folder / user_id / "GarminHealthAPI"

        pattern = os.path.join(moveiq_folder, "*MoveIQ*.csv")
        moveiq_files = glob.glob(pattern)
        if len(moveiq_files) > 0:
            query = f"""
                                SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                            """
            moveiq = (
                self.con.execute(query)
                .df()
                .drop(["ParticipantID", "tz_source"], axis=1)
                .rename(columns={"Timestamp": "utc_time"})
            )
            moveiq["utc_time"] = pd.to_datetime(
                moveiq["utc_time"], utc=True
            ).dt.tz_localize(None)
            moveiq["local_time"] = pd.to_datetime(moveiq["local_time"])

            moveiq["activity_subtype"] = moveiq["activity_subtype"].replace(
                "NULL", pd.NA
            )

            moveiq = moveiq.drop_duplicates(subset="utc_time", keep="last")
            if not (start_date is None):
                moveiq = moveiq[moveiq["local_time"] >= start_date]
            if not (end_date is None):
                moveiq = moveiq[moveiq["local_time"] < end_date]
            return moveiq.sort_values("utc_time", ignore_index=True)
        else:
            return pd.DataFrame(
                columns=[
                    "utc_time",
                    "local_time",
                    "activity_type",
                    "activity_subtype",
                    "duration_seconds",
                ]
            )

    def load_pulse_ox(
        self, user_id=None, start_date=None, end_date=None
    ) -> pd.DataFrame:
        """
        Tesserae dataset doesn't have pulse ox data.
        This function will always return an empty :class:`pandas.DataFrame`
        """
        return pd.DataFrame(columns=["utc_time", "local_time", "timezone", "pulse_ox"])

    def load_respiration(
        self, user_id=None, start_date=None, end_date=None
    ) -> pd.DataFrame:
        """
        Tesserae dataset doesn't have respiration data.
        This function will always return an empty :class:`pandas.DataFrame`
        """
        return pd.DataFrame(
            columns=["utc_time", "local_time", "timezone", "breaths_per_minute"]
        )

    def load_sleep_stage(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        sleep_stage_folder = self.root_folder / user_id / "GarminHealthAPI"

        pattern = os.path.join(sleep_stage_folder, "*Sleep*.csv")
        sleep_stage_files = glob.glob(pattern)
        if len(sleep_stage_files) == 0:
            return pd.DataFrame(
                columns=[
                    "sleep_summary_id",
                    "local_time",
                    "utc_time",
                    "duration_in_ms",
                    "type",
                ]
            )
        query = f"""
                            SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                        """
        sleep_summary = self.con.execute(query).df()

        # Let's create the same sleep summary identifier
        sleep_summary["sleep_summary_id"] = (
            sleep_summary["ParticipantID"] + "-" + sleep_summary.index.astype(str)
        )

        sleep_summary["sleep_levels_map"] = (
            sleep_summary["sleep_levels_map"]
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
        sleep_stage = pd.concat(
            [
                pd.json_normalize(json_norm[sleep_stage_type].explode())
                .assign(type=sleep_stage_type)
                .dropna(how="any")
                for sleep_stage_type in json_norm.columns
            ],
            ignore_index=True,
        ).sort_values(by="startTime", ignore_index=True)

        for col in ["startTime", "endTime"]:
            sleep_stage[col] = pd.to_datetime(sleep_stage[col])

        sleep_stage = sleep_stage.rename(
            columns={
                "startTime": "local_time",
                "startTimeTimestamp": "unixTimestampInMs",
            }
        ).drop(
            [
                "startTimeWithTimezone",
                "endTimeWithTimezone",
                "endTimeInSeconds",
                "endTimeTimestamp",
                "startTimeInSeconds",
            ],
            axis=1,
        )

        sleep_stage["duration_in_ms"] = (
            sleep_stage["endTime"] - sleep_stage["local_time"]
        ).dt.total_seconds() * 1000

        sleep_stage = sleep_stage.drop(["endTime"], axis=1)

        sleep_stage["unixTimestampInMs"] = sleep_stage["unixTimestampInMs"] * 1000
        sleep_stage["utc_time"] = pd.to_datetime(
            sleep_stage["unixTimestampInMs"], unit="ms", utc=True
        ).dt.tz_localize(None)
        # sleep_stage["timezoneOffsetInMs"] = (
        #    sleep_stage["isoDate"] - sleep_stage["utc_time"]
        # ) / pd.Timedelta(milliseconds=1)

        # sleep_stage["type"] = sleep_stage["type"].replace({"rem": "REM"})

        # Filter by isoDate
        if start_date is not None:
            sleep_stage = sleep_stage.loc[
                sleep_stage["local_time"] >= pd.Timestamp(start_date)
            ]
        if end_date is not None:
            sleep_stage = sleep_stage.loc[
                sleep_stage["local_time"] < pd.Timestamp(end_date)
            ]

        if drop_duplicates:
            sleep_summary = self.load_sleep_summary(
                user_id=user_id, start_date=None, end_date=None, drop_duplicates=True
            )
            sleep_stage = sleep_stage[
                sleep_stage["sleep_summary_id"].isin(
                    sleep_summary["sleep_summary_id"].unique()
                )
            ]
        return sleep_stage.sort_values(by="local_time", ascending=True).drop(
            ["unixTimestampInMs"], axis=1
        )

    def load_sleep_summary(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        sleep_summary_folder = self.root_folder / user_id / "GarminHealthAPI"

        pattern = os.path.join(sleep_summary_folder, "*Sleep*.csv")
        sleep_summary_files = glob.glob(pattern)
        if len(sleep_summary_files) == 0:
            return pd.DataFrame(
                columns=[
                    "sleep_summary_id",
                    "calendar_date",
                    "local_time",
                    "utc_time",
                    "duration_in_ms",
                    "awake_duration_in_ms",
                ]
            )
        query = f"""
            SELECT * FROM read_csv('{pattern}', union_by_name=True);
        """
        # , types={{'Timestamp':'VARCHAR'}});
        sleep_summary = self.con.execute(query).df()
        # We need to assign a sleep summary Id
        sleep_summary["sleep_summary_id"] = (
            sleep_summary["ParticipantID"] + "-" + sleep_summary.index.astype(str)
        )

        sleep_summary["utc_time"] = pd.to_datetime(
            sleep_summary["Timestamp"], utc=True
        ).dt.tz_localize(None)

        sleep_summary = sleep_summary.rename(
            columns={
                "awake_seconds": "awake_duration_in_ms",
                "rem_sleep_seconds": "rem_sleep_duration_in_ms",
                "deep_sleep_seconds": "deep_sleep_duration_in_ms",
                "unmeasurable_sleep_seconds": "unmeasurable_sleep_duration_in_ms",
                "light_sleep_seconds": "light_sleep_duration_in_ms",
                "duration_seconds": "duration_in_ms",
            }
        ).drop(["ParticipantID", "tz_source", "Timestamp"], axis=1)
        # From seconds to ms
        for col in [
            "awake_duration_in_ms",
            "rem_sleep_duration_in_ms",
            "deep_sleep_duration_in_ms",
            "unmeasurable_sleep_duration_in_ms",
            "light_sleep_duration_in_ms",
            "duration_in_ms",
        ]:
            sleep_summary[col] *= 1000
        # We need to manually compute the calendarDate
        sleep_summary["calendar_date"] = (
            pd.to_datetime(sleep_summary["local_time"])
            + pd.to_timedelta(
                sleep_summary["duration_in_ms"] + sleep_summary["awake_duration_in_ms"],
                unit="ms",
            )
        ).dt.normalize()

        # Filter by calendarDate
        if start_date is not None:
            sleep_summary = sleep_summary.loc[
                sleep_summary["calendar_date"] >= pd.Timestamp(start_date).normalize()
            ]
        if end_date is not None:
            sleep_summary = sleep_summary.loc[
                sleep_summary["calendar_date"] < pd.Timestamp(end_date).normalize()
            ]

        if drop_duplicates:
            sleep_summary["validation_map"] = sleep_summary["validation"].astype(
                pd.CategoricalDtype(
                    categories=[
                        "MANUAL",
                        "AUTO_MANUAL",
                        "AUTO_TENTATIVE",
                        "ENHANCED_TENTATIVE",
                        "ENHANCED_FINAL",
                    ],
                    ordered=True,
                )
            )
            sleep_summary = sleep_summary.sort_values(
                by=["duration_in_ms", "validation_map"], ascending=[True, True]
            ).drop_duplicates(subset="calendar_date", keep="last", ignore_index=True)

        return sleep_summary

    def load_stress(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        stress_folder = self.root_folder / user_id / "GarminHealthAPI"

        pattern = os.path.join(stress_folder, "*Stress*.csv")
        stress_files = glob.glob(pattern)
        if len(stress_files) > 0:
            query = f"SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});"
            stress = (
                (self.con.execute(query).df())
                .rename(columns={"Timestamp": "utc_time", "hr": "stress_level"})
                .drop(["ParticipantID"], axis=1)
            )

            if start_date is not None:
                stress = stress[stress["local_time"] >= pd.Timestamp(start_date)]
            if end_date is not None:
                stress = stress[stress["local_time"] < pd.Timestamp(end_date)]
            return stress
        else:
            return pd.DataFrame()

    ##################################################
    #               Garmin PA                       #
    ##################################################

    def load_bbi(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        bbi_folder = self.root_folder / user_id / "GarminPA"
        pattern = os.path.join(bbi_folder, "*BBI*.csv")
        bbi_files = glob.glob(pattern)
        if len(bbi_files) == 0:
            return schema.empty_bbi_frame(["source"])
        query = f"""
                SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
            """
        bbi = (
            self.con.execute(query)
            .df()
            .drop(
                ["ParticipantID", "tz_source"],
                axis=1,
            )
            .rename(columns={"Timestamp": "utc_time"})
        )
        if len(bbi) == 0:
            return schema.empty_bbi_frame(["source"])
        bbi["utc_time"] = pd.to_datetime(bbi["utc_time"], utc=True).dt.tz_localize(None)
        bbi["timezone"] = bbi["local_time"] - bbi["utc_time"]
        # we may have some milliseconds difference between utc and local time
        bbi["timezone"] = bbi["timezone"].dt.round("1min")

        if not (start_date is None):
            bbi = bbi[bbi["local_time"] >= pd.Timestamp(start_date)]
        if not (end_date is None):
            bbi = bbi[bbi["local_time"] < pd.Timestamp(end_date)]
        bbi["source"] = "garmin_pa_bbi"
        return schema.prepare_bbi(bbi)

    def load_calories(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        calories_folder = self.root_folder / user_id / "GarminPA"
        pattern = os.path.join(calories_folder, "*Calories*.csv")
        calories_files = glob.glob(pattern)
        if len(calories_files) == 0:
            return pd.DataFrame()
        query = f"""
                    SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                """
        calories = (
            self.con.execute(query)
            .df()
            .drop(
                ["ParticipantID", "tz_source"],
                axis=1,
            )
            .rename(columns={"Timestamp": "utc_time"})
        )
        calories["utc_time"] = pd.to_datetime(
            calories["utc_time"], utc=True
        ).dt.tz_localize(None)

        if not (start_date is None):
            calories = calories[calories["local_time"] >= pd.Timestamp(start_date)]
        if not (end_date is None):
            calories = calories[calories["local_time"] < pd.Timestamp(end_date)]
        return calories

    def load_device_battery(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")

        # Load all files
        intensity_folder = self.root_folder / user_id / "GarminPA"

        pattern = os.path.join(intensity_folder, "*_GarminBattery_*.csv")
        garmin_battery_files = glob.glob(pattern)
        if len(garmin_battery_files) > 0:
            query = f"""
                            SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                        """
            battery = (
                self.con.execute(query)
                .df()
                .drop(["ParticipantID", "tz_source"], axis=1)
                .rename(columns={"Timestamp": "utc_time"})
            ).replace("NULL", np.nan)

            battery["utc_time"] = pd.to_datetime(
                battery["utc_time"], utc=True
            ).dt.tz_localize(None)
            battery["local_time"] = pd.to_datetime(battery["local_time"])

            battery = battery.drop_duplicates(subset="utc_time", keep="last")
            if not (start_date is None):
                battery = battery[battery["local_time"] >= pd.Timestamp(start_date)]
            if not (end_date is None):
                battery = battery[battery["local_time"] < pd.Timestamp(end_date)]
            return battery
        else:
            return pd.DataFrame(columns=["utc_time", "local_time", "level"])

    def load_floors(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        floors_folder = self.root_folder / user_id / "GarminPA"
        pattern = os.path.join(floors_folder, "*Floors*.csv")
        floors_files = glob.glob(pattern)
        if len(floors_files) == 0:
            return pd.DataFrame()
        query = f"""
                SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
            """
        floors = (
            self.con.execute(query)
            .df()
            .drop(
                ["ParticipantID", "tz_source"],
                axis=1,
            )
            .rename(columns={"Timestamp": "utc_time"})
        )
        floors["utc_time"] = pd.to_datetime(
            floors["utc_time"], utc=True
        ).dt.tz_localize(None)

        if not (start_date is None):
            floors = floors[floors["local_time"] >= pd.Timestamp(start_date)]
        if not (end_date is None):
            floors = floors[floors["local_time"] < pd.Timestamp(end_date)]
        return floors

    def load_heart_rate_from_garmin_pa(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
        time_col: str = "local_time",
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        schema.validate_heart_rate_time_col(time_col)
        # Load all files
        heart_rate_folder = self.root_folder / user_id / "GarminPA"

        pattern = os.path.join(heart_rate_folder, "*_HR_*.csv")
        hr_files = glob.glob(pattern)
        if len(hr_files) > 0:
            query = f"""
                            SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                        """
            hr = (
                self.con.execute(query)
                .df()
                .drop(["ParticipantID", "tz_source"], axis=1)
                .rename(columns={"Timestamp": "utc_time", "hr": "beats_per_minute"})
            )

            hr["utc_time"] = pd.to_datetime(hr["utc_time"], utc=True).dt.tz_localize(
                None
            )
            hr["local_time"] = pd.to_datetime(hr["local_time"])
            hr["timezone"] = hr["local_time"] - hr["utc_time"]
            hr["timezone_offset"] = hr["timezone"]
            hr["heart_rate"] = pd.to_numeric(hr["beats_per_minute"], errors="coerce")
            hr = hr[hr["heart_rate"] > 0]
            hr = hr.drop_duplicates(subset="utc_time", keep="last")
            hr = schema.prepare_heart_rate_samples(hr)
            return (
                schema.filter_heart_rate_by_time(hr, start_date, end_date, time_col)
                .sort_values("utc_time")
                .reset_index(drop=True)
            )
        else:
            return schema.empty_heart_rate_frame(
                ["timezone_offset", "tz_source", "beats_per_minute"]
            )

    def load_hrv_series(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        hrv_series_folder = self.root_folder / user_id / "GarminPA"
        pattern = os.path.join(hrv_series_folder, "*HRVSeries*.csv")
        hrv_series_files = glob.glob(pattern)
        if len(hrv_series_files) == 0:
            return pd.DataFrame(
                columns=[
                    "utc_time",
                    "local_time",
                    "stop_time",
                    "rmssd",
                    "sdnn",
                    "duration",
                ]
            )
        query = f"""
                    SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                """
        hrv_series = (
            self.con.execute(query)
            .df()
            .drop(
                ["ParticipantID", "tz_source"],
                axis=1,
            )
            .rename(columns={"Timestamp": "utc_time"})
        )
        hrv_series["utc_time"] = pd.to_datetime(
            hrv_series["utc_time"], utc=True
        ).dt.tz_localize(None)

        # print(hrv_series["duration"].describe())

        if not (start_date is None):
            hrv_series = hrv_series[
                hrv_series["local_time"] >= pd.Timestamp(start_date)
            ]
        if not (end_date is None):
            hrv_series = hrv_series[hrv_series["local_time"] < pd.Timestamp(end_date)]
        return hrv_series

    def load_intensity(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")

        # Load all files
        intensity_folder = self.root_folder / user_id / "GarminPA"

        pattern = os.path.join(intensity_folder, "*_Intensity_*.csv")
        intensity_files = glob.glob(pattern)
        if len(intensity_files) > 0:
            query = f"""
                        SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                    """
            intensity = (
                self.con.execute(query)
                .df()
                .drop(["ParticipantID", "tz_source"], axis=1)
                .rename(columns={"Timestamp": "utc_time"})
            ).replace("NULL", np.nan)

            intensity["utc_time"] = pd.to_datetime(
                intensity["utc_time"], utc=True
            ).dt.tz_localize(None)
            intensity["local_time"] = pd.to_datetime(intensity["local_time"])

            intensity = intensity.drop_duplicates(subset="utc_time", keep="last")
            if not (start_date is None):
                intensity = intensity[
                    intensity["local_time"] >= pd.Timestamp(start_date)
                ]
            if not (end_date is None):
                intensity = intensity[intensity["local_time"] < pd.Timestamp(end_date)]
            return intensity
        else:
            return pd.DataFrame(
                columns=[
                    "utc_time",
                    "local_time",
                    "intensity_minutes_iphone",
                    "daily_vigorous_minutes",
                    "weekly_minutes",
                ]
            )

    def load_steps(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")

        # Load all files
        steps_folder = self.root_folder / user_id / "GarminPA"

        pattern = os.path.join(steps_folder, "*_Steps_*.csv")
        steps_files = glob.glob(pattern)
        if len(steps_files) > 0:
            query = f"""
                    SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                """
            steps = (
                self.con.execute(query)
                .df()
                .drop(["ParticipantID", "tz_source"], axis=1)
                .rename(columns={"Timestamp": "utc_time"})
            )

            steps["utc_time"] = pd.to_datetime(
                steps["utc_time"], utc=True
            ).dt.tz_localize(None)
            steps["local_time"] = pd.to_datetime(steps["local_time"])

            steps = steps.drop_duplicates(subset="utc_time", keep="last")
            if not (start_date is None):
                steps = steps[steps["local_time"] >= pd.Timestamp(start_date)]
            if not (end_date is None):
                steps = steps[steps["local_time"] < pd.Timestamp(end_date)]
            return steps
        else:
            return pd.DataFrame(columns=["utc_time", "local_time", "step_count"])

    ###################################################
    #                 Mixed metrics
    ###################################################
    def load_heart_rate(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
        time_col: str = "local_time",
        source="garmin_health_api",
    ):
        if source == "garmin_pa":
            return self.load_heart_rate_from_garmin_pa(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                time_col=time_col,
            )
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        schema.validate_heart_rate_time_col(time_col)
        # Load all files
        heart_rate_folder = self.root_folder / user_id / "GarminHealthAPI"

        pattern = os.path.join(heart_rate_folder, "*HR*.csv")
        hr_files = glob.glob(pattern)
        if len(hr_files) > 0:
            query = f"""
                            SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                        """
            hr = (
                self.con.execute(query)
                .df()
                .rename(columns={"Timestamp": "utc_time", "hr": "beats_per_minute"})
            )

            hr["utc_time"] = pd.to_datetime(hr["utc_time"], utc=True).dt.tz_localize(
                None
            )
            hr["local_time"] = pd.to_datetime(hr["local_time"])
            hr["timezone"] = hr["local_time"] - hr["utc_time"]
            hr["heart_rate"] = pd.to_numeric(hr["beats_per_minute"], errors="coerce")
            hr = hr[hr["heart_rate"] > 0]
            hr = hr.drop_duplicates(subset="utc_time", keep="last")
            hr = schema.prepare_heart_rate_samples(hr)
            return (
                schema.filter_heart_rate_by_time(hr, start_date, end_date, time_col)
                .sort_values("utc_time")
                .reset_index(drop=True)
            )
        else:
            return schema.empty_heart_rate_frame(["tz_source", "beats_per_minute"])

    ###################################################
    #                 Summaries
    ###################################################
    def load_daily_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        daily_summary_folder = self.root_folder / user_id / "GarminSummary"

        pattern = os.path.join(daily_summary_folder, "*DailySummaries*.csv")
        daily_summary_files = glob.glob(pattern)
        if len(daily_summary_files) > 0:
            query = f"""
                SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
            """
            daily_summary = self.con.execute(query).df()

            daily_summary = daily_summary.rename(
                columns={"local_time": "calendar_date"}
            ).drop(["ParticipantID", "tz_source", "Timestamp"], axis=1)

            daily_summary["calendar_date"] = pd.to_datetime(
                daily_summary["calendar_date"]
            )
            daily_summary["local_time"] = pd.to_datetime(daily_summary["calendar_date"])
            daily_summary["utc_time"] = pd.to_datetime(daily_summary["calendar_date"])

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
                ).drop_duplicates(
                    subset="calendar_date", keep="last", ignore_index=True
                )
            return daily_summary
        else:
            return pd.DataFrame()

    #####################################################
    #               Additional summaries
    ####################################################
    def load_activity_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        activity_summary_folder = self.root_folder / user_id / "GarminSummary"

        pattern = os.path.join(activity_summary_folder, "*ActivitySummary*.csv")
        activity_summary_files = glob.glob(pattern)
        if len(activity_summary_files) > 0:
            query = f"""
                    SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                """
            activity_summary = (
                self.con.execute(query).df().drop(["ParticipantID"], axis=1)
            ).rename(
                columns={
                    "date": "calendar_date",
                    "active_secs": "active_seconds",
                    "highly_active_secs": "highly_active_seconds",
                    "total_activity_secs": "total_activity_seconds",
                }
            )

            activity_summary["calendar_date"] = pd.to_datetime(
                activity_summary["calendar_date"]
            )
            activity_summary["timezone_offset"] = activity_summary["Timestamp"].apply(
                lambda x: self._convert_timezone_offset(x)
            )

            activity_summary = activity_summary.drop(["Timestamp"], axis=1)

            if not (start_date is None):
                activity_summary = activity_summary.loc[
                    activity_summary["calendar_date"]
                    >= pd.Timestamp(start_date).normalize()
                ]
            if not (end_date is None):
                activity_summary = activity_summary.loc[
                    activity_summary["calendar_date"]
                    < pd.Timestamp(end_date).normalize()
                ]

            return activity_summary
        else:
            return pd.DataFrame(
                columns=[
                    "calendar_date",
                    "num_samples",
                    "active_seconds",
                    "highly_active_seconds",
                    "total_activity_seconds",
                ]
            )

    def load_adjusted_imputed_sleep_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        adj_imp_sleep_summary_folder = self.root_folder / user_id / "GarminSummary"

        pattern = os.path.join(adj_imp_sleep_summary_folder, "*AdjImputedSleep*.csv")
        adj_imp_sleep_summary_files = glob.glob(pattern)
        if len(adj_imp_sleep_summary_files) > 0:
            query = f"""
                SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
            """
            adj_imp_sleep_summary = (
                self.con.execute(query).df().drop(["ParticipantID"], axis=1)
            ).rename(columns={"date": "calendar_date"})
            adj_imp_sleep_summary["calendar_date"] = pd.to_datetime(
                adj_imp_sleep_summary["calendar_date"]
            )
            adj_imp_sleep_summary["timezone_offset"] = adj_imp_sleep_summary[
                "Timestamp"
            ].apply(lambda x: self._convert_timezone_offset(x))

            adj_imp_sleep_summary = adj_imp_sleep_summary.drop(["Timestamp"], axis=1)
            for col in ["adjusted_bed_time", "adjusted_wakeup_time"]:
                adj_imp_sleep_summary[col] = adj_imp_sleep_summary[
                    "calendar_date"
                ] + pd.to_timedelta(
                    adj_imp_sleep_summary[col].replace("NULL", None).astype(float),
                    unit="h",
                )
            adj_imp_sleep_summary["adjusted_sleep_duration"] = pd.to_timedelta(
                adj_imp_sleep_summary["adjusted_sleep_duration"]
                .replace("NULL", None)
                .astype(float),
                unit="h",
            )

            if not (start_date is None):
                adj_imp_sleep_summary = adj_imp_sleep_summary.loc[
                    adj_imp_sleep_summary["calendar_date"]
                    >= pd.Timestamp(start_date).normalize()
                ]
            if not (end_date is None):
                adj_imp_sleep_summary = adj_imp_sleep_summary.loc[
                    adj_imp_sleep_summary["calendar_date"]
                    < pd.Timestamp(end_date).normalize()
                ]

        return adj_imp_sleep_summary

    def load_heart_rate_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        hr_summary_folder = self.root_folder / user_id / "GarminSummary"

        pattern = os.path.join(hr_summary_folder, "*HRSummary*.csv")
        hr_summary_files = glob.glob(pattern)
        if len(hr_summary_files) > 0:
            query = f"""
                        SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                    """
            hr_summary = (
                self.con.execute(query).df().drop(["ParticipantID"], axis=1)
            ).rename(
                columns={
                    "date": "calendar_date",
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
            hr_summary["calendar_date"] = pd.to_datetime(hr_summary["calendar_date"])
            hr_summary["timezone_offset"] = hr_summary["Timestamp"].apply(
                lambda x: self._convert_timezone_offset(x)
            )
            hr_summary["timezone_change"] = hr_summary["timezone_change"].replace(
                "NULL", None
            )
            hr_summary = hr_summary.drop(["Timestamp"], axis=1)

            if not (start_date is None):
                hr_summary = hr_summary.loc[
                    hr_summary["calendar_date"] >= pd.Timestamp(start_date).normalize()
                ]
            if not (end_date is None):
                hr_summary = hr_summary.loc[
                    hr_summary["calendar_date"] < pd.Timestamp(end_date).normalize()
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
        # Load all files
        hrv_summary_folder = self.root_folder / user_id / "GarminSummary"

        pattern = os.path.join(hrv_summary_folder, "*HRVSummary*.csv")
        hrv_summary_files = glob.glob(pattern)
        if len(hrv_summary_files) > 0:
            query = f"""
                        SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                    """
            hrv_summary = (
                self.con.execute(query)
                .df()
                .drop(["ParticipantID"], axis=1)
                .rename(columns={"date": "calendar_date"})
                .replace("NULL", np.nan)
            )

            hrv_summary["calendar_date"] = pd.to_datetime(hrv_summary["calendar_date"])
            hrv_summary["timezone_offset"] = hrv_summary["Timestamp"].apply(
                lambda x: self._convert_timezone_offset(x)
            )

            hrv_summary = hrv_summary.drop(["Timestamp"], axis=1)

            if not (start_date is None):
                hrv_summary = hrv_summary.loc[
                    hrv_summary["calendar_date"] >= pd.Timestamp(start_date).normalize()
                ]
            if not (end_date is None):
                hrv_summary = hrv_summary.loc[
                    hrv_summary["calendar_date"] < pd.Timestamp(end_date).normalize()
                ]

            return hrv_summary
        else:
            return pd.DataFrame(
                columns=[
                    "utc_time",
                    "local_time",
                    "calendar_date",
                    "calc_code",
                    "calc_desc",
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
            )

    def load_stress_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        stress_summary_folder = self.root_folder / user_id / "GarminSummary"

        pattern = os.path.join(stress_summary_folder, "*Stress*.csv")
        stress_summary_files = glob.glob(pattern)
        if len(stress_summary_files) > 0:
            query = f"""
                        SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                    """
            stress_summary = (
                self.con.execute(query).df().drop(["ParticipantID"], axis=1)
            ).rename(
                columns={
                    "date": "calendar_date",
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

            stress_summary["calendar_date"] = pd.to_datetime(
                stress_summary["calendar_date"]
            )
            stress_summary["timezone_offset"] = stress_summary["Timestamp"].apply(
                lambda x: self._convert_timezone_offset(x)
            )
            stress_summary["timezone_change"] = stress_summary[
                "timezone_change"
            ].replace("NULL", None)
            stress_summary = stress_summary.drop(["Timestamp"], axis=1)

            if not (start_date is None):
                stress_summary = stress_summary.loc[
                    stress_summary["calendar_date"]
                    >= pd.Timestamp(start_date).normalize()
                ]
            if not (end_date is None):
                stress_summary = stress_summary.loc[
                    stress_summary["calendar_date"] < pd.Timestamp(end_date).normalize()
                ]

        return stress_summary

    #############################################################
    #                   Phone
    #############################################################
    def _load_metric(
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
        # Load all files
        if folder == "Gimbal":
            folder = self.root_folder / user_id / "data-pub-onreq" / user_id / "Gimbal"
        else:
            folder = self.root_folder / user_id / folder

        pattern = os.path.join(folder, f"*_{metric}_*.csv")
        files = glob.glob(pattern)
        if len(files) > 0:
            query = f"""
                        SELECT * FROM read_csv('{pattern}', union_by_name=True, types={{'Timestamp':'VARCHAR'}});
                    """
            df = (
                self.con.execute(query)
                .df()
                .drop(["ParticipantID"], axis=1)
                # .rename(columns={"Timestamp": "utc_time"})
            )
            if "utc_time" in df.columns:
                df = df.drop("utc_time", axis=1)
            df = df.rename(columns={"Timestamp": "utc_time"})
            df["timezone_offset"] = df["utc_time"].apply(
                lambda x: self._convert_timezone_offset(x)
            )
            df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True).dt.tz_localize(
                None
            )
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
        else:
            return pd.DataFrame()

    def load_phone_pedometer(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        pedometer = self._load_metric(
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
        return self._load_metric(
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
        return self._load_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Contextual",
            metric="WeatherSummary",
            date_col="date",
        )

    def load_noaa_storm_events(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        return self._load_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Contextual",
            metric="NOAAStormEvents",
            date_col="start_time",
        )

    #############################################################
    #                   Beacon
    #############################################################
    def load_beacon_callback(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        return self._load_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Gimbal",
            metric="BeaconCallback",
            date_col="local_time",
        )

    def load_beacon_place(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        gimbal_beacon_place = self._load_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Gimbal",
            metric="BeaconPlace",
            date_col="local_time",
        )
        if len(gimbal_beacon_place) > 0:
            for col in ["departure_time", "arrival_time"]:
                gimbal_beacon_place[col] = pd.to_datetime(
                    gimbal_beacon_place[col].replace("NULL", np.nan)
                )
            print(gimbal_beacon_place.columns)
        return gimbal_beacon_place

    def load_beacon_sighting(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        gimbal_beacon_sighting = self._load_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Gimbal",
            metric="BeaconSighting",
            date_col="local_time",
        )
        return gimbal_beacon_sighting

    def load_beacon_summary(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        gimbal_beacon_summary = self._load_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="GimbalSummary",
            metric="BeaconSummary",
            date_col="date",
        )
        return gimbal_beacon_summary

    def load_home_office_hours(
        self, user_id, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        home_office_hours = self._load_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="GimbalSummary",
            metric="HomeOfficeHours",
            date_col="date",
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
        interaction_summary = self._load_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="GimbalSummary",
            metric="InteractionSummary",
            date_col="date",
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
        return self._load_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Mapping",
            metric="Beacons",
            date_col="local_time",
        )

    def load_mapping_participant_info(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        return self._load_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Mapping",
            metric="ParticipantInfo",
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
        home_office_hours = self._load_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Mixed",
            metric="HomeOfficeHoursLoc",
            date_col="date",
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
        hrv_summary_baseline = self._load_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Mixed",
            metric="HRVSummaryBaseline",
            date_col="date",
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
        hrv = self._load_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Mixed",
            metric="HRVWorkDeskHomeSummary",
            date_col="date",
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
        return self._load_metric(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            folder="Mixed",
            metric="PASummary",
            date_col="date",
        )

    #############################################################
    #                   Surveys
    #############################################################
    def load_daily_survey(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        ground_truth_folder = self.root_folder / user_id / "GroundTruthSurvey"

        pattern = os.path.join(ground_truth_folder, "*DailiesScored*.csv")
        daily_survey_files = glob.glob(pattern)
        if len(daily_survey_files) > 0:
            daily_survey = (
                pd.concat(
                    [
                        pd.read_csv(
                            f,
                            parse_dates=[
                                "date",
                                "survey_sent_datetime",
                                "survey_start_datetime",
                                "survey_end_datetime",
                            ],
                        )
                        for f in daily_survey_files
                    ],
                    ignore_index=True,
                )
                .drop(["ParticipantID", "timezone_change"], axis=1)
                .rename(columns={"date": "calendar_date"})
            )

            daily_survey["timezoneOffsetInMs"] = daily_survey["Timestamp"].apply(
                lambda x: self._convert_timezone_offset(x)
            )

            col_dict = {}
            for col in daily_survey.columns:
                if not "survey" in col and not (
                    col == "calendar_date"
                    or col == "timezoneOffsetInMs"
                    or col == "Timestamp"
                ):
                    col_dict[col] = "survey_" + col
            daily_survey = daily_survey.rename(columns=col_dict).drop(
                "Timestamp", axis=1
            )
            if start_date is not None:
                daily_survey = daily_survey.loc[
                    daily_survey["calendar_date"] >= pd.Timestamp(start_date)
                ]
            if end_date is not None:
                daily_survey = daily_survey.loc[
                    daily_survey["calendar_date"] < pd.Timestamp(end_date)
                ]
            return daily_survey.reset_index(drop=True)
        else:
            return pd.DataFrame()

    def load_demographics_raw(self, user_id: str) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        ground_truth_folder = self.root_folder / user_id / "GroundTruthSurvey"

        pattern = os.path.join(ground_truth_folder, "*IGTBDemographics*.csv")
        demographics_files = glob.glob(pattern)
        if len(demographics_files) > 0:
            demographics = (
                pd.concat(
                    [
                        pd.read_csv(
                            f,
                            parse_dates=[
                                "igtb.datetime",
                            ],
                        )
                        for f in demographics_files
                    ],
                    ignore_index=True,
                )
                .drop(["ParticipantID", "timezone_change"], axis=1)
                .rename(columns={"igtb.datetime": "local_time"})
            )
        else:
            return pd.DataFrame()
        demographics["timezoneOffsetInMs"] = demographics["Timestamp"].apply(
            lambda x: self._convert_timezone_offset(x)
        )

        col_dict = {}
        for col in demographics.columns:
            if not "demographics" in col and not (
                col == "local_time" or col == "timezoneOffsetInMs" or col == "Timestamp"
            ):
                col_dict[col] = "demographics_" + col
        demographics["calendar_date"] = demographics["local_time"].dt.normalize()
        demographics = demographics.rename(columns=col_dict).drop("Timestamp", axis=1)
        return demographics.reset_index(drop=True)

    def load_demographics_scored(self, user_id: str) -> pd.DataFrame:
        if not user_id in self.user_ids:
            raise ValueError(f"User id {user_id} is not valid")
        # Load all files
        ground_truth_folder = self.root_folder / user_id / "GroundTruthSurvey"

        pattern = os.path.join(ground_truth_folder, "*IGTBScored*.csv")
        demographics_files = glob.glob(pattern)
        if len(demographics_files) > 0:
            demographics = (
                pd.concat(
                    [
                        pd.read_csv(
                            f,
                            parse_dates=[
                                "igtb.datetime",
                            ],
                        )
                        for f in demographics_files
                    ],
                    ignore_index=True,
                )
                .drop(["ParticipantID", "timezone_change"], axis=1)
                .rename(columns={"igtb.datetime": "local_time"})
            )
        else:
            return pd.DataFrame()
        demographics["timezoneOffsetInMs"] = demographics["Timestamp"].apply(
            lambda x: self._convert_timezone_offset(x)
        )

        col_dict = {}
        for col in demographics.columns:
            if not "demographics" in col and not (
                col == "local_time" or col == "timezoneOffsetInMs" or col == "Timestamp"
            ):
                col_dict[col] = "demographics_" + col
        demographics["calendar_date"] = demographics["local_time"].dt.normalize()
        demographics = demographics.rename(columns=col_dict).drop("Timestamp", axis=1)
        return demographics.reset_index(drop=True)

    def load_provider_feature(
        self, feature: str, user_id: str, start_date=None, end_date=None, **kwargs
    ) -> pd.DataFrame:
        """Translate Tesserae's Garmin stress stream to canonical samples."""
        if feature != "garmin_stress_level":
            return super().load_provider_feature(
                feature, user_id, start_date, end_date, **kwargs
            )
        stress = self.load_stress(user_id, start_date, end_date).copy()
        if stress.empty:
            return schema.prepare_provider_feature_samples(
                pd.DataFrame(columns=[feature]), feature
            )
        stress[schema.LOCAL_TIME_COL] = pd.to_datetime(
            stress["isoDate"], errors="coerce"
        )
        stress[schema.UTC_TIME_COL] = pd.to_datetime(
            stress["Timestamp"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        stress[schema.TIMEZONE_COL] = pd.to_timedelta(
            pd.to_numeric(stress["timezoneOffsetInMs"], errors="coerce"), unit="ms"
        )
        stress[feature] = pd.to_numeric(stress["stressLevel"], errors="coerce")
        stress.loc[~stress[feature].between(0, 100), feature] = pd.NA
        return schema.prepare_provider_feature_samples(stress, feature)

    def load_activity_samples(
        self, user_id: str, start_date=None, end_date=None
    ) -> pd.DataFrame:
        """Return native Garmin Health API epochs as canonical step intervals."""
        epochs = self.load_epochs(user_id, start_date, end_date).copy()
        if epochs.empty:
            return schema.empty_activity_frame(extra_columns=("source",))

        epochs[schema.DURATION_IN_MS_COL] = (
            pd.to_numeric(epochs["duration_seconds"], errors="coerce") * 1000
        )
        epochs[schema.ACTIVITY_STEPS_COL] = epochs["steps"]
        return prepare_epoch_activity_samples(epochs)

    # Utility functions
    def get_user_list(self) -> list:
        return self.user_ids

    def get_user_start_date(self, user_id: str) -> pd.Timestamp:
        raw_demographics = self.load_demographics_raw(user_id=user_id)
        if len(raw_demographics) > 0:
            return raw_demographics["calendarDate"].min()
        else:
            return None

    def _convert_timezone_offset(self, x) -> int:
        x = x[-5:]
        hours = x[1:3]
        minutes = x[3:]
        if hours[0] == 0:
            hours = int(hours[1])
        else:
            hours = int(hours)

        if minutes[0] == 0:
            minutes = int(minutes[1])
        else:
            minutes = int(minutes)

        delta = pd.Timedelta(hours=hours, minutes=minutes)
        if x[0] == "-":
            return -delta.total_seconds() * 1000
        else:
            return delta.total_seconds() * 1000


# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
# __all__ = ["TesseraeCSVLoader"]
