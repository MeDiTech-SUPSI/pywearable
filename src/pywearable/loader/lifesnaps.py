import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

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
        return pd.DataFrame(columns=["utc_time", "local_time", "breaths_per_minute"])

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
        return pd.DataFrame(columns=["utc_time", "local_time", "breaths_per_minute"])

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
        return pd.DataFrame(columns=["utc_time", "local_time", "spo2"])

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
        return pd.DataFrame(columns=["utc_time", "local_time", "spo2"])

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
        time_col: str = "utc_time",
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

        Returns
        -------
        :class:`pandas.DataFrame`
            Empty dataframe.
        """
        self._check_user_id(user_id)
        file_path = self.root_folder / user_id / "heart_rate.parquet"
        if file_path.exists():
            heart_rate = pd.read_parquet(file_path).rename(
                columns={"dateTime": "utc_time", "bpm": "heart_rate"}
            )
            heart_rate["utc_time"] = pd.to_datetime(heart_rate["utc_time"])
            heart_rate["local_time"] = heart_rate["utc_time"]
            heart_rate["timezone"] = pd.Timedelta(0)

            if not (start_date is None):
                heart_rate = heart_rate[
                    heart_rate[time_col] >= pd.Timestamp(start_date)
                ]
            if not (end_date is None):
                heart_rate = heart_rate[heart_rate[time_col] < pd.Timestamp(end_date)]
            return heart_rate
        else:
            return pd.DataFrame(
                columns=["local_time", "utc_time", "timezone", "heart_rate"],
            )

    def load_respiration_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        user_id : _type_
            _description_
        start_date : :class:`pandas.Timestamp` or ``None``, optional
            Start date for respiration data retrieval, by default ``None``
        end_date : :class:`pandas.Timestamp` or ``None``, optional
            End date for respiration data retrieval, by default ``None``

        Returns
        -------
        :class:`pandas.DataFrame`
            _description_
        """
        file_path = self.root_folder / user_id / "respiratory_rate_summary.parquet"
        if file_path.exists():
            respiration = pd.read_parquet(file_path)
            respiration["timestamp"] = pd.to_datetime(respiration["timestamp"])

            # We need to merge it with sleep_summary to get the exact date
            sleep_summary = self.load_sleep_summary(user_id=user_id)
            if len(sleep_summary) > 0:
                respiration = pd.merge(
                    left=respiration,
                    right=sleep_summary.loc[:, ["calendarDate", "endTime"]],
                    left_on="timestamp",
                    right_on="endTime",
                ).drop(["endTime"], axis=1)

            respiration = respiration.rename(columns={"timestamp": "isoDate"})
            if not (start_date is None):
                respiration = respiration[
                    respiration["isoDate"] >= pd.Timestamp(start_date)
                ]
            if not (end_date is None):
                respiration = respiration[
                    respiration["isoDate"] < pd.Timestamp(end_date)
                ]
            return respiration
        return pd.DataFrame()

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

    def load_sleep_stage(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        file_path = self.root_folder / user_id / "sleep_stages.parquet"
        if file_path.exists():
            sleep_stages = pd.read_parquet(file_path).rename(
                columns={
                    "logId": "sleep_summary_id",
                    "dateTime": "utc_time",
                    "seconds": "duration_in_ms",
                    "level": "type",
                }
            )
            sleep_stages["duration_in_ms"] = sleep_stages["duration_in_ms"] * 1000
            sleep_stages["utc_time"] = pd.to_datetime(sleep_stages["utc_time"])
            sleep_stages["local_time"] = sleep_stages["utc_time"]
            sleep_stages["timezone_offset"] = pd.Timedelta(0)
            if drop_duplicates:
                sleep_stages = sleep_stages.drop_duplicates(
                    subset=["sleep_summary_id", "utc_time", "type"]
                )

            if not (start_date is None):
                sleep_stages = sleep_stages[
                    sleep_stages["local_time"] >= pd.Timestamp(start_date)
                ]
            if not (end_date is None):
                sleep_stages = sleep_stages[
                    sleep_stages["local_time"] < pd.Timestamp(end_date)
                ]
            return sleep_stages
        else:
            return pd.DataFrame(
                columns=["sleep_summary_id", "local_time", "type", "duration_in_ms"]
            )

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
            hrv_details = (
                pd.read_parquet(file_path)
                .rename(
                    columns={
                        "timestamp": "utc_time",
                        "rmssd": "RMSSD",
                        "low_frequency": "LF",
                        "high_frequency": "HF",
                    }
                )
                .sort_values("utc_time", ascending=True, ignore_index=True)
            )

            hrv_details["utc_time"] = pd.to_datetime(hrv_details["utc_time"])
            hrv_details["local_time"] = hrv_details["utc_time"]
            hrv_details["timezone_offset"] = pd.Timedelta(0)
            if not (start_date is None):
                hrv_details = hrv_details[
                    hrv_details["local_time"] >= pd.Timestamp(start_date)
                ]
            if not (end_date is None):
                hrv_details = hrv_details[
                    hrv_details["local_time"] < pd.Timestamp(end_date)
                ]
            return hrv_details
        else:
            return pd.DataFrame(
                columns=[
                    "local_time",
                    "utc_time",
                    "timezone_offset",
                    "RMSSD",
                    "coverage",
                    "LF",
                    "HF",
                ]
            )

    def load_sleep_summary(
        self,
        user_id: str,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
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

            sleep["utc_time"] = pd.to_datetime(sleep["utc_time"])
            sleep["local_time"] = sleep["utc_time"]
            sleep["timezone_offset"] = pd.Timedelta(0)

            if "deep_minutes" in sleep.columns:
                sleep["duration_in_ms"] = (
                    (
                        sleep[["deep_minutes", "light_minutes", "rem_minutes"]]
                        .fillna(0)
                        .sum(axis=1)
                    )
                    * 60
                    * 1000
                )
            if "wake_minutes" in sleep.columns:
                sleep["awake_duration_in_ms"] = (
                    sleep["wake_minutes"].fillna(0) * 60 * 1000
                )

            if not (start_date is None):
                sleep = sleep[
                    sleep["calendar_date"] >= pd.Timestamp(start_date).normalize()
                ]
            if not (end_date is None):
                sleep = sleep[
                    sleep["calendar_date"] < pd.Timestamp(end_date).normalize()
                ]
            return sleep
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
        file_path = self.root_folder / user_id / "steps.parquet"
        if file_path.exists():
            steps = pd.read_parquet(file_path).rename(
                columns={"dateTime": "utc_time", "value": "steps"}
            )
            steps["steps"] = steps["steps"].astype(float)
            steps["utc_time"] = pd.to_datetime(steps["utc_time"])
            steps["local_time"] = steps["utc_time"]
            steps["timezone_offset"] = pd.Timedelta(0)
            if not (start_date is None):
                steps = steps[steps["local_time"] >= pd.Timestamp(start_date)]
            if not (end_date is None):
                steps = steps[steps["local_time"] < pd.Timestamp(end_date)]
            return steps
        else:
            return pd.DataFrame(
                columns=["local_time", "utc_time", "timezone_offset", "steps"]
            )

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

    def load_survey(self, survey_name: str) -> pd.DataFrame:
        return pd.read_parquet(self.root_folder / f"{survey_name}.parquet")

    def load_survey_scored(self, survey_name: str) -> pd.DataFrame:
        scored_survey = pd.read_parquet(
            self.root_folder / f"{survey_name}_scored.parquet"
        ).drop("type", axis=1)
        if survey_name == "personality":
            raw_survey = pd.read_parquet(self.root_folder / f"bfpt.parquet")
        elif survey_name == "ttm":
            raw_survey = pd.read_parquet(self.root_folder / f"ttmspbf.parquet")
        else:
            raw_survey = pd.read_parquet(self.root_folder / f"{survey_name}.parquet")

        scored_survey["submitdate"] = pd.to_datetime(scored_survey["submitdate"])
        raw_survey["submitdate"] = pd.to_datetime(
            raw_survey["submitdate"], format="%d/%m/%Y %H:%M"
        )

        scored_survey = scored_survey.sort_values(["user_id", "submitdate"])
        raw_survey = raw_survey.sort_values(["user_id", "submitdate"])
        raw_survey["calendarDate"] = raw_survey["submitdate"].dt.normalize()
        # We need to merge to have the exact filling time
        scored_survey_list: list[pd.DataFrame] = []
        for user in scored_survey["user_id"].unique():
            scored_survey_list.append(
                pd.merge_asof(
                    left=scored_survey[scored_survey["user_id"] == user].drop(
                        "user_id", axis=1
                    ),
                    right=raw_survey.loc[
                        raw_survey["user_id"] == user,
                        ["submitdate", "datestamp"],
                    ],
                    on="submitdate",
                    allow_exact_matches=True,
                    direction="forward",
                    tolerance=pd.Timedelta(hours=24),
                ).assign(user_id=user)
            )

        scored_survey = pd.concat(scored_survey_list, axis=0).rename(
            columns={"datestamp": "utc_time"}
        )
        scored_survey["utc_time"] = pd.to_datetime(
            scored_survey["utc_time"], format="%d/%m/%Y %H:%M"
        )
        scored_survey["local_time"] = scored_survey["utc_time"]

        return scored_survey

    def load_profile(self, user_id: str) -> pd.DataFrame:
        file_path = self.root_folder / user_id / "profile.parquet"
        if file_path.exists():
            return pd.read_parquet(file_path)
        else:
            return pd.DataFrame()

    def load_ema(self, user_id: Union[str, None] = None) -> pd.DataFrame:
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
        if not (user_id is None):
            sema = sema[sema["user_id"] == user_id]
        return sema
