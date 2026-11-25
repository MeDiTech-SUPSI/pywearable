from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd

from .base import BaseLoader


class MCPhasesLoader(BaseLoader):
    def __init__(self, root_folder: Path):
        """Loader for the mcPHASES dataset.

        The mCPHASES dataset contains data from
        Fitbit,

        Parameters
        ----------
        root_folder : str or os.PathLike or pathlib.Path
            Path to the roo folder containing the unzipped
            files from the mcPHASES dataset.
        """
        self.root_folder = root_folder
        self.subject_info = None
        self.start_dates = {}

    def load_heart_rate(
        self,
        user_id: str,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        chunksize=100000,
        time_col: str = "utc_time",
        source: Literal["raw", "processed"] = "processed",
    ) -> pd.DataFrame:
        if source == "raw":
            iter_csv = pd.read_csv(
                self.root_folder / "heart_rate.csv", iterator=True, chunksize=chunksize
            )
            hr = pd.concat([chunk[chunk["id"] == user_id] for chunk in iter_csv])
        elif source == "processed":
            path_to_file = self.root_folder / "_processed" / f"{user_id}" / "hr.csv"
            if not path_to_file.exists():
                return pd.DataFrame()
            hr = pd.read_csv(self.root_folder / "_processed" / f"{user_id}" / "hr.csv")

        hr = hr.sort_values(["day_in_study", "timestamp"])

        hr = hr.rename(columns={"bpm": "heart_rate", "timestamp": "utc_time"})
        hr["day_in_study"] = hr["day_in_study"].astype(float)
        hr["utc_time"] = pd.to_timedelta(hr["utc_time"])
        hr = hr.sort_values(["day_in_study", "utc_time"], ascending=True)
        user_start_date = self._get_user_start_date(user_id=user_id, heart_rate=hr)

        hr["utc_time"] = (
            user_start_date
            + hr["utc_time"]
            + pd.to_timedelta(hr["day_in_study"] - 1, unit="d")
        )
        hr["local_time"] = hr["utc_time"]
        hr["timezone_offset"] = pd.Timedelta(0)
        if not (start_date is None):
            hr = hr[hr[time_col] >= pd.Timestamp(start_date)]
        if not (end_date is None):
            hr = hr[hr[time_col] < pd.Timestamp(end_date)]
        return hr

    def load_subject_info(self) -> pd.DataFrame:
        if self.subject_info is None:
            self.subject_info = pd.read_csv(self.root_folder / "subject-info.csv")
        return self.subject_info

    def load_sleep_stages(
        self, user_id, start_date=None, end_date=None
    ) -> pd.DataFrame:
        sleep = self.load_sleep_summaries(
            user_id=user_id, start_date=start_date, end_date=end_date
        )
        # We need to normalize sleep levels

    def load_sleep_summaries(
        self,
        user_id,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
        source: Literal["raw", "processed"] = "raw",
        chunksize=100000,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        if source == "raw":
            iter_csv = pd.read_csv(
                self.root_folder / "sleep.csv", iterator=True, chunksize=chunksize
            )
            sleep = pd.concat([chunk[chunk["id"] == user_id] for chunk in iter_csv])
        elif source == "processed":
            path_to_file = self.root_folder / "_processed" / f"{user_id}" / "sleep.csv"
            if not path_to_file.exists():
                return pd.DataFrame()
            sleep = pd.read_csv(
                self.root_folder / "_processed" / f"{user_id}" / "sleep.csv"
            )
        user_start_date = self._get_user_start_date(user_id=user_id)

        sleep = sleep.rename(
            columns={
                "sleep_start_timestamp": "utc_time",
                "sleep_end_timestamp": "utc_end_time",
            }
        )
        sleep["sleep_start_day_in_study"] = sleep["sleep_start_day_in_study"].astype(
            float
        )
        sleep["utc_time"] = pd.to_timedelta(sleep["utc_time"])
        sleep["utc_end_time"] = pd.to_timedelta(sleep["utc_end_time"])

        sleep["utc_time"] = (
            user_start_date
            + sleep["utc_time"]
            + pd.to_timedelta(sleep["sleep_start_day_in_study"] - 1, unit="d")
        )

        sleep["utc_end_time"] = (
            user_start_date
            + sleep["utc_end_time"]
            + pd.to_timedelta(sleep["sleep_end_day_in_study"] - 1, unit="d")
        )
        sleep["local_time"] = sleep["utc_time"]
        sleep["local_end_time"] = sleep["utc_end_time"]
        sleep["timezone_offset"] = pd.Timedelta(0)
        sleep["calendar_date"] = sleep["utc_end_time"].dt.normalize()
        """ 
        'id', 'study_interval', 'is_weekend', 'sleep_start_day_in_study',
       'sleep_start_timestamp', 'sleep_end_day_in_study',
       'sleep_end_timestamp', 'duration', 'minutestofallasleep',
       'minutesasleep', 'minutesawake', 'minutesafterwakeup', 'timeinbed',
       'efficiency', 'type', 'infocode', 'levels', 'mainsleep'

        """

        return sleep

    def get_user_list(self) -> list:
        return self.load_subject_info()["id"].unique()

    def get_user_ids(self) -> list:
        return self.get_user_list()

    def _get_user_start_date(
        self, user_id: str, heart_rate: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.Timestamp:
        """Pick the start date (anchor + 0..6 days) whose weekends best match is_weekend."""
        if not (user_id in self.start_dates.keys()):
            if heart_rate is None:
                heart_rate = self.load_heart_rate(user_id=user_id, **kwargs)
            # one is_weekend value per study day (majority vote in case of noise)
            per_day = heart_rate.groupby("day_in_study")["is_weekend"].agg(
                lambda s: s.mode().iat[0]
            )
            days = per_day.index.to_numpy() - 1
            observed = per_day.to_numpy().astype(bool)

            anchor = pd.Timestamp(2022, 1, 1)
            scores = []
            for k in range(7):
                start = anchor + pd.Timedelta(days=k)
                dow = (start.dayofweek + days) % 7  # Mon=0 ... Sun=6
                scores.append(((dow >= 5) == observed).mean())

            k_best = int(np.argmax(scores))
            if scores[k_best] < 1:
                print(
                    f"warning: {user_id} best weekend match only {scores[k_best]:.0%}"
                )
            best_guess = anchor + pd.Timedelta(days=k_best)
            self.start_dates[user_id] = best_guess
        return self.start_dates[user_id]
        return self.start_dates[user_id]
