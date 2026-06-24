"""
This module contains all the functions related to the analysis
of Labfront adherence.
"""

import datetime
import math
import os
from typing import Union

import pandas as pd

from . import constants, utils
from .loader.base import BaseLoader


def get_daily_wearing_time(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    loading_days: int = 7,
    ceil_to_24: bool = False,
) -> pd.Series:
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    intervals = math.ceil(
        (end_date - start_date).total_seconds() / (loading_days * 24 * 60 * 60)
    )
    hr_hours_count = pd.Series(name="daily-hr")
    total_sec_list = []
    for i in range(intervals):
        if i == 0:
            # For the first interval, we consider from start_date
            # up to (days-1) days and set timestamp just before midnight,
            tmp_start_date = start_date
            if intervals == 1:
                # Set to real end_date if we have only one interval of days
                tmp_end_date = end_date
            else:
                tmp_end_date = start_date + pd.Timedelta(days=loading_days - 1)
                tmp_end_date = tmp_end_date.replace(
                    hour=0, minute=0, second=0
                ) - pd.Timedelta(milliseconds=1)
        else:
            tmp_start_date = tmp_end_date + pd.Timedelta(milliseconds=1)
            if i < (intervals - 1):
                tmp_end_date = tmp_start_date + pd.Timedelta(
                    days=loading_days, milliseconds=-1
                )
            else:
                tmp_end_date = end_date
        # Get heart rate
        hr = loader.load_heart_rate(
            user_id=user_id, start_date=tmp_start_date, end_date=tmp_end_date
        )
        if len(hr) > 0:
            # We have some heart-rate data

            # Resample to have it on a consistent time resolution
            hr = (
                hr.resample(rule="60s", on=constants.ISODATE_COL)[constants.HR_COLUMN]
                .mean()
                .to_frame()
            )
            hr[constants.CALENDAR_DATE_COL] = hr.index.date

            # Compute total daily hours
            total_sec = (
                hr.groupby([constants.CALENDAR_DATE_COL])[constants.HR_COLUMN].count()
            ).rename(
                "daily-hr"
            ) / 60  # Convert to decimal hours
            if ceil_to_24:
                total_sec = total_sec.clip(upper=24)
            total_sec_list.append(total_sec)
    if len(total_sec_list) > 0:
        hr_hours_count = pd.concat(
            total_sec_list,
            axis=0,
        )
    hr_hours_count = hr_hours_count.reindex(
        pd.date_range(start=start_date, end=end_date, inclusive="left", normalize=True)
    )
    return hr_hours_count
