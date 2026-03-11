"""
Extraction of cycle sleep (bedtime to wake-up) statistics.
"""

from typing import Union

import pandas as pd


from . import base
from ..loader import BaseLoader

CYCLE_SLEEP_STATISTICS_LIST = base.registry.get_feature_names_for_cycle("sleep")


@base.registry.register_cycle("sleep")
class SleepCycle(base.BaseCycle):
    def get_intervals(
        self,
        loader: BaseLoader,
        user_id: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ):
        """
        Return start as bedtime and end as wakeup
        """
        sleep_summaries = loader.load_sleep_summary(
            user_id=user_id,
            start_date=pd.Timestamp(start_date),
            end_date=pd.Timestamp(end_date),
            drop_duplicates=True,
        )
        if len(sleep_summaries) == 0:
            return pd.DataFrame(
                index=pd.date_range(
                    start=start_date, end=end_date, normalize=True, inclusive="left"
                )
            )
        sleep_summaries["end"] = sleep_summaries["isoDate"] + pd.to_timedelta(
            sleep_summaries["durationInMs"] + sleep_summaries["awakeDurationInMs"],
            unit="ms",
        )
        sleep_summaries = (
            sleep_summaries.set_index("calendarDate")
            .asfreq("1d")
            .loc[:, ["isoDate", "end"]]
            .rename(columns={"isoDate": "start"})
        )
        sleep_summaries = sleep_summaries.dropna(how="any")
        print(sleep_summaries)
        return sleep_summaries


def get_cycle_sleep_statistics(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    hr_coverage: float,
) -> Union[pd.Series, pd.DataFrame]:
    from . import get_cycle_statistics

    return get_cycle_statistics(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        hr_coverage=hr_coverage,
        cycle="sleep",
    )
