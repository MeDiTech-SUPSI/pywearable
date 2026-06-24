"""
Extraction of physiological cycle (bedtime to bedtime) statistics.
"""

from typing import Union

import pandas as pd

from . import base
from ..loader.base import BaseLoader

CYCLE_PHYSIOLOGICAL_STATISTICS_LIST = base.registry.get_feature_names_for_cycle(
    "physiological"
)


@base.registry.register_cycle("physiological")
class PhysiologicalCycle(base.BaseCycle):
    def get_intervals(
        self,
        loader: BaseLoader,
        user_id: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ):
        """
        Return start as bedtime and end as next bedtime
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
        sleep_summaries = (
            sleep_summaries.set_index("calendarDate")
            .asfreq("1d")
            .loc[:, ["isoDate"]]
            .rename(columns={"isoDate": "start"})
        )
        sleep_summaries["end"] = sleep_summaries["start"].shift(-1)
        sleep_summaries = sleep_summaries.dropna(how="any")
        return sleep_summaries


def get_cycle_physiological_statistics(
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
        cycle="physiological",
    )
