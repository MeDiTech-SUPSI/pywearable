"""
Extraction of physiological cycle (bedtime to bedtime) statistics.
"""

from typing import Union

import pandas as pd

from . import base
from ..loader.base import BaseLoader

CYCLE_DAILY_STATISTICS_LIST = base.registry.get_feature_names_for_cycle("daily")


@base.registry.register_cycle("daily")
class DailyCycle(base.BaseCycle):
    def get_intervals(
        self,
        loader: BaseLoader,
        user_id: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ):
        """
        Return start as midnight and end as next midnight
        """
        date_range = pd.date_range(
            start=start_date, end=end_date, freq="d", normalize=True, inclusive="left"
        )
        intervals = pd.DataFrame(
            {"start": date_range, "end": date_range.shift(-1)},
            index=pd.date_range(
                start=start_date,
                end=end_date,
                freq="d",
                normalize=True,
                inclusive="left",
            ),
        )
        return intervals


def get_cycle_daily_statistics(
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
        cycle="daily",
    )
