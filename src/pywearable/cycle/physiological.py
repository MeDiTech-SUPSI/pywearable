"""
Extraction of physiological cycle (bedtime to bedtime) statistics.
"""

from typing import Union

import pandas as pd

from . import utils

CYCLE_PHYSIOLOGICAL_STATISTICS_DICT = {
    "cycle_physiological_duration": utils.compute_cycle_duration,
    "cycle_physiological_wearing_time": utils.compute_cycle_wearing_time,
    "cycle_physiological_start": utils.compute_cycle_start,
    "cycle_physiological_end": utils.compute_cycle_end,
    "cycle_physiological_heart_rate_count": utils.compute_cycle_heart_rate_count,
    "cycle_physiological_heart_rate_max": utils.compute_cycle_heart_rate_max,
    "cycle_physiological_heart_rate_mean": utils.compute_cycle_heart_rate_mean,
    "cycle_physiological_heart_rate_median": utils.compute_cycle_heart_rate_median,
    "cycle_physiological_heart_rate_min": utils.compute_cycle_heart_rate_min,
    "cycle_physiological_heart_rate_rest": utils.compute_cycle_heart_rate_rest,
    "cycle_physiological_heart_rate_std": utils.compute_cycle_heart_rate_std,
    "cycle_physiological_steps": utils.compute_cycle_steps,
    "cycle_physiological_stress_count": utils.compute_cycle_stress_count,
    "cycle_physiological_stress_max": utils.compute_cycle_stress_max,
    "cycle_physiological_stress_mean": utils.compute_cycle_stress_mean,
    "cycle_physiological_stress_median": utils.compute_cycle_stress_median,
    "cycle_physiological_stress_min": utils.compute_cycle_stress_min,
    "cycle_physiological_stress_std": utils.compute_cycle_stress_std,
    "cycle_physiological_high_stress_ms": utils.compute_cycle_high_stress_duration,
    "cycle_physiological_high_stress_percentage": utils.compute_cycle_high_stress_percentage,
    "cycle_physiological_low_stress_ms": utils.compute_cycle_low_stress_duration,
    "cycle_physiological_low_stress_percentage": utils.compute_cycle_low_stress_percentage,
    "cycle_physiological_medium_stress_ms": utils.compute_cycle_medium_stress_duration,
    "cycle_physiological_medium_stress_percentage": utils.compute_cycle_medium_stress_percentage,
    "cycle_physiological_rest_stress_ms": utils.compute_cycle_rest_stress_duration,
    "cycle_physiological_rest_stress_percentage": utils.compute_cycle_rest_stress_percentage,
    "cycle_physiological_unmeasurable_stress_ms": utils.compute_cycle_unmeasurable_stress_duration,
    "cycle_physiological_unmeasurable_stress_percentage": utils.compute_cycle_unmeasurable_stress_percentage,
    "cycle_physiological_sedentary_time_ms": utils.compute_cycle_epochs_sedentary_duration,
    "cycle_physiological_sedentary_time_percentage": utils.compute_cycle_epochs_sedentary_percentage,
    "cycle_physiological_active_time_ms": utils.compute_cycle_epochs_active_duration,
    "cycle_physiological_active_time_percentage": utils.compute_cycle_epochs_active_percentage,
    "cycle_physiological_highly_active_time_ms": utils.compute_cycle_epochs_highly_active_duration,
    "cycle_physiological_highly_active_time_percentage": utils.compute_cycle_epochs_highly_active_percentage,
}


def get_cycle_physiological_statistics(
    loader, user_id, start_date, end_date, hr_coverage
) -> Union[pd.Series, pd.DataFrame]:
    if statistics is None:
        statistics = list(CYCLE_PHYSIOLOGICAL_STATISTICS_DICT.keys())
    return utils.get_cycle_statistics(
        loader=loader,
        user_id=user_id,
        cycle_statistics=statistics,
        start_date=start_date,
        end_date=end_date,
        hr_coverage=hr_coverage,
    )
