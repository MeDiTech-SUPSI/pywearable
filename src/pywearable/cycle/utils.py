from typing import Literal, Tuple

import numpy as np
import pandas as pd

from .. import constants

STRESS_DURATION_UNMEASURABLE_COL = "UNMEASURABLE"
STRESS_DURATION_REST_COL = "REST"
STRESS_DURATION_LOW_COL = "LOW"
STRESS_DURATION_MEDIUM_COL = "MEDIUM"
STRESS_DURATION_HIGH_COL = "HIGH"


import datetime
import os
from typing import Union

import numpy as np
import pandas as pd

from .. import constants, loader

CYCLE_NEXT_BEDTIME_COL = "next_bedtime"
CYCLE_NEXT_WAKEUP_TIME_COL = "next_wakeup"
CYCLE_DIARY_END_COL = "cycle_diary_end"

SLEEP_SUMMARY_WAKEUP_COL = "wakeup"


def get_time_columns(
    cycle: Literal["awake", "sleep", "physiological"],
) -> Tuple[str, str]:
    if cycle == "physiological":
        lower_time_col = constants.ISODATE_COL
        upper_time_col = CYCLE_NEXT_BEDTIME_COL
    elif cycle == "sleep":
        lower_time_col = constants.ISODATE_COL
        upper_time_col = SLEEP_SUMMARY_WAKEUP_COL
    elif cycle == "awake":
        lower_time_col = SLEEP_SUMMARY_WAKEUP_COL
        upper_time_col = CYCLE_NEXT_BEDTIME_COL
    else:
        raise ValueError(
            f"cycle {cycle} not among possible values of ['physiological','sleep','awake']"
        )
    return lower_time_col, upper_time_col


def compute_cycle_duration(
    sleep_summary: pd.DataFrame,
    cycle: Literal[
        "sleep", "awake", "diary", "physiological", "diary_awake", "wake_to_wake"
    ],
    **kwargs,
) -> pd.Series:
    stat_name = f"cycle_{cycle}_duration"
    if len(sleep_summary) == 0:
        return pd.Series(name=stat_name)
    lower_time_col, upper_time_col = get_time_columns(cycle=cycle)
    cycle_duration = pd.Series(
        index=sleep_summary[constants.CALENDAR_DATE_COL],
        name=stat_name,
    )
    for index, row in sleep_summary.iterrows():
        calendar_date = row[constants.CALENDAR_DATE_COL]
        # Get hr hours
        cycle_duration.loc[calendar_date] = (
            row[upper_time_col] - row[lower_time_col]
        ).total_seconds() / (60 * 60)
    return cycle_duration


def compute_cycle_start(
    sleep_summary: pd.DataFrame,
    cycle: Literal[
        "sleep", "awake", "diary", "physiological", "diary_awake", "wake_to_wake"
    ],
    **kwargs,
) -> pd.Series:
    stat_name = f"cycle_{cycle}_start"
    if len(sleep_summary) == 0:
        return pd.Series(name=stat_name)
    lower_time_col, __ = get_time_columns(cycle=cycle)
    cycle_start = sleep_summary.set_index(constants.CALENDAR_DATE_COL)[
        lower_time_col
    ].rename(stat_name)
    return cycle_start


def compute_cycle_end(
    sleep_summary: pd.DataFrame,
    cycle: Literal[
        "sleep", "awake", "diary", "physiological", "diary_awake", "wake_to_wake"
    ],
    **kwargs,
) -> pd.Series:
    stat_name = f"cycle_{cycle}_end"
    if len(sleep_summary) == 0:
        return pd.Series(name=stat_name)
    __, upper_time_col = get_time_columns(cycle=cycle)
    cycle_end = sleep_summary.set_index(constants.CALENDAR_DATE_COL)[
        upper_time_col
    ].rename(stat_name)
    return cycle_end


#####################################################################
#                      Heart Rate Statistics                        #
#####################################################################


def compute_cycle_heart_rate_mean(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    cycle: Literal[
        "physiological", "sleep", "awake", "wake_to_wake", "diary", "diary_awake"
    ],
    **kwargs,
) -> pd.Series:
    return compute_cycle_heart_rate_statistic(
        heart_rate=heart_rate, sleep_summary=sleep_summary, stat="mean", cycle=cycle
    )


def compute_cycle_heart_rate_std(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    cycle: Literal["physiological", "sleep", "awake", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_heart_rate_statistic(
        heart_rate=heart_rate, sleep_summary=sleep_summary, stat="std", cycle=cycle
    )


def compute_cycle_heart_rate_min(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    cycle: Literal["physiological", "sleep", "awake", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_heart_rate_statistic(
        heart_rate=heart_rate, sleep_summary=sleep_summary, stat="min", cycle=cycle
    )


def compute_cycle_heart_rate_max(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    cycle: Literal["physiological", "sleep", "awake", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_heart_rate_statistic(
        heart_rate=heart_rate, sleep_summary=sleep_summary, stat="max", cycle=cycle
    )


def compute_cycle_heart_rate_count(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    cycle: Literal["physiological", "sleep", "awake", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_heart_rate_statistic(
        heart_rate=heart_rate, sleep_summary=sleep_summary, stat="count", cycle=cycle
    )


def compute_cycle_heart_rate_median(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    cycle: Literal["physiological", "sleep", "awake", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_heart_rate_statistic(
        heart_rate=heart_rate, sleep_summary=sleep_summary, stat="median", cycle=cycle
    )


def compute_cycle_heart_rate_rest(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    cycle: Literal["physiological", "sleep", "awake", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_heart_rate_statistic(
        heart_rate=heart_rate, sleep_summary=sleep_summary, stat="rest", cycle=cycle
    )


def compute_cycle_heart_rate_statistic(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    cycle: Literal["physiological", "sleep", "awake", "wake_to_wake"],
    stat: Literal["count", "mean", "median", "max", "min", "std", "rest"],
    **kwargs,
) -> pd.Series:
    stat_name = f"cycle_{cycle}_heart_rate_{stat}"
    if len(sleep_summary) == 0:
        return pd.Series(name=stat_name)
    if len(heart_rate) == 0:
        return pd.Series(
            name=stat_name,
            index=sleep_summary[constants.CALENDAR_DATE_COL],
        )
    # Sort heart rate values - Do not drop duplicates as we may have multiple
    # timezones
    heart_rate = heart_rate.sort_values(by=[constants.ISODATE_COL], ignore_index=True)

    hr_stat = pd.Series(
        index=sleep_summary[constants.CALENDAR_DATE_COL], name=stat_name
    )
    lower_time_col, upper_time_col = get_time_columns(cycle=cycle)
    for index, row in sleep_summary.iterrows():
        calendar_date = row[constants.CALENDAR_DATE_COL]
        if pd.isnull(row[lower_time_col]) or pd.isnull(row[upper_time_col]):
            hr_stat.loc[calendar_date] = np.nan
        else:
            filtered_cycle_heart_rate = heart_rate.loc[
                (heart_rate[constants.ISODATE_COL] >= row[lower_time_col])
                & (heart_rate[constants.ISODATE_COL] < row[upper_time_col]),
                :,
            ]
            if stat == "rest":
                # Set iso date as index
                filtered_cycle_heart_rate = filtered_cycle_heart_rate.set_index(
                    constants.ISODATE_COL
                )
                # Perform 30 minute average
                mean_hr = filtered_cycle_heart_rate.rolling(window="30Min")[
                    constants.HR_COLUMN
                ].mean()
                # Return minimum 30 minute average
                hr_stat.loc[calendar_date] = mean_hr.min()
            else:
                hr_stat.loc[calendar_date] = filtered_cycle_heart_rate[
                    constants.HR_COLUMN
                ].agg(stat)
    return hr_stat


################################################################################
#                                 Steps                                        #
################################################################################


def compute_cycle_steps(
    sleep_summary: pd.DataFrame,
    epochs: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
):
    stat_name = f"cycle_{cycle}_steps"
    if len(sleep_summary) == 0:
        return pd.Series(name=stat_name)
    if len(epochs) == 0:
        return pd.Series(
            name=stat_name,
            index=sleep_summary[constants.CALENDAR_DATE_COL],
        )
    steps = pd.Series(
        index=sleep_summary[constants.CALENDAR_DATE_COL],
        name=stat_name,
    )
    lower_time_col, upper_time_col = get_time_columns(cycle=cycle)
    for index, row in sleep_summary.iterrows():
        calendar_date = row[constants.CALENDAR_DATE_COL]
        if pd.isnull(lower_time_col) or pd.isnull(row[upper_time_col]):
            steps.loc[calendar_date] = np.nan
        filtered_cycle_epochs = epochs.loc[
            (epochs[constants.ISODATE_COL] >= row[lower_time_col])
            & (epochs[constants.ISODATE_COL] < row[upper_time_col]),
            :,
        ]
        # Get steps sum
        steps.loc[calendar_date] = filtered_cycle_epochs[
            constants.ACTIVITY_STEPS_COL
        ].sum()
    return steps


def compute_cycle_high_stress_duration(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_stress_range_duration(
        sleep_summary=sleep_summary,
        stress=stress,
        stress_range="high",
        cycle=cycle,
        return_as_percentage=False,
    )


def compute_cycle_high_stress_percentage(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_stress_range_duration(
        sleep_summary=sleep_summary,
        stress=stress,
        stress_range="high",
        cycle=cycle,
        return_as_percentage=True,
    )


def compute_cycle_low_stress_duration(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_stress_range_duration(
        sleep_summary=sleep_summary,
        stress=stress,
        stress_range="low",
        cycle=cycle,
        return_as_percentage=False,
    )


def compute_cycle_low_stress_percentage(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_stress_range_duration(
        sleep_summary=sleep_summary,
        stress=stress,
        stress_range="low",
        cycle=cycle,
        return_as_percentage=True,
    )


def compute_cycle_medium_stress_duration(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_stress_range_duration(
        sleep_summary=sleep_summary,
        stress=stress,
        stress_range="medium",
        cycle=cycle,
        return_as_percentage=False,
    )


def compute_cycle_medium_stress_percentage(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_stress_range_duration(
        sleep_summary=sleep_summary,
        stress=stress,
        stress_range="medium",
        cycle=cycle,
        return_as_percentage=True,
    )


def compute_cycle_rest_stress_duration(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_stress_range_duration(
        sleep_summary=sleep_summary,
        stress=stress,
        stress_range="rest",
        cycle=cycle,
        return_as_percentage=False,
    )


def compute_cycle_rest_stress_percentage(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_stress_range_duration(
        sleep_summary=sleep_summary,
        stress=stress,
        stress_range="rest",
        cycle=cycle,
        return_as_percentage=True,
    )


def compute_cycle_unmeasurable_stress_duration(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_stress_range_duration(
        sleep_summary=sleep_summary,
        stress=stress,
        stress_range="unmeasurable",
        cycle=cycle,
        return_as_percentage=False,
    )


def compute_cycle_unmeasurable_stress_percentage(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    return compute_cycle_stress_range_duration(
        sleep_summary=sleep_summary,
        stress=stress,
        stress_range="unmeasurable",
        cycle=cycle,
        return_as_percentage=True,
    )


def compute_cycle_stress_range_duration(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    stress_range: Literal["unmeasurable", "rest", "low", "medium", "high"],
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    return_as_percentage: bool = False,
) -> pd.Series:
    stat_name = f"cycle_{cycle}_{stress_range}_stress_{'percentage' if return_as_percentage else 'ms'}"

    if len(sleep_summary) == 0:
        return pd.Series(name=stat_name)
    min_sleep_datetime = sleep_summary[constants.ISODATE_COL].min()
    max_sleep_datetime = sleep_summary[constants.ISODATE_COL].max()
    if len(stress) > 0:
        min_stress_datetime = stress[constants.ISODATE_COL].min()
        max_stress_datetime = stress[constants.ISODATE_COL].max()
    else:
        min_stress_datetime = min_sleep_datetime
        max_stress_datetime = max_sleep_datetime
    if (
        (len(stress) == 0)
        or (min_stress_datetime > max_sleep_datetime)
        or (max_stress_datetime < min_sleep_datetime)
    ):
        return_df = pd.Series(
            name=stat_name,
            index=sleep_summary[constants.CALENDAR_DATE_COL],
        )
        return return_df
    lower_time_col, upper_time_col = get_time_columns(cycle=cycle)
    stress["stressLevelInStressRange"] = pd.cut(
        stress[constants.STRESS_STRESS_LEVEL_COL],
        bins=[-10, -1, 25, 50, 75, 100],
        labels=[
            STRESS_DURATION_UNMEASURABLE_COL,
            STRESS_DURATION_REST_COL,
            STRESS_DURATION_LOW_COL,
            STRESS_DURATION_MEDIUM_COL,
            STRESS_DURATION_HIGH_COL,
        ],
        include_lowest=True,
    )
    stress_ms_df_list = []

    for index, row in sleep_summary.iterrows():
        calendar_date = row[constants.CALENDAR_DATE_COL]
        # Awake stress
        filtered_stress: pd.Series = stress.loc[
            (stress[constants.ISODATE_COL] >= row[lower_time_col])
            & (stress[constants.ISODATE_COL] < row[upper_time_col])
            # & (stress[constants.STRESS_STRESS_LEVEL_COL] >= 0)
            ,
            "stressLevelInStressRange",
        ]
        if len(filtered_stress) > 0:
            # Convert to milliseconds
            stress_ms = (filtered_stress.value_counts() * 3 * 60 * 1000).to_frame()

            stress_ms[constants.CALENDAR_DATE_COL] = calendar_date
            stress_ms_df_list.append(stress_ms)
    if len(stress_ms_df_list) == 0:
        return_df = pd.Series(
            name=stat_name,
            index=sleep_summary[constants.CALENDAR_DATE_COL],
        )
        return return_df
    # Concatenate single dataframes
    stress_ms_df = pd.concat(stress_ms_df_list)
    stress_ms_df = pd.pivot(
        data=stress_ms_df.reset_index(names="stressLevelInStressRange"),
        index=constants.CALENDAR_DATE_COL,
        columns="stressLevelInStressRange",
        values="count",
    )
    stress_ms_df.columns.name = None
    if return_as_percentage:
        return (stress_ms_df[stress_range.upper()] / stress_ms_df.sum(axis=1)).rename(
            stat_name
        )
    return stress_ms_df[stress_range.upper()].rename(stat_name)


#######################################################################################
#                                       Stress                                        #
#######################################################################################


def compute_cycle_stress_count(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
):
    return compute_cycle_stress_statistic(
        sleep_summary=sleep_summary, stress=stress, cycle=cycle, stat="count"
    )


def compute_cycle_stress_mean(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
):
    return compute_cycle_stress_statistic(
        sleep_summary=sleep_summary, stress=stress, cycle=cycle, stat="mean"
    )


def compute_cycle_stress_median(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
):
    return compute_cycle_stress_statistic(
        sleep_summary=sleep_summary, stress=stress, cycle=cycle, stat="median"
    )


def compute_cycle_stress_max(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
):
    return compute_cycle_stress_statistic(
        sleep_summary=sleep_summary, stress=stress, cycle=cycle, stat="max"
    )


def compute_cycle_stress_min(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
):
    return compute_cycle_stress_statistic(
        sleep_summary=sleep_summary, stress=stress, cycle=cycle, stat="min"
    )


def compute_cycle_stress_std(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
):
    return compute_cycle_stress_statistic(
        sleep_summary=sleep_summary, stress=stress, cycle=cycle, stat="std"
    )


def compute_cycle_stress_statistic(
    sleep_summary: pd.DataFrame,
    stress: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    stat: Literal["count", "mean", "median", "max", "min", "std"],
    **kwargs,
) -> pd.Series:
    stat_name = f"cycle_{cycle}_stress_{stat}"
    if len(sleep_summary) == 0:
        return pd.Series(name=stat_name)
    if len(stress) == 0:
        return pd.Series(
            name=stat_name,
            index=sleep_summary[constants.CALENDAR_DATE_COL],
        )
    lower_time_col, upper_time_col = get_time_columns(cycle=cycle)
    stress_stat_values = pd.Series(
        index=sleep_summary[constants.CALENDAR_DATE_COL],
        name=stat_name,
    )
    for _, row in sleep_summary.iterrows():
        calendar_date = row[constants.CALENDAR_DATE_COL]
        if pd.isnull(lower_time_col) or pd.isnull(row[upper_time_col]):
            stress_stat_values.loc[calendar_date] = np.nan
        else:
            stress_stat_values.loc[calendar_date] = stress.loc[
                (stress[constants.ISODATE_COL] >= row[lower_time_col])
                & (stress[constants.ISODATE_COL] <= row[upper_time_col])
                & (stress[constants.STRESS_STRESS_LEVEL_COL] >= 0),
                constants.STRESS_STRESS_LEVEL_COL,
            ].agg(stat)
    return stress_stat_values


################################################################################
#                           Activity levels                                    #
################################################################################


def compute_cycle_epochs_sedentary_duration(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    epochs: pd.DataFrame,
    cycle: Literal[
        "awake", "sleep", "physiological", "diary", "diary_awake", "wake_to_wake"
    ],
    hr_coverage: float = 0.8,
    **kwargs,
) -> pd.Series:
    return compute_cycle_epoch_activity_level_duration(
        sleep_summary=sleep_summary,
        heart_rate=heart_rate,
        epochs=epochs,
        activity_level="sedentary",
        cycle=cycle,
        return_as_percentage=False,
        hr_coverage=hr_coverage,
    )


def compute_cycle_epochs_sedentary_percentage(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    epochs: pd.DataFrame,
    cycle: Literal[
        "awake", "sleep", "physiological", "diary", "diary_awake", "wake_to_wake"
    ],
    hr_coverage: float = 0.8,
    **kwargs,
) -> pd.Series:
    return compute_cycle_epoch_activity_level_duration(
        sleep_summary=sleep_summary,
        heart_rate=heart_rate,
        epochs=epochs,
        activity_level="sedentary",
        cycle=cycle,
        return_as_percentage=True,
        hr_coverage=hr_coverage,
    )


def compute_cycle_epochs_active_duration(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    epochs: pd.DataFrame,
    cycle: Literal[
        "awake", "sleep", "physiological", "diary", "diary_awake", "wake_to_wake"
    ],
    hr_coverage: float = 0.8,
    **kwargs,
) -> pd.Series:
    return compute_cycle_epoch_activity_level_duration(
        sleep_summary=sleep_summary,
        heart_rate=heart_rate,
        epochs=epochs,
        activity_level="active",
        cycle=cycle,
        return_as_percentage=False,
        hr_coverage=hr_coverage,
    )


def compute_cycle_epochs_active_percentage(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    epochs: pd.DataFrame,
    cycle: Literal[
        "awake", "sleep", "physiological", "diary", "diary_awake", "wake_to_wake"
    ],
    hr_coverage: float = 0.8,
    **kwargs,
) -> pd.Series:
    return compute_cycle_epoch_activity_level_duration(
        sleep_summary=sleep_summary,
        heart_rate=heart_rate,
        epochs=epochs,
        activity_level="active",
        cycle=cycle,
        return_as_percentage=True,
        hr_coverage=hr_coverage,
    )


def compute_cycle_epochs_highly_active_duration(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    epochs: pd.DataFrame,
    cycle: Literal[
        "awake", "sleep", "physiological", "diary", "diary_awake", "wake_to_wake"
    ],
    hr_coverage: float = 0.8,
    **kwargs,
) -> pd.Series:
    return compute_cycle_epoch_activity_level_duration(
        sleep_summary=sleep_summary,
        heart_rate=heart_rate,
        epochs=epochs,
        activity_level="highly_active",
        cycle=cycle,
        return_as_percentage=False,
        hr_coverage=hr_coverage,
    )


def compute_cycle_epochs_highly_active_percentage(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    epochs: pd.DataFrame,
    cycle: Literal[
        "awake", "sleep", "physiological", "diary", "diary_awake", "wake_to_wake"
    ],
    hr_coverage: float = 0.8,
    **kwargs,
) -> pd.Series:
    return compute_cycle_epoch_activity_level_duration(
        sleep_summary=sleep_summary,
        heart_rate=heart_rate,
        epochs=epochs,
        activity_level="highly_active",
        cycle=cycle,
        return_as_percentage=True,
        hr_coverage=hr_coverage,
    )


def compute_cycle_epoch_activity_level_duration(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    epochs: pd.DataFrame,
    activity_level: Literal["sedentary", "active", "highly_active"],
    cycle: Literal[
        "awake", "sleep", "physiological", "diary", "diary_awake", "wake_to_wake"
    ],
    return_as_percentage: bool = False,
    hr_coverage: float = 0.8,
) -> pd.Series:
    stat_name = f"cycle_{cycle}_epochs_{activity_level}_{'percentage' if return_as_percentage else 'ms'}"
    if len(sleep_summary) == 0:
        return pd.Series(name=stat_name)
    if len(heart_rate) == 0 or len(epochs) == 0:
        return pd.Series(
            name=stat_name,
            index=sleep_summary[constants.CALENDAR_DATE_COL],
        )

    lower_time_col, upper_time_col = get_time_columns(cycle=cycle)

    epochs_with_hr_list = []
    for _, row in sleep_summary.iterrows():
        lower_time = row[lower_time_col].floor("15min")
        upper_time = row[upper_time_col].ceil("15min")
        if pd.isnull(lower_time) or pd.isnull(upper_time):
            continue
        # Filter heart rate for this cycle
        filtered_heart_rate = heart_rate.loc[
            (heart_rate[constants.ISODATE_COL] >= lower_time)
            & (heart_rate[constants.ISODATE_COL] < upper_time),
            :,
        ].copy()
        # Filter epochs for this cycle
        filtered_epochs = epochs.loc[
            (epochs[constants.ISODATE_COL] >= lower_time)
            & (epochs[constants.ISODATE_COL] < upper_time),
            :,
        ].copy()

        heart_rate_counts = (
            filtered_heart_rate.value_counts(subset=constants.ISODATE_COL)
            .sort_index()
            .resample("15Min")
            .count()
            .rename("hr_count")
        )
        filtered_epochs = filtered_epochs.groupby(
            [constants.ISODATE_COL, constants.EPOCH_INTENSITY_COL]
        )[constants.DURATION_IN_MS_COL].sum()
        epochs_with_hr = pd.merge(
            left=filtered_epochs,
            right=heart_rate_counts,
            left_index=True,
            right_index=True,
            how="left",
        )
        epochs_with_hr[constants.CALENDAR_DATE_COL] = row[constants.CALENDAR_DATE_COL]

        epochs_with_hr_list.append(epochs_with_hr)

    # Concatenate epochs
    if len(epochs_with_hr_list) == 0:
        return pd.Series(
            name=stat_name, index=sleep_summary[constants.CALENDAR_DATE_COL]
        )
    epochs_with_hr = pd.concat(epochs_with_hr_list, axis=0)
    epochs_with_hr = epochs_with_hr[
        epochs_with_hr["hr_count"] >= ((15 * 4) * hr_coverage)
    ]
    intensity_durations = epochs_with_hr.groupby(
        [constants.CALENDAR_DATE_COL, constants.EPOCH_INTENSITY_COL],
        observed=False,
    )[constants.DURATION_IN_MS_COL].sum()
    intensity_durations = intensity_durations.unstack()
    intensity_durations.columns.name = None

    for col in ["SEDENTARY", "ACTIVE", "HIGHLY_ACTIVE"]:
        if not (col in intensity_durations.columns):
            intensity_durations[col] = 0
        else:
            intensity_durations[col] = intensity_durations[col].fillna(0)

    if return_as_percentage:
        return (
            intensity_durations[activity_level.upper()]
            / intensity_durations.sum(axis=1)
        ).rename(stat_name)
    else:
        return intensity_durations[activity_level.upper()].rename(stat_name)


def compute_cycle_wearing_time(
    sleep_summary: pd.DataFrame,
    heart_rate: pd.DataFrame,
    cycle: Literal["awake", "sleep", "physiological", "wake_to_wake"],
    **kwargs,
) -> pd.Series:
    stat_name = f"cycle_{cycle}_wearing_time"
    if len(sleep_summary) == 0:
        return pd.Series(name=stat_name)
    if len(heart_rate) == 0:
        return pd.Series(
            name=stat_name,
            index=sleep_summary[constants.CALENDAR_DATE_COL],
        )
    cycle_wearing_time = pd.Series(
        index=sleep_summary[constants.CALENDAR_DATE_COL],
        name=stat_name,
    )
    lower_time_col, upper_time_col = get_time_columns(cycle=cycle)
    for index, row in sleep_summary.iterrows():
        calendar_date = row[constants.CALENDAR_DATE_COL]
        if pd.isnull(row[lower_time_col]) or pd.isnull(row[upper_time_col]):
            cycle_wearing_time.loc[calendar_date] = np.nan
        else:
            filtered_cycle_heart_rate = heart_rate.loc[
                (heart_rate[constants.ISODATE_COL] >= row[lower_time_col])
                & (heart_rate[constants.ISODATE_COL] < row[upper_time_col]),
                :,
            ]
            # Get hr hours
            cycle_wearing_time.loc[calendar_date] = (
                filtered_cycle_heart_rate[constants.HR_COLUMN].count() * 15 / (60 * 60)
            )
    return cycle_wearing_time


def get_date_range(start_date, end_date):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    date_range = pd.date_range(
        start=start_date,
        end=end_date,
        normalize=True,
        freq="D",
        inclusive="left",
    )

    return date_range


############################################
#               Main Function              #
############################################
def get_cycle_statistics(
    loader: loader.BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cycle_statistics: Union[str, list, None] = None,
    hr_coverage: float = 0.8,
) -> Union[pd.DataFrame, pd.Series]:
    """Extract cycle-based statistics.

    This function allows to extract cycle-based statistics for
    the given ``user_id``. Several cycles are defined:

        - physiological: standard cycle, from bedtime to bedtime
        - sleep: from bedtime to wakeup
        - awake: from bedtime to wakeup
        - wake to wake: from wakeup to wakeup
        - diary: from bedtime to the filling of the daily diary
        - diary awake: from wakeup to the filling of the daily diary

    Parameters
    ----------
    loader : :class:`loader.RenewalDataLoader`
        Initialized data loader.
    user_id : str
        Unique user identifier.
    voucher : str
        Voucher identifier, only for compatibility with other functions and unused, by default ``None``.
    cycle_statistics : :class:`str` or :class:`list` or ``None``, optional
        List of statistics to extract, by default ``None``
    cycle_statistics : Union[str, list, None], optional
        _description_, by default None
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or :class:`pandas.Timestamp` or ``None``, optional
        Date of the first awake cycle, by default ``None``
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or :class:`pandas.Timestamp` or ``None``, optional
        Date of the last awake cycle, by default ``None``
    questionnaire_folder : :class:`os.PathLike` or ``None``, optional
        _description_, by default None
    hr_coverage : float, optional
        Percentage of HR values required to consider an epoch as valid based on the
        amount of available heart rate samples, by default 0.8

    Returns
    -------
    :class:`pandas.DataFrame` or :class:`pandas.Series`
        Daily cycle statistics.

    Raises
    ------
    TypeError
        If ``cycle_statistics`` is not of the correct type.
    """
    date_range = get_date_range(start_date=start_date, end_date=end_date)
    from . import awake, physiological, sleep

    # Check statistics
    CYCLE_STATISTICS_DICT = {
        **awake.CYCLE_AWAKE_STATISTICS_DICT,
        **sleep.CYCLE_SLEEP_STATISTICS_DICT,
        **physiological.CYCLE_PHYSIOLOGICAL_STATISTICS_DICT,
    }
    if type(cycle_statistics) is str:
        cycle_statistics = [cycle_statistics]
    elif cycle_statistics is None:
        cycle_statistics = list(CYCLE_STATISTICS_DICT.keys())
    elif not (type(cycle_statistics) is list):
        raise TypeError(
            f"cycle_statistics must be one of [list, str, None], not {type(cycle_statistics)}"
        )

    # Load sleep summaries
    sleep_summaries = loader.load_sleep_summary(
        user_id=user_id,
        start_date=pd.Timestamp(start_date) - pd.Timedelta(days=2),
        end_date=pd.Timestamp(end_date) + pd.Timedelta(days=2),
        drop_duplicates=True,
    )

    if len(sleep_summaries) == 0:
        if len(cycle_statistics) == 1:
            return pd.Series(index=date_range, name=cycle_statistics[0])
        else:
            return pd.DataFrame(
                index=date_range,
                columns=cycle_statistics,
            )

    sleep_summaries = (
        sleep_summaries.set_index(constants.CALENDAR_DATE_COL)
        .asfreq("1d")
        .reset_index(names=constants.CALENDAR_DATE_COL)
    )

    # Get wakeup column
    sleep_summaries[SLEEP_SUMMARY_WAKEUP_COL] = sleep_summaries[
        constants.ISODATE_COL
    ] + pd.to_timedelta(
        sleep_summaries[constants.DURATION_IN_MS_COL]
        + sleep_summaries[constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL].fillna(0),
        unit="ms",
    )

    # Put next day bedtime on the same row
    sleep_summaries[CYCLE_NEXT_BEDTIME_COL] = sleep_summaries[
        constants.ISODATE_COL
    ].shift(-1)
    # Put next wake-up time on the same row
    sleep_summaries[CYCLE_NEXT_WAKEUP_TIME_COL] = sleep_summaries[
        SLEEP_SUMMARY_WAKEUP_COL
    ].shift(-1)
    sleep_summaries[CYCLE_DIARY_END_COL] = pd.NaT

    earliest_bedtime = sleep_summaries[constants.ISODATE_COL].min()

    last_cycle_end_date = np.max(
        np.array(
            [
                sleep_summaries[CYCLE_NEXT_BEDTIME_COL].max(),
                sleep_summaries[CYCLE_DIARY_END_COL].max(),
                sleep_summaries[constants.ISODATE_COL].max(),
                sleep_summaries[SLEEP_SUMMARY_WAKEUP_COL].max(),
            ],
        )
    )
    heart_rate = loader.load_heart_rate(
        user_id=user_id,
        start_date=earliest_bedtime,
        end_date=last_cycle_end_date + pd.Timedelta(milliseconds=1),
    )
    stress = loader.load_stress(
        user_id=user_id,
        start_date=earliest_bedtime,
        end_date=last_cycle_end_date + pd.Timedelta(milliseconds=1),
    )
    epochs = loader.load_epoch(
        user_id=user_id,
        start_date=earliest_bedtime,
        end_date=last_cycle_end_date + pd.Timedelta(milliseconds=1),
    )
    cycle_statistics_list = []
    computed_statistics = []
    for statistic in cycle_statistics:
        ref_cycle = statistic.split("_")[1]
        if not (statistic in computed_statistics):
            ser: pd.Series = CYCLE_STATISTICS_DICT[statistic](
                sleep_summary=sleep_summaries,
                heart_rate=heart_rate,
                stress=stress,
                epochs=epochs,
                cycle=ref_cycle,
                hr_coverage=hr_coverage,
            )
            cycle_statistics_list.append(ser)
            if type(ser) is pd.Series:
                computed_statistics.extend([ser.name])
            else:
                computed_statistics.extend(ser.columns)
    cycle_statistics_df = pd.concat(cycle_statistics_list, axis=1)

    cycle_statistics_df = cycle_statistics_df.reindex(date_range)
    if len(cycle_statistics) == 1:
        return cycle_statistics_df[cycle_statistics[0]]
    else:
        return cycle_statistics_df
