import numpy as np
import pandas as pd
from typing import Literal, Union

from . import base
from .. import constants
from .. import cardiac
from .. import utils


@base.registry.register_feature("duration")
def compute_cycle_duration(cycle_data: dict, **kwargs) -> pd.Timedelta:
    if len(cycle_data["intervals"]) > 0:
        return cycle_data["intervals"]["end"] - cycle_data["intervals"]["start"]
    else:
        return pd.NaT


@base.registry.register_feature("start")
def compute_cycle_start(cycle_data: dict, **kwargs) -> pd.Timestamp:
    if len(cycle_data["intervals"]) > 0:
        return cycle_data["intervals"]["start"]
    else:
        return pd.NaT


@base.registry.register_feature("end")
def compute_cycle_end(cycle_data: dict, **kwargs) -> pd.Timestamp:
    if len(cycle_data["intervals"]) > 0:
        return cycle_data["intervals"]["end"]
    else:
        return pd.NaT


#####################################################################
#                      Heart Rate Statistics                        #
#####################################################################


@base.registry.register_feature("heart_rate_mean")
def compute_cycle_heart_rate_mean(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_heart_rate_statistic(cycle_data=cycle_data, stat="mean")


@base.registry.register_feature("heart_rate_std")
def compute_cycle_heart_rate_std(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_heart_rate_statistic(cycle_data=cycle_data, stat="std")


@base.registry.register_feature("heart_rate_min")
def compute_cycle_heart_rate_min(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_heart_rate_statistic(cycle_data=cycle_data, stat="min")


@base.registry.register_feature("heart_rate_max")
def compute_cycle_heart_rate_max(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_heart_rate_statistic(cycle_data=cycle_data, stat="max")


@base.registry.register_feature("heart_rate_count")
def compute_cycle_heart_rate_count(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_heart_rate_statistic(cycle_data=cycle_data, stat="count")


@base.registry.register_feature("heart_rate_median")
def compute_cycle_heart_rate_median(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_heart_rate_statistic(cycle_data=cycle_data, stat="median")


@base.registry.register_feature("heart_rate_rest")
def compute_cycle_heart_rate_rest(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_heart_rate_statistic(cycle_data=cycle_data, stat="rest")


def compute_cycle_heart_rate_statistic(
    cycle_data: dict,
    stat: Literal["count", "mean", "median", "max", "min", "std", "rest"],
    **kwargs,
) -> float:
    if not "heart_rate" in cycle_data.keys():
        return np.nan
    elif len(cycle_data["heart_rate"]) == 0:
        return np.nan
    else:
        heart_rate: pd.DataFrame = cycle_data["heart_rate"]
        # Sort heart rate values - Do not drop duplicates as we may have multiple
        # timezones
        heart_rate = heart_rate.sort_values(
            by=[constants.ISODATE_COL], ignore_index=True
        )
        if stat == "rest":
            # Set iso date as index
            heart_rate = heart_rate.set_index(constants.ISODATE_COL)
            # Perform 30 minute average
            mean_hr = heart_rate.rolling(window="30Min")[constants.HR_COLUMN].mean()
            # Return minimum 30 minute average
            return mean_hr.min()
        else:
            return heart_rate[constants.HR_COLUMN].agg(stat)


################################################################################
#                                 Steps                                        #
################################################################################


@base.registry.register_feature("steps_sum")
def compute_cycle_steps(cycle_data: dict, **kwargs):
    if not "epochs" in cycle_data.keys():
        return np.nan
    elif len(cycle_data["epochs"]) == 0:
        return np.nan
    else:
        return cycle_data["epochs"].loc[:, constants.ACTIVITY_STEPS_COL].sum()


################################################################################
#                         Stress Ranges                                        #
################################################################################
@base.registry.register_feature("high_stress_duration_in_ms")
def compute_cycle_high_stress_duration_in_ms(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_stress_range_duration(
        cycle_data=cycle_data,
        stress_range="high",
        return_as_percentage=False,
    )


@base.registry.register_feature("high_stress_duration_in_percentage")
def compute_cycle_high_stress_duration_in_percentage(
    cycle_data: dict, **kwargs
) -> float:
    return compute_cycle_stress_range_duration(
        cycle_data=cycle_data,
        stress_range="high",
        return_as_percentage=True,
    )


@base.registry.register_feature("low_stress_duration_in_ms")
def compute_cycle_low_stress_duration_in_ms(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_stress_range_duration(
        cycle_data=cycle_data,
        stress_range="low",
        return_as_percentage=False,
    )


@base.registry.register_feature("low_stress_duration_in_percentage")
def compute_cycle_low_stress_duration_in_percentage(
    cycle_data: dict, **kwargs
) -> float:
    return compute_cycle_stress_range_duration(
        cycle_data=cycle_data,
        stress_range="low",
        return_as_percentage=True,
    )


@base.registry.register_feature("medium_stress_duration_in_ms")
def compute_cycle_medium_stress_duration_in_ms(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_stress_range_duration(
        cycle_data=cycle_data,
        stress_range="medium",
        return_as_percentage=False,
    )


@base.registry.register_feature("medium_stress_duration_in_percentage")
def compute_cycle_medium_stress_duration_in_percentage(
    cycle_data: dict, **kwargs
) -> float:
    return compute_cycle_stress_range_duration(
        cycle_data=cycle_data,
        stress_range="medium",
        return_as_percentage=True,
    )


@base.registry.register_feature("rest_stress_duration_in_ms")
def compute_cycle_rest_stress_duration_in_ms(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_stress_range_duration(
        cycle_data=cycle_data,
        stress_range="rest",
        return_as_percentage=False,
    )


@base.registry.register_feature("rest_stress_duration_in_percentage")
def compute_cycle_rest_stress_duration_in_percentage(
    cycle_data: dict, **kwargs
) -> float:
    return compute_cycle_stress_range_duration(
        cycle_data=cycle_data,
        stress_range="rest",
        return_as_percentage=True,
    )


@base.registry.register_feature("unmeasurable_stress_duration_in_ms")
def compute_cycle_unmeasurable_stress_duration_in_ms(
    cycle_data: dict, **kwargs
) -> float:
    return compute_cycle_stress_range_duration(
        cycle_data=cycle_data,
        stress_range="unmeasurable",
        return_as_percentage=False,
    )


@base.registry.register_feature("unmeasurable_stress_duration_in_percentage")
def compute_cycle_unmeasurable_stress_duration_in_percentage(
    cycle_data: dict, **kwargs
) -> float:
    return compute_cycle_stress_range_duration(
        cycle_data=cycle_data,
        stress_range="unmeasurable",
        return_as_percentage=True,
    )


def compute_cycle_stress_range_duration(
    cycle_data: dict, stress_range: str, return_as_percentage: bool, **kwargs
) -> float:
    if not "stress" in cycle_data.keys():
        return np.nan
    elif len(cycle_data["stress"]) == 0:
        return np.nan
    else:
        stress: pd.DataFrame = cycle_data["stress"]
        total_duration = len(stress)
        stress_ser: pd.Series = pd.cut(
            stress[constants.STRESS_STRESS_LEVEL_COL],
            bins=[-10, -1, 25, 50, 75, 100],
            labels=["UNMEASURABLE", "REST", "LOW", "MEDIUM", "HIGH"],
            include_lowest=True,
        )
        stress_ser = stress_ser.loc[stress_ser.str.lower() == stress_range.lower()]

        stress_ms = len(stress_ser) * 3 * 60 * 1000

        if return_as_percentage:
            return stress_ms / (total_duration * 3 * 60 * 1000)


#######################################################################################
#                                       Stress                                        #
#######################################################################################


@base.registry.register_feature("stress_count")
def compute_cycle_stress_count(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_stress_statistic(cycle_data=cycle_data, stat="count")


@base.registry.register_feature("stress_mean")
def compute_cycle_stress_mean(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_stress_statistic(cycle_data=cycle_data, stat="mean")


@base.registry.register_feature("stress_median")
def compute_cycle_stress_median(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_stress_statistic(cycle_data=cycle_data, stat="median")


@base.registry.register_feature("stress_max")
def compute_cycle_stress_max(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_stress_statistic(cycle_data=cycle_data, stat="max")


@base.registry.register_feature("stress_min")
def compute_cycle_stress_min(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_stress_statistic(cycle_data=cycle_data, stat="min")


@base.registry.register_feature("stress_std")
def compute_cycle_stress_std(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_stress_statistic(cycle_data=cycle_data, stat="std")


def compute_cycle_stress_statistic(
    cycle_data: dict,
    stat: Literal["count", "mean", "median", "max", "min", "std"],
    **kwargs,
) -> float:
    if not "stress" in cycle_data.keys():
        return np.nan
    elif len(cycle_data["stress"]) == 0:
        return np.nan
    else:
        stress: pd.DataFrame = cycle_data["stress"]

        return stress.loc[
            (stress[constants.STRESS_STRESS_LEVEL_COL] >= 0),
            constants.STRESS_STRESS_LEVEL_COL,
        ].agg(stat)


################################################################################
#                           Activity levels                                    #
################################################################################


@base.registry.register_feature("epochs_sedentary_duration_in_ms")
def compute_cycle_epochs_sedentary_duration(
    cycle_data: dict,
    hr_coverage: float = 0.8,
    **kwargs,
) -> float:
    return compute_cycle_epoch_activity_level_duration(
        cycle_data=cycle_data,
        activity_level="sedentary",
        return_as_percentage=False,
        hr_coverage=hr_coverage,
    )


@base.registry.register_feature("epochs_sedentary_duration_in_percentage")
def compute_cycle_epochs_sedentary_percentage(
    cycle_data: dict, hr_coverage: float = 0.8, **kwargs
) -> float:
    return compute_cycle_epoch_activity_level_duration(
        cycle_data=cycle_data,
        activity_level="sedentary",
        return_as_percentage=True,
        hr_coverage=hr_coverage,
    )


@base.registry.register_feature("epochs_active_duration_in_ms")
def compute_cycle_epochs_active_duration(
    cycle_data: dict, hr_coverage: float = 0.8, **kwargs
) -> float:
    return compute_cycle_epoch_activity_level_duration(
        cycle_data=cycle_data,
        activity_level="active",
        return_as_percentage=False,
        hr_coverage=hr_coverage,
    )


@base.registry.register_feature("epochs_active_duration_in_percentage")
def compute_cycle_epochs_active_percentage(
    cycle_data: dict, hr_coverage: float = 0.8, **kwargs
) -> float:
    return compute_cycle_epoch_activity_level_duration(
        cycle_data=cycle_data,
        activity_level="active",
        return_as_percentage=True,
        hr_coverage=hr_coverage,
    )


@base.registry.register_feature("epochs_highly_active_duration_in_ms")
def compute_cycle_epochs_highly_active_duration(
    cycle_data: dict, hr_coverage: float = 0.8, **kwargs
) -> float:
    return compute_cycle_epoch_activity_level_duration(
        cycle_data=cycle_data,
        activity_level="highly_active",
        return_as_percentage=False,
        hr_coverage=hr_coverage,
    )


@base.registry.register_feature("epochs_highly_active_duration_in_percentage")
def compute_cycle_epochs_highly_active_percentage(
    cycle_data: dict, hr_coverage: float = 0.8, **kwargs
) -> float:
    return compute_cycle_epoch_activity_level_duration(
        cycle_data=cycle_data,
        activity_level="highly_active",
        return_as_percentage=True,
        hr_coverage=hr_coverage,
    )


def compute_cycle_epoch_activity_level_duration(
    cycle_data: dict,
    activity_level: str,
    return_as_percentage: bool = False,
    hr_coverage: float = 0.8,
    **kwargs,
) -> float:
    if (not "heart_rate" in cycle_data.keys()) or (not "epochs" in cycle_data.keys()):
        return np.nan
    elif (len(cycle_data["heart_rate"]) == 0) or (len(cycle_data["epochs"]) == 0):
        return np.nan
    else:
        heart_rate: pd.DataFrame = cycle_data["heart_rate"]
        epochs: pd.DataFrame = cycle_data["epochs"]
        heart_rate_counts = (
            heart_rate.value_counts(subset=constants.ISODATE_COL)
            .sort_index()
            .resample("15Min")
            .count()
            .rename("hr_count")
        )
        epochs = epochs.groupby([constants.ISODATE_COL, constants.EPOCH_INTENSITY_COL])[
            constants.DURATION_IN_MS_COL
        ].sum()

        epochs_with_hr = pd.merge(
            left=epochs,
            right=heart_rate_counts,
            left_index=True,
            right_index=True,
            how="left",
        )

        # Concatenate epochs
        if len(epochs_with_hr) == 0:
            return np.nan

        epochs_with_hr = epochs_with_hr[
            epochs_with_hr["hr_count"] >= ((15 * 4) * hr_coverage)
        ]
        intensity_durations = epochs_with_hr.groupby(
            [constants.EPOCH_INTENSITY_COL],
            observed=False,
        )[constants.DURATION_IN_MS_COL].sum()
        intensity_durations = intensity_durations.reindex(
            ["ACTIVE", "HIGHLY_ACTIVE", "SEDENTARY"], fill_value=0
        )
        if return_as_percentage:
            return intensity_durations[
                activity_level.upper()
            ] / intensity_durations.sum(axis=0)
        else:
            return intensity_durations[activity_level.upper()]


@base.registry.register_feature("wearing_time")
def compute_cycle_wearing_time(cycle_data: dict, **kwargs) -> float:
    if not "heart_rate" in cycle_data.keys():
        return pd.NaT
    elif len(cycle_data["heart_rate"]) == 0:
        return pd.NaT
    else:
        # Get hr hours
        return pd.Timedelta(
            cycle_data["heart_rate"][constants.HR_COLUMN].count() * 15, unit="seconds"
        )


#######################################################################################
#                                       BBI                                           #
#######################################################################################
@base.registry.register_feature("bbi_raw_duration")
def compute_cycle_bbi_raw_duration(cycle_data: dict, **kwargs) -> pd.Timedelta:
    if not "bbi" in cycle_data.keys():
        return pd.NaT
    elif len(cycle_data["bbi"]) == 0:
        return pd.NaT
    else:
        # Get total BBI hours
        return pd.Timedelta(cycle_data["bbi"]["bbi"].sum(), unit="ms")


@base.registry.register_feature("bbi_sleep_duration")
def compute_cycle_bbi_sleep_duration(cycle_data: dict, **kwargs) -> pd.Timedelta:
    if not "bbi" in cycle_data.keys():
        return pd.NaT
    elif len(cycle_data["bbi"]) == 0:
        return pd.NaT
    else:
        # Filter to keep only sleep stages
        bbi: pd.DataFrame = cycle_data["bbi"]
        filtered_bbi = utils.filter_out_sleep_stages(
            df=bbi,
            stages_to_filter=["awake"],
            sleep_stages=cycle_data["sleep_stages"],
        )
        # Get total BBI hours
        return pd.Timedelta(filtered_bbi["bbi"].sum(), unit="ms")


@base.registry.register_feature("bbi_high_confidence_duration")
def compute_cycle_bbi_high_confidence_duration(
    cycle_data: dict, **kwargs
) -> pd.Timedelta:
    if not "bbi" in cycle_data.keys():
        return pd.NaT
    elif len(cycle_data["bbi"]) == 0:
        return pd.NaT
    else:
        # Filter to keep only high confidence
        bbi: pd.DataFrame = cycle_data["bbi"]
        bbi = bbi.loc[bbi["confidence"] == "HIGH", :]

        return pd.Timedelta(bbi["bbi"].sum(), unit="ms")


@base.registry.register_feature("bbi_sleep_and_high_confidence_duration")
def compute_cycle_bbi_sleep_and_high_confidence_duration(
    cycle_data: dict, **kwargs
) -> pd.Timedelta:
    if not "bbi" in cycle_data.keys():
        return pd.NaT
    elif len(cycle_data["bbi"]) == 0:
        return pd.NaT
    else:
        # Filter to keep only high confidence and during sleep
        bbi: pd.DataFrame = cycle_data["bbi"]
        bbi = bbi.loc[bbi["confidence"] == "HIGH", :]
        filtered_bbi = utils.filter_out_sleep_stages(
            df=bbi,
            stages_to_filter=["awake"],
            sleep_stages=cycle_data["sleep_stages"],
        )
        # Get total BBI hours
        return pd.Timedelta(filtered_bbi["bbi"].sum(), unit="ms")


@base.registry.register_feature("bbi_raw_rmssd")
def compute_cycle_bbi_raw_rmssd(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="rmssd",
        sleep_stages_to_filter=None,
        confidence=None,
    )


@base.registry.register_feature("bbi_sleep_rmssd")
def compute_cycle_bbi_raw_rmssd(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="rmssd",
        sleep_stages_to_filter=["awake"],
        confidence=None,
    )


@base.registry.register_feature("bbi_high_confidence_rmssd")
def compute_cycle_bbi_raw_rmssd(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="rmssd",
        sleep_stages_to_filter=None,
        confidence="high",
    )


@base.registry.register_feature("bbi_sleep_and_high_confidence_rmssd")
def compute_cycle_bbi_raw_rmssd(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="rmssd",
        sleep_stages_to_filter=["awake"],
        confidence="high",
    )


@base.registry.register_feature("bbi_raw_sdnn")
def compute_cycle_bbi_raw_sdnn(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="sdnn",
        sleep_stages_to_filter=None,
        confidence=None,
    )


@base.registry.register_feature("bbi_sleep_sdnn")
def compute_cycle_bbi_raw_sdnn(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="sdnn",
        sleep_stages_to_filter=["awake"],
        confidence=None,
    )


@base.registry.register_feature("bbi_high_confidence_sdnn")
def compute_cycle_bbi_raw_sdnn(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="sdnn",
        sleep_stages_to_filter=None,
        confidence="high",
    )


@base.registry.register_feature("bbi_sleep_and_high_confidence_sdnn")
def compute_cycle_bbi_raw_sdnn(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="sdnn",
        sleep_stages_to_filter=["awake"],
        confidence="high",
    )


@base.registry.register_feature("bbi_raw_lf")
def compute_cycle_bbi_raw_lf(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="lf",
        sleep_stages_to_filter=None,
        confidence=None,
    )


@base.registry.register_feature("bbi_sleep_lf")
def compute_cycle_bbi_raw_lf(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="lf",
        sleep_stages_to_filter=["awake"],
        confidence=None,
    )


@base.registry.register_feature("bbi_high_confidence_lf")
def compute_cycle_bbi_raw_lf(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="lf",
        sleep_stages_to_filter=None,
        confidence="high",
    )


@base.registry.register_feature("bbi_sleep_and_high_confidence_lf")
def compute_cycle_bbi_raw_lf(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="lf",
        sleep_stages_to_filter=["awake"],
        confidence="high",
    )


@base.registry.register_feature("bbi_raw_hf")
def compute_cycle_bbi_raw_hf(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="hf",
        sleep_stages_to_filter=None,
        confidence=None,
    )


@base.registry.register_feature("bbi_sleep_hf")
def compute_cycle_bbi_raw_hf(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="hf",
        sleep_stages_to_filter=["awake"],
        confidence=None,
    )


@base.registry.register_feature("bbi_high_confidence_hf")
def compute_cycle_bbi_raw_hf(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="hf",
        sleep_stages_to_filter=None,
        confidence="high",
    )


@base.registry.register_feature("bbi_sleep_and_high_confidence_hf")
def compute_cycle_bbi_raw_hf(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="hf",
        sleep_stages_to_filter=["awake"],
        confidence="high",
    )


@base.registry.register_feature("bbi_raw_lfhf")
def compute_cycle_bbi_raw_lfhf(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="lfhf",
        sleep_stages_to_filter=None,
        confidence=None,
    )


@base.registry.register_feature("bbi_sleep_lfhf")
def compute_cycle_bbi_raw_lfhf(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="lfhf",
        sleep_stages_to_filter=["awake"],
        confidence=None,
    )


@base.registry.register_feature("bbi_high_confidence_lfhf")
def compute_cycle_bbi_raw_lfhf(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="lfhf",
        sleep_stages_to_filter=None,
        confidence="high",
    )


@base.registry.register_feature("bbi_sleep_and_high_confidence_lfhf")
def compute_cycle_bbi_raw_lfhf(cycle_data: dict, **kwargs) -> float:
    return compute_cycle_bbi_statistic(
        cycle_data=cycle_data,
        statistic="lfhf",
        sleep_stages_to_filter=["awake"],
        confidence="high",
    )


def compute_cycle_bbi_statistic(
    cycle_data: dict,
    statistic: str,
    confidence: Union[str, list] = "high",
    sleep_stages_to_filter: Union[str, list, None] = "awake",
    **kwargs,
) -> float:
    if not "bbi" in cycle_data.keys():
        return np.nan
    elif len(cycle_data["bbi"]) == 0:
        return np.nan
    else:
        bbi: pd.DataFrame = cycle_data["bbi"]
        # Filter by confidence if needed
        if not (confidence is None):
            if type(confidence) is str:
                confidence = [confidence.lower()]
            bbi = bbi.loc[
                bbi["confidence"].str.lower().isin([x.lower() for x in confidence]), :
            ]
        # Filter out sleep stages if needed
        if not (sleep_stages_to_filter is None):
            bbi = utils.filter_out_sleep_stages(
                df=bbi,
                stages_to_filter=sleep_stages_to_filter,
                sleep_stages=cycle_data["sleep_stages"],
            )
        if len(bbi) == 0:
            return np.nan
        hrv_stat = cardiac._compute_hrv_statistic(
            {pd.Timestamp(1900, 1, 1): bbi}, metric=statistic.upper()
        )
        return hrv_stat[pd.Timestamp(1900, 1, 1)]


@base.registry.register_feature("rem_sleep_duration")
def compute_cycle_rem_sleep_duration(cycle_data: dict) -> pd.Timedelta:
    return compute_cycle_sleep_stage_duration(cycle_data=cycle_data, sleep_stage="rem")


@base.registry.register_feature("deep_sleep_duration")
def compute_cycle_rem_sleep_duration(cycle_data: dict) -> pd.Timedelta:
    return compute_cycle_sleep_stage_duration(cycle_data=cycle_data, sleep_stage="deep")


@base.registry.register_feature("light_sleep_duration")
def compute_cycle_rem_sleep_duration(cycle_data: dict) -> pd.Timedelta:
    return compute_cycle_sleep_stage_duration(
        cycle_data=cycle_data, sleep_stage="light"
    )


def compute_cycle_sleep_stage_duration(
    cycle_data: dict, sleep_stage: str
) -> pd.Timedelta:
    if not "sleep_stages" in cycle_data.keys():
        return pd.NaT
    elif len(cycle_data["sleep_stages"]) == 0:
        return pd.NaT
    sleep_stages: pd.DataFrame = cycle_data["sleep_stages"]
    filtered_sleep_stages = sleep_stages.loc[
        sleep_stages["type"].str.lower() == sleep_stage.lower(), :
    ]
    if len(filtered_sleep_stages) == 0:
        return pd.Timedelta(0)
    else:
        return pd.Timedelta(filtered_sleep_stages["durationInMs"].sum(), unit="ms")
