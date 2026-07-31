from dataclasses import dataclass

import pandas as pd


@dataclass
class ResamplingSpec:
    rule: str
    time_col: str
    calendar_based: bool


def get_resampling_spec(
    resolution: str,
) -> ResamplingSpec:

    if resolution == "day" or resolution == "daily":
        return ResamplingSpec(
            rule="d",
            time_col="local_time",
            calendar_based=True,
        )
    elif resolution == "week" or resolution == "weekly":
        return ResamplingSpec(
            rule="W",
            time_col="local_time",
            calendar_based=True,
        )
    elif resolution == "month" or resolution == "monthly":
        return ResamplingSpec(rule="MS", time_col="local_time", calendar_based=True)
    elif resolution == "year" or resolution == "yearly":
        return ResamplingSpec(rule="YS", time_col="local_time", calendar_based=True)
    else:
        try:
            pd.Timedelta(resolution)
        except ValueError:
            raise ValueError(f"Resolution {resolution} is not a valid choice.")
        return ResamplingSpec(
            rule=resolution,
            time_col="utc_time",
            calendar_based=False,
        )
