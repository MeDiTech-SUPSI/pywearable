from dataclasses import dataclass

import pandas as pd


@dataclass
class ResamplingSpec:
    rule: str
    time_col: str
    calendar_based: bool


def get_resampling_spec(
    resolution: str,
    time_reference: str = "auto",
) -> ResamplingSpec:

    if time_reference not in {"auto", "utc", "local"}:
        raise ValueError(
            "time_reference must be one of ('auto', 'utc', 'local'), "
            f"not {time_reference}"
        )

    if resolution == "day" or resolution == "daily":
        rule = "d"
        calendar_based = True
    elif resolution == "week" or resolution == "weekly":
        rule = "W"
        calendar_based = True
    elif resolution == "month" or resolution == "monthly":
        rule = "MS"
        calendar_based = True
    elif resolution == "year" or resolution == "yearly":
        rule = "YS"
        calendar_based = True
    else:
        try:
            pd.Timedelta(resolution)
        except ValueError:
            raise ValueError(f"Resolution {resolution} is not a valid choice.")
        rule = resolution
        calendar_based = False

    if time_reference == "auto":
        time_reference = "local" if calendar_based else "utc"
    return ResamplingSpec(
        rule=rule,
        time_col=f"{time_reference}_time",
        calendar_based=calendar_based,
    )
