from . import base
from . import awake, daily, physiological, sleep
from ..loader.base import BaseLoader
import pandas as pd


def get_cycle_statistics(
    loader: BaseLoader,
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cycle="physiological",
    statistics=None,
    params=None,
    **kwargs,
):
    if not cycle in base.registry.cycles.keys():
        raise ValueError(
            f"Cycle {cycle} is not supported. Choose among f{list(base.registry.cycles.keys())}."
        )
    cycle_strategy: base.BaseCycle = base.registry.cycles[cycle]()
    intervals = cycle_strategy.get_intervals(
        loader=loader, user_id=user_id, start_date=start_date, end_date=end_date
    )
    params = params or {}
    if statistics is None:
        statistics = base.registry.features.keys()
    else:
        if type(statistics) is str:
            statistics = [statistics.replace(f"cycle_{cycle}_", "")]
        elif type(statistics) is list:
            statistics = [x.replace(f"cycle_{cycle}_", "") for x in statistics]

        for stat in statistics:
            if not stat in base.registry.features.keys():
                raise ValueError(f"Unknown statistic: cycle_{cycle}_{stat}")

    if len(intervals) == 0:
        return pd.DataFrame(
            columns=[f"cycle_{cycle}_{x}" for x in statistics],
            index=pd.date_range(
                start=start_date,
                end=end_date,
                normalize=True,
                inclusive="left",
                freq="d",
            ),
        )
    else:
        # TODO here may want to implement loading with different overlaps as data maybe huge to load into memory
        min_date = intervals["start"].min()
        max_date = intervals["end"].max() + pd.Timedelta(milliseconds=1)
        heart_rate = loader.load_heart_rate(
            user_id=user_id,
            start_date=min_date,
            end_date=max_date,
        )
        sleep_summaries = loader.load_sleep_summary(
            user_id=user_id,
            start_date=min_date,
            end_date=max_date,
        )
        sleep_stages = loader.load_sleep_stage(
            user_id=user_id,
            start_date=min_date,
            end_date=max_date,
        )
        stress = loader.load_stress(
            user_id=user_id,
            start_date=min_date,
            end_date=max_date,
        )
        epochs = loader.load_epoch(
            user_id=user_id,
            start_date=min_date,
            end_date=max_date,
        )
        bbi = loader.load_bbi(
            user_id=user_id,
            start_date=min_date,
            end_date=max_date,
        )
        data_context = {
            "heart_rate": heart_rate,
            "sleep_summaries": sleep_summaries,
            "sleep_stages": sleep_stages,
            "stress": stress,
            "epochs": epochs,
            "intervals": intervals,
            "bbi": bbi,
        }
        results = []
        for date, interval in intervals.iterrows():
            # Slice data for the given cycle
            window_data = {
                "heart_rate": data_context["heart_rate"].loc[
                    (data_context["heart_rate"].loc[:, "isoDate"] >= interval["start"])
                    & (data_context["heart_rate"].loc[:, "isoDate"] < interval["end"]),
                    :,
                ],
                "stress": data_context["stress"].loc[
                    (data_context["stress"].loc[:, "isoDate"] >= interval["start"])
                    & (data_context["stress"].loc[:, "isoDate"] < interval["end"]),
                    :,
                ],
                "epochs": data_context["epochs"].loc[
                    (data_context["epochs"].loc[:, "isoDate"] >= interval["start"])
                    & (data_context["epochs"].loc[:, "isoDate"] < interval["end"]),
                    :,
                ],
                "bbi": data_context["bbi"].loc[
                    (data_context["bbi"].loc[:, "isoDate"] >= interval["start"])
                    & (data_context["bbi"].loc[:, "isoDate"] < interval["end"]),
                    :,
                ],
                "sleep_stages": data_context["sleep_stages"].loc[
                    (
                        data_context["sleep_stages"].loc[:, "isoDate"]
                        >= interval["start"]
                    )
                    & (
                        data_context["sleep_stages"].loc[:, "isoDate"] < interval["end"]
                    ),
                    :,
                ],
                "intervals": data_context["intervals"].loc[date],
            }
            for key, df in kwargs.items():
                if (df is not None) and (type(df) is pd.DataFrame):
                    # We slice the custom dataframe using the same time window
                    window_data[key] = df[
                        (df.index >= interval["start"]) & (df.index <= interval["end"])
                    ]
            metrics = {}
            for statistic in statistics:
                func_params = params.get(statistic, {})
                metrics[f"cycle_{cycle}_{statistic}"] = base.registry.compute_feature(
                    statistic
                )(window_data, **func_params)

            results.append(metrics)
        results = pd.DataFrame(results, index=intervals.index)
        results = results.reindex(
            pd.date_range(
                start=start_date,
                end=end_date,
                normalize=True,
                inclusive="left",
                freq="d",
            )
        )
        return results
