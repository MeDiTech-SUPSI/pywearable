"""Build the tutorial notebooks included in the documentation."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "_notebooks"


def markdown(source):
    """Create a dedented Markdown cell."""
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source):
    """Create a dedented code cell."""
    return nbf.v4.new_code_cell(dedent(source).strip())


def write_notebook(path, cells):
    """Write one Python tutorial notebook."""
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    nbf.write(notebook, path)


def build_analysis_examples():
    """Build the real-data analysis tutorial."""
    cells = [
        markdown(
            """
            # Analyze wearable data with pywearable

            This tutorial follows one de-identified Oura participant from raw
            loader access to domain features, sleep-anchored cycles,
            hierarchies, trajectory post-processing, and visualization.

            You will learn to:

            - check which streams a loader exposes;
            - extract activity, heart-rate, HRV, respiration, SpO2, sleep,
              provider-native, and adherence trajectories;
            - distinguish calendar resolutions, sleep cycles, and hierarchical
              aggregation;
            - impute selected gaps, add a historical trend, and choose a plot.

            The saved outputs use measured values, while participant IDs and
            calendar dates are replaced in tables and figures. To rerun the
            notebook, set `PYWEARABLE_OURA_ROOT` to an Oura API v2 export root.
            """
        ),
        markdown(
            """
            ## Setup

            The root directory contains one subdirectory per participant. This
            local example selects the first directory beginning with `A_`; the
            complete identifier is never displayed or stored in a title.
            """
        ),
        code(
            """
            import os
            import sys
            import warnings
            from pathlib import Path

            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt
            import pandas as pd
            from matplotlib.ticker import FuncFormatter
            from IPython.display import display

            REPO_ROOT = Path.cwd().resolve()
            if not (REPO_ROOT / "src" / "pywearable").is_dir():
                REPO_ROOT = next(
                    parent
                    for parent in Path.cwd().resolve().parents
                    if (parent / "src" / "pywearable").is_dir()
                )
            sys.path.insert(0, str(REPO_ROOT / "src"))

            from pywearable import (
                activity,
                adherence,
                cycle,
                heart_rate,
                hrv,
                provider_features,
                respiration,
                schema,
                sleep,
                trajectory,
                visualization,
            )
            from pywearable.loader import OuraLoader

            warnings.filterwarnings(
                "ignore",
                message=r".*Oura heart-rate samples have no unambiguous local offset.*",
            )

            data_root = Path(
                os.environ.get(
                    "PYWEARABLE_OURA_ROOT",
                    REPO_ROOT / "data_wearable" / "oura_20240822",
                )
            )
            if not data_root.is_dir():
                raise FileNotFoundError(
                    "Set PYWEARABLE_OURA_ROOT to an Oura API v2 export root."
                )

            loader = OuraLoader(data_root)
            source_user = next(user for user in loader.get_user_ids() if user.startswith("A_"))
            public_user = "example-participant"
            print("Loaded one Oura participant")
            """
        ),
        markdown(
            """
            ## Start with loader capabilities

            “What can I extract?” always depends on the selected loader. Oura
            supplies daily activity summaries, subdaily heart rate,
            provider-computed HRV, nightly respiratory-rate and SpO2 summaries,
            sleep summaries and stages, and readiness features. It does not
            provide timestamped step intervals, so peak-hour activity and the
            multi-stream weekly overview are not available from this loader.
            """
        ),
        code(
            """
            capability_table = pd.DataFrame(
                {
                    "loader setting": [
                        "activity_source",
                        "activity_primary_resolution",
                        "hrv_source",
                        "respiratory_rate_source",
                        "spo2_source",
                        "provider_features",
                    ],
                    "value": [
                        loader.activity_source,
                        loader.activity_primary_resolution,
                        loader.hrv_source,
                        loader.respiratory_rate_source,
                        loader.spo2_source,
                        f"{len(loader.provider_features)} Oura readiness features",
                    ],
                }
            )
            display(capability_table)
            """
        ),
        markdown(
            """
            ## Choose one well-supported example interval

            The code below finds the 70-day activity interval with the most
            observed days. This keeps the tutorial reproducible without
            publishing the source calendar dates.
            """
        ),
        code(
            """
            raw_activity = loader.load_activity_summary(source_user)
            source_start = raw_activity[schema.CALENDAR_DATE_COL].min().normalize()
            source_end = raw_activity[schema.CALENDAR_DATE_COL].max().normalize() + pd.Timedelta(days=1)

            all_steps = activity.get_activity_statistics(
                loader,
                statistic=activity.ACTIVITY_STATISTIC_STEPS,
                user_id=source_user,
                start_date=source_start,
                end_date=source_end,
                resolution="daily",
            )
            step_series = all_steps.xs(source_user, level=schema.USER_COL)[
                activity.ACTIVITY_STATISTIC_STEPS
            ]
            rolling_support = step_series.notna().rolling(70, min_periods=70).sum()
            final_day = rolling_support.idxmax()
            analysis_start = final_day - pd.Timedelta(days=69)
            analysis_end = final_day + pd.Timedelta(days=1)
            step_times = all_steps.index.get_level_values(schema.CALENDAR_DATE_COL)
            steps = all_steps[(step_times >= analysis_start) & (step_times < analysis_end)]

            PUBLIC_MONDAY = pd.Timestamp("2026-01-05")

            def public_trajectory(frame):
                # Replace identifiers and shift datetimes while retaining values.
                rows = frame.reset_index()
                rows[schema.USER_COL] = public_user
                datetime_columns = [
                    column
                    for column in rows
                    if pd.api.types.is_datetime64_any_dtype(rows[column])
                ]
                if datetime_columns:
                    origin = min(rows[column].dropna().min() for column in datetime_columns)
                    anchor = PUBLIC_MONDAY + pd.Timedelta(days=origin.weekday())
                    for column in datetime_columns:
                        rows[column] = anchor + (rows[column] - origin)
                return rows

            print(f"Selected {len(steps)} study days; {step_series.loc[analysis_start:final_day].notna().sum()} contain steps")
            display(public_trajectory(steps).head())
            """
        ),
        markdown(
            """
            ## Extract features from every supported domain

            Each domain returns the same trajectory shape: a `(user_id, time)`
            index, feature columns, and matching coverage columns. The feature
            names describe the measurement; the loader determines whether the
            required stream exists.
            """
        ),
        code(
            """
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hourly_hr = heart_rate.get_heart_rate_statistics(
                    loader,
                    statistic=heart_rate.HEART_RATE_STATISTIC_MEAN,
                    user_id=source_user,
                    start_date=analysis_start,
                    end_date=analysis_end,
                    resolution="1h",
                    time_reference="local",
                )
                daily_hr = heart_rate.get_heart_rate_statistics(
                    loader,
                    statistic=[
                        heart_rate.HEART_RATE_STATISTIC_MEAN,
                        heart_rate.HEART_RATE_STATISTIC_REST,
                    ],
                    user_id=source_user,
                    start_date=analysis_start,
                    end_date=analysis_end,
                    resolution="daily",
                )

            nightly_sleep = sleep.get_sleep_statistics(
                loader,
                statistic=[
                    sleep.SLEEP_STATISTIC_TST,
                    sleep.SLEEP_STATISTIC_WASO,
                    sleep.SLEEP_STATISTIC_BEDTIME,
                    sleep.SLEEP_STATISTIC_SLEEP_SCORE,
                    sleep.SLEEP_STATISTIC_CPD_MIDPOINT,
                    sleep.SLEEP_STATISTIC_CPD_DURATION,
                ],
                user_id=source_user,
                start_date=analysis_start,
                end_date=analysis_end,
                resolution="nightly",
            )
            nightly_hrv = hrv.get_hrv_statistics(
                loader,
                statistic=hrv.HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES,
                user_id=source_user,
                start_date=analysis_start,
                end_date=analysis_end,
                resolution="nightly",
            )
            breathing = respiration.get_respiration_statistics(
                loader,
                statistic=[
                    respiration.RESPIRATORY_RATE_STATISTIC_MEAN,
                    respiration.SPO2_STATISTIC_MEAN,
                ],
                user_id=source_user,
                start_date=source_start,
                end_date=source_end,
                resolution="nightly",
            )
            readiness = provider_features.get_provider_features(
                loader,
                feature=provider_features.OURA_READINESS_SCORE,
                user_id=source_user,
                start_date=analysis_start,
                end_date=analysis_end,
            )
            availability = adherence.get_adherence(
                loader,
                metric=[adherence.ADHERENCE_HEART_RATE, adherence.ADHERENCE_SLEEP],
                user_id=source_user,
                start_date=analysis_start,
                end_date=analysis_start + pd.Timedelta(days=7),
                resolution="daily",
                time_reference="local",
            )

            trajectories = {
                "Activity": steps,
                "Heart rate": daily_hr,
                "HRV": nightly_hrv,
                "Respiration and SpO2": breathing,
                "Sleep": nightly_sleep,
                "Oura readiness": readiness,
                "Adherence": availability,
            }
            summary = pd.DataFrame(
                {
                    "rows": {name: len(frame) for name, frame in trajectories.items()},
                    "feature columns": {
                        name: sum(
                            not column.endswith("__coverage")
                            and column not in {"period_start", "period_end"}
                            for column in frame.columns
                        )
                        for name, frame in trajectories.items()
                    },
                    "non-missing values": {
                        name: int(frame.notna().sum().sum())
                        for name, frame in trajectories.items()
                    },
                }
            )
            display(summary)
            """
        ),
        markdown(
            """
            ## Sleep-anchored cycles

            A night runs from one main-sleep bedtime to its end, an awake cycle
            runs from that end to the following bedtime, and a physiological
            cycle runs from one bedtime to the next. The table reports elapsed
            duration only, keeping the source dates private.
            """
        ),
        code(
            """
            cycle_rows = []
            for cycle_type in (
                cycle.CYCLE_NIGHT,
                cycle.CYCLE_AWAKE,
                cycle.CYCLE_PHYSIOLOGICAL,
            ):
                intervals = cycle.get_intervals(
                    loader,
                    cycle_type,
                    user_id=source_user,
                    start_date=analysis_start,
                    end_date=analysis_end,
                ).xs(source_user, level=schema.USER_COL)
                valid = intervals[intervals["interval_valid"]].head(5)
                cycle_rows.append(
                    pd.DataFrame(
                        {
                            "cycle": cycle_type,
                            "study row": range(1, len(valid) + 1),
                            "duration (hours)": (
                                valid["end_time"] - valid["start_time"]
                            ).dt.total_seconds().div(3600).round(2).to_numpy(),
                        }
                    )
                )
            display(pd.concat(cycle_rows, ignore_index=True))
            """
        ),
        markdown(
            """
            ## Hierarchies preserve the aggregation path

            Here nightly total sleep time is the child level. Weekly means and
            standard deviations are calculated from valid nights, so each
            night receives equal weight. The lower-level observations and
            parent estimates remain visible together.
            """
        ),
        code(
            """
            hierarchy_ax = visualization.plot_hierarchical_trajectory(
                loader,
                sleep.SLEEP_STATISTIC_TST,
                ("nightly", "weekly"),
                user_id=source_user,
                start_date=analysis_start,
                end_date=analysis_end,
                child_label="Nightly TST",
                center_label="Weekly mean",
                range_label="Weekly mean ± SD",
                title="Nightly total sleep time and weekly context",
                ylabel="Total sleep time (minutes)",
                show=False,
            )

            def relative_week(value, _):
                timestamp = pd.Timestamp(mdates.num2date(value)).tz_localize(None)
                return f"Week {(timestamp.normalize() - analysis_start).days // 7 + 1}"

            hierarchy_ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            hierarchy_ax.xaxis.set_major_formatter(FuncFormatter(relative_week))
            hierarchy_ax.tick_params(axis="x", rotation=0)
            hierarchy_ax.set_xlabel("Study week")
            plt.show()
            """
        ),
        markdown(
            """
            ## Imputation and trends are explicit trajectory operations

            Imputation fills eligible gaps and adds `__is_imputed`. Trend
            estimation then adds participant-specific baseline, normal-range,
            deviation, and support columns. Coverage continues to describe the
            source evidence.
            """
        ),
        code(
            """
            public_steps = public_trajectory(steps).set_index(steps.index.names)[steps.columns]
            filled_steps = trajectory.impute_trajectory(
                public_steps,
                activity.ACTIVITY_STATISTIC_STEPS,
                method="rolling_median",
                window=7,
                min_periods=4,
            )
            trended_steps = trajectory.compute_trajectory_trend(
                filled_steps,
                activity.ACTIVITY_STATISTIC_STEPS,
                baseline_window=7,
                min_periods_baseline=4,
                normal_range_window=21,
                min_periods_normal_range=10,
            )
            trend_axes = visualization.plot_trajectory_trend(
                trended_steps,
                activity.ACTIVITY_STATISTIC_STEPS,
                show_coverage=True,
                title="Daily steps with imputation and historical context",
                ylabel="Steps",
                show=False,
            )
            plt.show()
            """
        ),
        markdown(
            """
            ## Calendar and within-day views

            Calendar heatmaps suit daily outcomes. Coverage heatmaps and weekly
            profiles require a subdaily trajectory; here they use hourly heart
            rate. The profile summarizes repeated weekdays and is not a weekly
            chronological aggregation.
            """
        ),
        code(
            """
            visualization.plot_calendar_heatmap(
                public_steps,
                activity.ACTIVITY_STATISTIC_STEPS,
                title="Daily steps",
                colorbar_label="Steps",
                show=False,
            )
            plt.show()

            public_hourly = public_trajectory(hourly_hr).set_index(hourly_hr.index.names)[hourly_hr.columns]
            visualization.plot_coverage_heatmap(
                public_hourly,
                heart_rate.HEART_RATE_STATISTIC_MEAN,
                title="Hourly heart-rate coverage",
                show=False,
            )
            plt.show()

            visualization.plot_weekly_profile(
                public_hourly,
                heart_rate.HEART_RATE_STATISTIC_MEAN,
                aggregation="median",
                night_window=(23, 7),
                title="Median heart-rate profile by weekday",
                ylabel="Heart rate (bpm)",
                show=False,
            )
            plt.show()
            """
        ),
        markdown(
            """
            ## Loader-aware sleep views

            Composite plots accept a loader because they coordinate sleep
            summaries, stages, and heart-rate samples. `plot_sleep_summary`
            takes inclusive sleep-calendar bounds, so the first selected night
            retains stages from the preceding civil date.
            """
        ),
        code(
            """
            sleep_dates = loader.load_sleep_summary(
                source_user, analysis_start, analysis_end
            )[schema.CALENDAR_DATE_COL]
            summary_start = sleep_dates.iloc[0]
            summary_end = sleep_dates.iloc[min(13, len(sleep_dates) - 1)]
            sleep_axes = visualization.plot_sleep_summary(
                loader,
                source_user,
                summary_start,
                summary_end,
                cpd=("midpoint", "duration"),
                title="Sleep stages, timing, score, and composite phase deviation",
                fontsize=9,
                show=False,
            )
            sleep_axis = sleep_axes["sleep"]
            sleep_axis.set_yticklabels(
                [f"Night {index + 1}" for index in range(len(sleep_axis.get_yticks()))]
            )
            sleep_axis.set_ylabel("Observed night")
            plt.show()
            """
        ),
        code(
            """
            heart_rate_night = None
            for row in loader.load_sleep_summary(source_user, analysis_start, analysis_end).itertuples(index=False):
                samples = loader.load_heart_rate(
                    source_user,
                    start_date=row.local_time,
                    end_date=row.local_end_time,
                    time_col=schema.LOCAL_TIME_COL,
                )
                if samples[schema.HEART_RATE_COL].notna().any():
                    heart_rate_night = row.calendar_date
                    break
            assert heart_rate_night is not None
            visualization.plot_sleep_heart_rate(
                loader,
                source_user,
                heart_rate_night,
                show_coverage=True,
                title="Heart rate through one observed sleep session",
                show=False,
            )
            plt.show()
            """
        ),
        markdown(
            """
            ## Period comparison and adherence monitoring

            Period comparison expects an already aggregated trajectory. The
            adherence summary aligns participants to their first evidence and
            can use a fixed follow-up length. The Oura loader cannot produce
            `plot_weekly_overview` because that plot requires timestamped step
            intervals, whereas Oura supplies daily step totals.
            """
        ),
        code(
            """
            weekly_steps = activity.get_activity_statistics(
                loader,
                statistic=activity.ACTIVITY_STATISTIC_STEPS,
                user_id=source_user,
                start_date=analysis_start,
                end_date=analysis_end,
                resolution=("daily", "weekly"),
                aggregation="sum",
            )
            public_weekly = public_trajectory(weekly_steps).set_index(weekly_steps.index.names)[weekly_steps.columns]
            weekly_feature = next(
                column
                for column in public_weekly
                if column.endswith("__weekly_sum")
            )
            visualization.plot_period_comparison(
                public_weekly,
                weekly_feature,
                title="Weekly steps",
                ylabel="Steps",
                show=False,
            )
            plt.show()

            adherence_axes = visualization.plot_adherence_summary(
                loader,
                metric=adherence.ADHERENCE_HEART_RATE,
                user_id=source_user,
                follow_up_days=28,
                include_without_evidence=False,
                title="First 28 days of heart-rate and sleep availability",
                show=False,
            )
            for panel in ("wearing_days", "available_nights"):
                adherence_axes[panel].set_yticklabels([public_user])
            plt.show()
            """
        ),
        markdown(
            """
            ## Exercise

            Choose one feature above and answer the same question at two
            resolutions. For example, compare direct daily heart-rate means
            with a weekly hierarchy built from qualified daily means.

            1. Decide the primary and parent resolutions.
            2. Set a coverage threshold appropriate for the primary rows.
            3. Plot child observations against the parent estimate.
            4. Explain what would change if raw samples were pooled directly.
            """
        ),
        code(
            """
            # Your analysis here.
            # daily = heart_rate.get_heart_rate_statistics(...)
            # weekly = heart_rate.get_heart_rate_statistics(
            #     ..., resolution=("daily", "weekly"), aggregation="mean"
            # )
            """
        ),
        markdown(
            """
            ## Practical checks

            - Confirm that the loader exposes the stream required by a feature.
            - Treat `end_date` as exclusive for feature extraction; the sleep
              summary visualization documents its inclusive calendar bounds.
            - Inspect coverage before setting thresholds.
            - Use a hierarchy when child windows should receive equal weight.
            - Keep imputation and trend settings in the analysis record.
            - Retain provider names for provider-defined outcomes such as Oura
              readiness.
            """
        ),
    ]
    write_notebook(OUTPUT / "analysis_examples.ipynb", cells)


def build_custom_loader():
    """Build the synthetic custom-loader tutorial."""
    cells = [
        markdown(
            """
            # Implement a custom pywearable loader

            A loader adapts one project or provider to pywearable's canonical
            schemas. Domain functions then calculate features without knowing
            the original file names or field conventions.

            This tutorial builds a small loader from synthetic tables. It
            covers:

            - the required `BaseLoader` methods;
            - capability attributes for optional streams;
            - canonical schema preparation and half-open date filtering;
            - empty and unsupported streams;
            - end-to-end feature extraction and visualization.
            """
        ),
        markdown(
            """
            ## The loader contract

            Subclasses implement `get_user_ids`, `load_sleep_summary`,
            `load_sleep_stage`, and `load_heart_rate`. Optional methods such as
            `load_activity_samples` are implemented only when the source
            provides that stream. Capability attributes describe source grain
            and available clocks.

            Loader methods normalize fields and units. Feature calculations,
            aggregation, imputation, and trends remain in the domain modules.
            """
        ),
        code(
            """
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display

            REPO_ROOT = Path.cwd().resolve()
            if not (REPO_ROOT / "src" / "pywearable").is_dir():
                REPO_ROOT = next(
                    parent
                    for parent in Path.cwd().resolve().parents
                    if (parent / "src" / "pywearable").is_dir()
                )
            sys.path.insert(0, str(REPO_ROOT / "src"))

            from pywearable import activity, heart_rate, schema, sleep, visualization
            from pywearable.loader import BaseLoader
            """
        ),
        markdown(
            """
            ## Create representative source tables

            Real adapters would read CSV, Parquet, a database, or an API. These
            tables deliberately use source-specific names so the translation
            step remains visible.
            """
        ),
        code(
            """
            rng = np.random.default_rng(7)
            participant = "participant-01"
            start = pd.Timestamp("2026-01-01")
            end = pd.Timestamp("2026-01-15")

            hr_time = pd.date_range(start, end, freq="5min", inclusive="left")
            source_hr = pd.DataFrame(
                {
                    "participant": participant,
                    "recorded_at": hr_time,
                    "bpm": 66
                    + 9 * np.sin(2 * np.pi * (hr_time.hour + hr_time.minute / 60) / 24)
                    + rng.normal(0, 2.5, len(hr_time)),
                }
            )

            activity_time = pd.date_range(start, end, freq="15min", inclusive="left")
            hour = activity_time.hour.to_numpy()
            source_activity = pd.DataFrame(
                {
                    "participant": participant,
                    "interval_start": activity_time,
                    "step_count": rng.poisson(
                        2
                        + 35 * np.exp(-((hour - 8) / 2.0) ** 2)
                        + 50 * np.exp(-((hour - 18) / 2.5) ** 2)
                    ),
                }
            )

            summary_rows = []
            stage_rows = []
            stages = [("awake", 20), ("light", 90), ("deep", 100), ("light", 80), ("rem", 70), ("light", 90), ("awake", 30)]
            for offset, calendar_date in enumerate(pd.date_range(start + pd.Timedelta(days=1), end, inclusive="left")):
                bedtime = calendar_date - pd.Timedelta(days=1) + pd.Timedelta(hours=23, minutes=(offset % 4) * 10)
                summary_id = f"night-{offset + 1}"
                summary_rows.append(
                    {
                        "night_id": summary_id,
                        "sleep_day": calendar_date,
                        "bedtime": bedtime,
                        "session_end": bedtime + pd.Timedelta(hours=8),
                        "sleep_minutes": 430,
                        "awake_minutes": 50,
                    }
                )
                stage_start = bedtime
                for stage_name, minutes in stages:
                    stage_rows.append(
                        {
                            "night_id": summary_id,
                            "stage_start": stage_start,
                            "minutes": minutes,
                            "stage": stage_name,
                        }
                    )
                    stage_start += pd.Timedelta(minutes=minutes)

            source_sleep = pd.DataFrame(summary_rows)
            source_stages = pd.DataFrame(stage_rows)
            display(source_hr.head(2), source_activity.head(2), source_sleep.head(2), source_stages.head(2))
            """
        ),
        markdown(
            """
            ## Implement the adapter

            Each method returns canonical columns and applies `[start_date,
            end_date)` to the clock documented by that method. Schema preparers
            normalize types, add optional columns, and validate required fields.
            Returning a typed empty canonical frame is preferable to inventing
            observations when a supported interval has no rows.
            """
        ),
        code(
            """
            class SyntheticLoader(BaseLoader):
                activity_source = "samples"
                activity_primary_resolution = "15min"
                activity_time_columns = (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL)
                heart_rate_time_columns = (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL)

                def __init__(self, heart_rate_rows, activity_rows, sleep_rows, stage_rows):
                    self._heart_rate = heart_rate_rows.copy()
                    self._activity = activity_rows.copy()
                    self._sleep = sleep_rows.copy()
                    self._stages = stage_rows.copy()

                def get_user_ids(self):
                    return sorted(self._heart_rate["participant"].unique())

                @staticmethod
                def _filter(frame, column, start_date, end_date):
                    values = pd.to_datetime(frame[column])
                    mask = pd.Series(True, index=frame.index)
                    if start_date is not None:
                        mask &= values >= pd.Timestamp(start_date)
                    if end_date is not None:
                        mask &= values < pd.Timestamp(end_date)
                    return frame.loc[mask].copy()

                def load_heart_rate(self, user_id, start_date=None, end_date=None, time_col="local_time"):
                    rows = self._heart_rate[self._heart_rate["participant"].eq(user_id)].copy()
                    canonical = pd.DataFrame(
                        {
                            schema.UTC_TIME_COL: rows["recorded_at"],
                            schema.LOCAL_TIME_COL: rows["recorded_at"],
                            schema.TIMEZONE_COL: pd.Timedelta(0),
                            schema.HEART_RATE_COL: rows["bpm"],
                        }
                    )
                    canonical = schema.prepare_heart_rate_samples(canonical)
                    return self._filter(canonical, time_col, start_date, end_date)

                def load_activity_samples(self, user_id, start_date=None, end_date=None, **kwargs):
                    rows = self._activity[self._activity["participant"].eq(user_id)].copy()
                    canonical = pd.DataFrame(
                        {
                            schema.UTC_TIME_COL: rows["interval_start"],
                            schema.LOCAL_TIME_COL: rows["interval_start"],
                            schema.TIMEZONE_COL: pd.Timedelta(0),
                            schema.DURATION_IN_MS_COL: 15 * 60_000,
                            schema.ACTIVITY_STEPS_COL: rows["step_count"],
                        }
                    )
                    canonical = schema.prepare_activity_samples(canonical)
                    return self._filter(
                        canonical, schema.LOCAL_TIME_COL, start_date, end_date
                    )

                def load_sleep_summary(self, user_id, start_date=None, end_date=None, drop_duplicates=True):
                    rows = self._sleep.copy()
                    canonical = pd.DataFrame(
                        {
                            schema.SLEEP_SUMMARY_ID_COL: rows["night_id"],
                            schema.CALENDAR_DATE_COL: rows["sleep_day"],
                            schema.LOCAL_TIME_COL: rows["bedtime"],
                            schema.LOCAL_END_TIME_COL: rows["session_end"],
                            schema.DURATION_IN_MS_COL: rows["sleep_minutes"] * 60_000,
                            schema.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL: rows["awake_minutes"] * 60_000,
                        }
                    )
                    canonical = schema.prepare_sleep_summary(canonical)
                    canonical = self._filter(
                        canonical, schema.CALENDAR_DATE_COL, start_date, end_date
                    )
                    if drop_duplicates:
                        canonical = canonical.drop_duplicates(schema.CALENDAR_DATE_COL)
                    return canonical.reset_index(drop=True)

                def load_sleep_stage(self, user_id, start_date=None, end_date=None):
                    rows = self._stages.copy()
                    canonical = pd.DataFrame(
                        {
                            schema.SLEEP_SUMMARY_ID_COL: rows["night_id"],
                            schema.LOCAL_TIME_COL: rows["stage_start"],
                            schema.DURATION_IN_MS_COL: rows["minutes"] * 60_000,
                            schema.SLEEP_STAGE_SLEEP_TYPE_COL: rows["stage"],
                        }
                    )
                    canonical = schema.prepare_sleep_stages(canonical)
                    stage_end = canonical[schema.LOCAL_TIME_COL] + pd.to_timedelta(
                        canonical[schema.DURATION_IN_MS_COL], unit="ms"
                    )
                    mask = pd.Series(True, index=canonical.index)
                    if start_date is not None:
                        mask &= stage_end > pd.Timestamp(start_date)
                    if end_date is not None:
                        mask &= canonical[schema.LOCAL_TIME_COL] < pd.Timestamp(end_date)
                    return canonical.loc[mask].reset_index(drop=True)


            loader = SyntheticLoader(source_hr, source_activity, source_sleep, source_stages)
            print(loader.get_user_ids())
            """
        ),
        markdown(
            """
            ## Validate the canonical boundary

            Before using domain functions, inspect each loader method directly.
            Required columns, dtypes, sorting, duplicate handling, and empty
            intervals are part of the loader contract.
            """
        ),
        code(
            """
            hr = loader.load_heart_rate(participant, start, start + pd.Timedelta(hours=1))
            activity_samples = loader.load_activity_samples(participant, start, start + pd.Timedelta(hours=1))
            summaries = loader.load_sleep_summary(participant, start, end)
            stage_intervals = loader.load_sleep_stage(participant, start, end)

            assert set(schema.HEART_RATE_REQUIRED_COLUMNS) <= set(hr)
            assert set(schema.ACTIVITY_SAMPLE_REQUIRED_COLUMNS) <= set(activity_samples)
            assert set(schema.SLEEP_SUMMARY_REQUIRED_COLUMNS) <= set(summaries)
            assert tuple(stage_intervals.columns) == schema.SLEEP_STAGE_REQUIRED_COLUMNS
            assert loader.load_heart_rate(participant, end, end + pd.Timedelta(days=1)).empty
            print("Canonical loader checks passed")
            """
        ),
        markdown(
            """
            ## Use the normal pywearable APIs

            Once the adapter returns canonical streams, no project-specific
            code is needed for feature extraction.
            """
        ),
        code(
            """
            daily_steps = activity.get_activity_statistics(
                loader,
                statistic=activity.ACTIVITY_STATISTIC_STEPS,
                user_id=participant,
                start_date=start,
                end_date=end,
                resolution="daily",
            )
            daily_hr = heart_rate.get_heart_rate_statistics(
                loader,
                statistic=[
                    heart_rate.HEART_RATE_STATISTIC_MEAN,
                    heart_rate.HEART_RATE_STATISTIC_REST,
                ],
                user_id=participant,
                start_date=start,
                end_date=end,
                resolution="daily",
            )
            nightly_sleep = sleep.get_sleep_statistics(
                loader,
                statistic=[
                    sleep.SLEEP_STATISTIC_TIB,
                    sleep.SLEEP_STATISTIC_TST,
                    sleep.SLEEP_STATISTIC_WASO,
                    sleep.SLEEP_STATISTIC_BEDTIME,
                ],
                user_id=participant,
                start_date=start,
                end_date=end,
                resolution="nightly",
            )
            display(daily_steps.head(), daily_hr.head(), nightly_sleep.head())
            """
        ),
        markdown(
            """
            ## Verify a composite view

            The weekly overview needs timestamped activity samples, heart rate,
            and sleep stages. Implementing those three streams makes the
            loader-aware visualization available automatically.
            """
        ),
        code(
            """
            visualization.plot_weekly_overview(
                loader,
                participant,
                start,
                start + pd.Timedelta(days=7),
                title="Synthetic loader: one integrated week",
                show=False,
            )
            plt.show()
            """
        ),
        markdown(
            """
            ## Exercise

            Add an optional provider-native daily score.

            1. Add its name to `provider_features`.
            2. Implement `load_provider_feature` with a daily canonical table.
            3. Call `provider_features.get_available_provider_features` and
               `provider_features.get_provider_features`.
            4. Add tests for empty bounds and an unknown feature name.
            """
        ),
        code(
            """
            # Suggested scaffold:
            # class ExtendedSyntheticLoader(SyntheticLoader):
            #     provider_features = frozenset({"my_provider_score"})
            #
            #     def load_provider_feature(self, feature, user_id, start_date=None, end_date=None):
            #         ...
            """
        ),
        markdown(
            """
            ## Production checklist

            - Preserve the provider's native resolution and units.
            - Expose UTC and local clocks only when they are known.
            - Apply `[start_date, end_date)` consistently.
            - Keep observed zero distinct from missing data.
            - Return canonical empty frames for supported but empty intervals.
            - Raise a clear error for unsupported provider-native features.
            - Link every sleep-stage interval to a `sleep_summary_id`.
            - Test duplicate records, timezone changes, empty intervals, and
              one end-to-end domain function for every implemented stream.
            """
        ),
    ]
    write_notebook(OUTPUT / "custom_loader.ipynb", cells)


def main():
    """Build both documentation notebooks."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_analysis_examples()
    build_custom_loader()


if __name__ == "__main__":
    main()
