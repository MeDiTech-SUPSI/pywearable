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
    """Build the guided custom-loader tutorial."""
    cells = [
        markdown(
            """
            # Implement a custom pywearable loader

            A loader adapts one project or provider to pywearable's canonical
            schemas. Domain functions then calculate features without knowing
            the original file names or field conventions.

            Here we implement `WyvernLoader` for the **Wyvern Band**, a fictional
            wearable. The example export contains three participants and seven
            days, with deliberately inconsistent field names, units, and file
            formats.

            We will inspect the provider files, translate deliberately
            inconsistent fields into canonical schemas, validate the loader
            boundary, and then use the normal feature and visualization APIs.
            """
        ),
        code(
            """
            import json
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
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

            from pywearable import (
                activity,
                heart_rate,
                provider_features,
                respiration,
                schema,
                sleep,
                visualization,
            )
            from pywearable.loader import BaseLoader

            DATA_ROOT = REPO_ROOT / "docs" / "_notebooks" / "mock_provider_data"
            START = pd.Timestamp("2026-03-02")
            END = pd.Timestamp("2026-03-09")

            WYVERN_SLEEP_HEALTH_SCORE = "wyvern_sleep_health_score"
            WYVERN_DAILY_STRAIN_SCORE = "wyvern_daily_strain_score"
            """
        ),
        markdown(
            """
            ## Inspect the fictional provider export

            The export uses one directory per participant and mixes TSV, CSV,
            and JSON files. Source names mix camel case and snake case, and
            none is assumed to be canonical.
            """
        ),
        code(
            """
            participant_dirs = sorted(
                path for path in DATA_ROOT.glob("demo-*") if path.is_dir()
            )
            participants_source = pd.concat(
                [
                    pd.read_csv(path / "participant.tsv", sep="\t")
                    for path in participant_dirs
                ],
                ignore_index=True,
            )
            pulse_source = pd.concat(
                [
                    pd.read_csv(path / "pulse.tsv", sep="\t")
                    for path in participant_dirs
                ],
                ignore_index=True,
            )
            movement_source = pd.concat(
                [pd.read_csv(path / "movement.csv") for path in participant_dirs],
                ignore_index=True,
            )
            sleep_source = pd.json_normalize(
                [
                    night
                    for path in participant_dirs
                    for night in json.loads(
                        (path / "sleep.json").read_text()
                    )["nights"]
                ]
            )
            score_source = pd.DataFrame(
                [
                    record
                    for path in participant_dirs
                    for record in json.loads(
                        (path / "daily_scores.json").read_text()
                    )["records"]
                ]
            )
            inventory = pd.DataFrame(
                [
                    ("demo-*/participant.tsv", "TSV", len(participant_dirs), len(participants_source)),
                    ("demo-*/pulse.tsv", "TSV", len(participant_dirs), len(pulse_source)),
                    ("demo-*/movement.csv", "CSV", len(participant_dirs), len(movement_source)),
                    ("demo-*/sleep.json", "JSON", len(participant_dirs), len(sleep_source)),
                    ("demo-*/daily_scores.json", "JSON", len(participant_dirs), len(score_source)),
                ],
                columns=["path pattern", "format", "files", "records"],
            )

            display(inventory)
            display(
                pulse_source.head(3),
                movement_source.head(3),
                sleep_source[
                    [
                        "memberCode",
                        "sleepDate",
                        "startedAt",
                        "sleepSeconds",
                        "nightlyRespiration.meanBreathsPerMinute",
                    ]
                ].head(3),
                score_source.head(3),
            )
            """
        ),
        markdown(
            """
            ## Plan the translation

            | Source field | Canonical field | Conversion |
            |---|---|---|
            | `capturedAtLocal`, `utcOffsetMinutes`, `pulseBpm` | `local_time`, `utc_time`, `timezone`, `heart_rate` | Parse the local clock, subtract the offset for UTC, and retain numeric beats per minute. |
            | `movement.csv`: `intervalStart`, `intervalSeconds`, `stepCount`; `participant.tsv`: `utcOffsetMinutes` | `local_time`, `utc_time`, `timezone`, `duration_in_ms`, `activity_steps` | Use the participant metadata offset for both clocks, convert seconds to milliseconds, and preserve zero-step intervals. |
            | `nightKey`, `sleepDate`, `startedAt`, `endedAt` | `sleep_summary_id`, `calendar_date`, `local_time`, `local_end_time` | Retain the provider's night identifier and sleep-calendar label; parse session bounds. |
            | `sleepSeconds`, `awakeSeconds`, `unmeasurableSeconds`, stage totals | `duration_in_ms`, `awake_duration_in_ms`, `unmeasurable_sleep_duration_in_ms`, `light_sleep_duration_in_ms`, `deep_sleep_duration_in_ms`, `rem_sleep_duration_in_ms` | Convert seconds to milliseconds and aggregate stage durations by night. |
            | `stageSegments[].startedAt`, `durationSeconds`, `stageCode` | `local_time`, `duration_in_ms`, `type`, linked by `sleep_summary_id` | Flatten stage intervals and map provider codes to canonical sleep-stage values. |
            | `nightlyRespiration.meanBreathsPerMinute`, `stdBreathsPerMinute` | `respiratory_rate_mean`, `respiratory_rate_std`, indexed by `calendar_date` | Expose the available nightly summary without inventing respiration samples. |
            | `sleepHealthScore` | `overall_sleep_score` and `wyvern_sleep_health_score` | Make the score available in sleep views while retaining its provider-native feature name. |
            | `dailyStrain_score` | `wyvern_daily_strain_score` | Retain the fictional provider prefix and native daily grain. |

            A `BaseLoader` subclass implements the required sleep and
            heart-rate methods. Optional methods are added only for streams
            present in the export. Capability attributes tell pywearable which
            clocks and source grains are available.
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
            class WyvernLoader(BaseLoader):
                '''Adapt participant-folder exports from the fictional Wyvern Band.'''

                activity_source = "samples"
                activity_primary_resolution = "15min"
                activity_time_columns = (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL)
                heart_rate_time_columns = (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL)
                respiratory_rate_source = "provider_summary"
                provider_features = frozenset(
                    {WYVERN_SLEEP_HEALTH_SCORE, WYVERN_DAILY_STRAIN_SCORE}
                )
                provider_feature_definitions = {
                    WYVERN_SLEEP_HEALTH_SCORE: {"grain": "daily", "numeric": True},
                    WYVERN_DAILY_STRAIN_SCORE: {"grain": "daily", "numeric": True},
                }

                _stage_codes = {
                    "AWK": schema.SLEEP_STAGE_AWAKE_STAGE_VALUE,
                    "LT": schema.SLEEP_STAGE_LIGHT_STAGE_VALUE,
                    "DP": schema.SLEEP_STAGE_DEEP_STAGE_VALUE,
                    "REM": schema.SLEEP_STAGE_REM_STAGE_VALUE,
                    "UNM": schema.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE,
                    "ASL": "asleep",
                }

                def __init__(self, data_root):
                    '''Read every participant directory into source-shaped tables.

                    Each ``demo-*`` directory is self-contained. TSV, CSV, and
                    JSON records are concatenated here, but no canonical
                    conversion occurs until a ``load_*`` method is called.
                    '''
                    data_root = Path(data_root)
                    participant_dirs = sorted(
                        path for path in data_root.glob("demo-*") if path.is_dir()
                    )
                    self._participants = pd.concat(
                        [
                            pd.read_csv(path / "participant.tsv", sep="\t")
                            for path in participant_dirs
                        ],
                        ignore_index=True,
                    )
                    self._heart_rate = pd.concat(
                        [
                            pd.read_csv(path / "pulse.tsv", sep="\t")
                            for path in participant_dirs
                        ],
                        ignore_index=True,
                    )
                    self._activity = pd.concat(
                        [pd.read_csv(path / "movement.csv") for path in participant_dirs],
                        ignore_index=True,
                    )

                    sleep_nights = [
                        night
                        for path in participant_dirs
                        for night in json.loads(
                            (path / "sleep.json").read_text()
                        )["nights"]
                    ]
                    self._sleep = pd.json_normalize(sleep_nights)
                    stage_rows = []
                    for night in sleep_nights:
                        for segment in night["stageSegments"]:
                            stage_rows.append(
                                {
                                    "memberCode": night["memberCode"],
                                    "nightKey": night["nightKey"],
                                    **segment,
                                }
                            )
                    self._stages = pd.DataFrame(stage_rows)

                    self._scores = pd.DataFrame(
                        [
                            record
                            for path in participant_dirs
                            for record in json.loads(
                                (path / "daily_scores.json").read_text()
                            )["records"]
                        ]
                    )

                def get_user_ids(self):
                    '''Return participant identifiers read from ``participant.tsv``.'''
                    return sorted(self._participants["memberCode"].unique())

                @staticmethod
                def _filter(frame, column, start_date, end_date):
                    '''Apply half-open ``[start_date, end_date)`` canonical bounds.'''
                    values = pd.to_datetime(frame[column])
                    mask = pd.Series(True, index=frame.index)
                    if start_date is not None:
                        mask &= values >= pd.Timestamp(start_date)
                    if end_date is not None:
                        mask &= values < pd.Timestamp(end_date)
                    return frame.loc[mask].copy()

                @staticmethod
                def _utc_from_local(local_time, offset_minutes):
                    '''Subtract provider UTC offsets from naive local timestamps.'''
                    return pd.to_datetime(local_time) - pd.to_timedelta(
                        offset_minutes, unit="m"
                    )

                def load_heart_rate(
                    self,
                    user_id,
                    start_date=None,
                    end_date=None,
                    time_col=schema.LOCAL_TIME_COL,
                ):
                    '''Convert ``pulse.tsv`` rows to canonical heart-rate samples.

                    ``capturedAtLocal`` becomes ``local_time``; the per-row
                    minute offset supplies both ``timezone`` and ``utc_time``;
                    ``pulseBpm`` becomes numeric ``heart_rate``. Filtering uses
                    the caller-selected canonical clock.
                    '''
                    rows = self._heart_rate[
                        self._heart_rate["memberCode"].eq(user_id)
                    ].copy()
                    local_time = pd.to_datetime(rows["capturedAtLocal"])
                    offsets = pd.to_timedelta(rows["utcOffsetMinutes"], unit="m")
                    canonical = pd.DataFrame(
                        {
                            schema.UTC_TIME_COL: self._utc_from_local(
                                local_time, rows["utcOffsetMinutes"]
                            ),
                            schema.LOCAL_TIME_COL: local_time,
                            schema.TIMEZONE_COL: offsets,
                            schema.HEART_RATE_COL: rows["pulseBpm"],
                        }
                    )
                    canonical = schema.prepare_heart_rate_samples(canonical)
                    return self._filter(canonical, time_col, start_date, end_date)

                def load_activity_samples(self, user_id, start_date=None, end_date=None, **kwargs):
                    '''Convert ``movement.csv`` intervals to canonical activity samples.

                    The participant metadata supplies the UTC offset missing
                    from each movement row. ``intervalSeconds`` is converted to
                    ``duration_in_ms`` and ``stepCount`` to ``activity_steps``;
                    observed zero-step intervals remain observations, not nulls.
                    '''
                    rows = self._activity[
                        self._activity["member_code"].eq(user_id)
                    ].copy()
                    local_time = pd.to_datetime(rows["intervalStart"])
                    offset_minutes = self._participants.set_index("memberCode").loc[
                        user_id, "utcOffsetMinutes"
                    ]
                    canonical = pd.DataFrame(
                        {
                            schema.UTC_TIME_COL: self._utc_from_local(
                                local_time, offset_minutes
                            ),
                            schema.LOCAL_TIME_COL: local_time,
                            schema.TIMEZONE_COL: pd.to_timedelta(
                                offset_minutes, unit="m"
                            ),
                            schema.DURATION_IN_MS_COL: rows["intervalSeconds"] * 1_000,
                            schema.ACTIVITY_STEPS_COL: rows["stepCount"],
                        }
                    )
                    canonical = schema.prepare_activity_samples(canonical)
                    return self._filter(
                        canonical, schema.LOCAL_TIME_COL, start_date, end_date
                    )

                def load_sleep_summary(self, user_id, start_date=None, end_date=None, drop_duplicates=True):
                    '''Convert nightly JSON records to canonical sleep summaries.

                    Provider session identifiers, calendar labels, and local
                    bounds are retained. Durations are converted from seconds
                    to milliseconds, flattened stage intervals are totaled by
                    type, and ``sleepHealthScore`` is joined as canonical
                    ``overall_sleep_score`` for sleep-specific displays.
                    '''
                    rows = self._sleep[self._sleep["memberCode"].eq(user_id)].copy()
                    score_rows = self._scores[
                        self._scores["memberCode"].eq(user_id)
                    ][["memberCode", "day", "sleepHealthScore"]]
                    rows = rows.merge(
                        score_rows,
                        left_on=["memberCode", "sleepDate"],
                        right_on=["memberCode", "day"],
                        how="left",
                    )
                    stage_totals = (
                        self._stages[self._stages["memberCode"].eq(user_id)]
                        .groupby(["nightKey", "stageCode"])["durationSeconds"]
                        .sum()
                        .unstack(fill_value=0)
                    )
                    rows = rows.merge(stage_totals, on="nightKey", how="left")

                    def stage_duration(code):
                        return rows.get(code, pd.Series(0, index=rows.index)) * 1_000

                    canonical = pd.DataFrame(
                        {
                            schema.SLEEP_SUMMARY_ID_COL: rows["nightKey"],
                            schema.CALENDAR_DATE_COL: rows["sleepDate"],
                            schema.LOCAL_TIME_COL: rows["startedAt"],
                            schema.LOCAL_END_TIME_COL: rows["endedAt"],
                            schema.DURATION_IN_MS_COL: rows["sleepSeconds"] * 1_000,
                            schema.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL: rows["awakeSeconds"] * 1_000,
                            schema.SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL: rows["unmeasurableSeconds"] * 1_000,
                            schema.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL: stage_duration("LT"),
                            schema.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL: stage_duration("DP"),
                            schema.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL: stage_duration("REM"),
                            schema.SLEEP_SUMMARY_OVERALL_SLEEP_SCORE_COL: rows["sleepHealthScore"],
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
                    '''Convert flattened stage segments to canonical intervals.

                    ``startedAt`` becomes ``local_time``, seconds become
                    ``duration_in_ms``, and Wyvern stage codes map to canonical
                    ``type`` values. ``nightKey`` preserves the required link
                    to ``sleep_summary_id``. Bounds use interval overlap.
                    '''
                    rows = self._stages[
                        self._stages["memberCode"].eq(user_id)
                    ].copy()
                    canonical = pd.DataFrame(
                        {
                            schema.SLEEP_SUMMARY_ID_COL: rows["nightKey"],
                            schema.LOCAL_TIME_COL: rows["startedAt"],
                            schema.DURATION_IN_MS_COL: rows["durationSeconds"] * 1_000,
                            schema.SLEEP_STAGE_SLEEP_TYPE_COL: rows["stageCode"].map(
                                self._stage_codes
                            ),
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

                def load_respiration_summary(
                    self, user_id, start_date=None, end_date=None
                ):
                    '''Expose the nightly respiration summary stored in ``sleep.json``.

                    The two provider fields map directly to canonical
                    ``respiratory_rate_mean`` and ``respiratory_rate_std`` at
                    ``calendar_date`` grain. No sample-level stream is created.
                    '''
                    rows = self._sleep[self._sleep["memberCode"].eq(user_id)].copy()
                    canonical = pd.DataFrame(
                        {
                            schema.CALENDAR_DATE_COL: rows["sleepDate"],
                            schema.RESPIRATORY_RATE_MEAN_COL: rows[
                                "nightlyRespiration.meanBreathsPerMinute"
                            ],
                            schema.RESPIRATORY_RATE_STD_COL: rows[
                                "nightlyRespiration.stdBreathsPerMinute"
                            ],
                        }
                    )
                    canonical = schema.prepare_respiration_summary(canonical)
                    return self._filter(
                        canonical, schema.CALENDAR_DATE_COL, start_date, end_date
                    ).reset_index(drop=True)

                def load_provider_feature(
                    self, feature, user_id, start_date=None, end_date=None, **kwargs
                ):
                    '''Convert a named daily score to a canonical provider feature.

                    ``daily_scores.json`` uses ``day`` plus two differently
                    styled score names. The requested source column is renamed
                    to its registered Wyvern feature and returned at canonical
                    ``calendar_date`` grain, preserving missing score values.
                    '''
                    source_columns = {
                        WYVERN_SLEEP_HEALTH_SCORE: "sleepHealthScore",
                        WYVERN_DAILY_STRAIN_SCORE: "dailyStrain_score",
                    }
                    if feature not in source_columns:
                        raise ValueError(f"WyvernLoader does not provide {feature!r}")
                    rows = self._scores[self._scores["memberCode"].eq(user_id)]
                    canonical = pd.DataFrame(
                        {
                            schema.CALENDAR_DATE_COL: rows["day"],
                            feature: rows[source_columns[feature]],
                        }
                    )
                    canonical = schema.prepare_provider_feature_summary(
                        canonical, feature
                    )
                    return self._filter(
                        canonical, schema.CALENDAR_DATE_COL, start_date, end_date
                    ).reset_index(drop=True)


            loader = WyvernLoader(DATA_ROOT)
            print("Participants:", loader.get_user_ids())
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
            participant = loader.get_user_ids()[0]
            hr = loader.load_heart_rate(
                participant, START, START + pd.Timedelta(hours=1)
            )
            activity_samples = loader.load_activity_samples(
                participant, START, START + pd.Timedelta(hours=1)
            )
            summaries = loader.load_sleep_summary(participant, START, END)
            stage_intervals = loader.load_sleep_stage(participant, START, END)
            respiration_summary = loader.load_respiration_summary(
                participant, START, END
            )

            assert set(schema.HEART_RATE_REQUIRED_COLUMNS) <= set(hr)
            assert set(schema.ACTIVITY_SAMPLE_REQUIRED_COLUMNS) <= set(activity_samples)
            assert set(schema.SLEEP_SUMMARY_REQUIRED_COLUMNS) <= set(summaries)
            assert set(schema.SLEEP_STAGE_REQUIRED_COLUMNS) <= set(stage_intervals)
            assert {
                schema.CALENDAR_DATE_COL,
                schema.RESPIRATORY_RATE_MEAN_COL,
            } <= set(respiration_summary)
            assert loader.load_respiration(participant, START, END).empty
            assert loader.load_heart_rate(
                participant, END, END + pd.Timedelta(days=1)
            ).empty
            assert provider_features.get_available_provider_features(loader) == [
                WYVERN_DAILY_STRAIN_SCORE,
                WYVERN_SLEEP_HEALTH_SCORE,
            ]

            display(hr.head(3), activity_samples.head(3), summaries.head(3))
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
                user_id="all",
                start_date=START,
                end_date=END,
                resolution="daily",
            )
            daily_hr = heart_rate.get_heart_rate_statistics(
                loader,
                statistic=[
                    heart_rate.HEART_RATE_STATISTIC_MEAN,
                    heart_rate.HEART_RATE_STATISTIC_REST,
                ],
                user_id=participant,
                start_date=START,
                end_date=END,
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
                start_date=START,
                end_date=END,
                resolution="nightly",
            )
            nightly_respiration = respiration.get_respiration_statistics(
                loader,
                statistic=[
                    respiration.RESPIRATORY_RATE_STATISTIC_MEAN,
                    respiration.RESPIRATORY_RATE_STATISTIC_STD,
                ],
                user_id=participant,
                start_date=START,
                end_date=END,
                resolution="cycle_night",
            )
            daily_scores = provider_features.get_provider_features(
                loader,
                feature=[WYVERN_SLEEP_HEALTH_SCORE, WYVERN_DAILY_STRAIN_SCORE],
                user_id="all",
                start_date=START,
                end_date=END,
            )

            display(
                daily_steps.head(6),
                daily_hr.head(3),
                nightly_sleep.head(3),
                nightly_respiration.head(3),
                daily_scores.head(6),
            )
            """
        ),
        markdown(
            """
            ## Visualize one integrated week

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
                START,
                END,
                title="Wyvern mock export: one integrated week",
                show=False,
            )
            plt.show()
            """
        ),
        markdown(
            """
            ## Compare the provider-native scores

            These scores remain provider-specific: pywearable can align and
            aggregate them, but does not equate them with similarly named
            scores from another device. A sleep-health score is absent when
            no nightly sleep result is available; daytime strain can still be
            present when activity data exists.
            """
        ),
        code(
            """
            score_participant = "demo-02"
            score_axes = visualization.plot_trajectory(
                daily_scores,
                [WYVERN_SLEEP_HEALTH_SCORE, WYVERN_DAILY_STRAIN_SCORE],
                user_id=score_participant,
                title=f"Provider-native daily scores: {score_participant}",
                ylabel="Score (0-100)",
                show_imputed=False,
                show_coverage=True,
                show=False,
            )
            score_axes["trajectory"].set_ylim(0, 100)
            plt.show()
            """
        ),
        markdown(
            """
            ## Inspect nightly sleep

            The compact sleep view uses the canonical summaries and native
            stage intervals. Because the loader maps `sleepHealthScore` to
            canonical `overall_sleep_score`, it can also be displayed beside
            each night; unavailable nights and scores remain blank.
            """
        ),
        code(
            """
            sleep_participant = "demo-02"
            visualization.plot_sleep_summary(
                loader,
                sleep_participant,
                start_date=START,
                end_date=END - pd.Timedelta(days=1),
                show_scores=True,
                title=f"Seven sleep-calendar dates: {sleep_participant}",
                show=False,
            )
            plt.show()
            """
        ),
        markdown(
            """
            ## Inspect adherence across participants

            To inspect adherence during the study, aggregate heart-rate
            coverage into 10-minute windows and compare its local-time pattern
            for each participant.
            """
        ),
        code(
            """
            ten_minute_heart_rate = heart_rate.get_heart_rate_statistics(
                loader,
                statistic=heart_rate.HEART_RATE_STATISTIC_MEAN,
                user_id="all",
                start_date=START,
                end_date=END,
                resolution="10min",
            )

            participants = loader.get_user_ids()
            fig, axes = plt.subplots(
                len(participants), 1, figsize=(12, 8), constrained_layout=True
            )
            for user_id, axis in zip(participants, axes):
                visualization.plot_coverage_heatmap(
                    ten_minute_heart_rate,
                    heart_rate.HEART_RATE_STATISTIC_MEAN,
                    user_id=user_id,
                    interval="10min",
                    title=f"{user_id}: 10-minute heart-rate coverage",
                    ax=axis,
                    show=False,
                )
            plt.show()
            """
        ),
        markdown(
            """
            ## What the adapter is responsible for

            - Preserve the provider's native resolution and units.
            - Expose UTC and local clocks only when they are known.
            - Apply `[start_date, end_date)` consistently.
            - Keep observed zero distinct from missing data.
            - Return canonical empty frames for supported but empty intervals.
            - Raise a clear error for unsupported provider-native features.
            - Link every sleep-stage interval to a `sleep_summary_id`.
            - Test duplicate records, timezone changes, empty intervals, and
              one end-to-end domain function for every implemented stream.

            Once those responsibilities are met, downstream analysis remains
            independent of the original source structure.
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
