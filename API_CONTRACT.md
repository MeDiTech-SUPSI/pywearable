# Proposed public API contract

This document inventories the Python API proposed for pywearable's first
public release. It is a review document, not a promise that every currently
importable name is public. The proposed classifications are encoded in each
curated module's `__all__` and protected by a public-API smoke test; both can
still be adjusted during review.

Sphyncs remains in the repository because it supports a private dataset, but it
is outside the first-release contract.

## Stability labels

- **Public**: documented and supported as part of the first-release API.
- **Extension**: public contract for loader authors and advanced integrations.
- **Compatibility**: retained for existing users, but the listed public API is
  preferred. It may be deprecated only through an announced migration.
- **Experimental**: importable but outside the first-release compatibility
  promise.
- **Internal**: implementation detail; no compatibility promise.

## Shared trajectory contract

Unless a row below says otherwise, analytical functions return a
`pandas.DataFrame` with a two-level `(user_id, time)` index. The time level is
named `utc_time`, `local_time`, or `calendar_date` according to the selected
resolution and clock. Feature columns are accompanied by
`<feature>__coverage` columns. Hierarchical outputs encode their calculation
path in the feature name.

Date bounds are half-open: `[start_date, end_date)`. Calendar and sleep-cycle
windows use participant-local labels. Fixed subdaily windows use UTC when the
loader provides it and `time_reference="auto"`; callers can request local time
where supported. Selecting cycles by date does not clip their physical
boundaries.

Missing means unavailable or invalid, while numeric zero remains an observed
value. Coverage is always kept separate from the value. At parent hierarchy
levels, valid children have equal weight and coverage is the fraction of
expected children that are valid. Domain-specific primary coverage is defined
in the owning function's docstring.

## API matrix

| Module and symbols | Status | Return, index, and required input | Bounds | Coverage and missingness | Owner | Documentation and tests |
| --- | --- | --- | --- | --- | --- | --- |
| `activity.get_activity_statistics` and activity feature identifiers | Public | Canonical trajectory; requires a loader exposing activity samples or daily summaries | Shared contract | Sample coverage combines valid heart-rate occupancy and positive-step intervals; summary coverage is feature availability | `activity` | Current docstring; direct tests |
| `activity.get_activity_statistic`, `activity.get_daily_activity_statistics` | Public | Permanent convenience entry points returning the same trajectory as the canonical function | Forwarded unchanged | Forwarded unchanged | `activity` | Short docstrings; wrapper tests are indirect |
| `adherence.get_adherence` and adherence metric identifiers | Public | Canonical trajectory with `<metric>_adherence`; explicit start and end are required | Shared contract; sleep labels are local | Values are in `[0, 1]`; missing denotes unsupported/inapplicable streams, while supported empty streams can be zero | `adherence` | Current docstring; extensive direct tests |
| `adherence.get_daily_wearing_time` | Compatibility | `Series` indexed by local date, containing hours | Half-open local bounds | Derived from daily heart-rate adherence | `adherence` | Current docstring; direct compatibility tests |
| `cycle.get_intervals` and cycle identifiers | Public | `DataFrame` indexed by `(user_id, calendar_date)` with `start_time`, `end_time`, and validity columns | Bounds select labels and do not clip intervals | Invalid or incomplete cycles remain explicit; `missing_end` controls the documented fallback | `cycle` | Current docstring; direct tests |
| `heart_rate.get_heart_rate_statistics` and heart-rate feature identifiers | Public | Canonical trajectory from canonical heart-rate samples | Shared contract | Primary coverage is inferred cadence occupancy; unknown coverage is retained unless a threshold is requested | `heart_rate` | Current docstring; extensive direct tests |
| `heart_rate.get_*_heart_rate_statistics`, `heart_rate.get_daily_{rest,max,min,mean}_heart_rate` | Public | Permanent convenience entry points returning the same trajectory as the canonical function | Forwarded unchanged | Forwarded unchanged | `heart_rate` | Mostly documented and directly tested; yearly wrapper lacks a docstring |
| `hrv.compute_rmssd`, `compute_sdnn`, `compute_pnn20`, `compute_pnn50`, `compute_hrv_statistics` | Public | Scalar computations from BBI values; `compute_hrv_statistics` returns a statistic mapping | Not applicable | Invalid BBI values and discontinuities follow the function contracts | `hrv` | Current docstrings; direct numerical tests |
| `hrv.get_hrv_statistics` and HRV feature identifiers | Public | Canonical trajectory from raw BBI or provider HRV data | Shared contract | Raw coverage is represented BBI duration; provider coverage remains unknown when absent and is never invented | `hrv` | Current docstring; extensive direct tests |
| `hrv.get_hrv_statistic`, resolution helpers, and named nightly helpers | Public | Permanent convenience entry points; DataFrame helpers return canonical trajectories and named nightly helpers return a `Series` | Forwarded unchanged | Forwarded unchanged | `hrv` | Current short docstrings; direct wrapper tests |
| `respiration.get_respiration_statistics` and respiratory/SpO2 feature identifiers | Public | Canonical trajectory from samples or provider nightly summaries | Shared contract | Sample cadence coverage or provider-supplied coverage; unsupported summary features remain missing | `respiration` | Current docstring; extensive direct tests |
| `respiration.get_respiration_statistic`, `get_five_minute_respiration_statistics`, `get_night_respiration_statistics`, `get_awake_respiration_statistics` | Public | Permanent convenience entry points returning the same trajectory as the canonical function | Forwarded unchanged | Forwarded unchanged | `respiration` | Current short docstrings; direct wrapper tests |
| `sleep.get_sleep_statistics` and sleep feature identifiers | Public | Canonical nightly or calendar hierarchy trajectory from main-sleep summaries and stages | Half-open local calendar-date bounds | Nightly coverage is binary feature availability; parent coverage is valid expected nights | `sleep` | Core docstring and extensive direct tests |
| `sleep.get_sleep_wake_trajectory`, `sleep.get_sleep_regularity` | Public | Sleep/wake trajectory indexed by `(user_id, local_time)`; regularity is indexed by `(user_id, calendar_date)` | Half-open local bounds; regularity reads the preceding day for comparison | Unknown stages and missing nights remain missing; SRI exposes comparison coverage | `sleep` | Current docstrings; direct tests |
| `sleep.get_sleep_statistic`, calendar-resolution helpers, named sleep-feature getters, `get_sleep_regularity_index` | Public | Permanent convenience entry points returning a canonical feature trajectory, except `get_sleep_regularity_index`, which returns a `Series` | Forwarded unchanged | Forwarded unchanged | `sleep` | Direct and indirect tests; eight named getters lack docstrings |
| `provider_features.get_available_provider_features`, `get_provider_features`, and provider feature identifiers | Public | Available names as `list[str]`; extraction as canonical trajectory; requested features must share a native grain | Shared contract | Coverage is sample/day/child availability; vendor metrics are not equated across providers | `provider_features` | Current docstrings; direct tests across providers |
| `trajectory.impute_trajectory`, `compute_trajectory_trend`, `aggregate_trajectory_by_weekday` | Public | Copy of a canonical trajectory, or weekday trajectory indexed by `(user_id, weekday)`; requires matching coverage columns | Operates on input rows | Coverage is never overwritten; imputation is flagged separately and excluded from trend inputs by default | `trajectory` | Current docstrings; extensive direct tests |
| All names in `visualization.__all__` | Public | Matplotlib `Axes`, axis mappings, or arrays; generic functions require canonical trajectories and composite functions require a loader | Loader-aware plots document their own half-open bounds | Missing, zero, observed, and imputed values remain visually distinct | `visualization` | Current docstrings; direct behavioral tests |
| Canonical schema constants and `schema.prepare_*` functions | Extension | Canonical loader-facing `DataFrame` schemas; required columns depend on stream | Preparers normalize rows but do not define an analysis interval | Invalid provider values become missing or rows are rejected as documented | `schema` | Function docstrings and direct schema/loader tests; no curated `__all__` yet |
| `loader.BaseLoader` portable methods and capability attributes | Extension | Canonical stream DataFrames; see schema contract; `resolve_user_ids` normalizes participant selection | Loader methods target half-open bounds on the selected clock | Empty supported streams return typed empty frames; unavailable capabilities are advertised by attributes | `loader.base` | Contract is partly documented and tested; several abstract/legacy methods need docstrings |
| `loader.LabfrontLoader`, `LifeSnapsParquetLoader`, `OuraLoader`, `TesseraeCSVLoader`, `TesseraeParquetLoader` | Public | Implement the portable `BaseLoader` contract; provider-only methods are compatibility conveniences | Target shared loader bounds | Translate provider missingness and conflicts into canonical schemas | Provider loader modules | Direct loader tests; class/method documentation is uneven |
| `loader.garmin.prepare_epoch_activity_samples` | Extension | Canonical activity-sample `DataFrame` from Garmin epochs | Uses input rows | Collapses categories without double-counting duration; conflicts become missing | `loader.garmin` | Short docstring; direct activity tests |
| `hierarchy.*`, `utils.time.*`, underscored names, and `sleep.compute_sleep_validation` | Internal | Implementation-specific | Internal | Internal | Respective modules | Unit-tested where useful; not user-facing documentation targets |
| `loader.SphyncsDataLoader` | Experimental | Private-dataset loader | Loader-specific | Loader-specific | `loader.sphyncs` | Retained and partly tested; excluded from first release |
| `loader.LifeSnapsMongoDBLoader` | Internal | Unimplemented placeholder | Not applicable | Not applicable | `loader.lifesnaps` | No implementation, docs, or tests; should not be exported |
| `pywearable` console command and conversion subcommands | Pending decision | Filesystem conversion commands | Command-specific | Command-specific | `cli` | A maintainer decision is required; public documentation and tests are currently absent |

## Exact symbol groups

The grouped rows above mean the following names are intentional parts of the
proposed contract.

### Feature and cycle identifiers

- Activity: `ACTIVITY_STATISTIC_STEPS`,
  `ACTIVITY_STATISTIC_PEAK_60MIN_STEPS`,
  `ACTIVITY_STATISTIC_PEAK_60MIN_START`, `ACTIVITY_STATISTICS`.
- Adherence: `ADHERENCE_HEART_RATE`, `ADHERENCE_SLEEP`, `ADHERENCE_BBI`,
  `ADHERENCE_METRICS`.
- Cycles: `CYCLE_NIGHT`, `CYCLE_AWAKE`, `CYCLE_PHYSIOLOGICAL`, `CYCLE_TYPES`.
- Heart rate: `HEART_RATE_STATISTIC_COUNT`, `HEART_RATE_STATISTIC_MAX`,
  `HEART_RATE_STATISTIC_MEAN`, `HEART_RATE_STATISTIC_MEDIAN`,
  `HEART_RATE_STATISTIC_MIN`, `HEART_RATE_STATISTIC_P10`,
  `HEART_RATE_STATISTIC_P90`, `HEART_RATE_STATISTIC_STD`,
  `HEART_RATE_STATISTIC_REST`, `HEART_RATE_STATISTICS`.
- HRV: `HRV_STATISTIC_ROOT_MEAN_SQUARED_SUCCESSIVE_DIFFERENCES`,
  `HRV_STATISTIC_STANDARD_DEVIATION_NORMAL_TO_NORMAL`, `HRV_STATISTIC_PNN20`,
  `HRV_STATISTIC_PNN50`, `HRV_STATISTIC_VERY_LOW_FREQUENCY_POWER`,
  `HRV_STATISTIC_LOW_FREQUENCY_POWER`, `HRV_STATISTIC_HIGH_FREQUENCY_POWER`,
  `HRV_STATISTIC_LOW_HIGH_FREQUENCY_RATIO`, `HRV_STATISTIC_BBI_MEDIAN`,
  `HRV_TIME_DOMAIN_STATISTICS`, `HRV_FREQUENCY_DOMAIN_STATISTICS`,
  `HRV_STATISTICS`.
- Respiration: all `RESPIRATORY_RATE_STATISTIC_*` and `SPO2_STATISTIC_*`
  identifiers, plus `RESPIRATORY_RATE_STATISTICS`, `SPO2_STATISTICS`, and
  `RESPIRATION_STATISTICS`.
- Sleep: all `SLEEP_STATISTIC_*` identifiers, plus
  `SLEEP_TEMPORAL_STATISTICS`, `SLEEP_NIGHTLY_RESOLUTIONS`,
  `SLEEP_WAKE_TRAJECTORY_COLUMNS`, and `SLEEP_STATISTICS_DICT`.
- Provider features: all non-underscored `GARMIN_*`, `FITBIT_*`, and `OURA_*`
  feature identifiers, plus `PROVIDER_FEATURES`.

### Named sleep convenience getters

`get_time_in_bed`, `get_sleep_period_time`, `get_total_sleep_time`,
`get_sleep_efficiency`, `get_sleep_maintenance_efficiency`,
`get_deep_latency`, `get_light_latency`, `get_rem_latency`,
`get_light_sleep_duration`, `get_deep_sleep_duration`, `get_rem_duration`,
`get_nrem_duration`, `get_awake_duration`, `get_unmeasurable_duration`,
`get_light_sleep_percentage`, `get_deep_sleep_percentage`,
`get_rem_percentage`, `get_nrem_percentage`, `get_wake_after_sleep_onset`,
`get_sleep_onset_latency`, `get_sleep_score`, `get_sleep_timestamps`,
`get_bedtime`, `get_wakeup_time`, `get_sleep_midpoint`, `get_awake_count`,
`get_light_sleep_count`, `get_deep_sleep_count`, `get_rem_count`,
`get_cpd_midpoint`, `get_awakening_count`, `get_awakening_index`,
`get_median_awakening_duration`, `get_stage_transition_count`,
`get_stage_transition_index`, `get_stage_coverage`, `get_sleep_onset`,
`get_getup_time`, `get_getup_latency`, and `get_cpd_duration`.

### Portable loader interface

The stable cross-provider interface is the capability attributes and methods
declared by `BaseLoader`: `get_user_ids`, `resolve_user_ids`, `load_sleep_summary`,
`load_sleep_stage`, `load_sleep_sessions`, `load_sleep_session_stages`,
`load_heart_rate`, `load_bbi`, `load_hrv_samples`, `load_hrv_summary`,
`load_activity_samples`, `load_activity_summary`, `load_respiration`,
`load_respiration_summary`, `load_pulse_ox`, `load_spo2_summary`,
`load_sleep_respiration`, `load_sleep_pulse_ox`, and
`load_provider_feature`.

`BaseLoader.load_stress` and `load_epoch` are retained as compatibility
methods. They expose provider-specific shapes and are not part of the portable
cross-provider interface.

Provider-specific discovery, questionnaire, survey, demographic, raw-table,
and conversion methods remain available on their loaders but are compatibility
conveniences rather than portable cross-provider API.

### Canonical schema extension interface

All non-underscored uppercase constants in `schema` are canonical field,
required-column, or categorical-value definitions. The public preparation
functions are `prepare_activity_samples`, `prepare_activity_summary`,
`prepare_bbi`, `prepare_heart_rate_samples`, `prepare_provider_feature_samples`,
`prepare_provider_feature_summary`, `prepare_respiration_samples`,
`prepare_respiration_summary`, `prepare_spo2_samples`, `prepare_spo2_summary`,
`prepare_sleep_summary`, `prepare_sleep_stages`, and
`prepare_sleep_sessions`.

`empty_heart_rate_frame`, `empty_bbi_frame`, `empty_respiration_frame`,
`empty_spo2_frame`, `empty_activity_frame`, `empty_activity_summary`,
`validate_heart_rate_time_col`, `normalize_time_bound`,
and `filter_heart_rate_by_time` are loader-building helpers. They are extension
API because custom loaders need the same normalization behavior. Provider
field names must be translated to canonical names before calling a schema
preparer.

Sleep-stage run coalescing is internal. `prepare_sleep_stages` already merges
exactly adjacent, equal-stage intervals within one sleep session, so loader
authors receive the behavior without depending on the lower-level helper.

## Export decisions applied

- `sleep`, `heart_rate`, `schema`, and `loader` now define curated
  `__all__` lists matching this inventory.
- Feature identifiers already used by callers are included in the `hrv`,
  `respiration`, and `provider_features` export lists.
- `pywearable.loader` exports the five first-release loader implementations.
  The private Sphyncs loader remains available from
  `pywearable.loader.sphyncs`, while the unimplemented MongoDB placeholder is
  no longer advertised. Their historical package-level imports still resolve,
  but neither name belongs to `pywearable.loader.__all__`.
- `pywearable.utils` now restricts wildcard exports to the named legacy
  compatibility helpers instead of also exposing imported dependencies.
- The root `pywearable` package is intentionally not a flat namespace. The
  supported style is module-qualified, such as
  `from pywearable import sleep` followed by `sleep.get_sleep_statistics(...)`.

## Work required to freeze the contract

1. Resolve whether the command-line interface belongs to the first release.
2. Adjust the candidate export lists if that review changes a classification.
3. Fill the identified public docstring gaps and add focused tests where the
   matrix currently says coverage is only indirect.
4. Use the frozen inventory as the input to the later ownership, constants,
   structure, and documentation passes. File moves must preserve public import
   paths with compatibility re-exports where necessary.

Before release, the loader documentation must include a small guide showing
how to implement a project- or cohort-specific `BaseLoader`, declare stream
capabilities, return canonical schemas, and validate them with the public
`schema.prepare_*` helpers.
