# pywearable loader implementation instructions

Follow these instructions when adding a data loader for a previously unseen
source dataset. They define the discovery, implementation, and validation
contract for the task.

The current code is authoritative. If this guide and the repository disagree,
inspect the implementation and tests, report the discrepancy, and follow the
current code unless an API change is explicitly requested.

## Sources of truth to read first

Before inspecting provider data or editing code, read:

1. Repository-level and directory-level `AGENTS.md` files, when present.
2. [`BaseLoader`](../src/pywearable/loader/base.py), including its capability
   attributes, abstract methods, defaults, and date-bound semantics.
3. The canonical schemas and preparers in
   [`schema.py`](../src/pywearable/schema.py).
4. One existing loader that most closely resembles the source format or
   provider. Use it as a contract example, not as code to copy blindly.
5. The domain module and tests for every stream being implemented.
6. [`provider_features.py`](../src/pywearable/provider_features.py) when the
   source contains provider-native features.

Do not start by changing a domain module. A new provider should normally be
implemented entirely at the loader boundary.

## Operating rules

- Treat raw provider data as read-only.
- Do not infer units, timezone semantics, sleep-date conventions, stage codes,
  or missing-value meanings from a plausible column name.
- Distinguish observed zero from missing data.
- Preserve native observations and resolution. Do not interpolate, expand, or
  fabricate samples in a loader.
- Translate source representation only. Aggregation and derived features
  belong in pywearable domain modules.
- Keep changes surgical. Do not refactor unrelated code or modify public APIs
  unless that expansion is explicitly approved.
- Do not commit or push without explicit authorization.
- Stop for clarification only when an unresolved choice would materially change
  the interpretation or public behavior of the loader.

## Required workflow

### Phase 1: inspect the source read-only

Determine, without editing the source:

- how participants and files are discovered;
- which streams exist and which are absent;
- file formats, nesting, naming conventions, and schema variants;
- identifier fields and whether identifiers are stable across streams;
- timestamp fields, units, timezone information, and daylight-saving behavior;
- source grain: event, fixed interval, daily summary, nightly summary, or
  provider-defined;
- value units and missing-value conventions;
- duplicate, revision, and conflict indicators;
- sleep-session identifiers, main-sleep indicators, sleep-calendar labels,
  and stage codes;
- whether summaries and samples represent the same underlying stream;
- whether provider-native scores have a documented range and favorable
  direction.

Inspect a small representative sample across multiple participants and dates.

### Phase 2: produce the discovery report

Before implementation, report:

```text
## Source layout
- Participant organization:
- File patterns and formats:
- Participant discovery rule:

## Available streams
| Stream | Source grain | Time fields | Units | Status |

## Proposed mappings
| Source fields | Canonical fields | Conversion | Filtering clock |

## Material ambiguities
- None, or a concise list of decisions requiring clarification.

## Planned files
- Loader module:
- Package export:
- Tests:
- Test fixtures, if required:
```

### Phase 3: record a source manifest

Create this manifest in the working notes or discovery response. Do not commit
it by default. Repeat the `streams` entry for every supported source.

```yaml
provider:
  name: <provider name>
  loader_class: <ProviderNameLoader>
  loader_module: <src/pywearable/loader/...>
  source_root: <source path>

participants:
  layout: <one directory per participant | shared files | other>
  discovery: <glob, index file, API call, or other rule>
  source_id_field: <field>

streams:
  <stream_name>:
    support: <supported | unsupported | unresolved>
    file_pattern: <pattern>
    format: <csv | tsv | json | parquet | other>
    source_grain: <event | fixed interval | daily | nightly | other>
    source_fields:
      participant_id: <field or null>
      timestamp: <field or null>
      end_timestamp: <field or null>
      duration: <field or null>
      timezone: <field or null>
      value: <field or fields>
    source_units:
      timestamp: <unit and epoch, or parsed datetime>
      duration: <unit or null>
      value: <unit or units>
    time_semantics:
      clock: <utc | local | both | date label>
      timezone_representation: <offset | zone name | absent>
      bounds: <source selection semantics>
    canonical:
      method: <load_* method>
      preparer: <schema.prepare_* function>
      columns: [<canonical columns>]
    missing_values: <source sentinels and interpretation>
    duplicate_policy: <documented rule>
    unresolved: []
```

The manifest is a decision record. Every implemented conversion should be
traceable to it.

## pywearable loader contract

### Required interface

Every concrete `BaseLoader` subclass implements:

- `get_user_ids()`;
- `load_heart_rate()`;
- `load_sleep_summary()`;
- `load_sleep_stage()`.

When the source does not support one of the required streams, return the
corresponding typed canonical empty frame rather than inventing data.
Optional streams should retain the `BaseLoader` default unless the source
actually provides them.

### Date and time behavior

- Date bounds are half-open: `[start_date, end_date)`.
- Heart-rate bounds apply to the requested `time_col`.
- Sleep-summary bounds select participant-local `calendar_date`; they do not
  clip the physical sleep session.
- When `drop_duplicates=True`, sleep summaries contain at most one main-sleep
  row per participant-local calendar date.
- Sleep-stage intervals retain native positive durations and reference a valid
  `sleep_summary_id`.
- For interval data, include an interval when it overlaps the requested
  physical range: its end is after `start_date` and its start is before
  `end_date`.
- Advertise only clocks the source genuinely supplies. Retain required
  canonical clock columns, using missing values when a clock cannot be known.
- Canonical `timezone` values are UTC offsets represented as
  `pandas.Timedelta`, not timezone names or strings.

Material uncertainty about UTC, local time, offsets, daylight-saving behavior,
or sleep-calendar assignment is a stop condition.

### Canonical preparers

Construct canonical columns explicitly, then call the relevant preparer:

| Stream | Loader method | Canonical preparer |
|---|---|---|
| Heart rate | `load_heart_rate` | `schema.prepare_heart_rate_samples` |
| Activity intervals | `load_activity_samples` | `schema.prepare_activity_samples` |
| Daily activity | `load_activity_summary` | `schema.prepare_activity_summary` |
| Respiration samples | `load_respiration` | `schema.prepare_respiration_samples` |
| Nightly respiration | `load_respiration_summary` | `schema.prepare_respiration_summary` |
| SpO2 samples | `load_pulse_ox` | `schema.prepare_spo2_samples` |
| Nightly SpO2 | `load_spo2_summary` | `schema.prepare_spo2_summary` |
| Beat-to-beat intervals | `load_bbi` | `schema.prepare_bbi` |
| Sleep summaries | `load_sleep_summary` | `schema.prepare_sleep_summary` |
| Sleep stages | `load_sleep_stage` | `schema.prepare_sleep_stages` |
| Provider samples | `load_provider_feature` | `schema.prepare_provider_feature_samples` |
| Provider daily summaries | `load_provider_feature` | `schema.prepare_provider_feature_summary` |

Read the required-column constants and preparer implementation in the current
`schema.py`; do not rely on memory or duplicate their validation logic.

### Capability attributes

Set only attributes supported by the source:

- `heart_rate_time_columns`;
- `activity_source`: `"samples"`, `"daily_summary"`, or `None`;
- `activity_time_columns` and `activity_primary_resolution`;
- `hrv_source`: `"bbi"` or `"provider_samples"`;
- `hrv_time_columns`;
- `respiratory_rate_source`: `"samples"` or `"provider_summary"`;
- `respiratory_rate_time_columns`;
- `spo2_source`: `"samples"` or `"provider_summary"`;
- `spo2_time_columns`;
- `provider_features`, `provider_feature_definitions`, and
  `provider_feature_time_columns`.

For a custom provider feature, declare:

```python
provider_features = frozenset({"provider_feature_name"})
provider_feature_definitions = {
    "provider_feature_name": {
        "grain": "samples",  # or "daily"
        "numeric": True,
    }
}
```

Use a stable provider prefix for names that are not shared canonical features.
Do not relabel a provider-specific score as a standardized metric unless its
semantics genuinely match the canonical field.

### Source-to-canonical conversion

For every implemented method:

1. Select one participant without exposing other participants' records.
2. Parse source fields without mutating cached source tables.
3. Convert timestamps, durations, and values explicitly.
4. Construct a DataFrame using `schema` constants for canonical column names.
5. Call the relevant `schema.prepare_*` function.
6. Apply documented half-open bounds to the correct canonical clock or date.
7. Sort deterministically and apply only the documented duplicate rule.
8. Return the prepared canonical empty frame when the selected interval has no
   rows.

Do not use a preparer to conceal an unresolved source interpretation. Parsing
and semantic conversion belong before preparation.

## Material stop conditions

Stop and request clarification before implementing the affected stream when
any of these cannot be established from data or documentation:

- seconds versus milliseconds or another unit;
- UTC versus participant-local timestamps;
- fixed offset versus timezone-aware local time;
- cumulative values versus interval increments;
- zero versus missing versus device-not-worn;
- main sleep versus naps or other sessions;
- the date to which an overnight session belongs;
- sleep-stage code meanings;
- revisions or duplicates that disagree;
- a score's scale, direction, or grain;
- whether two similarly named files represent the same stream.

Continue implementing unaffected streams while waiting when that work remains
valid under every interpretation.

## Testing requirements

Add focused tests for each supported method. Use the smallest fixtures needed
to exercise the source format and loader contract.

### Tests required for every stream

- the method returns all required canonical columns;
- canonical dtypes and sorting are correct after preparation;
- participant filtering cannot leak another participant's rows;
- `start_date` is inclusive and `end_date` is exclusive;
- an empty requested interval returns a typed canonical empty frame;
- source units are converted correctly;
- observed zero remains distinct from missing;
- the documented duplicate rule is enforced;
- the input tables are not mutated by repeated calls.

### Additional temporal tests

- UTC and local clocks differ by the declared offset;
- filtering by every advertised clock behaves correctly;
- unknown clocks remain missing and are not reconstructed without evidence;
- fixed-duration intervals preserve their native duration;
- intervals crossing a boundary are handled with the documented overlap rule.

### Additional sleep tests

- every stage references an available `sleep_summary_id`;
- stage codes map only to recognized canonical values;
- stage durations are positive and remain at native resolution;
- calendar-date filtering does not clip an overnight session;
- `drop_duplicates=True` leaves at most one main sleep per calendar date;
- missing stages or summaries are represented as missing, not fabricated.

### Integration tests

For each implemented domain, run at least one public pywearable function using
the loader. Examples include extracting daily steps, heart-rate statistics,
nightly sleep features, respiration statistics, or provider features. Add a
visualization smoke test only when the loader enables a visualization-specific
capability.

Run the focused loader tests first, then the complete test suite. Use the
repository's documented environment and commands. Report environment failures
separately from assertion failures.

## Definition of done

The loader is complete only when:

- source discovery and every semantic decision are documented;
- supported streams and unsupported streams are explicit;
- source-to-canonical mappings are traceable to evidence;
- capability attributes match the implemented source grains and clocks;
- every implemented method prepares and filters canonical output correctly;
- focused and full tests pass, or genuine external blockers are reported;
- no unrelated core behavior was changed;
- public exports and concise documentation are updated when appropriate;
- the final response lists assumptions, files changed, validation performed,
  and any remaining limitations.

## Required final response from the coding agent

Use this structure:

```text
## Implemented
- Loader class and module
- Supported streams and source grains
- Important source-to-canonical decisions

## Validation
- Focused tests
- Full test suite
- End-to-end domain checks

## Assumptions and limitations
- Confirmed assumptions
- Unsupported streams
- Remaining external blockers, if any

## Files changed
- Concise list with purpose
```

Do not describe the work as complete if required tests were not run or a
material source ambiguity remains unresolved.

## Common failure modes

- Copying an existing provider loader and retaining its undocumented
  assumptions.
- Treating filename or column-name similarity as evidence of semantics.
- Reconstructing UTC from local time without a trustworthy offset.
- Turning daily summaries into fake sample streams.
- Expanding native sleep-stage intervals into one-minute rows.
- Treating missing activity as zero steps.
- Calculating derived metrics inside the loader.
- Returning arbitrary DataFrames instead of calling schema preparers.
- Filtering sleep summaries by physical timestamps instead of sleep-calendar
  labels.
- Adding broad abstractions for a single provider.
- Changing pywearable core behavior to make one loader easier to implement.

The goal is not merely to make a new class run. The goal is to establish a
small, auditable translation boundary whose semantics remain correct when
used by every downstream pywearable domain function.
