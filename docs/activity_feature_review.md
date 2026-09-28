# Activity features for pywearable: evidence and implementation priorities

Research date: 23 September 2026. Scope: a time-boxed literature and repository review, with a maximum budget of 20 minutes. This is a research proposal, not an implementation or a systematic review. Repository snapshot: `8538c4b`, branch `experimental-hierarchical-statistics`, including the working-tree files present during inspection.

## Recommendation

Build a small common activity layer around **step volume, accumulation patterns, timing, and longitudinal variability**, with additional features enabled only when the source supplies the required measurements. Keep physiological intensity, vendor activity categories, and step-derived proxies distinct.

The broadly feasible starting point is daily steps and their summaries across days. Intraday step distributions are feasible for several supported sources. Genuine cadence and short stepping bouts are valuable extensions, but cannot be recovered from daily totals or interpolated 15-minute epochs. Oura offers a different extension path through activity classes and MET estimates. Raw-acceleration features belong in a later, separately supported capability.

The useful dimensions are:

| Dimension | Main question | Recommended measurements |
|---|---|---|
| Volume | How much movement? | Daily steps; steps within an explicit analysis window |
| Intensity | How demanding or fast? | Native minute-level cadence; separately, validated intensity durations or vendor MET estimates |
| Accumulation | How is movement distributed into episodes? | Step-defined bout frequency, length, steps per bout, fragmentation |
| Inactivity pattern | How sustained are periods of little movement? | Non-stepping or low-motion bout lengths and interruptions; sedentary estimates only with an appropriate classifier |
| Timing | When does movement occur? | Hourly profile, period-specific steps, peak activity window, later rest–activity rhythm features |
| Variability | How consistent is behavior across days? | Mean/median, SD/IQR/CV, weekday or workday contrasts, later IS/IV |

These are descriptive research features. Their computability does not establish clinical validity or equivalence between brands. Outcome studies motivate what to measure; they do not validate a different device's implementation of that measurement.

## What the repository actually supports

The existing [activity module](C:/Users/alvise.deirossi/_dev/git/pywearable/src/pywearable/activity.py:20) exposes step/distance series, daily steps/distance/goals, moderate and vigorous durations, and averages of Garmin activity categories, including weekday/weekend summaries. These are a useful starting vocabulary, but the implementation is largely a Labfront/Garmin-specific proposal.

| Source in the repository | Inspected capability | Realistic feature scope and required work |
|---|---|---|
| Labfront / Garmin Health API | `load_daily_summary`, `load_garmin_connect_epoch`, and `load_steps`; Connect step conversion assigns 15-minute duration | Daily volume and coarse intraday distribution. Audit category/duration semantics before intensity mapping. Fifteen-minute counts cannot resolve minute cadence or short interruptions. |
| Labfront / Garmin SDK | `load_steps(source=...SDK)` reads device step files without a common interval normalization | Potentially finer features, conditional on actual interval and cumulative-versus-incremental semantics; the method name alone does not establish resolution. |
| LifeSnaps Parquet / Fitbit | `load_steps` reads timestamped values from `steps.parquet` | Volume and intraday features; cadence/bouts only after verifying native sample duration and missingness. It currently exposes no interval duration and assigns zero timezone offset. `LifeSnapsMongoDBLoader` is a placeholder. |
| Tesserae CSV / Garmin | `load_epochs` reads steps, intensity, MET, motion intensity, distance, and activity calories | Useful native epoch input, but loader naming and duration normalization need reconciliation. It multiplies two duration columns by 100 and then drops them; do not treat this output as a complete interval contract. |
| Tesserae Parquet / Garmin | `load_steps` reads Health API/PA data, removes nonpositive values, and spreads counts over a 5-second grid | Revisit this transformation before activity analysis. Its own comment identifies a 15-minute source; generated 5-second values are not native measurements. Removing zero intervals also damages inactivity analysis. |
| SPHYNCS / Fitbit-derived biometrics | `load_epoch` sums `biometrics.parquet` steps into 15-minute bins; daily summary returns an empty frame | Coarse volume/distribution after confirming source count semantics. Preserve native data in a future adapter for finer analysis. Empty-bin sums must not manufacture observed zero activity. |
| Oura API v2 exports | `load_steps_summary` implemented; `load_activity_summary` and `load_intensity` explicitly unimplemented | Daily steps now; additional vendor summaries, classes, and MET series after mapping. No intraday step-count stream was established by this review. |

Code anchors: [Labfront dispatch](C:/Users/alvise.deirossi/_dev/git/pywearable/src/pywearable/loader/labfront/loader.py:3021), [Connect steps](C:/Users/alvise.deirossi/_dev/git/pywearable/src/pywearable/loader/labfront/loader.py:2090), [LifeSnaps](C:/Users/alvise.deirossi/_dev/git/pywearable/src/pywearable/loader/lifesnaps.py:754), [Tesserae CSV](C:/Users/alvise.deirossi/_dev/git/pywearable/src/pywearable/loader/tesserae/tesserae_csv.py:494), [Tesserae Parquet](C:/Users/alvise.deirossi/_dev/git/pywearable/src/pywearable/loader/tesserae/tesserae_parquet.py:422), [SPHYNCS](C:/Users/alvise.deirossi/_dev/git/pywearable/src/pywearable/loader/sphyncs/loader.py:521), [Oura steps](C:/Users/alvise.deirossi/_dev/git/pywearable/src/pywearable/loader/oura/loader.py:495), [Oura activity stubs](C:/Users/alvise.deirossi/_dev/git/pywearable/src/pywearable/loader/oura/loader.py:584).

A schema-only inspection of one existing local Oura `daily_activity` record confirmed fields for low/medium/high activity time, sedentary/resting/non-wear time, active/total calories, equivalent walking distance, `class_5_min`, and a MET series whose `interval` was 60 seconds. This establishes potential input availability in that export, not validation of the categories or completeness across participants. No participant measurement values are reproduced here. Class decoding and duration units still require verification against the provider specification.

Five design problems should be addressed before extending the legacy functions:

1. They call `load_garmin_connect_epoch` directly even though [BaseLoader](C:/Users/alvise.deirossi/_dev/git/pywearable/src/pywearable/loader/base.py:276) has a generic `load_epoch` stub and no completed common activity contract.
2. Some functions subtract 15 minutes from the requested end; date filtering should have one explicit interval convention.
3. Category proportions divide by summed `activeTimeInMs`; this has not been established as observed wear time or waking time.
4. The merged [daily intensity output](C:/Users/alvise.deirossi/_dev/git/pywearable/src/pywearable/activity.py:616) is moderate minutes plus twice vigorous minutes. That is an equivalent-minute measure, not elapsed MVPA duration.
5. Oura currently converts daily zero steps to missing. That policy needs source-specific justification; a general activity layer must preserve genuine zeros and distinguish them from absent data.

No common raw-accelerometry loading capability was found in the inspected loaders. Garmin's SDK can expose accelerometry, but provider capability does not mean this repository currently loads it. See the official [Garmin SDK overview](https://developer.garmin.com/health-sdk/) and [Health API overview](https://developer.garmin.com/gc-developer-program/health-api/). Fitbit's official API supports 1- or 15-minute intraday activity, subject to access, but that does not prove which resolution a particular archived dataset contains. See [Fitbit intraday documentation](https://dev.fitbit.com/build/reference/web-api/intraday/get-activity-intraday-by-date-range/).

## What the literature supports

### Volume is the common foundation; cadence is a useful conditional addition

Daily steps have substantial prospective evidence. Saint-Maurice et al. studied 4,840 US adults; higher step counts were associated with lower mortality, while intensity did not retain a significant association after adjustment for total steps. In contrast, del Pozo Cruz et al.'s UK Biobank analysis of 78,500 adults found additional associations for peak-30 cadence. These differences favor including both descriptive volume and cadence when possible, without treating cadence as a universally superior health indicator. [Saint-Maurice et al., 2020](https://pubmed.ncbi.nlm.nih.gov/32207799/); [del Pozo Cruz et al., 2022](https://pubmed.ncbi.nlm.nih.gov/36094529/).

The harmonized analysis by Paluch et al. included 15 cohorts and 47,471 adults and supports daily steps as a central measure, with age-dependent dose–response relationships. It does not justify hard-coding 10,000 steps as a universal normal/abnormal cutoff. Offer an explicit user-selected threshold if goal attainment is needed. [Paluch et al., 2022](https://pubmed.ncbi.nlm.nih.gov/35247352/).

Cadence thresholds around 100 steps/minute have laboratory and overground support as heuristics for moderate-intensity walking in studied adult populations. They are not general classifiers of all physical activity: cycling, resistance training, load carrying, slow pathological gait, and upper-body work need other information. Name the output `minutes_above_cadence_threshold`, not generic MVPA. [CADENCE-Adults, ages 21–40](https://pubmed.ncbi.nlm.nih.gov/30654810/); [overground validation](https://pubmed.ncbi.nlm.nih.gov/31698337/); [CADENCE-Adults, ages 61–85](https://pubmed.ncbi.nlm.nih.gov/34556146/).

### Bout structure contains information lost by daily totals

Two people can accumulate the same total through a few long episodes or many short episodes. Wanigatunga et al. found associations between activity fragmentation and mortality in 548 older adults. This motivates recording transitions and bout distributions, but the cohort and activity classification do not establish a universal prognostic meaning for step-defined fragmentation in working-age adults. [Wanigatunga et al., 2019](https://pubmed.ncbi.nlm.nih.gov/31577355/). Chastin and Granat provide an earlier methodological basis for describing accumulation patterns beyond total duration. [Chastin and Granat, 2010](https://pubmed.ncbi.nlm.nih.gov/19854651/).

There is direct consumer-device evidence: Delobelle et al. assessed two Fitbit models against ActiGraph/activPAL in 69 adults and older adults and reported good detection of several short stepping and prolonged sedentary events under tested definitions. That supports a conditional bout feature family, not unrestricted transfer across models, age groups, or thresholds. [Delobelle et al., 2024](https://biblio.ugent.be/publication/01J0XE00V0VC2895KWY1D590ZP).

The recent review by Chanteau et al. is a useful corrective: among 48 epidemiological studies, it identified 68 distinct methodological combinations, with only four referenced validation studies suitable for assessing bout detection. Resolution, sensor placement, threshold, and allowed interruptions are part of the feature definition. They cannot be hidden implementation details. [Chanteau et al., 2026](https://pubmed.ncbi.nlm.nih.gov/42384116/).

### Inactivity interruptions matter, but zero steps do not establish sedentary behavior

Sedentary behavior combines wakefulness, low energy expenditure, and a sitting/reclining/lying posture. A non-stepping interval could instead be standing, cycling, sleeping, or non-wear. The safe generic outputs are non-stepping time or low-motion time; reserve sedentary terminology for a documented provider estimate or appropriate validated classifier. [SBRN terminology consensus, 2017](https://pubmed.ncbi.nlm.nih.gov/28599680/).

Healy et al. reported associations between accelerometer-defined sedentary interruptions and metabolic measures; Diaz et al. associated longer sedentary accumulation with mortality. A randomized crossover experiment by Dunstan et al. found that brief walking interruptions reduced acute postprandial glucose and insulin responses compared with uninterrupted sitting. Together these justify measuring interruption patterns, without assuming a wrist zero-step classifier measures the same exposure. [Healy et al., 2008](https://pubmed.ncbi.nlm.nih.gov/18252901/); [Diaz et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28892811/); [Dunstan et al., 2012](https://pubmed.ncbi.nlm.nih.gov/22374636/).

Activity fragmentation and interruption of sedentary time are also different constructs: frequent termination of active episodes and frequent termination of sedentary episodes should not share one vaguely named fragmentation score or one presumed direction of health benefit.

### Timing and regularity are useful for longitudinal and sleep-related research

Rest–activity research routinely uses M10, L5, relative amplitude, interdaily stability (IS), and intradaily variability (IV). Gonçalves et al. show that sampling scale changes rhythm estimates. A large NHANES analysis provides population-level use of these measures, while pyActigraphy supplies inspectable implementations. [Gonçalves et al., 2014](https://pubmed.ncbi.nlm.nih.gov/26483921/); [NHANES lifespan study, 2023](https://pubmed.ncbi.nlm.nih.gov/37610882/); [pyActigraphy methods paper, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8555797/).

Start with interpretable hourly profiles and movement within explicit clock or sleep-relative windows. Timing has also been investigated in prospective mortality research, but observational timing associations do not prescribe an optimal exercise hour. [Timing of physical activity, 2023](https://www.nature.com/articles/s41467-023-36546-5).

Daily variation deserves its own output rather than disappearing into a grand average. SD, IQR, and CV are inexpensive when daily totals exist. Inoue et al. studied the number of days achieving a step threshold, demonstrating the relevance of weekly distribution as a separate description. For reliability, Hilden et al.'s year-long Fitbit study found that the required number of days differed across activity metrics; a universal three-day or seven-day rule is not established. [Inoue et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10051082/); [Hilden et al., 2023](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282162).

### Raw acceleration offers stronger measurement standardization, but is a separate capability

Rowlands et al. proposed average acceleration and intensity gradient as complementary descriptions of volume and intensity distribution. MX metrics describe the acceleration threshold exceeded during the most active accumulated X minutes. They offer an attractive later extension when calibrated acceleration is available. They cannot be reconstructed from step totals, vendor activity labels, or estimated calories. [Rowlands et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29360664/); [Rowlands et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6895365/).

A scoping review maps further methods, including distributional, temporal, and complexity measures. This supports considering entropy, power-law summaries, and functional profiles as later research options, not expanding the initial API indiscriminately. [Backes et al., 2022](https://pubmed.ncbi.nlm.nih.gov/34695249/).

### Availability of a vendor estimate does not make it interchangeable

Energy expenditure should remain a provider estimate rather than a common ground-truth outcome. Shcherbina et al.'s validation demonstrated substantial errors for the devices tested. Oura-specific validation found systematic differences despite correlations with reference measurements. A 2026 free-living comparison again found that agreement depended on the metric and device; its Oura subgroup contained only five people. These results justify preserving provenance, not a numerical correction factor to apply to all newer models. [Shcherbina et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28538708/); [Kristiansson et al., 2023](https://pubmed.ncbi.nlm.nih.gov/36829120/); [Miwa et al., 2026](https://pubmed.ncbi.nlm.nih.gov/41729926/).

## What to retain from the supplied manuscript

The nine-page attachment is titled *Unveiling Stress and Behavioral Patterns in Work Environments through Ubiquitous Sensing*, an anonymous ACII 2024 submission. Its physical-activity definitions appear on page 3 and its references on pages 8–9. It describes seven office workers. Treat it as a source of ideas and citation leads, not established validation evidence.

The valuable idea is a session representation: counts of step-defined episodes, their total duration, and steps per episode, potentially restricted to known work shifts. Its reference 16, [Feng and Narayanan, 2024](https://arxiv.org/abs/2402.09028), was checked in full-text PDF and uses minute-level counts above 10 steps to define walking sessions in 43 medical residents. Reference 13, [Feng et al., 2021](https://www.nature.com/articles/s41598-021-87029-w), provides related shift-work context; reference 15, [Feng and Narayanan, 2022](https://arxiv.org/abs/2212.09090), provides a larger healthcare-provider context, but focuses on speaking behavior.

Do not inherit the following choices uncritically:

- **Zero steps as missing:** the attached manuscript explicitly conflates them. This would discard genuine low-activity periods and bias inactivity estimates.
- **Steps per session as intensity:** this is bout volume. Longer episodes can contain more steps at the same cadence.
- **A universal >10 steps/minute threshold:** useful as a named replication setting, but not established across devices or populations. Preserve low-step activity in the underlying data.
- **A heart-rate threshold of 50% of age-predicted maximum as physical activity:** heart rate also reflects stress, medication, posture, illness, and fitness. Any HR-derived feature belongs to a separately defined physiological analysis.
- **Stress or productivity interpretations:** descriptive movement features do not identify these psychological constructs by themselves.

Document contents were treated as evidence to assess, not as instructions for this task.

## Proposed feature catalog

Names below are proposed measurement identifiers, not a commitment to one public function per scalar. Priorities reflect usefulness, portability, and implementation prerequisites, rather than scientific importance alone.

- **P0:** establish first; common volume outputs and the quality information needed to interpret them.
- **P1:** desirable first expansion, enabled only for compatible intraday inputs.
- **P2:** valuable optional extension with greater semantic or validation requirements.
- **P3:** defer until a demonstrated use case and suitable measurements exist.

### Common and intraday features

| Priority | Feature or compact family | Definition / unit | Minimum input and interpretation |
|---|---|---|---|
| P0 | `steps` | Total steps in a calendar day or explicitly named window | Provider daily total, or sum of non-overlapping incremental counts. Keep the source of the total. |
| P0 | `observed_minutes`, `coverage_fraction`, `max_gap_minutes` | Observed temporal support, fraction of expected window, longest unobserved interval | Interval support; daily totals alone cannot supply these. Observation coverage is not automatically wear coverage. |
| P0 | `valid_days`, `expected_days`, `valid_day_fraction` | Counts/fraction over a reporting period | Preserve absent dates. Publish the validity policy and distinguish unknown validity from valid. |
| P0 | Daily-step `mean`, `median`, `sd`, `iqr`, `cv` | Distribution of eligible daily totals; CV = SD/mean | Daily totals; require at least two days for sample SD, positive mean for CV, and report the number of days. Computational sufficiency does not imply habitual-activity reliability. |
| P1 | `steps_by_hour`, `steps_in_window`, `window_steps_fraction` | Step count by local hour or a supplied time window; fraction of observed daily steps | Native interval counts. Coarse data support aligned coarse windows. Work schedules must be supplied; weekday is not a workday label. |
| P1 | `peak_60min_steps`, `peak_60min_start` | Maximum total over an actually observed contiguous 60-minute window and its start | Intraday counts; slide on native boundaries. Accept 15-minute inputs, but retain that resolution in metadata. |
| P1 | `stepping_epoch_minutes`, `stepping_epoch_fraction` | Duration/fraction of valid epochs classified as containing stepping | Prefer native minute-level counts; explicitly an epoch occupancy measure, not exact time spent walking. Fifteen-minute occupancy is a separate coarse feature. |
| P1 | `step_bout_count`, `step_bouts_per_observed_hour` | Number/rate of maximal adjacent stepping-epoch runs | Genuine minute-level data preferred; explicit threshold, minimum duration, and interruption policy. Call them step-defined bouts, not validated walking episodes. |
| P1 | `step_bout_duration_mean`, `median`, `p90`, `max` | Duration distribution of eligible step-defined bouts, minutes | Same as above; incomplete bouts must be marked/censored. Median and maximum are sufficient initial outputs; others are cheap optional summaries. |
| P1 | `step_bout_steps_mean`, `step_fraction_in_bouts_ge_10min` | Mean steps per bout; fraction of observed steps accumulated in sufficiently long bouts | Native counts plus segmentation. Ten minutes is a descriptive parameter, not a minimum duration for activity to count. |
| P1 | `peak_1min_cadence`, `peak_30min_cadence` | Highest minute count; mean of the 30 largest minute counts, steps/min | Native one-minute counts or finer counts exactly aggregated into complete minutes. Peak-30 minutes need not be consecutive. |
| P1 | `minutes_above_cadence_threshold` | Minutes meeting a stated threshold, e.g. 100 steps/min | Same input. Report as ambulatory intensity proxy, with explicit threshold and intended population. |
| P1 | `nonstepping_bout_max_minutes`, `nonstepping_minutes_in_bouts_ge_30min` | Longest observed non-stepping run; time in prolonged runs | Minute-level counts and defensible observation/wear status. If restricted to wake, also require a wake mask. These are not sedentary minutes. |
| P1 | `nonstepping_to_stepping_transitions_per_hour` | Observed interruptions normalized by eligible observation time | Same input; transitions across gaps are excluded. Preserve the two states in the name. |
| P1 | `weekday_weekend_steps_difference` | Mean weekend steps minus mean weekday steps | Eligible daily totals in both groups, with group counts. For shift workers prefer supplied workday/non-workday labels. |
| P2 | `step_active_to_inactive_probability` and reverse | Conditional transition probabilities per fixed epoch | Native regular grid, explicit step threshold, adjacent valid observations. These are stepping proxies for the activity-fragmentation literature. |
| P2 | `steps_target_day_fraction` | Eligible days reaching a caller-specified target / eligible days | Daily steps; missing days are not failures. A vendor goal can be offered separately, preserving changes in that goal. |
| P2 | `steps_t25`, `steps_t50`, `steps_t75` | Clock times when cumulative observed daily steps reach 25%, 50%, 75% | Adequately covered intraday data. Proposed descriptive timing summaries; no disease-specific validation asserted here. Report bin-level timing uncertainty. |
| P2 | `steps_trend` or deviation from personal baseline | Change per day or difference from a predeclared baseline | Longer daily series, with device changes and missingness controlled. Baselines for prediction must use past data only. |

The bout/cadence/inactivity families have the strongest argument for inclusion beyond the current proposal, but require the most care about source resolution. Prefer a small, interpretable set over dozens of nearly redundant threshold variants.

### Intensity, rhythms, and specialized extensions

| Priority | Feature family | Computation and eligibility |
|---|---|---|
| P1, provider-scoped | Vendor low/moderate/high activity durations | Expose original categories, units, and provider definition. Do not silently equate Garmin `ACTIVE`, Fitbit activity levels, and Oura classes. |
| P2, harmonized only after validation | `light_minutes`, `moderate_minutes`, `vigorous_minutes`, `mvpa_minutes` | Time-weighted classification with documented thresholds/algorithm, input channel, population, and wake/non-wear handling. MVPA = moderate + vigorous elapsed minutes. |
| P2 | `moderate_equivalent_minutes` | Moderate + 2 × vigorous, kept distinct from MVPA duration. Do not combine already weighted vendor scores with elapsed minutes. |
| P2, provider-scoped | `mean_met`, `gross_met_minutes`, `net_met_minutes` | Duration-weighted vendor MET estimates, with explicit activity/wake scope. Gross sums MET × minutes; net sums max(MET−1, 0) × minutes under the stated baseline convention. |
| P2 | `sedentary_minutes`, prolonged sedentary bouts, sedentary breaks | Require a documented sedentary classifier; separately label a provider estimate or posture-validated measure. Daily sedentary totals cannot establish intraday breaks. |
| P2 | `m10_activity`, `l5_activity`, corresponding onset times, `relative_amplitude` | Adequately covered 24-hour series, including night. Preserve whether the input is counts, step rate, MET, or acceleration. Distinguish averaged-profile metrics from averages of daily metrics. |
| P2 | `interdaily_stability`, `intradaily_variability` | Regular time series with multiple sufficiently covered days; fixed resolution and formula conventions. Step-derived results are not automatically interchangeable with actigraphy results. |
| P2, provider-scoped | Distance and activity/total energy expenditure | Keep provider values as optional measurements. Oura equivalent walking distance is not actual traveled distance; active calories are not total calories. |
| P3 | Mean calibrated acceleration / ENMO; intensity gradient; acceleration MX | Require raw tri-axial acceleration or a validated, equivalently processed acceleration summary; units, calibration, placement, epoch, and non-wear processing are essential. |
| P3 | Bout Gini, duration-weighted median, power-law exponent | Potentially useful accumulation descriptors, but choose only after simple bout summaries prove insufficient. Tail fitting and minimum-bout assumptions make power-law estimates less robust. |
| P3 | Entropy, multiscale entropy, DFA, rhythm acrophase/cosinor | Research extensions with explicit signal, length, missingness, and parameter requirements. Do not add as an undifferentiated “activity complexity” score. |
| P3 | Activity type, gait speed, gait variability, stairs, VILPA, training load | Separate capabilities requiring suitable sensors or validated event labels. Step counts alone cannot establish these. |

WHO's 2020 guidance removed the old requirement for activity to occur in bouts of at least ten minutes. Keep total activity independent of any bout-length filter. [WHO, 2020](https://www.who.int/news/item/25-11-2020-every-move-counts-towards-better-health-says-who).

## Definitions that should be fixed before implementation

### Steps, duration, and coverage

Represent an interval as `[start, end)` with an incremental step count `s_i` and actual duration `d_i`. Sum counts only after removing duplicate transmissions and distinguishing them from legitimate category-split records. Cumulative counters require differencing and reset handling; do not sum them directly. Prefer native source intervals over artificial interpolation.

For fully observed, non-overlapping intervals in a window:

```text
steps = sum(s_i)
observed_minutes = duration(union of observed intervals) / 60 seconds
coverage_fraction = observed_duration / expected_window_duration
window_steps_fraction = steps_in_window / observed_daily_steps
```

Zero denominators produce an undefined result, not an arbitrary zero. Preserve `steps=0` when it is genuinely observed. Unsupported features, absent recordings, and insufficient coverage should carry different status/reason values even if their numeric value is missing.

Do not allocate part of an aggregate count to a clipped interval unless a stated approximation is explicitly requested. For example, a 15-minute count spanning 08:00–08:15 cannot establish exact steps in 08:07–08:12. Native-aligned windows avoid this ambiguity. Provider daily totals and epoch-summed totals may disagree; expose/reconcile the provenance rather than adding them together.

### Bout detection and censoring

For a fixed native one-minute grid, let `a_t = 1(s_t > theta)`. Start with an explicit `theta=0` stepping-occupancy definition and a named `theta=10` replication option for the attached paper; neither is claimed to validate continuous walking. A bout is a maximal consecutive run of qualifying, observed epochs. Default to no allowed interruption. Missingness always splits observation, never counts as a rest interval or a permitted pause.

Retain a bout table with start, end, duration, steps, threshold, native resolution, and left/right censoring flags. A gap or reporting-window boundary can truncate a bout. Count observed starts when start counts are requested; calculate completed-bout duration distributions separately and report how many bouts were censored. An observed maximum across truncated bouts is a lower bound on the true maximum. Avoid silently discarding all censored data without documenting the resulting selection.

Bout count normalized per observed hour controls exposure length, but does not remove bias when non-wear occurs preferentially during exercise or work. Sensitivity analysis should compare reasonable thresholds and resolutions before selecting a study-specific setting.

For transition probabilities, count only adjacent valid pairs separated by exactly one epoch:

```text
p_active_to_inactive = N(a_t=1, a_(t+1)=0) / N(a_t=1 with valid successor)
p_inactive_to_active = N(a_t=0, a_(t+1)=1) / N(a_t=0 with valid successor)
```

The reciprocal of mean active-bout length is a common fragmentation summary in the literature. It equals a transition-style rate only under the corresponding interval and boundary conventions; it should not be silently substituted for the pair-based estimator above when gaps or censored bouts exist. Always store the epoch duration. A per-15-minute probability and a per-minute probability are different quantities.

### Cadence

Divide steps by the actual interval duration to obtain an interval-average step rate. However, `150 steps / 15 minutes = 10 steps/minute` does not mean the person walked at 10 steps/minute: the steps might all have occurred in one minute. Label this `mean_step_rate_15min`, not walking cadence.

Peak-30 cadence averages the 30 highest **minute-level** counts in an eligible day, allowing nonconsecutive minutes. Include observed zero-count minutes if needed; require at least 30 observed minutes and the independently defined day-coverage criterion. Do not pad missing minutes with zero. Distinguish this from the highest contiguous 30-minute mean, and from the maximum one-minute value. A source upsampled from 15 minutes remains ineligible.

### Intensity and behavior classes

Intensity durations should accumulate time, not the number of rows, because durations may differ. A MET-based threshold scheme must state its boundaries and whether it describes waking activity. Vendor estimated MET and acceleration-derived classifications must retain their distinct method labels. A generic mapping should follow a documented validation decision, not similar category names.

For a 24-hour time budget, behavior classes should be mutually exclusive on observed intervals. Sleep, waking low activity, light activity, and higher intensity cannot simply be added from unrelated overlapping provider summaries. Keep missing/non-wear explicit. Downstream modeling of a fixed time budget should account for its compositional nature rather than treating every fraction as independent.

### Rhythms and longitudinal aggregation

M10 is the mean activity in the most active **consecutive ten-hour** window; L5 is the corresponding least-active **five-hour** mean. Specify whether the search is over a multi-day mean 24-hour profile or separate days. The window may wrap midnight. `RA = (M10 − L5) / (M10 + L5)` when the denominator is positive. These are unrelated to acceleration `M30`, which refers to the most active accumulated 30 minutes.

For a complete hourly series `x_i`, `N` observations, `p=24`, grand mean `x_bar`, and mean activity `h_j` at hour j, a conventional specification is:

```text
IS = N * sum_j((h_j - x_bar)^2) / (p * sum_i((x_i - x_bar)^2))
IV = N * sum_i=2..N((x_i - x_(i-1))^2)
     / ((N-1) * sum_i((x_i - x_bar)^2))
```

Choose and version the convention: sample- versus population-variance implementations can differ. This formula assumes complete, equally spaced input; deleting gaps and concatenating the survivors would invent adjacency. An initial implementation can require complete eligible days/blocks and explicitly flag limitations rather than silently impute. A constant series has undefined IS/IV. A single day cannot support a meaningful measure of *interdaily* stability. A pragmatic seven-day analysis window is a project choice, not a universal validated reliability threshold. Compare against established [pyActigraphy definitions and implementation](https://ghammad.github.io/pyActigraphy/_modules/pyActigraphy/metrics/metrics.html).

Nighttime zero steps often make step-based L5 equal to zero and RA equal to one, reducing its informativeness. Thus rhythm features on steps should remain explicitly `step`-based; replacing actigraphy counts with steps or MET estimates is a measurement change. Sleep and non-wear can both look inactive, so full-day missingness handling is essential.

Aggregate daily totals as totals or averages according to the intended question. A mean of weekly means weights weeks equally; a mean of all eligible daily values weights days equally. Both can be useful but are not the same. Similarly, pooled bout-duration distributions and means of daily mean bout lengths answer different questions. Clock times require circular aggregation. Record the device/source and split or flag periods containing a device change.

## Quality rules and architectural fit

The minimum input contract should distinguish:

| Input | Essential information |
|---|---|
| Daily activity summary | User, provider calendar date, measurement, value, unit, source/method, any supplied completeness/non-wear information |
| Intraday activity | User, interval start/end or duration, clock semantics, incremental count/rate/category, source, native resolution, observed versus imputed status |
| Optional state information | Wear state and method; sleep/wake intervals; provenance and confidence/availability of each |

A daily summary does not imply 24 hours of wear. An API that emits a value every minute can still contain inferred or default-filled intervals. Conversely, absence of heart-rate readings is not automatically non-wear. The current [adherence function](C:/Users/alvise.deirossi/_dev/git/pywearable/src/pywearable/adherence.py:19) estimates time from available HR minutes; reuse only if its meaning is explicitly “HR observation coverage,” not universal activity wear time.

Use real elapsed time for durations and local time for behavioral grouping, preserving UTC/offset information where available. Daylight-saving days can last 23 or 25 hours. Do not fabricate UTC for local-only datasets. Supply coverage for each requested window, since a day with good overall coverage may have no morning or work-shift data. Avoid universal automatic extrapolation from partial days to full-day step totals.

For the initial API, derive daily measurements and a reusable bout table, then use the repository's [hierarchical aggregation utilities](C:/Users/alvise.deirossi/_dev/git/pywearable/src/pywearable/hierarchy.py:267) where their semantics fit. Specialized nonlinear metrics such as IS, IV, and peak windows must be recomputed from the appropriate underlying series, rather than averaged through the hierarchy indiscriminately.

The existing [cycle intervals](C:/Users/alvise.deirossi/_dev/git/pywearable/src/pywearable/cycle.py:116) can supply physiological or sleep-relative analysis windows. Its `cycle_awake` interval currently includes naps: that is a window between main sleep sessions, not a complete wake-state mask. Actual waking-only inactivity metrics need nap exclusion or an explicit alternative definition. Daily summaries cannot be divided accurately into these cycles without intraday measurements.

Return measurement status alongside values: supported and available, unsupported input, missing recording, insufficient coverage, or undefined denominator. A small capability description per loader is preferable to broad exception handlers that turn every failure into `None`. This is a proposed design direction; no new abstraction is needed until implementing the first common loader methods.

## Suggested implementation order and acceptance criteria

**Increment 1 — common steps and observation semantics.** Normalize native intervals and daily summaries; preserve zeros and gaps; expose daily steps, coverage where known, valid-day counts, and daily mean/median/SD/IQR/CV. Add explicitly requested date-type contrasts. This reaches the broadest supported-source set with the fewest assumptions.

**Increment 2 — intraday distribution and stepping episodes.** Add hourly/window totals, peak-hour timing, a reusable bout table, selected bout summaries, and non-stepping interruptions. Enable minute cadence only on genuine qualifying input. Retain a distinct coarse feature vocabulary for 15-minute streams. Revising Tesserae's artificial upsampling and SPHYNCS's loss of native resolution is a prerequisite for fine-scale use.

**Increment 3 — provider intensity and rhythms.** Map the Oura summaries/classes/MET series and Garmin equivalents without erasing their provenance. Add harmonized intensity outputs only where defensible. Implement M10/L5/RA and IS/IV with explicit input-channel names, completeness requirements, and reference comparisons.

**Increment 4 — specialized research inputs.** Add raw-acceleration metrics only with an actual accelerometry adapter and an agreed processing pipeline. Evaluate entropy, richer accumulation statistics, exercise events, and physiological coupling against concrete research questions.

Before calling the first implementation reliable, verify:

1. **Conservation and boundaries:** native incremental counts sum correctly; cumulative resets and duplicates do not inflate totals; midnight, DST, offset changes, and `[start,end)` behavior are consistent.
2. **Zero versus missing:** an observed all-zero day remains zero; an absent day remains missing; sparse positive-only input does not manufacture inactivity. Coverage cannot exceed the union of valid intervals.
3. **Known synthetic episodes:** hand-calculated bouts, censored edges, transitions, and peak windows agree; a gap never joins episodes or creates a transition.
4. **Resolution eligibility:** resampled 15-minute data are rejected for one-minute cadence; coarse and fine bout definitions are visibly distinct.
5. **Rhythm references:** periodic and constant test signals produce expected behavior; compare a fixed documented convention with a reference implementation.
6. **Source validity:** after mathematical checks, inspect actual native exports and compare device-specific outputs where reference data are available. Correct arithmetic is not device validation.

No production code or tests were changed for this review. The existing uncommitted code changes were left intact.

## Search scope and remaining uncertainty

The review combined repository inspection, extraction of the supplied nine-page manuscript, backward citation checking, and targeted searches across PubMed/PMC, journal sites, author repositories, and official vendor documentation. Search families covered daily steps/cadence, bouts/fragmentation, sedentary interruptions, timing/rest–activity rhythms, longitudinal reliability, acceleration metrics, and consumer-device validity. Foundational studies and relevant 2024–2026 work were included. Primary cohort, experimental, validation, and methods papers informed the recommendations; reviews supplied broader methodological context.

Access depth varied: the attachment and the 2024 medical-resident PDF were read for their methods; several other sources were assessed through publisher text or indexed abstracts. PMC, Nature, and ScienceDirect occasionally blocked full-text retrieval. The official Oura v2 page did not expose readable specification text through the web tool, so potential Oura activity fields were corroborated against local export structure, without claiming validated mappings. No authenticated Notion, Zotero, or other personal-service connector was used.

The main unresolved questions are the native resolution and counter semantics of each dataset export, reliable wear-state availability, exact vendor category definitions, and cross-device validity in the intended population. These require adapter-level investigation and empirical validation, not more feature names.

Two adjacent papers were located but do not support core claims here: the [2025 iSPHYNCS comparison of Fitbit and MotionWatch for rhythm analysis](https://www.sciencedirect.com/science/article/abs/pii/S2352648325000753), whose full text was inaccessible, and [Fang et al., 2026](https://www.nature.com/articles/s41467-026-69552-4), which studies long-term activity consistency over decades rather than wearable-derived daily bout continuity. The latter should not be cited as validation of short activity bouts merely because its title refers to sustained activity.

The recommended first deliverable is therefore a **small, capability-aware activity module with defensible measurement names**, accompanied by an explicit source matrix. A common interface is realistic; claiming every supported wearable provides the same activity measurement is not.
