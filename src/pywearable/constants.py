METRIC_BBI = "bbi"
METRIC_DAILY_SUMMARY = "daily-summary"
METRIC_HEART_RATE = "heart-rate"
METRIC_STRESS = "stress"
METRIC_SLEEP_SUMMARY = "sleep-summary"
METRIC_SLEEP_STAGE = "sleep-stage"
METRIC_PULSE_OX = "pulse-ox"
METRIC_RESPIRATION = "respiration"
METRIC_SLEEP_RESPIRATION = "sleep-respiration"
METRIC_SLEEP_PULSE_OX = "sleep-pulse-ox"

#################################################
#          CSV FILES - COLUMN NAMES             #
#################################################
# General columns #

#: Column for date in ISO format
ISODATE_COL = "isoDate"

#: Column for unix timestamp in milliseconds
UNIXTIMESTAMP_IN_MS_COL = "unixTimestampInMs"

#: Column for timezone offset in milliseconds
TIMEZONEOFFSET_IN_MS_COL = "timezoneOffsetInMs"

#: Column for calendar date
CALENDAR_DATE_COL = "calendarDate"
SLEEP_SUMMARY_ID_COL = "sleepSummaryId"

#: Column duration expressed in milliseconds
DURATION_IN_MS_COL = "durationInMs"

#: Column for entries in which sleep is detected
IS_SLEEPING_COL = "isSleeping"

# spo2 columns #
SPO2_SPO2_COL = "spo2"

# Sleep summary columns #
_SLEEP_SUMMARY_OVERALL_SLEEP_SCORE_COL = "overallSleepScore"
_SLEEP_SUMMARY_DURATION_IN_MS_COL = "durationInMs"
_SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL = "lightSleepDurationInMs"
_SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL = "deepSleepDurationInMs"
_SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL = "remSleepDurationInMs"
_SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL = "awakeDurationInMs"
_SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL = "unmeasurableSleepDurationInMs"

SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL = "sleepSummaryId"

# Sleep stage columns #
SLEEP_STAGE_SLEEP_TYPE_COL = "type"
SLEEP_STAGE_DURATION_IN_MS_COL = "durationInMs"
SLEEP_STAGE_SLEEP_SUMMARY_ID_COL = "sleepSummaryId"

# Stress columns #
STRESS_STRESS_LEVEL_COL = "stressLevel"
STRESS_BODY_BATTERY_COL = "bodyBattery"

# Recovery columns #
_RECOVERY_RECOVERY_COL = "recovery"

# Daily summary columns #
DAILY_SUMMARY_CALENDAR_DATE_COL = "calendarDate"
DAILY_SUMMARY_STEPS_COL = "steps"
DAILY_SUMMARY_DISTANCE_COL = "distanceInMeters"
DAILY_SUMMARY_STEPS_GOAL_COL = "stepsGoal"
DAILY_SUMMARY_MODERATE_INTENSITY_COL = "moderateIntensityDurationInMs"
DAILY_SUMMARY_VIGOROUS_INTENSITY_COL = "vigorousIntensityDurationInMs"
DAILY_SUMMARY_AVG_STRESS_IN_STRESS_LVL_COL = "averageStressInStressLevel"
DAILY_SUMMARY_MAX_STRESS_IN_STRESS_LVL_COL = "maxStressInStressLevel"
DAILY_SUMMARY_REST_STRESS_DURATION_IN_MS_COL = "restStressDurationInMs"
DAILY_SUMMARY_LOW_STRESS_DURATION_IN_MS_COL = "lowStressDurationInMs"
DAILY_SUMMARY_MEDIUM_STRESS_DURATION_IN_MS_COL = "mediumStressDurationInMs"
DAILY_SUMMARY_HIGH_STRESS_DURATION_IN_MS_COL = "highStressDurationInMs"
DAILY_SUMMARY_UNRELIABLE_STRESS_DURATION_IN_MS_COL = "activityStressDurationInMs"
DAILY_SUMMARY_STRESS_SCORE_COL = "stressQualifier"

# Respiratory data columns #
RESPIRATION_BREATHS_PER_MINUTE_COL = "breathsPerMinute"
RESPIRATION_SLEEP_SUMMARY_ID_COL = "sleepSummaryId"
RESPIRATION_CALENDAR_DATE_COL = "calendarDate"

# Cardiac data columns #
SPO2_COLUMN = "spo2"
HR_COLUMN = "beatsPerMinute"
RESTING_HR_COLUMN = "restingHeartRateInBeatsPerMinute"
AVG_HR_COLUMN = "averageHeartRateInBeatsPerMinute"
MAX_HR_COLUMN = "maxHeartRateInBeatsPerMinute"
MIN_HR_COLUMN = "minHeartRateInBeatsPerMinute"

# Epochs #
EPOCH_INTENSITY_COL = "intensity"
EPOCH_ACTIVE_TIME_MS_COL = "activeTimeInMs"

# User column (for index and related)
USER_COL = "user"

#################################################
#           CSV FILES - COLUMN VALUES          #
#################################################

# Sleep stage #
_SLEEP_STAGE_REM_STAGE_VALUE = "rem"
_SLEEP_STAGE_LIGHT_STAGE_VALUE = "light"
_SLEEP_STAGE_DEEP_STAGE_VALUE = "deep"
_SLEEP_STAGE_AWAKE_STAGE_VALUE = "awake"
_SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE = "unmeasurable"

# Hypnogram
_SLEEP_STAGE_REM_STAGE_MAPPED_VALUE = 4
_SLEEP_STAGE_LIGHT_STAGE_MAPPED_VALUE = 1
_SLEEP_STAGE_DEEP_STAGE_MAPPED_VALUE = 3
_SLEEP_STAGE_AWAKE_STAGE_MAPPED_VALUE = 0
_SLEEP_STAGE_UNMEASURABLE_STAGE_MAPPED_VALUE = -1

# Epochs #
EPOCH_ACTIVITY_ACTIVE_VALUE = "ACTIVE"
EPOCH_ACTIVITY_HIGHLY_ACTIVE_VALUE = "HIGHLY_ACTIVE"
EPOCH_ACTIVITY_SEDENTARY_VALUE = "SEDENTARY"

# Todo/Questionnaire values #
QUESTIONNAIRE_QUESTION_TYPE_RADIO_VALUE = "radio"
QUESTIONNAIRE_QUESTION_TYPE_MULTI_SELECT_VALUE = "multi_select"
QUESTIONNAIRE_QUESTION_TYPE_TEXT_VALUE = "text"
