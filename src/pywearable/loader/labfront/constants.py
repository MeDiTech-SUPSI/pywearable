##################################################
#               FILE INDEX KEYS                  #
##################################################
FILE_INDEX_METRIC_KEY = "metric"
FILE_INDEX_FILENAME_KEY = "fileName"
FILE_INDEX_PARTICIPANT_ID_KEY = "participantId"
FILE_INDEX_PARTICIPANT_LABFRONT_ID_KEY = "participantLabfrontId"
FILE_INDEX_PARTICIPANT_FULL_ID_KEY = "participantFullId"
FILE_INDEX_FIRST_TIMESTAMP_KEY = "firstTimestamp"
FILE_INDEX_LAST_TIMESTAMP_KEY = "lastTimestamp"
FILE_INDEX_IS_QUESTIONNAIRE_KEY = "isQuestionnaire"
FILE_INDEX_IS_TODO_KEY = "isTodo"
FILE_INDEX_TASK_NAME_KEY = "taskName"
FILE_INDEX_TASK_ID_KEY = "taskId"
FILE_INDEX_TASK_FULL_ID_KEY = "taskFullId"
FILE_INDEX_TASK_FOLDER = "taskFolder"
FILE_INDEX_LAST_EDIT_TIMESTAMP_KEY = "lastEditTimestamp"
FILE_INDEX_BASE_FOLDER_KEY = "baseFolder"

##################################################
#                 DATA SOURCES                   #
##################################################
DATA_SOURCE_GARMIN_HEALTH_API = "garmin_health_api"
DATA_SOURCE_GARMIN_SDK = "garmin_sdk"

METRIC_BBI = "bbi"
METRIC_DAILY_SUMMARY = "daily-summary"
METRIC_HEART_RATE = "heart-rate"
METRIC_STRESS = "stress"
METRIC_PULSE_OX = "pulse-ox"
METRIC_RESPIRATION = "respiration"
METRIC_SLEEP_RESPIRATION = "sleep-respiration"
METRIC_SLEEP_STAGE = "sleep-stage"
METRIC_SLEEP_SUMMARY = "sleep-summary"

#################################################
#              CSV FILES - ROWS                 #
#################################################
CSV_STATS_SKIP_ROWS = 3  # Number of rows to skip in CSV files

##################################################
#               FOLDER NAMES                     #
##################################################

# Garmin Connect metrics - Folder names #
GARMIN_CONNECT_BASE_FOLDER = "garmin-connect"
GARMIN_CONNECT_ACTIVITY_DETAILS_LAP_FOLDER = (
    GARMIN_CONNECT_BASE_FOLDER + "-activity-details-lap"
)
GARMIN_CONNECT_ACTIVITY_DETAILS_SAMPLE_FOLDER = (
    GARMIN_CONNECT_BASE_FOLDER + "-activity-details-sample"
)
GARMIN_CONNECT_ACTIVITY_DETAILS_SUMMARY_FOLDER = (
    GARMIN_CONNECT_BASE_FOLDER + "-activity-details-summary"
)
GARMIN_CONNECT_ACTIVITY_MOVEIQ_SUMMARY_FOLDER = (
    GARMIN_CONNECT_BASE_FOLDER + "-activity-move-iq-summary"
)
GARMIN_CONNECT_ACTIVITY_SUMMARY_FOLDER = (
    GARMIN_CONNECT_BASE_FOLDER + "-activity-summary"
)
GARMIN_CONNECT_BODY_COMPOSITION_FOLDER = (
    GARMIN_CONNECT_BASE_FOLDER + "-body-composition"
)
GARMIN_CONNECT_HEART_RATE_FOLDER = GARMIN_CONNECT_BASE_FOLDER + "-daily-heart-rate"
GARMIN_CONNECT_DAILY_SUMMARY_FOLDER = GARMIN_CONNECT_BASE_FOLDER + "-daily-summary"
GARMIN_CONNECT_EPOCH_FOLDER = GARMIN_CONNECT_BASE_FOLDER + "-epoch"
GARMIN_CONNECT_DAILY_PULSE_OX_FOLDER = GARMIN_CONNECT_BASE_FOLDER + "-pulse-ox"
GARMIN_CONNECT_RESPIRATION_FOLDER = GARMIN_CONNECT_BASE_FOLDER + "-respiration"
GARMIN_CONNECT_SLEEP_PULSE_OX_FOLDER = GARMIN_CONNECT_BASE_FOLDER + "-sleep-pulse-ox"
GARMIN_CONNECT_SLEEP_RESPIRATION_FOLDER = (
    GARMIN_CONNECT_BASE_FOLDER + "-sleep-respiration"
)
GARMIN_CONNECT_SLEEP_STAGE_FOLDER = GARMIN_CONNECT_BASE_FOLDER + "-sleep-stage"
GARMIN_CONNECT_SLEEP_SUMMARY_FOLDER = GARMIN_CONNECT_BASE_FOLDER + "-sleep-summary"
GARMIN_CONNECT_STRESS_FOLDER = GARMIN_CONNECT_BASE_FOLDER + "-stress"
GARMIN_CONNECT_USER_METRICS_FOLDER = GARMIN_CONNECT_BASE_FOLDER + "-user-metrics"

# Garmin Device metrics - Folder names #
GARMIN_DEVICE_FOLDER = "garmin-device"
GARMIN_DEVICE_BBI_FOLDER = GARMIN_DEVICE_FOLDER + "-bbi"
GARMIN_DEVICE_HEART_RATE_FOLDER = GARMIN_DEVICE_FOLDER + "-heart-rate"
GARMIN_DEVICE_PULSE_OX_FOLDER = GARMIN_DEVICE_FOLDER + "-pulse-ox"
GARMIN_DEVICE_RESPIRATION_FOLDER = GARMIN_DEVICE_FOLDER + "-respiration"
GARMIN_DEVICE_STEP_FOLDER = GARMIN_DEVICE_FOLDER + "-step"
GARMIN_DEVICE_STRESS_FOLDER = GARMIN_DEVICE_FOLDER + "-stress"

# Questionnaire and Todo - Folder names #
QUESTIONNAIRE_FOLDER = "questionnaire"
TODO_FOLDER = "todo"

# Labfront-specific CSV columns
GARMIN_CONNECT_TIMEZONEOFFSET_IN_MS_COL = "timezoneOffsetInMs"
ISODATE_COL = "isoDate"
UNIXTIMESTAMP_IN_MS_COL = "unixTimestampInMs"
TIMEZONEOFFSET_IN_MS_COL = "timezoneOffsetInMs"
CALENDAR_DATE_COL = "calendarDate"
SLEEP_SUMMARY_ID_COL = "sleepSummaryId"
DURATION_IN_MS_COL = "durationInMs"
IS_SLEEPING_COL = "isSleeping"
SPO2_SPO2_COL = "spo2"

##################################################
#          Questionnaire/Task Columns            #
##################################################
TODO_NAME_COL = "todoName"
TODO_ID_COL = "todoId"
QUESTIONNAIRE_NAME_COL = "questionnaireName"
QUESTIONNAIRE_ID_COL = "questionnaireId"
TASK_ID_COL = "taskId"
QUESTIONNAIRE_QUESTION_TYPE_TEXT_VALUE = "text"

##################################################
#            Sleep Summary Columns               #
##################################################
GARMIN_CONNECT_SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL = "deepSleepDurationInMs"
GARMIN_CONNECT_SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL = "lightSleepDurationInMs"
GARMIN_CONNECT_SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL = "remSleepInMs"
GARMIN_CONNECT_SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL = "awakeDurationInMs"
GARMIN_CONNECT_SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL = (
    "unmeasurableSleepInMs"
)

SLEEP_SUMMARY_DURATION_IN_MS_COL = "durationInMs"
SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL = "deepSleepDurationInMs"
SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL = "lightSleepDurationInMs"
SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL = "remSleepDurationInMs"
SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL = "awakeDurationInMs"
SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL = (
    "unmeasurableSleepDurationInMs"
)
SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL = "sleepSummaryId"

##################################################
#         Sleep Stages - Type values             #
##################################################
SLEEP_STAGE_LIGHT_STAGE_VALUE = "light"
SLEEP_STAGE_DEEP_STAGE_VALUE = "deep"
SLEEP_STAGE_REM_STAGE_VALUE = "rem"
SLEEP_STAGE_AWAKE_STAGE_VALUE = "awake"
SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE = "unmeasurable"
SLEEP_STAGE_SLEEP_TYPE_COL = "type"
SLEEP_STAGE_DURATION_IN_MS_COL = "durationInMs"
SLEEP_STAGE_SLEEP_SUMMARY_ID_COL = "sleepSummaryId"

##################################################
#                 Stress columns                 #
##################################################

_STRESS_STRESS_LEVEL_COL = "stressLevel"
_STRESS_BODY_BATTERY_COL = "bodyBattery"
_STRESS_BODY_BATTERY_BEDTIME_COL = "bodyBatteryBedtime"
_STRESS_BODY_BATTERY_WAKING_COL = "bodyBatteryWakeup"
STRESS_STRESS_LEVEL_COL = "stressLevel"
STRESS_BODY_BATTERY_COL = "bodyBattery"

##################################################
#             Daily summary columns              #
##################################################

DAILY_SUMMARY_AVG_STRESS_IN_STRESS_LVL_COL = "averageStressInStressLevel"
DAILY_SUMMARY_MAX_STRESS_IN_STRESS_LVL_COL = "maxStressInStressLevel"
DAILY_SUMMARY_REST_STRESS_DURATION_IN_MS_COL = "restStressDurationInMs"
DAILY_SUMMARY_LOW_STRESS_DURATION_IN_MS_COL = "lowStressDurationInMs"
DAILY_SUMMARY_MEDIUM_STRESS_DURATION_IN_MS_COL = "mediumStressDurationInMs"
DAILY_SUMMARY_HIGH_STRESS_DURATION_IN_MS_COL = "highStressDurationInMs"
DAILY_SUMMARY_UNRELIABLE_STRESS_DURATION_IN_MS_COL = "activityStressDurationInMs"
DAILY_SUMMARY_STRESS_SCORE_COL = "stressQualifier"

RESPIRATION_BREATHS_PER_MINUTE_COL = "breathsPerMinute"

##################################################
#                 Activity columns               #
##################################################

ACTIVITY_TOTAL_STEPS_COL = "totalSteps"
ACTIVITY_STEPS_COL = "steps"
