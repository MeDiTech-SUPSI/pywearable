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
FILE_INDEX_QUESTIONNAIRE_NAME_KEY = "questionnaireName"
FILE_INDEX_QUESTIONNAIRE_ID_KEY = "questionnaireName"
FILE_INDEX_TODO_NAME_KEY = "todoName"
FILE_INDEX_TODO_ID_KEY = "todoId"
FILE_INDEX_TASK_ID_KEY = "taskId"

##################################################
#                 DATA SOURCES                   #
##################################################
DATA_SOURCE_GARMIN_HEALTH_API = "garmin_health_api"
DATA_SOURCE_GARMIN_SDK = "garmin_sdk"

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
GARMIN_DEVICE_TIMEZONEOFFSET_IN_MS_COL = "timezone"
FIRST_SAMPLE_UNIXTIMESTAMP_IN_MS_COL = "firstSampleUnixTimestampInMs"
LAST_SAMPLE_UNIXTIMESTAMP_IN_MS_COL = "lastSampleUnixTimestampInMs"

##################################################
#          Questionnaire/Task Columns            #
##################################################
TODO_NAME_COL = "todoName"
TODO_ID_COL = "todoId"
QUESTIONNAIRE_NAME_COL = "questionnaireName"
QUESTIONNAIRE_ID_COL = "questionnaireId"
TASK_ID_COL = "taskId"
QUESTIONNAIRE_QUESTION_DESCRIPTION_COL = "questionDescription"
QUESTIONNAIRE_QUESTION_TYPE_COL = "questionType"
QUESTIONNAIRE_SECTION_INDEX_COL = "sectionIndex"
QUESTIONNAIRE_QUESTION_INDEX_COL = "questionIndex"
TASK_SCHEDULE_COL = "taskScheduleRepeat"

##################################################
#            Sleep Summary Columns               #
##################################################
SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL = "deepSleepDurationInMs"
SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL = "lightSleepDurationInMs"
SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL = "remSleepInMs"
SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL = "awakeDurationInMs"
SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL = "unmeasurableSleepInMs"

##################################################
#         Sleep Stages - Type values             #
##################################################
SLEEP_STAGE_LIGHT_STAGE_VALUE = "light"
SLEEP_STAGE_DEEP_STAGE_VALUE = "deep"
SLEEP_STAGE_REM_STAGE_VALUE = "rem"
SLEEP_STAGE_AWAKE_STAGE_VALUE = "awake"
SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE = "unmeasurable"
