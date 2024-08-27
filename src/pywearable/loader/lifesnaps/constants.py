##################################
#           Metric Names         #
##################################
METRIC_SPO2 = "daily-spo2"
METRIC_COMP_TEMP = "temprature-computed"
METRIC_DEVICE_TEMP = "temperature-device"
METRIC_DAILY_HRV_SUMMARY = "hrv-daily-summary"
METRIC_HRV_DETAILS = "hrv-details"
METRIC_PROFILE = "profile"
METRIC_RESPIRATORY_RATE_SUMMARY = "respiratory-rate-summary"
METRIC_STRESS = "stress"
METRIC_WRIST_TEMPERATURE = "temperature-wrist"
METRIC_ALTITUDE = "altitude"
METRIC_BADGE = "badge"
METRIC_CALORIES = "calories"
METRIC_HRV_HISTOGRAM = "hrv-histogram"
METRIC_DISTANCE = "distance"
METRIC_EST_OXY_VARIATION = "estimated-oxygen-variation"
METRIC_HEART_RATE = "heart-rate"
METRIC_JOURNAL_ENTRIES = "journal_entries"
METRIC_LIGHTLY_ACTIVE_MINUTES = "lightly-active-minutes"
METRIC_MODERATELY_ACTIVE_MINUTES = "moderately-active-minutes"
METRIC_SEDENTARY_MINUTES = "sedentary-minutes"
METRIC_VERY_ACTIVE_MINUTES = "very_active_minutes"
METRIC_STEPS = "steps"
METRIC_WATER_LOGS = "water-logs"
METRIC_RESTING_HEART_RATE = "resting-heart-rate"
METRIC_TIME_IN_HR_ZONES = "time-in-hr-zones"
METRIC_DEMOGRAPHIC_VO2_MAX = "demographic-vo2-max"
METRIC_ECG = "ecg"
METRIC_MINDFULNESS_GOALS = "mindfulness-goals"
# Questionnaires
METRIC_QUESTIONNAIRE_BREQ = "questionnaire-breq"
METRIC_QUESTIONNAIRE_STAI = "questionnaire-stai"
METRIC_QUESTIONNAIRE_PANAS = "questionnaire-panas"
METRIC_QUESTIONNAIRE_PERSONALITY = "questionnaire-personality"
METRIC_QUESTIONNAIRE_TTM = "questionnaire-ttmspbf"
# SEMA
METRIC_SEMA = "sema"

##################################
#        Sleep Columns           #
##################################
SLEEP_DEEP_DURATION_IN_MS_COL = "deepSleepDurationInMs"
SLEEP_LIGHT_DURATION_IN_MS_COL = "lightSleepDurationInMs"
SLEEP_REM_DURATION_IN_MS_COL = "remSleepInMs"
SLEEP_AWAKE_DURATION_IN_MS_COL = "awakeDurationInMs"

##################################
#     HRV Histogram Columns      #
##################################
HRV_HISTOGRAM_BUCKET_WIDTHS_COL = "bucketWidths"
HRV_HISTOGRAM_BUCKET_VALUES_COL = "bucketValues"

##################################
#           ECG Columns          #
##################################
ECG_SAMPLE_VALUE_COL = "value"

##################################
#           Database             #
##################################
DB_NAME = "rais_anonymized"
DB_TYPE_KEY = "type"
DB_ID_KEY = "id"
DB_DATA_KEY = "data"

##################################
#       Surveys Collection        #
##################################
DB_SURVEYS_COLLECTION_NAME = "surveys"
DB_SURVEYS_COLLECTION_DATE_FORMAT = "%d/%m/%Y %H:%M"
DB_SURVEYS_COLLECTION_DATE_KEY = "submitdate"
DB_SURVEYS_COLLECTION_USER_KEY = "user_id"


# --------------------------------#
#          Document types         #
# --------------------------------#
DB_SURVEYS_COLLECTION_DATA_TYPE_BREQ = "breq"
DB_SURVEYS_COLLECTION_DATA_TYPE_STAI = "stai"
DB_SURVEYS_COLLECTION_DATA_TYPE_PANAS = "panas"
DB_SURVEYS_COLLECTION_DATA_TYPE_BFPT = "bfpt"
DB_SURVEYS_COLLECTION_DATA_TYPE_TTMSPBF = "ttmspbf"

##################################
#       SEMA Collection        #
##################################
DB_SEMA_COLLECTION_NAME = "sema"
DB_SEMA_COLLECTION_DATE_FORMAT = "%Y-%m-%dT%H:%M%S"
DB_SEMA_COLLECTION_DATE_KEY = "COMPLETED_TS"
DB_SEMA_COLLECTION_USER_KEY = "user_id"

##################################
#       FitBit Collection        #
##################################
DB_FITBIT_COLLECTION_NAME = "fitbit"
DB_FITBIT_DATE_FORMAT = "%d/%m/%Y %H:%M"

# --------------------------------#
#          Document types         #
# --------------------------------#
DB_FITBIT_COLLECTION_DATA_TYPE_SLEEP = "sleep"
DB_FITBIT_COLLECTION_DATA_TYPE_COMP_TEMP = "Computed Temperature"
DB_FITBIT_COLLECTION_DATA_TYPE_DAILY_HRV_SUMMARY = (
    "Daily Heart Rate Variability Summary"
)
DB_FITBIT_COLLECTION_DATA_TYPE_DAILY_SPO2 = "Daily SpO2"
DB_FITBIT_COLLECTION_DATA_TYPE_DEVICE_TEMP = "Device Temperature"
DB_FITBIT_COLLECTION_DATA_TYPE_AFIB_ECG_READINGS = "Afib ECG Readings"
DB_FITBIT_COLLECTION_DATA_TYPE_HRV_DETAILS = "Heart Rate Variability Details"
DB_FITBIT_COLLECTION_DATA_TYPE_HRV_HISTOGRAM = "Heart Rate Variability Histogram"
DB_FITBIT_COLLECTION_DATA_TYPE_PROFILE = "Profile"
DB_FITBIT_COLLECTION_DATA_TYPE_RESPIRATORY_RATE_SUMMARY = "Respiratory Rate Summary"
DB_FITBIT_COLLECTION_DATA_TYPE_STRESS_SCORE = "Stress Score"
DB_FITBIT_COLLECTION_DATA_TYPE_WRIST_TEMPERATURE = "Wrist Temperature"
DB_FITBIT_COLLECTION_DATA_TYPE_ALTITUDE = "altitude"
DB_FITBIT_COLLECTION_DATA_TYPE_BADGE = "badge"
DB_FITBIT_COLLECTION_DATA_TYPE_CALORIES = "calories"
DB_FITBIT_COLLECTION_DATA_TYPE_DEMOGRAPHIC_VO2_MAX = "demographic_vo2_max"
DB_FITBIT_COLLECTION_DATA_TYPE_DISTANCE = "distance"
DB_FITBIT_COLLECTION_DATA_TYPE_ESTIMATED_OXYGEN_VARIATION = "estimated_oxygen_variation"
DB_FITBIT_COLLECTION_DATA_TYPE_HEART_RATE = "heart_rate"
DB_FITBIT_COLLECTION_DATA_TYPE_JOURNAL_ENTRIES = "journal_entries"
DB_FITBIT_COLLECTION_DATA_TYPE_LIGHTLY_ACTIVE_MINUTES = "lightly_active_minutes"
DB_FITBIT_COLLECTION_DATA_TYPE_MODERATELY_ACTIVE_MINUTES = "moderately_active_minutes"
DB_FITBIT_COLLECTION_DATA_TYPE_SEDENTARY_MINUTES = "sedentary_minutes"
DB_FITBIT_COLLECTION_DATA_TYPE_VERY_ACTIVE_MINUTES = "very_active_minutes"
DB_FITBIT_COLLECTION_DATA_TYPE_STEPS = "steps"
DB_FITBIT_COLLECTION_DATA_TYPE_WATER_LOGS = "water_logs"
DB_FITBIT_COLLECTION_DATA_TYPE_RESTING_HEART_RATE = "resting_heart_rate"
DB_FITBIT_COLLECTION_DATA_TYPE_TIME_IN_HR_ZONES = "time_in_heart_rate_zones"
DB_FITBIT_COLLECTION_DATA_TYPE_MINDFULNESS_GOALS = "mindfulness_goals"
"""
'exercise', 'mindfulness_eda_data_sessions', 
'mindfulness_sessions', ]
"""


# --------------------------------#
#         Sleep Documents         #
# --------------------------------#
DB_FITBIT_COLLECTION_SLEEP_DATA_LOG_ID_KEY = "logId"
DB_FITBIT_COLLECTION_SLEEP_DATA_DATE_OF_SLEEP_KEY = "dateOfSleep"
DB_FITBIT_COLLECTION_SLEEP_DATA_START_TIME_KEY = "startTime"
DB_FITBIT_COLLECTION_SLEEP_DATA_END_TIME_KEY = "endTime"
DB_FITBIT_COLLECTION_SLEEP_DATA_DURATION_KEY = "duration"
DB_FITBIT_COLLECTION_SLEEP_DATA_MIN_TO_FALL_ASLEEP_KEY = "minutesToFallAsleep"
DB_FITBIT_COLLECTION_SLEEP_DATA_MIN_ASLEEP_KEY = "minutesAsleep"
DB_FITBIT_COLLECTION_SLEEP_DATA_MIN_AWAKE_KEY = "minutesAwake"
DB_FITBIT_COLLECTION_SLEEP_DATA_MIN_AFTER_WAKEUP_KEY = "minutesAfterWakeup"
DB_FITBIT_COLLECTION_SLEEP_DATA_TIME_IN_BED_KEY = "timeInBed"
DB_FITBIT_COLLECTION_SLEEP_DATA_EFFICIENCY_KEY = "efficiency"
DB_FITBIT_COLLECTION_SLEEP_DATA_MAIN_SLEEP_KEY = "mainSleep"
DB_FITBIT_COLLECTION_SLEEP_DATA_LEVELS_KEY = "levels"
DB_FITBIT_COLLECTION_SLEEP_DATA_LEVELS_SUMMARY_KEY = "summary"
DB_FITBIT_COLLECTION_SLEEP_DATA_LEVELS_SUMMARY_DEEP_KEY = "deep"
DB_FITBIT_COLLECTION_SLEEP_DATA_LEVELS_SUMMARY_LIGHT_KEY = "light"
DB_FITBIT_COLLECTION_SLEEP_DATA_LEVELS_SUMMARY_REM_KEY = "rem"
DB_FITBIT_COLLECTION_SLEEP_DATA_LEVELS_SUMMARY_WAKE_KEY = "wake"
DB_FITBIT_COLLECTION_SLEEP_DATA_LEVELS_DATA_KEY = "data"
DB_FITBIT_COLLECTION_SLEEP_DATA_LEVELS_DATA_DATETIME_KEY = "dateTime"
DB_FITBIT_COLLECTION_SLEEP_DATA_LEVELS_DATA_SECONDS_KEY = "seconds"
DB_FITBIT_COLLECTION_SLEEP_DATA_LEVELS_DATA_LEVEL_KEY = "level"
DB_FITBIT_COLLECTION_SLEEP_DATA_LEVELS_SHORT_DATA_KEY = "shortData"
DB_FITBIT_COLLECTION_SLEEP_DATA_TYPE_KEY = "type"
DB_FITBIT_COLLECTION_SLEEP_DATA_INFO_CODE_KEY = "infoCode"
DB_FITBIT_COLLECTION_SLEEP_DATA_DEEP_COUNT_KEY = "deepCount"
DB_FITBIT_COLLECTION_SLEEP_DATA_DEEP_MIN_KEY = "deepMinutes"
DB_FITBIT_COLLECTION_SLEEP_DATA_DEEP_30_DAYS_AVG_MIN_KEY = "deepThirtydayavgminutes"
DB_FITBIT_COLLECTION_SLEEP_DATA_WAKE_COUNT_KEY = "wakeCount"
DB_FITBIT_COLLECTION_SLEEP_DATA_WAKE_MIN_KEY = "wakeMinutes"
DB_FITBIT_COLLECTION_SLEEP_DATA_WAKE_30_DAYS_AVG_MIN_KEY = "wakeThirtydayavgminutes"
DB_FITBIT_COLLECTION_SLEEP_DATA_LIGHT_COUNT_KEY = "lightCount"
DB_FITBIT_COLLECTION_SLEEP_DATA_LIGHT_MIN_KEY = "lightMinutes"
DB_FITBIT_COLLECTION_SLEEP_DATA_LIGHT_30_DAYS_AVG_MIN_KEY = "lightThirtydayavgminutes"
DB_FITBIT_COLLECTION_SLEEP_DATA_REM_COUNT_KEY = "remCount"
DB_FITBIT_COLLECTION_SLEEP_DATA_REM_MIN_KEY = "remMinutes"
DB_FITBIT_COLLECTION_SLEEP_DATA_REM_30_DAYS_AVG_MIN_KEY = "remThirtydayavgminutes"
DB_FITBIT_COLLECTION_SLEEP_DATA_RESTLESS_COUNT_KEY = "restlessCount"
DB_FITBIT_COLLECTION_SLEEP_DATA_RESTLESS_MIN_KEY = "restlessMinutes"
DB_FITBIT_COLLECTION_SLEEP_DATA_AWAKE_COUNT_KEY = "awakeCount"
DB_FITBIT_COLLECTION_SLEEP_DATA_AWAKE_MIN_KEY = "awakeMinutes"
DB_FITBIT_COLLECTION_SLEEP_DATA_ASLEEP_COUNT_KEY = "asleepCount"
DB_FITBIT_COLLECTION_SLEEP_DATA_ASLEEP_MIN_KEY = "asleepMinutes"
DB_FITBIT_COLLECTION_SLEEP_DATA_STAGE_WAKE_VALUE = "wake"
DB_FITBIT_COLLECTION_SLEEP_DATA_STAGE_AWAKE_VALUE = "awake"
DB_FITBIT_COLLECTION_SLEEP_DATA_STAGE_DEEP_VALUE = "deep"
DB_FITBIT_COLLECTION_SLEEP_DATA_STAGE_LIGHT_VALUE = "light"
DB_FITBIT_COLLECTION_SLEEP_DATA_STAGE_REM_VALUE = "rem"
DB_FITBIT_COLLECTION_SLEEP_DATA_STAGE_ASLEEP_VALUE = "asleep"
DB_FITBIT_COLLECTION_SLEEP_DATA_STAGE_RESTLESS_VALUE = "restless"

# ---------------------------------------------#
#      Converted Temperature Documents         #
# ---------------------------------------------#
DB_FITBIT_COLLECTION_COMP_TEMP_SLEEP_START_KEY = "sleep_start"
DB_FITBIT_COLLECTION_COMP_TEMP_SLEEP_END_KEY = "sleep_end"

# --------------------------------#
#          SPo2 Documents         #
# --------------------------------#
DB_FITBIT_COLLECTION_SPO2_TIMESTAMP_KEY = "timestamp"

# --------------------------------#
#  Device Temperature Documents   #
# --------------------------------#
DB_FITBIT_COLLECTION_DEVICE_TEMP_RECORDED_TIME_KEY = "recorded_time"

# --------------------------------#
#   Daily HRV Summary Documents   #
# --------------------------------#
DB_FITBIT_COLLECTION_DAILY_HRV_SUMMARY_TIMESTAMP_KEY = "timestamp"
DB_FITBIT_COLLECTION_DAILY_HRV_SUMMARY_RMSSD_KEY = "rmssd"
DB_FITBIT_COLLECTION_DAILY_HRV_SUMMARY_NREMHR_KEY = "nremhr"
DB_FITBIT_COLLECTION_DAILY_HRV_SUMMARY_ENTROPY_KEY = "entropy"

# --------------------------------#
#    HRV Details Documents        #
# --------------------------------#
DB_FITBIT_COLLECTION_HRV_DETAILS_TIMESTAMP_KEY = "timestamp"
DB_FITBIT_COLLECTION_HRV_DETAILS_COVERAGE_KEY = "coverage"
DB_FITBIT_COLLECTION_HRV_DETAILS_LOW_FREQUENCY_KEY = "low_frequency"
DB_FITBIT_COLLECTION_HRV_DETAILS_HIGH_FREQUENCY_KEY = "high_frequency"
DB_FITBIT_COLLECTION_HRV_DETAILS_RMSSD_KEY = "rmssd"

# --------------------------------#
#       Profile Documents         #
# --------------------------------#
DB_FITBIT_COLLECTION_PROFILE_GENDER_COL = "gender"
DB_FITBIT_COLLECTION_PROFILE_BMI_COL = "bmi"
DB_FITBIT_COLLECTION_PROFILE_AGE_COL = "age"

# ------------------------------------#
#  Respiratory Rate Summary Documents #
# ------------------------------------#
DB_FITBIT_COLLECTION_RESP_RATE_SUMMARY_FULL_SLEEP_BREATHING_RATE_COL = (
    "full_sleep_breathing_rate"
)
DB_FITBIT_COLLECTION_RESP_RATE_SUMMARY_TIMESTAMP_COL = "timestamp"

# --------------------------------#
#     Stress Score Documents      #
# --------------------------------#
DB_FITBIT_COLLECTION_STRESS_SCORE_DATE_COL = "DATE"

# --------------------------------#
#     Wrist Temperature Docs      #
# --------------------------------#
DB_FITBIT_COLLECTION_WRIST_TEMP_RECORDED_TIME_COL = "recorded_time"
DB_FITBIT_COLLECTION_WRIST_TEMP_TEMP_COL = "temperature"

# --------------------------------#
#     Wrist Temperature Docs      #
# --------------------------------#
DB_FITBIT_COLLECTION_ALTITUDE_DATETIME_COL = "dateTime"
DB_FITBIT_COLLECTION_ALTITUDE_ALTITUDE_COL = "value"

# --------------------------------#
#     Wrist Temperature Docs      #
# --------------------------------#
DB_FITBIT_COLLECTION_BADGE_DATETIME_COL = "dateTime"
DB_FITBIT_COLLECTION_BADGE_TYPE_COL = "badgeType"
DB_FITBIT_COLLECTION_BADGE_VALUE_COL = "value"

# --------------------------------#
#          Calories Docs          #
# --------------------------------#
DB_FITBIT_COLLECTION_CALORIES_DATETIME_COL = "dateTime"
DB_FITBIT_COLLECTION_CALORIES_VALUE_COL = "value"

# --------------------------------#
#          Distance Docs          #
# --------------------------------#
DB_FITBIT_COLLECTION_DISTANCE_DATETIME_COL = "dateTime"
DB_FITBIT_COLLECTION_DISTANCE_VALUE_COL = "value"

# ----------------------------------#
#  Estimated Oxygen Variation Docs  #
# ----------------------------------#
DB_FITBIT_COLLECTION_EST_OXY_VAR_DATETIME_COL = "timestamp"
DB_FITBIT_COLLECTION_EST_OXY_VAR_VALUE_COL = "Infrared to Red Signal Ratio"

# ----------------------------------#
#        Heart Rate Docs            #
# ----------------------------------#
DB_FITBIT_COLLECTION_HEART_RATE_DATETIME_COL = "dateTime"
DB_FITBIT_COLLECTION_HEART_RATE_VALUE_KEY = "value"
DB_FITBIT_COLLECTION_HEART_RATE_VALUE_BPM_COL = "bpm"
DB_FITBIT_COLLECTION_HEART_RATE_VALUE_CONFIDENCE_COL = "confidence"

# ----------------------------------#
#       Journal Entries Docs        #
# ----------------------------------#
DB_FITBIT_COLLECTION_JOURNAL_ENTRIES_LOG_TIME_COL = "log_time"
DB_FITBIT_COLLECTION_JOURNAL_ENTRIES_LOG_TYPE_COL = "log_type"
DB_FITBIT_COLLECTION_JOURNAL_ENTRIES_PLATFORM_COL = "platform"
DB_FITBIT_COLLECTION_JOURNAL_ENTRIES_SOURCE_COL = "source"

# ----------------------------------#
#    Lightly Active Minutes Docs    #
# ----------------------------------#
DB_FITBIT_COLLECTION_LIGHTLY_ACTIVE_MIN_DATETIME_COL = "dateTime"
DB_FITBIT_COLLECTION_LIGHTLY_ACTIVE_MIN_VALUE_COL = "value"

# ----------------------------------#
#   Moderately Active Minutes Docs  #
# ----------------------------------#
DB_FITBIT_COLLECTION_MODERATELY_ACTIVE_MIN_DATETIME_COL = "dateTime"
DB_FITBIT_COLLECTION_MODERATELY_ACTIVE_MIN_VALUE_COL = "value"

# ----------------------------------#
#    Very Active Minutes Docs       #
# ----------------------------------#
DB_FITBIT_COLLECTION_VERY_ACTIVE_MIN_DATETIME_COL = "dateTime"
DB_FITBIT_COLLECTION_VERY_ACTIVE_MIN_VALUE_COL = "value"

# ----------------------------------#
#     Sedentary Minutes Docs        #
# ----------------------------------#
DB_FITBIT_COLLECTION_SEDENTARY_MIN_DATETIME_COL = "dateTime"
DB_FITBIT_COLLECTION_SEDENTARY_MIN_VALUE_COL = "value"

# ----------------------------------#
#           Steps Docs              #
# ----------------------------------#
DB_FITBIT_COLLECTION_STEPS_DATETIME_COL = "dateTime"
DB_FITBIT_COLLECTION_STEPS_VALUE_COL = "value"

# ----------------------------------#
#       Water Logs Docs             #
# ----------------------------------#
DB_FITBIT_COLLECTION_WATER_LOGS_DATE_COL = "date"
DB_FITBIT_COLLECTION_WATER_LOGS_WATER_AMOUNT_COL = "waterAmount"
DB_FITBIT_COLLECTION_WATER_LOGS_MEASUREMENT_UNIT_COL = "measurementUnit"

# ----------------------------------#
#    Resting Heart Rate Docs        #
# ----------------------------------#
DB_FITBIT_COLLECTION_RESTING_HEART_RATE_DATETIME_COL = "dateTime"
DB_FITBIT_COLLECTION_RESTING_HEART_RATE_VALUE_KEY = "value"
DB_FITBIT_COLLECTION_RESTING_HEART_RATE_VALUE_VALUE_COL = "value"
DB_FITBIT_COLLECTION_RESTING_HEART_RATE_VALUE_ERROR_COL = "error"
DB_FITBIT_COLLECTION_RESTING_HEART_RATE_VALUE_DATE_COL = "date"

# ----------------------------------#
#      Time in HR Zones Docs        #
# ----------------------------------#
DB_FITBIT_COLLECTION_TIME_IN_HR_ZONES_DATETIME_COL = "dateTime"

# ----------------------------------#
#        HRV Histogram Docs         #
# ----------------------------------#
DB_FITBIT_COLLECTION_HRV_HISTOGRAM_TIMESTAMP_COL = "timestamp"
DB_FITBIT_COLLECTION_HRV_HISTOGRAM_BUCKET_VALUES_COL = "bucket_values"

# ----------------------------------#
#    Demographic VO2 Max Docs       #
# ----------------------------------#
DB_FITBIT_COLLECTION_DEMOGRAPHIC_VO2_MAX_DATETIME_COL = "dateTime"

# ----------------------------------#
#       ECG Recordings Docs         #
# ----------------------------------#
DB_FITBIT_COLLECTION_AFIB_ECG_READINGS_DATETIME_COL = "reading_time"
DB_FITBIT_COLLECTION_AFIB_ECG_READINGS_WAVEFORM_SAMPLES_COL = "waveform_samples"

# ----------------------------------#
#     Mindfulness Goals Docs        #
# ----------------------------------#
DB_FITBIT_COLLECTION_MINDFULNESS_GOALS_DATE_COL = "date"
