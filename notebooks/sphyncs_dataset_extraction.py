import pandas as pd

import pywearable.adherence
import pywearable.cardiac
import pywearable.sleep
from pywearable.loader import SpynchsDataLoader

loader = SpynchsDataLoader(root_folder=None)

user_ids = loader.get_user_ids()

START_DATE = "2021-01-01"
END_DATE = "2023-01-01"

# - pywearable.sleep.get_sleep_statistics
# - pywearable.cardiac.get_heart_rate_statistics

# - pywearable.cycle.get_cycle_awake_statistics
# - pywearable.cycle.get_cycle_sleep_statistics
# - pywearable.cycle.get_cycle_physiological_statistics

# - pywearable.adherence.get_daily_wearing_time

sleep_statistics_list = []
wearing_time_list = []
for user in user_ids:
    sleep_statistics_list.append(
        pywearable.sleep.get_sleep_statistics(
            loader=loader, user_id=user, start_date=START_DATE, end_date=END_DATE
        ).assign(user_id=user)
    )
    wearing_time_list.append(
        pywearable.adherence.get_daily_wearing_time(
            loader=loader, user_id=user, start_date=START_DATE, end_date=END_DATE
        )
        .to_frame()
        .assign(user_id=user)
    )


sleep_statistics = pd.concat(sleep_statistics_list, axis=0)
wearing_time = pd.concat(wearing_time_list, axis=0)

dataset = pd.concat((wearing_time, sleep_statistics), axis=1)
