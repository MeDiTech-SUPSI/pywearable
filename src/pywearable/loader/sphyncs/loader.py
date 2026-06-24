import os

import pandas as pd

from ..base import BaseLoader


class SphyncsDataLoader(BaseLoader):
    def __init__(self, root_folder: os.PathLike):
        self.root_folder = root_folder

    def load_heart_rate(
        self, user_id: str, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        """Load heart rate data for a specific user.

        Extracts heart rate measurements from the biometrics.parquet file
        and returns a standardized DataFrame with timestamps and BPM values.

        Parameters
        ----------
        user_id : str
            User identifier.
        start_date : pd.Timestamp, optional
            Start date for filtering data (inclusive). Default is None.
        end_date : pd.Timestamp, optional
            End date for filtering data (exclusive). Default is None.

        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns:
            
            - isoDate : datetime64[ns]
                Timestamp of the heart rate measurement.
            - beatsPerMinute : float
                Heart rate value in beats per minute.

        Raises
        ------
        FileNotFoundError
            If the biometrics.parquet file does not exist for the user.
        ValueError
            If required columns 'dateTime' or 'heart' are missing from the biometrics file.
        
        Notes
        -----
        The biometrics.parquet file should be located in the user's folder
        and must contain 'dateTime' and 'heart' columns.
        """
        # Construct path to the user's biometrics.parquet file
        user_folder = os.path.join(self.root_folder, user_id)
        biometrics_file = os.path.join(user_folder, "biometrics.parquet")
        
        # Check if file exists
        if not os.path.exists(biometrics_file):
            raise FileNotFoundError(
                f"Biometrics file not found for user {user_id}: {biometrics_file}"
            )
        
        # Load the parquet file
        df = pd.read_parquet(biometrics_file)
        
        # If dateTime is the index, reset it to make it a column
        if df.index.name == 'dateTime':
            df = df.reset_index()
        
        # Extract dateTime and heart columns
        if 'dateTime' not in df.columns or 'heart' not in df.columns:
            raise ValueError(
                f"Required columns 'dateTime' or 'heart' not found in {biometrics_file}"
            )
        
        # Create the heart rate dataframe
        heart_rate_df = pd.DataFrame({
            'isoDate': pd.to_datetime(df['dateTime']),
            'beatsPerMinute': df['heart']
        })
        
        # Set isoDate as index
        heart_rate_df = heart_rate_df.set_index('isoDate')
        
        # Filter by date range if provided
        if start_date is not None:
            heart_rate_df = heart_rate_df[heart_rate_df.index >= start_date]
        if end_date is not None:
            heart_rate_df = heart_rate_df[heart_rate_df.index < end_date]

        
        # Reset index to have isoDate as a column
        heart_rate_df = heart_rate_df.reset_index()
        
        return heart_rate_df

    def load_daily_summary(self, user_id: str, start_date=None, end_date=None) -> pd.DataFrame:
        """Return empty DataFrame - not available in SPHYNCS dataset."""
        return pd.DataFrame()

    def load_sleep_summary(
        self, user_id: str, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None, drop_duplicates: bool = True
    ) -> pd.DataFrame:
        """Load sleep summary data for a specific user.

        Aggregates sleep episode and sleep stage data to compute sleep metrics
        including durations for different sleep stages (deep, light, REM, awake).
        Extracts data from sleep_episodes.parquet and sleep_stages.parquet files.

        Parameters
        ----------
        user_id : str
            User identifier.
        start_date : pd.Timestamp, optional
            Start date for filtering data (inclusive). Default is None.
        end_date : pd.Timestamp, optional
            End date for filtering data (exclusive). Default is None.
        drop_duplicates : bool, default=True
            Whether to drop duplicate entries based on calendarDate.

        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns:
            
            - sleepSummaryId : str
                Unique identifier for each sleep summary (format: userId_YYYYMMDD_HHMMSS).
            - calendarDate : datetime64[ns]
                Calendar date for the sleep episode.
            - isoDate : datetime64[ns]
                Sleep start time (bedtime) in ISO format.
            - unixTimestampInMs : int64
                Unix timestamp of bedtime in milliseconds.
            - timezoneOffsetInMs : int
                Timezone offset in milliseconds (CET = 3600000).
            - durationInMs : int
                Overall sleep duration in milliseconds (excluding awake periods).
            - deepSleepDurationInMs : int
                Duration of deep sleep in milliseconds.
            - lightSleepDurationInMs : int
                Duration of light sleep in milliseconds.
            - remSleepDurationInMs : int
                Duration of REM sleep in milliseconds.
            - unmeasurableSleepDurationInMs : int
                Duration of unmeasurable/restless sleep in milliseconds.
            - awakeDurationInMs : int
                Duration of awake stages in milliseconds.
            - validation : int
                Validation flag (1 if contains 'restless' or 'asleep' stages, 0 otherwise).

        Raises
        ------
        FileNotFoundError
            If sleep_episodes.parquet or sleep_stages.parquet do not exist for the user.
        ValueError
            If required columns ('startTime', 'endTime', 'sleepStage') are missing.
        
        Notes
        -----
        - Returns an empty DataFrame if no data is available.
        - The timezone offset is set to CET (UTC+1 = 3600000 ms) for all entries.
        - Sleep stages are mapped from SPHYNCS format to standard format.
        """
        # Construct paths to the user's sleep data files
        user_folder = os.path.join(self.root_folder, user_id)
        episodes_file = os.path.join(user_folder, "sleep_episodes.parquet")
        stages_file = os.path.join(user_folder, "sleep_stages.parquet")
        
        # Check if files exist
        if not os.path.exists(episodes_file):
            return pd.DataFrame()
        if not os.path.exists(stages_file):
            return pd.DataFrame()
        
        # Load the parquet files
        episodes_df = pd.read_parquet(episodes_file)
        stages_df = pd.read_parquet(stages_file)
        
        # Validate required columns
        if 'startTime' not in episodes_df.columns or 'endTime' not in episodes_df.columns:
            return pd.DataFrame()
        if 'sleepStage' not in stages_df.columns:
            return pd.DataFrame()
        
        # Reset index if dateTime is the index
        if episodes_df.index.name == 'dateTime':
            episodes_df = episodes_df.reset_index()
        
        # Initialize result list
        sleep_summaries = []
        
        # Process each sleep episode
        for idx, episode in episodes_df.iterrows():
            calendar_date = episode['dateTime']
            start_time = episode['startTime']
            end_time = episode['endTime']
            
            # Create unique sleep summary ID using timestamp
            sleep_summary_id = f"{user_id}_{start_time.strftime('%Y%m%d_%H%M%S')}"
            
            # Filter stages for this sleep episode
            stage_mask = (stages_df.index >= start_time) & (stages_df.index <= end_time)
            episode_stages = stages_df[stage_mask].copy()
            
            if len(episode_stages) == 0:
                # No stage data for this episode, skip or use zeros
                continue
            
            # This aligns the summary with the artificial 'awake' stage created in load_sleep_stage.
            first_stage_time = episode_stages.index[0]
            initial_gap_ms = (first_stage_time - start_time).total_seconds() * 1000
            initial_gap_ms = int(initial_gap_ms)
            
            # Calculate duration for each stage
            # Add next timestamp to compute time differences
            episode_stages['next_time'] = episode_stages.index.to_series().shift(-1)
            
            # For the last stage, use the episode end time
            episode_stages.loc[episode_stages['next_time'].isna(), 'next_time'] = end_time
            
            # Calculate duration in milliseconds
            episode_stages['duration_ms'] = (
                (episode_stages['next_time'] - episode_stages.index).dt.total_seconds() * 1000
            ).astype(int)
            
            # Aggregate durations by stage type
            stage_durations = episode_stages.groupby('sleepStage')['duration_ms'].sum()
            
            # Check if episode contains 'restless' or 'asleep' stages
            unique_stages = episode_stages['sleepStage'].unique()
            has_unreliable = int('restless' in unique_stages or 'asleep' in unique_stages)
            
            # Map stages to expected columns
            # Stages: 'wake', 'light', 'deep', 'rem', 'asleep', 'restless', 'awake'
            deep_sleep_ms = int(stage_durations.get('deep', 0))
            light_sleep_ms = int(stage_durations.get('light', 0))
            rem_sleep_ms = int(stage_durations.get('rem', 0))
            awake_ms = int(stage_durations.get('wake', 0) + stage_durations.get('awake', 0)) + initial_gap_ms
            unmeasurable_ms = int(stage_durations.get('restless', 0) + stage_durations.get('asleep', 0))
            
            # Calculate total sleep duration (excluding awake periods)
            duration_ms = deep_sleep_ms + light_sleep_ms + rem_sleep_ms + unmeasurable_ms
            
            # Convert timestamp to unix timestamp in milliseconds
            timestamps = pd.Timestamp(episode['startTime'])
            unix_timestamp_ms = int(timestamps.value // 10**6)
            
            # Timezone offset (CET = UTC+1)
            timezone_offset_ms = 3600000
            
            # Create sleep summary entry
            summary = {
                'sleepSummaryId': sleep_summary_id,
                'calendarDate': pd.Timestamp(calendar_date.date()),
                'isoDate': pd.Timestamp(start_time),
                'unixTimestampInMs': unix_timestamp_ms,
                'timezoneOffsetInMs': timezone_offset_ms,
                'durationInMs': duration_ms,
                'deepSleepDurationInMs': deep_sleep_ms,
                'lightSleepDurationInMs': light_sleep_ms,
                'remSleepDurationInMs': rem_sleep_ms,
                'unmeasurableSleepDurationInMs': unmeasurable_ms,
                'awakeDurationInMs': awake_ms,
                'validation': has_unreliable
            }
            
            sleep_summaries.append(summary)
        
        # Create result DataFrame
        if len(sleep_summaries) == 0:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(sleep_summaries)
        
        # Filter by date range if provided
        if start_date is not None:
            start_date = pd.Timestamp(start_date)
            result_df = result_df[result_df['calendarDate'] >= start_date]
        if end_date is not None:
            end_date = pd.Timestamp(end_date)
            result_df = result_df[result_df['calendarDate'] < end_date]
        
        # Drop duplicates if requested (keep first occurrence by calendarDate)
        if drop_duplicates and len(result_df) > 0:
            result_df = result_df.drop_duplicates(subset=['calendarDate'], keep='first')
        
        return result_df.reset_index(drop=True)

    def load_sleep_stage(
        self, user_id: str, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        """Load sleep stage data for a specific user.

        Extracts sleep stage information from sleep_stages.parquet and maps it
        to sleep episodes, calculating the duration of each stage. Automatically
        detects and adds an initial 'awake' stage if there's a gap between bedtime
        and the first recorded stage.

        Parameters
        ----------
        user_id : str
            User identifier.
        start_date : pd.Timestamp, optional
            Start date for filtering data (inclusive). Default is None.
        end_date : pd.Timestamp, optional
            End date for filtering data (exclusive). Default is None.

        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns:
            
            - sleepSummaryId : str
                Reference to the corresponding sleep summary (format: userId_YYYYMMDD_HHMMSS).
            - isoDate : datetime64[ns]
                Timestamp of the stage start.
            - unixTimestampInMs : int64
                Unix timestamp in milliseconds.
            - durationInMs : int
                Duration of the stage in milliseconds.
            - type : str
                Type of sleep stage. Possible values: 'deep', 'light', 'rem', 'awake', 'unmeasurable'.

        Raises
        ------
        FileNotFoundError
            If sleep_episodes.parquet or sleep_stages.parquet do not exist for the user.
        ValueError
            If required columns ('startTime', 'endTime', 'sleepStage') are missing.
        
        Notes
        -----
        - Returns an empty DataFrame if no data is available.
        - Stage mapping from SPHYNCS to standard format:
          * 'deep' -> 'deep'
          * 'light' -> 'light'
          * 'rem' -> 'rem'
          * 'wake', 'awake' -> 'awake'
          * 'restless', 'asleep' -> 'unmeasurable'
        - If there's a time gap between bedtime and the first recorded stage,
          an initial 'awake' stage is automatically added to account for the difference.
        """
        # Construct paths to the user's sleep data files
        user_folder = os.path.join(self.root_folder, user_id)
        episodes_file = os.path.join(user_folder, "sleep_episodes.parquet")
        stages_file = os.path.join(user_folder, "sleep_stages.parquet")
        
        # Check if files exist
        if not os.path.exists(episodes_file):
            return pd.DataFrame()
        if not os.path.exists(stages_file):
            return pd.DataFrame()
        
        # Load the parquet files
        episodes_df = pd.read_parquet(episodes_file)
        stages_df = pd.read_parquet(stages_file)
        
        # Validate required columns
        if 'startTime' not in episodes_df.columns or 'endTime' not in episodes_df.columns:
            return pd.DataFrame()
        if 'sleepStage' not in stages_df.columns:
            return pd.DataFrame()
        
        # Reset index if dateTime is the index
        if episodes_df.index.name == 'dateTime':
            episodes_df = episodes_df.reset_index()
        
        # Stage mapping from SPHYNCS format to standard format
        stage_mapping = {
            'deep': 'deep',
            'light': 'light',
            'rem': 'rem',
            'wake': 'awake',
            'awake': 'awake',
            'restless': 'unmeasurable',
            'asleep': 'unmeasurable'
        }
        
        # Initialize result list
        all_stages = []
        
        # Process each sleep episode
        for idx, episode in episodes_df.iterrows():
            start_time = episode['startTime']
            end_time = episode['endTime']
            
            # Create same sleep summary ID as in load_sleep_summary
            sleep_summary_id = f"{user_id}_{start_time.strftime('%Y%m%d_%H%M%S')}"
            
            # Filter stages for this sleep episode
            stage_mask = (stages_df.index >= start_time) & (stages_df.index <= end_time)
            episode_stages = stages_df[stage_mask].copy()
            
            if len(episode_stages) == 0:
                continue
            
            # Reset index to have timestamp as a column
            episode_stages = episode_stages.reset_index()
            episode_stages = episode_stages.rename(columns={episode_stages.columns[0]: 'timestamp'})
            
            # Check if there's a gap between bedtime and first stage
            first_stage_time = episode_stages['timestamp'].iloc[0]
            time_diff_ms = (first_stage_time - start_time).total_seconds() * 1000
            
            # If there's a gap > 0, add an initial 'awake' stage
            stages_list = []
            if time_diff_ms > 0:
                # Calculate unix timestamp for bedtime
                bedtime_unix_ms = int(pd.Timestamp(start_time).value // 10**6)
                
                initial_awake = {
                    'sleepSummaryId': sleep_summary_id,
                    'isoDate': pd.Timestamp(start_time),
                    'unixTimestampInMs': bedtime_unix_ms,
                    'durationInMs': int(time_diff_ms),
                    'type': 'awake'
                }
                stages_list.append(initial_awake)
            
            # Calculate duration for each stage
            episode_stages['next_time'] = episode_stages['timestamp'].shift(-1)
            
            # For the last stage, use the episode end time
            episode_stages.loc[episode_stages['next_time'].isna(), 'next_time'] = end_time
            
            # Calculate duration in milliseconds
            episode_stages['durationInMs'] = (
                (episode_stages['next_time'] - episode_stages['timestamp']).dt.total_seconds() * 1000
            ).astype(int)
            
            # Map stage types to standard format
            episode_stages['type'] = episode_stages['sleepStage'].map(stage_mapping)
            
            # Handle unknown stage types (if any)
            episode_stages['type'] = episode_stages['type'].fillna('unmeasurable')
            
            # Convert timestamps to unix timestamp in milliseconds
            episode_stages['unixTimestampInMs'] = episode_stages['timestamp'].apply(
                lambda x: int(pd.Timestamp(x).value // 10**6)
            )
            
            # Create stage records
            for _, row in episode_stages.iterrows():
                stage_record = {
                    'sleepSummaryId': sleep_summary_id,
                    'isoDate': pd.Timestamp(row['timestamp']),
                    'unixTimestampInMs': row['unixTimestampInMs'],
                    'durationInMs': row['durationInMs'],
                    'type': row['type']
                }
                stages_list.append(stage_record)
            
            all_stages.extend(stages_list)
        
        # Combine all stages
        if len(all_stages) == 0:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(all_stages)
        
        # Filter by date range if provided
        if start_date is not None:
            start_date = pd.Timestamp(start_date)
            result_df = result_df[result_df['isoDate'] >= start_date]
        if end_date is not None:
            end_date = pd.Timestamp(end_date)
            result_df = result_df[result_df['isoDate'] < end_date]
        
        return result_df.reset_index(drop=True)

    def load_pulse_ox(
        self, user_id: str, start_date=None, end_date=None
    ) -> pd.DataFrame:
        """Return empty DataFrame - not available in SPHYNCS dataset."""
        return pd.DataFrame()

    def load_respiration(
        self, user_id: str, start_date=None, end_date=None
    ) -> pd.DataFrame:
        """Return empty DataFrame - not available in SPHYNCS dataset."""
        return pd.DataFrame()

    def load_sleep_pulse_ox(
        self, user_id: str, start_date=None, end_date=None
    ) -> pd.DataFrame:
        """Return empty DataFrame - not available in SPHYNCS dataset."""
        return pd.DataFrame()

    def load_sleep_respiration(
        self, user_id: str, start_date=None, end_date=None
    ) -> pd.DataFrame:
        """Return empty DataFrame - not available in SPHYNCS dataset."""
        return pd.DataFrame()

    def load_bbi(
        self, user_id: str, start_date=None, end_date=None
    ) -> pd.DataFrame:
        """Return empty DataFrame - not available in SPHYNCS dataset."""
        return pd.DataFrame()
    
    def load_stress(self, user_id: str, start_date=None, end_date=None) -> pd.DataFrame:
        """Return empty DataFrame - not available in SPHYNCS dataset."""
        return pd.DataFrame()
    
    def load_epoch(
        self, user_id: str, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        """Load epoch data with step counts only.

        Loads step data from the biometrics.parquet file, aggregates it into 
        15-minute epochs, and returns a standardized DataFrame. Intensity is 
        left as None since it's not calculated.

        Parameters
        ----------
        user_id : str
            User identifier.
        start_date : pd.Timestamp, optional
            Start date for filtering data (inclusive). Default is None.
        end_date : pd.Timestamp, optional
            End date for filtering data (exclusive). Default is None.

        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns:
            
            - isoDate : datetime64[ns]
                Timestamp of the 15-minute epoch.
            - intensity : NoneType
                Not calculated, left as None.
            - durationInMs : int
                Duration of the epoch in milliseconds (900000 for 15 minutes).
            - steps : float or int
                Sum of steps within the epoch.

        Raises
        ------
        FileNotFoundError
            If the biometrics.parquet file does not exist for the user.
        ValueError
            If required columns 'dateTime' or 'steps' are missing from the biometrics file.

        Notes
        -----
        The biometrics.parquet file should be located in the user's folder
        and must contain 'dateTime' and 'steps' columns.
        """
        # Construct path to the user's biometrics.parquet file
        user_folder = os.path.join(self.root_folder, user_id)
        biometrics_file = os.path.join(user_folder, "biometrics.parquet")
        
        # Check if file exists
        if not os.path.exists(biometrics_file):
            raise FileNotFoundError(
                f"Biometrics file not found for user {user_id}: {biometrics_file}"
            )
        
        # Load the parquet file
        df = pd.read_parquet(biometrics_file)
        
        # If dateTime is the index, reset it to make it a column
        if df.index.name == 'dateTime':
            df = df.reset_index()
            
        # Extract dateTime and steps columns
        if 'dateTime' not in df.columns or 'steps' not in df.columns:
            raise ValueError(
                f"Required columns 'dateTime' or 'steps' not found in {biometrics_file}"
            )
            
        # Create the steps dataframe
        steps_df = pd.DataFrame({
            'isoDate': pd.to_datetime(df['dateTime']),
            'steps': df['steps']
        })
        
        # Set isoDate as index for filtering and resampling
        steps_df = steps_df.set_index('isoDate')
        
        # Filter by date range if provided
        if start_date is not None:
            steps_df = steps_df[steps_df.index >= start_date]
        if end_date is not None:
            steps_df = steps_df[steps_df.index < end_date]
            
        # Resample to 15-minute epochs and sum the steps
        epochs = steps_df.resample('15Min')['steps'].sum().reset_index()
        
        # Format as expected by pywearable
        result = pd.DataFrame({
            'isoDate': epochs['isoDate'],
            'intensity': None,
            'durationInMs': 900000,
            'steps': epochs['steps']
        })
        
        return result

    def get_user_ids(self) -> list:
        """Get available user IDs from the dataset.

        This method retrieves all patient IDs by listing the subdirectories
        in the root folder. Each subdirectory name represents a unique patient ID.

        Returns
        -------
        list
            List of strings representing unique patient IDs (subdirectory names).
        """
        if not os.path.exists(self.root_folder):
            raise FileNotFoundError(
                f"Root folder does not exist: {self.root_folder}"
            )

        user_ids = []
        for entry in os.listdir(self.root_folder):
            entry_path = os.path.join(self.root_folder, entry)
            if os.path.isdir(entry_path):
                user_ids.append(entry)

        return sorted(user_ids)
