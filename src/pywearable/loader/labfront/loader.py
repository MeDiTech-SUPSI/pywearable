"""
This module contains all the functions related to the loading of data from Labfront.

"""

import datetime
import os
import re
import warnings
from pathlib import Path
from typing import Tuple, Union

import dateutil.parser
import numpy as np
import pandas as pd

from ... import schema
from ..base import BaseLoader
from ..garmin import (
    _normalize_garmin_sleep_columns,
    prepare_epoch_activity_samples,
)
from . import constants as lf_constants

_LABFRONT_METRICS_DICT = {
    lf_constants.METRIC_HEART_RATE: {
        lf_constants.DATA_SOURCE_GARMIN_HEALTH_API: lf_constants.GARMIN_CONNECT_HEART_RATE_FOLDER,
        lf_constants.DATA_SOURCE_GARMIN_SDK: lf_constants.GARMIN_DEVICE_HEART_RATE_FOLDER,
    },
    lf_constants.METRIC_STRESS: {
        lf_constants.DATA_SOURCE_GARMIN_HEALTH_API: lf_constants.GARMIN_CONNECT_STRESS_FOLDER,
        lf_constants.DATA_SOURCE_GARMIN_SDK: lf_constants.GARMIN_DEVICE_HEART_RATE_FOLDER,
    },
    lf_constants.METRIC_PULSE_OX: {
        lf_constants.DATA_SOURCE_GARMIN_HEALTH_API: lf_constants.GARMIN_CONNECT_DAILY_PULSE_OX_FOLDER,
        lf_constants.DATA_SOURCE_GARMIN_SDK: lf_constants.GARMIN_DEVICE_PULSE_OX_FOLDER,
    },
    lf_constants.METRIC_RESPIRATION: {
        lf_constants.DATA_SOURCE_GARMIN_HEALTH_API: lf_constants.GARMIN_CONNECT_RESPIRATION_FOLDER,
        lf_constants.DATA_SOURCE_GARMIN_SDK: lf_constants.GARMIN_DEVICE_RESPIRATION_FOLDER,
    },
}

# Temporary internal key used to keep export snapshots separate while their
# rows are reconciled. Public loader output never retains this column.
_LABFRONT_SOURCE_FILE_COL = "__labfront_source_file"
_SLEEP_STAGE_CONFLICT_POLICIES = ("drop_night", "raise", "last")

LABFRONT_FILE_INDEX_NAME = ".pywearable_file_index.csv"


class LabfrontLoader(BaseLoader):
    """Loader for Labfront data.

    This is the main object for loading and analyzing data collected
    with Labfront.

    Parameters
    ----------
    data_paths : str, pathlib.Path, or list
        Path or paths to folders containing Labfront data.
    reindex_files : bool, optional
        Rebuild the local Labfront file index, by default False.
    sleep_stage_on_conflict : {"drop_night", "raise", "last"}, optional
        How to handle repeated sleep-stage snapshots that cannot be resolved
        against a sleep summary. ``"drop_night"`` keeps the summary but omits
        the ambiguous stage rows, ``"raise"`` stops loading, and ``"last"``
        uses the last snapshot in Labfront file-index order. By default
        ``"drop_night"``.
    """

    hrv_source = "bbi"
    respiratory_rate_source = "samples"
    spo2_source = "samples"
    activity_source = "samples"
    activity_time_columns = (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL)
    activity_primary_resolution = "15min"
    provider_features = frozenset(
        {
            "garmin_stress_level",
            "garmin_body_battery",
            "garmin_daily_average_stress",
            "garmin_daily_max_stress",
            "garmin_daily_stress_duration_seconds",
            "garmin_daily_rest_stress_duration_seconds",
            "garmin_daily_activity_stress_duration_seconds",
            "garmin_daily_low_stress_duration_seconds",
            "garmin_daily_medium_stress_duration_seconds",
            "garmin_daily_high_stress_duration_seconds",
            "garmin_daily_stress_qualifier",
        }
    )
    provider_feature_time_columns = (schema.UTC_TIME_COL, schema.LOCAL_TIME_COL)

    def __init__(
        self,
        data_paths: Union[str, Path, list],
        reindex_files: bool = False,
        sleep_stage_on_conflict: str = "drop_night",
    ):
        """Constructor method"""
        self.sleep_stage_on_conflict = self._validate_sleep_stage_conflict_policy(
            sleep_stage_on_conflict
        )
        self.set_path(data_paths, reindex_files=reindex_files)

    @staticmethod
    def _validate_sleep_stage_conflict_policy(on_conflict: str) -> str:
        if on_conflict not in _SLEEP_STAGE_CONFLICT_POLICIES:
            raise ValueError(
                "on_conflict must be one of "
                f"{_SLEEP_STAGE_CONFLICT_POLICIES}, not {on_conflict}"
            )
        return on_conflict

    def set_path(self, data_paths: Union[str, Path, list], reindex_files: bool = False):
        """Set path to folder containing Labfront data.

        This function allows to change the path of Labfront data
        to be loaded.

        Parameters
        ----------
        data_path : str or Path
            Path to folder containing Labfront data.
        """
        if isinstance(data_paths, str):
            self.data_paths = [Path(data_paths)]
        elif isinstance(data_paths, Path):
            self.data_paths = [data_paths]
        elif isinstance(data_paths, list):
            self.data_paths = [Path(x) for x in data_paths]
        else:
            raise ValueError(
                f"data_paths must be either Path, str, or list, not {type(data_paths)}"
            )
        self.file_index = self.get_file_index(reindex_files=reindex_files)

    def get_user_id(self, full_id: str) -> str:
        """Extract user ID from full ID.

        Parameters
        ----------
        full_id : str
            Full ID, composed of User ID and Labfront ID.

        Returns
        -------
        str
            User ID.

        Raises
        ------
        ValueError
            If not user ID could be extracted.
        """
        if "_" in full_id:
            # Labfront separates user ID from Labfront ID using "_"
            return full_id.split("_")[0]
        else:
            raise ValueError("Could not extract user ID.")

    def get_user_ids(self) -> list:
        """Get available user IDs.

        Returns
        -------
        list
            List of available user IDs.
        """
        return (
            self.file_index.index.get_level_values(
                lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY
            )
            .unique()
            .tolist()
        )

    def get_labfront_ids(self) -> list:
        """Get available Labfront IDs.

        Returns
        -------
        list
            List of available Labfront IDs.
        """
        return (
            self.file_index.index.get_level_values(
                lf_constants.FILE_INDEX_PARTICIPANT_LABFRONT_ID_KEY
            )
            .unique()
            .tolist()
        )

    def get_full_ids(self) -> list:
        """Get available full IDs, composed of both user and Labfront IDs.

        Returns
        -------
        list
            List of available full IDs.
        """
        return (
            self.file_index.index.get_level_values(
                lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY
            )
            .unique()
            .tolist()
        )

    def get_full_id(self, id: str) -> str:
        """Get full participant ID.

        Parameters
        ----------
        id : str
            Identifier for the user. This can be either the user ID,
            the Labfront ID, or the full ID.

        Returns
        -------
        str
            Full ID for the user, composed of both user and Labfront IDs.

        Raises
        ------
        ValueError
            Multiple users exist with the same user ID.
        ValueError
            Invalid ID.
        """
        if "_" in id:
            # This is already a full ID
            return id
        elif (
            id
            in self.file_index.index.get_level_values(
                lf_constants.FILE_INDEX_PARTICIPANT_ID_KEY
            ).unique()
        ):
            # This is a user ID
            # Check if we have more than one participant with the same user id
            if (
                len(
                    self.file_index.loc[(slice(None), id, slice(None), slice(None)), :]
                    .index.get_level_values(
                        lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY
                    )
                    .unique()
                )
                > 1
            ):
                raise ValueError(
                    f"Multiple users exist with ID {id}. Specify either Labfront ID or full ID."
                )
            # Return full id corresponding to this user id
            return (
                self.file_index.loc[(slice(None), id, slice(None), slice(None)), :]
                .index.get_level_values(lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY)
                .unique()[0]
            )
        elif (
            id
            in self.file_index.index.get_level_values(
                lf_constants.FILE_INDEX_PARTICIPANT_LABFRONT_ID_KEY
            ).unique()
        ):
            # This is a Labfront ID
            return (
                self.file_index.loc[(slice(None), slice(None), id, slice(None)), :]
                .index.get_level_values(lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY)
                .unique()[0]
            )
        else:
            raise ValueError(f"Could not get full_id from {id}")

    def get_available_questionnaires(
        self, user_id: Union[str, list] = "all"
    ) -> list:
        """Get the list of available questionnaires.

        Parameters
        ----------
            user_id: str or list, optional
                IDs of the users for which to return available questionnaires, by default "all".
        Returns
        -------
        list
            Alphabetically sorted questionnaire names for the selected users.
        """
        required = {
            lf_constants.FILE_INDEX_IS_QUESTIONNAIRE_KEY,
            lf_constants.FILE_INDEX_TASK_NAME_KEY,
        }
        if not required.issubset(self.file_index.columns):
            return []

        if user_id == "all":
            participant_ids = self.get_full_ids()
        else:
            requested_ids = [user_id] if isinstance(user_id, str) else list(user_id)
            participant_ids = [self.get_full_id(value) for value in requested_ids]
        questionnaires = self.file_index.loc[participant_ids, :]
        questionnaires = questionnaires.loc[
            questionnaires[lf_constants.FILE_INDEX_IS_QUESTIONNAIRE_KEY].eq(True),
            lf_constants.FILE_INDEX_TASK_NAME_KEY,
        ].dropna()
        return sorted(questionnaires.astype(str).unique().tolist())

    def get_available_todos(
        self, user_id: Union[str, list] = "all", return_dict: bool = False
    ) -> list:
        """Get available todos for given users.

        Parameters
        ----------
        user_id : :class:`str` or list, optional
            ID(s) of the users for which todos should be retrieved, by default "all"
        return_dict : bool, optional
            Whether to return a dictionary of the name of the todos
            and their full ids, or simply a sorted list of the available todos, by default False

        Returns
        -------
        list
            alphabetically sorted names of the todos for the user(s).

        Raises
        ------
        FileNotFoundError
            if data path does not exist
        """
        if lf_constants.FILE_INDEX_TASK_NAME_KEY in self.file_index:
            return (
                self.file_index.loc[
                    self.file_index[lf_constants.FILE_INDEX_IS_TODO_KEY] == True,
                    lf_constants.FILE_INDEX_TASK_NAME_KEY,
                ]
                .unique()
                .tolist()
            )
        else:
            return []

    def get_stored_file_index(self, data_folder: Union[str, Path]) -> pd.DataFrame:
        if Path(data_folder / LABFRONT_FILE_INDEX_NAME).exists():
            # load it
            file_index = pd.read_csv(
                data_folder / LABFRONT_FILE_INDEX_NAME,
                parse_dates=[
                    lf_constants.FILE_INDEX_FIRST_TIMESTAMP_KEY,
                    lf_constants.FILE_INDEX_LAST_TIMESTAMP_KEY,
                ],
                date_format="ISO8601",
            )
            return file_index
        else:
            return None

    def get_file_index(self, reindex_files: bool = False) -> pd.DataFrame:
        """Create internal file index to speed up finding file.

        This function creates an internal file index with
        information about the metrics and the timerange for
        which data are available for each user. The
        resulting :class:`pd.DataFrame` has the following
        structure::

                          labfrontId userId fileName startDateTime endDateTime
            fullId metric

        This function also saves the file index information into
        the data folder. If reindex is True, then the files are
        all indexed again. If instead it is False, then the information
        already available in the file is used.

        Returns
        -------
        :class:`pd.DataFrame`
            _description_

        Raises
        ------
        FileNotFoundError
            _description_
        """
        global_file_index_list = []
        for data_folder in self.data_paths:
            data_folder = Path(data_folder)
            if not data_folder.exists():
                raise FileNotFoundError(f"Folder {data_folder} does not exist.")
            # Get stored file index
            file_index = self.get_stored_file_index(data_folder)
            if (file_index is None) or reindex_files:

                file_index = pd.DataFrame()
                participants_folders = [x for x in data_folder.iterdir() if x.is_dir()]
                file_index_list = []
                for participant_folder in participants_folders:
                    # For each participant
                    metrics_folders = [
                        x for x in participant_folder.iterdir() if x.is_dir()
                    ]
                    for metric_folder in metrics_folders:
                        if (
                            "movesense-stream" in metric_folder.name
                            or "women" in metric_folder.name
                        ):
                            continue
                        # Set is_questionnaire/is_todo
                        is_questionnaire = False
                        is_todo = False
                        if lf_constants.QUESTIONNAIRE_FOLDER in metric_folder.name:
                            is_questionnaire = True
                        elif lf_constants.TODO_FOLDER in metric_folder.name:
                            is_todo = True
                        # Get the number of files inside folder
                        n_files = len(os.listdir(metric_folder))
                        # Get subfolders (questionnaires/todos)
                        subfolders = [x for x in metric_folder.iterdir() if x.is_dir()]
                        if len(subfolders) == 0:
                            # Get csv files in metric folder
                            csv_files = [
                                x
                                for x in metric_folder.iterdir()
                                if x.is_file() and x.name.endswith("csv")
                            ]
                            subfolders = []
                        else:
                            # Get csv files from nested metric folder
                            items = [
                                (subfolder.name, x)
                                for subfolder in subfolders
                                for x in subfolder.iterdir()
                                if x.is_file() and x.name.endswith("csv")
                            ]
                            subfolders = [x[0] for x in items]
                            csv_files = [x[1] for x in items]
                        # Get first dates in the csv files
                        dates = [
                            self._get_first_and_last_date_in_csv_file(
                                csv_file,
                                n_files=n_files,
                                is_questionnaire=is_questionnaire,
                            )
                            for csv_file in csv_files
                        ]
                        first_dates = [date[0] for date in dates]
                        # Get last dates in the csv files
                        last_dates = [date[1] for date in dates]

                        # Get last edit times
                        last_edit_times = [
                            os.path.getmtime(csv_file) for csv_file in csv_files
                        ]

                        # Save everything into a DataFrame
                        metric_file_index = pd.DataFrame(
                            {
                                lf_constants.FILE_INDEX_FILENAME_KEY: [
                                    csv_file.name for csv_file in csv_files
                                ],
                                lf_constants.FILE_INDEX_FIRST_TIMESTAMP_KEY: first_dates,
                                lf_constants.FILE_INDEX_LAST_TIMESTAMP_KEY: last_dates,
                                lf_constants.FILE_INDEX_LAST_EDIT_TIMESTAMP_KEY: last_edit_times,
                                lf_constants.FILE_INDEX_IS_QUESTIONNAIRE_KEY: is_questionnaire,
                                lf_constants.FILE_INDEX_IS_TODO_KEY: is_todo,
                                lf_constants.FILE_INDEX_METRIC_KEY: metric_folder.name,
                                lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY: participant_folder.name,
                            }
                        )
                        # Get task info
                        if is_questionnaire or is_todo:
                            task_info = [
                                self._get_task_info(
                                    csv_file, is_questionnaire=is_questionnaire
                                )
                                for csv_file in csv_files
                            ]
                            metric_file_index[lf_constants.FILE_INDEX_TASK_NAME_KEY] = [
                                x[0] for x in task_info
                            ]
                            metric_file_index[lf_constants.FILE_INDEX_TASK_ID_KEY] = [
                                x[1] for x in task_info
                            ]
                            metric_file_index[
                                lf_constants.FILE_INDEX_TASK_FULL_ID_KEY
                            ] = [x[2] for x in task_info]
                            metric_file_index[lf_constants.FILE_INDEX_TASK_FOLDER] = (
                                subfolders
                            )
                        # Let's drop the row in which we have NaT on both timestamps
                        metric_file_index = metric_file_index.dropna(
                            subset=[
                                lf_constants.FILE_INDEX_FIRST_TIMESTAMP_KEY,
                                lf_constants.FILE_INDEX_LAST_TIMESTAMP_KEY,
                            ],
                            axis=0,
                            how="all",
                        )
                        if len(metric_file_index) == 0:
                            continue

                        # Reindex based on first and last dates
                        def reindex_by_dates_on_groupby(groupby: pd.DataFrame):
                            dates = pd.date_range(
                                groupby.index.min(),
                                groupby[
                                    lf_constants.FILE_INDEX_LAST_TIMESTAMP_KEY
                                ].max(),
                                freq="D",
                            )
                            return groupby.reindex(dates).ffill()

                        metric_file_index = (
                            metric_file_index.set_index(
                                lf_constants.FILE_INDEX_FIRST_TIMESTAMP_KEY
                            )
                            .groupby(lf_constants.FILE_INDEX_FILENAME_KEY)
                            .apply(reindex_by_dates_on_groupby, include_groups=False)
                            .reset_index(
                                names=[
                                    lf_constants.FILE_INDEX_FILENAME_KEY,
                                    lf_constants.FILE_INDEX_FIRST_TIMESTAMP_KEY,
                                ]
                            )
                        )
                        metric_file_index[
                            lf_constants.FILE_INDEX_LAST_TIMESTAMP_KEY
                        ] = metric_file_index[
                            lf_constants.FILE_INDEX_FIRST_TIMESTAMP_KEY
                        ] + pd.Timedelta(
                            days=1
                        )

                        # print(metric_file_index)
                        file_index_list.append(metric_file_index)
                file_index = pd.concat(file_index_list, axis=0, ignore_index=True)
                if (
                    lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY
                    in file_index.columns
                ):
                    # Create two separate columns for IDs
                    file_index[
                        [
                            lf_constants.FILE_INDEX_PARTICIPANT_ID_KEY,
                            lf_constants.FILE_INDEX_PARTICIPANT_LABFRONT_ID_KEY,
                        ]
                    ] = file_index[
                        lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY
                    ].str.split(
                        "_", expand=True
                    )
                # Store it into CSV
                file_index.to_csv(data_folder / ".pywearable_file_index.csv")
            file_index[lf_constants.FILE_INDEX_BASE_FOLDER_KEY] = data_folder
            file_index = file_index.set_index(
                [
                    lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY,
                    lf_constants.FILE_INDEX_PARTICIPANT_ID_KEY,
                    lf_constants.FILE_INDEX_PARTICIPANT_LABFRONT_ID_KEY,
                    lf_constants.FILE_INDEX_METRIC_KEY,
                ]
            )
            global_file_index_list.append(file_index)
        file_index = pd.concat(global_file_index_list, axis=0)
        # Sort by first timestamp and last edit timestamp
        file_index = file_index.sort_values(
            by=[
                lf_constants.FILE_INDEX_FIRST_TIMESTAMP_KEY,
                lf_constants.FILE_INDEX_LAST_EDIT_TIMESTAMP_KEY,
            ],
            ascending=True,
        )
        # Drop duplicate filenames and keep the latest edited ones
        # file_index = file_index.drop_duplicates(
        #    subset=[lf_constants.FILE_INDEX_FILENAME_KEY], keep="last"
        # )
        return file_index

    def _get_task_info(
        self, path_to_file, is_questionnaire: bool
    ) -> Tuple[str, str, str]:
        if is_questionnaire:
            return self._get_questionnaire_info(path_to_file=path_to_file)
        else:
            return self._get_todo_info(path_to_file=path_to_file)

    def _get_questionnaire_info(self, path_to_file) -> Tuple[str, str]:
        questionnaire_info_df = pd.read_csv(
            path_to_file, nrows=1, skiprows=lf_constants.CSV_STATS_SKIP_ROWS
        )
        questionnaire_name = questionnaire_info_df.loc[
            0, lf_constants.QUESTIONNAIRE_NAME_COL
        ]
        questionnaire_id = questionnaire_info_df.loc[
            0, lf_constants.QUESTIONNAIRE_ID_COL
        ]
        task_id = questionnaire_info_df.loc[0, lf_constants.TASK_ID_COL]
        return (questionnaire_name, questionnaire_id, task_id)

    def _get_todo_info(self, path_to_file) -> Tuple[str, str]:
        todo_info_df = pd.read_csv(
            path_to_file, nrows=1, skiprows=lf_constants.CSV_STATS_SKIP_ROWS
        )
        todo_name = todo_info_df.loc[0, lf_constants.TODO_NAME_COL]
        todo_id = todo_info_df.loc[0, lf_constants.TODO_ID_COL]
        task_id = todo_info_df.loc[0, lf_constants.TASK_ID_COL]
        return (todo_name, todo_id, task_id)

    def _get_first_and_last_date_in_csv_file(
        self,
        path_to_file: Union[str, Path],
        n_files: int,
        is_questionnaire: bool,
    ) -> datetime.date:
        print(path_to_file)
        # Get first date of the data in CSV file
        # Either we have it in the file Name (e.g. 240313) or we
        # need to get this information from the CSV file itself
        # -> Let's read the header and see if we have first an last unixTimestamp
        # -> If not, date must be retrieved from the name

        n_rows_to_skip = 1
        if is_questionnaire:
            n_rows_to_skip += self.get_key_length(file_path=path_to_file) + 1

        try:
            temp_df = pd.read_csv(
                path_to_file,
                skiprows=self.get_header_length(file_path=path_to_file)
                + n_rows_to_skip,
            )
        except pd.errors.EmptyDataError:
            return pd.NaT, pd.NaT
        if lf_constants.CALENDAR_DATE_COL in temp_df:
            column = lf_constants.CALENDAR_DATE_COL
            temp_df[column] = pd.to_datetime(
                temp_df[column], format="%Y-%m-%d"
            ).dt.tz_localize(None)
            first_date = temp_df[column].min()
            last_date = temp_df[column].max()
        elif lf_constants.ISODATE_COL in temp_df:
            column = lf_constants.ISODATE_COL
            try:
                temp_df[column] = pd.to_datetime(
                    temp_df[lf_constants.ISODATE_COL].str[:-6],
                    utc=True,
                ).dt.tz_localize(None)
            except:
                print(path_to_file)
                raise
            first_date = temp_df[column].min().normalize()
            last_date = temp_df[column].max().normalize()
        elif lf_constants.UNIXTIMESTAMP_IN_MS_COL in temp_df:
            column = lf_constants.UNIXTIMESTAMP_IN_MS_COL
            temp_df[column] = pd.to_datetime(
                temp_df[column], unit="ms", utc=True
            ).dt.tz_localize(None)
            first_date = temp_df[column].min() - pd.Timedelta(hours=14)
            last_date = temp_df[column].max() + pd.Timedelta(hours=14)
        else:
            raise ValueError(
                f"Could not find column {lf_constants.ISODATE_COL} or {lf_constants.CALENDAR_DATE_COL} or {lf_constants.UNIXTIMESTAMP_IN_MS_COL} in file {path_to_file}"
            )

        return first_date, last_date

    def read_csv_file_header_info(
        self, path_to_csv_file: Union[str, Path]
    ) -> pd.DataFrame:
        head = pd.read_csv(
            path_to_csv_file, skiprows=lf_constants.CSV_STATS_SKIP_ROWS, nrows=1
        )
        return head

    def get_files_from_timerange(
        self,
        user_id: str,
        metric: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        is_questionnaire: bool = False,
        is_todo: bool = False,
        task_name: str = "",
    ) -> list:
        """Get files containing daily data from within a given time range.

        This function retrieves the files that contain data in a given time range. By setting start
        and end times to the time range of interest, this function returns all the files that
        contain data within this time range. This function is based on unix timestamps, thus it
        does not take into account timezones. In order to find the files containing data within
        the timerange, 12 hours are removed (added) from the start_date (end_date).
        In case no files are available to be loaded, then an empty list is returned.

        Parameters
        ----------
        user_id : :class:`str`
            User ID.
        metric : str
            Metric of interest.
        start_date : datetime.datetime or datetime.date or str or None, optional
            Start date for data retrieval, by default None
        end_date : datetime.datetime or datetime.date or str or None, optional
            End date for data retrieval, by default None
        is_questionnaire : bool, optional
            Metric of interest is a questionnaire, by default False
        is_todo : bool, optional
            Metric of interest is a todo, by default False
        task_name : str, optional
            Name of the questionnaire or of the todo, by default None
        Returns
        -------
        list
            _description_

        Raises
        ------
        ValueError
            Invalid settings for questionnaire and todos, or invalid ID.
        """
        if is_questionnaire and is_todo:
            raise ValueError("Select only questionnaire or todo.")
        if (is_questionnaire or is_todo) and (task_name == ""):
            raise ValueError("Please specify name of questionnaire or of todo.")

        # Get full participant_id
        participant_id = self.get_full_id(user_id)

        # Get full task id
        if task_name != "":
            task_name = self.get_task_full_id(task_name)
        if not (
            metric
            in self.file_index.loc[(participant_id)].index.get_level_values(
                lf_constants.FILE_INDEX_METRIC_KEY
            )
        ):
            # Participant does not have files for this metric, return empty list
            return []

        # Get files for this participant and this metric
        participant_files = self.file_index.loc[
            (participant_id, slice(None), slice(None), metric), :
        ].reset_index(drop=True)

        # Return empty list if no files are available for this metric
        if len(participant_files) == 0:
            return []

        if task_name != "":
            participant_files = participant_files.loc[
                participant_files[lf_constants.FILE_INDEX_TASK_FULL_ID_KEY] == task_name
            ]
            # Return empty list if no files are available for this task
            if len(participant_files) == 0:
                return []
        if start_date is None:
            start_date = participant_files[
                lf_constants.FILE_INDEX_FIRST_TIMESTAMP_KEY
            ].min()
        if end_date is None:
            end_date = participant_files[
                lf_constants.FILE_INDEX_LAST_TIMESTAMP_KEY
            ].max()
        # Floor to the previous day because some Labfront files store
        # midnight in the file belonging to the previous day.
        start_date = pd.Timestamp(
            start_date - datetime.timedelta(microseconds=1)
        ).floor("1d")
        # We need to ceil end_date to the next date to
        # handle end-date non inclusiveness with dates
        end_date = pd.Timestamp(end_date + datetime.timedelta(microseconds=1)).ceil(
            "1d"
        )
        # Reset index and fill last timestamps with end date
        # TODO it may happen that end_date is lower than the firstTimestamp in file_index
        if len(participant_files) == 1:
            # We only have one file
            if (
                start_date
                >= participant_files.iloc[0].loc[
                    lf_constants.FILE_INDEX_LAST_TIMESTAMP_KEY
                ]
            ) or (
                end_date
                <= participant_files.iloc[0].loc[
                    lf_constants.FILE_INDEX_FIRST_TIMESTAMP_KEY
                ]
            ):
                return []
        else:
            # Return only files that contain start and end dates
            # TODO truncate does not work if we have a start date that
            # is greater than the minimum start date with only one file
            participant_files = participant_files.set_index(
                lf_constants.FILE_INDEX_FIRST_TIMESTAMP_KEY
            ).truncate(before=start_date, after=end_date)
        participant_files = participant_files.drop_duplicates(
            subset=[lf_constants.FILE_INDEX_FILENAME_KEY]
        )

        file_names = (
            participant_files[lf_constants.FILE_INDEX_BASE_FOLDER_KEY].astype(str)
            + os.path.sep
            + participant_id
            + os.path.sep
            + metric
            + os.path.sep
        )
        if is_questionnaire or is_todo:
            file_names += participant_files[lf_constants.FILE_INDEX_TASK_FOLDER].astype(
                str
            )
            file_names += os.path.sep
        file_names += participant_files[lf_constants.FILE_INDEX_FILENAME_KEY].astype(
            str
        )
        file_names = file_names.drop_duplicates()

        return file_names.values

    def get_data_from_datetime(
        self,
        user_id: str,
        metric: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        is_questionnaire: bool = False,
        is_todo: bool = False,
        task_name: str = "",
        filter_on_calendar_date: bool = False,
        include_source_file: bool = False,
        filter_rows: bool = True,
    ) -> pd.DataFrame:
        """Load data from a given user in a given time frame.

        This function allows to load data of a given metric from a specified user
        only within a given timeframe. If `start_date` and `end_date` are set to None, then
        all data available for the user of interest are returned.
        If no data are available, then an empty :class:`pandas.DataFrame` is returned.

        Parameters
        ----------
        user_id : :class:`str`
            User identifier.
        metric : str
            Metric of interest.
        start_date : datetime.datetime or datetime.date or str or None, optional
            Start date and time of interest, by default None
        end_date : datetime.datetime or datetime.date or str or None, optional
            End date and time of interest, by default None
        is_questionnaire : bool, optional
            Metric of interest is a questionnaire, by default False
        is_todo : bool, optional
            Metric of interest is a todo, by default False
        task_name : str, optional
            Name of the questionnaire or of the todo, by default None
        filter_on_calendar_date : bool, optional
            Filter rows using their calendar-date column rather than their
            timestamp column, by default False.
        include_source_file : bool, optional
            Add the source CSV path as a temporary internal column. Sleep-stage
            reconciliation uses it to distinguish repeated export snapshots;
            it is disabled by default and not part of public loader output.
        filter_rows : bool, optional
            Apply the requested bounds to individual rows after selecting the
            overlapping files, by default True. Snapshot reconciliation sets
            this to False so it can validate complete exports before clipping
            the selected rows.

        Returns
        -------
        pd.DataFrame
            Dataframe with the data of interest.

        Raises
        ------
        ValueError
            Not possible to load from both questionnaires and todos at the same time.
        ValueError
            No name specified for questionnaire or todo.
        ValueError
            The ID was not found among available IDs.
        """

        # Check questionnaire or todo
        if is_questionnaire and is_todo:
            raise ValueError("Select only questionnaire or todo.")
        if (is_questionnaire or is_todo) and (task_name is None):
            raise ValueError("Specify name of questionnaire or of todo.")
        if is_questionnaire or is_todo:
            task_name = self.get_task_full_id(task_name)

        # Check dates and times
        if not (start_date is None):
            start_date = pd.Timestamp(start_date)

        if not (end_date is None):
            end_date = pd.Timestamp(end_date)

        # Get full participant_id (user + labfront)
        participant_id = self.get_full_id(user_id)
        files = self.get_files_from_timerange(
            user_id=participant_id,
            metric=metric,
            start_date=start_date,
            end_date=end_date,
            is_questionnaire=is_questionnaire,
            is_todo=is_todo,
            task_name=task_name,
        )
        if len(files) == 0:
            return pd.DataFrame()

        n_rows_to_skip = self.get_header_length(Path(files[0]))
        if is_questionnaire:
            n_rows_to_skip += self.get_key_length(Path(files[0])) + 1
        # Load data from first file
        df_list = []
        for f in files:
            try:
                file_data = pd.read_csv(Path(f), skiprows=n_rows_to_skip)
                if include_source_file:
                    file_data[_LABFRONT_SOURCE_FILE_COL] = str(Path(f))
                df_list.append(file_data)
            except pd.errors.EmptyDataError:
                # We may have empty CSV files
                continue
        if len(df_list) > 0:
            data = pd.concat(df_list, ignore_index=True, axis=0)
        else:
            return pd.DataFrame()

        # Convert columns to datetime
        cols_to_sort = []
        if lf_constants.ISODATE_COL in data.columns:
            data[lf_constants.ISODATE_COL] = pd.to_datetime(
                data[lf_constants.ISODATE_COL].str[:-6],
                utc=True,
            ).dt.tz_localize(tz=None)

        if lf_constants.CALENDAR_DATE_COL in data.columns:
            data[lf_constants.CALENDAR_DATE_COL] = pd.to_datetime(
                data[lf_constants.CALENDAR_DATE_COL], format="%Y-%m-%d"
            )
            cols_to_sort.append(lf_constants.CALENDAR_DATE_COL)
        data = data.sort_values(by=cols_to_sort, ignore_index=True)
        # Get data only from given start and end dates
        if not filter_rows or ((start_date is None) and (end_date is None)):
            return data
        elif not (
            lf_constants.ISODATE_COL in data.columns
            or lf_constants.CALENDAR_DATE_COL in data.columns
        ):
            return data
        else:
            date_filter_col = lf_constants.ISODATE_COL
            if filter_on_calendar_date:
                start_date = (
                    None
                    if (start_date is None)
                    else pd.Timestamp(start_date).normalize()
                )
                end_date = (
                    None if (end_date is None) else pd.Timestamp(end_date).normalize()
                )
                date_filter_col = lf_constants.CALENDAR_DATE_COL
            if (start_date is None) and (not (end_date is None)):
                return data[(data[date_filter_col] < end_date)].reset_index(drop=True)
            elif (not (start_date is None)) and (end_date is None):
                return data[(data[date_filter_col] >= start_date)].reset_index(
                    drop=True
                )
            else:
                if start_date == end_date:
                    return data[data[date_filter_col] == start_date].reset_index(
                        drop=True
                    )
                else:
                    return data[
                        (data[date_filter_col] >= start_date)
                        & (data[date_filter_col] < end_date)
                    ].reset_index(drop=True)

    def get_header_length(self, file_path: Union[str, Path]) -> int:
        """Get header length of Labfront csv file.

        Parameters
        ----------
        file_path : str or Path
            Path to file.

        Returns
        -------
        int
            Lenght of the header.
        """
        # Read first line of file
        with open(file_path, "r", encoding="utf-8") as f:
            line = f.readline().split(",")
        header_length = int(line[1])
        return header_length

    def get_key_length(self, file_path: Union[str, Path]) -> int:
        """Get key length of Labfront questionnaire csv file.

        Parameters
        ----------
        file_path : str or Path
            Path to file.

        Returns
        -------
        int
            Lenght of the key header.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            while True:
                line = f.readline().split(",")
                if line[0] == "Key Length":
                    break
        key_length = int(line[1])
        return key_length

    def get_available_metrics(self, user_id: Union[str, list] = "all") -> list:
        """Get the list of available metrics for a given user.

        Parameters
        ----------
        user_id : :class:`str` or list, optional
            ID of the user(s), by default "all"

        Returns
        -------
        list
            alphabetically sorted names of the metrics for the participant(s).
        """
        participant_ids = self.resolve_user_ids(user_id)
        return sorted(
            self.file_index.loc[participant_ids, :]
            .index.get_level_values(lf_constants.FILE_INDEX_METRIC_KEY)
            .unique()
        )

    def get_task_full_id(self, task_id: str) -> str:
        """Get the full ID (name and ID) of a given questionnaire or todo.

        Parameters
        ----------
        task_id : str
            ID of the task.

        Returns
        -------
        str
            Full ID of the task.
        """
        if "_" in task_id:
            # This is already a full ID
            return task_id
        elif task_id in self.file_index[lf_constants.FILE_INDEX_TASK_NAME_KEY].unique():
            # This is a task name
            # Check if we have more than one task with the same name
            if (
                len(
                    self.file_index.loc[
                        self.file_index[lf_constants.FILE_INDEX_TASK_NAME_KEY]
                        == task_id,
                        lf_constants.FILE_INDEX_TASK_FULL_ID_KEY,
                    ].unique()
                )
                > 1
            ):
                raise ValueError(
                    f"""Multiple tasks exist with the same name {task_id}. Specify either Task ID or Task Full ID: {[x for x in self.file_index.loc[
                        self.file_index[lf_constants.FILE_INDEX_TASK_NAME_KEY]
                        == task_id,
                        lf_constants.FILE_INDEX_TASK_FULL_ID_KEY,
                    ].unique()]}"""
                )
            # Return full id corresponding to this task name
            return self.file_index.loc[
                self.file_index[lf_constants.FILE_INDEX_TASK_NAME_KEY] == task_id,
                lf_constants.FILE_INDEX_TASK_FULL_ID_KEY,
            ].unique()[0]
        elif task_id in self.file_index[lf_constants.FILE_INDEX_TASK_ID_KEY].unique():
            # This is a Task ID
            self.file_index[
                self.file_index.loc[lf_constants.FILE_INDEX_TASK_ID_KEY] == task_id,
                lf_constants.FILE_INDEX_TASK_FULL_ID_KEY,
            ].unique()[0]
        elif (
            task_id
            in self.file_index[lf_constants.FILE_INDEX_TASK_FULL_ID_KEY].unique()
        ):
            # This is a task full id
            return task_id
        else:
            raise ValueError(f"Could not get task full id from {task_id}")

    def get_questionnaire_questions(self, questionnaire_name: str) -> dict:
        # TODO change implementation with new file_index
        """Retrieve questions and answers for a given questionnaire.

        This function returns all the questions and answers for a given
        questionnaire. The return value is a dictionary, which has questions
        numbers (1_1, 1_2, 2_1, ...) as keys and question information as
        values.::

            {
                '1_1': {
                    'type': 'radio',
                    'description': 'SLEEP QUALITY: Compared to your usual wake-up, how would you rate quality of sleep?',
                    'options': ['Much better than usual', 'Better than usual', 'As usual', 'Worse than usual', 'Much worse than usual']
                },
                '1_2': {
                    'type': 'radio',
                    'description': 'ENERGY: Compared to your usual wake-up, how would you rate your energy?',
                    'options': ['Much better than usual', 'Better than usual', 'As usual', 'Worse than usual', 'Much worse than usual']
                }
            }

        Parameters
        ----------
        questionnaire_name : str
            Name of the questionnaire for which questions must be retrieved

        Returns
        -------
        dict
            Dictionary with question number as keys and question info as values

        Raises
        ------
        ValueError
            if questionnaire name does not exist
        """
        full_task_id = self.get_task_full_id(questionnaire_name)

        if (
            len(
                self.file_index[
                    self.file_index[lf_constants.FILE_INDEX_TASK_FULL_ID_KEY]
                    == full_task_id
                ]
            )
            == 0
        ):
            raise ValueError(
                f"Could not find questions for questionnaire {questionnaire_name}"
            )

        questionnaire_files = self.file_index[
            self.file_index[lf_constants.FILE_INDEX_TASK_FULL_ID_KEY] == full_task_id
        ].head(1)
        # print(questionnaire_files)
        questionnaire_info_file = (
            questionnaire_files[lf_constants.FILE_INDEX_BASE_FOLDER_KEY]
            .astype(str)
            .unique()[0]
            + os.path.sep
            + questionnaire_files.index.get_level_values(level=0)
            .astype(str)
            .unique()[0]
            + os.path.sep
            + "questionnaire"
            + os.path.sep
            + questionnaire_files[lf_constants.FILE_INDEX_TASK_FOLDER]
            .astype(str)
            .unique()[0]
            + os.path.sep
            + questionnaire_files[lf_constants.FILE_INDEX_FILENAME_KEY]
            .astype(str)
            .unique()[0]
        )

        header_length = self.get_header_length(questionnaire_info_file)
        key_length = self.get_key_length(questionnaire_info_file)

        questions_df = pd.read_csv(
            questionnaire_info_file, skiprows=header_length + 2, nrows=key_length - 2
        )

        questions_dict = {}
        # For each row, we need to get the question and all the options
        option_cols = []
        for col in questions_df.columns:
            regex_match = re.search(r"option\dName", col)
            if regex_match:
                option_cols.append(col)
        for idx, row in questions_df.iterrows():
            question_id = f"{row['sectionIndex']}_{row['questionIndex']}"
            questions_dict[question_id] = {
                "type": row["questionType"],
                "description": row["questionDescription"],
            }
            if row["questionType"] == lf_constants.QUESTIONNAIRE_QUESTION_TYPE_TEXT_VALUE:
                continue
            question_options = {}
            for col in option_cols:
                if not pd.isna(row[col]):
                    question_options[int(col.strip("option").strip("Name"))] = row[col]
            questions_dict[question_id]["options"] = question_options
        return questions_dict

    def load_garmin_connect_activity_details_lap(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_ACTIVITY_DETAILS_LAP_FOLDER,
            start_date=start_date,
            end_date=end_date,
        )
        return data

    def load_garmin_connect_activity_details_sample(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_ACTIVITY_DETAILS_SAMPLE_FOLDER,
            start_date=start_date,
            end_date=end_date,
        )
        return data

    def load_garmin_connect_activity_details_summary(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_ACTIVITY_DETAILS_SUMMARY_FOLDER,
            start_date=start_date,
            end_date=end_date,
        )
        return data

    def load_garmin_connect_activity_moveiq_summary(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_ACTIVITY_MOVEIQ_SUMMARY_FOLDER,
            start_date=start_date,
            end_date=end_date,
        )
        return data

    def load_garmin_connect_activity_summary(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_ACTIVITY_SUMMARY_FOLDER,
            start_date=start_date,
            end_date=end_date,
        )
        return data

    def load_garmin_connect_body_composition(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin Connect body composition data.

        This function loads Garmin Connect body composition data from a given
        participant and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing Garmin Connect body composition data.
        """
        data = self.get_data_from_datetime(
            user_id,
            lf_constants.GARMIN_CONNECT_BODY_COMPOSITION_FOLDER,
            start_date,
            end_date,
        )
        return data

    def load_garmin_connect_heart_rate(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin Connect heart rate data.

        This function loads heart rate data from a given
        user and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing heart rate data.
        """
        return self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_HEART_RATE_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=False,
        )

    def load_garmin_connect_pulse_ox(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.datetime] = None,
        end_date: Union[str, datetime.date, datetime.datetime] = None,
    ) -> pd.DataFrame:
        """Load Garmin Connect pulse ox data.

        Parameters
        ----------
        user_id : :class:`str`
            Identifier of the user. This can be the user ID, Labfront ID, or
            the full ID. If multiple users exist with the same user ID, then
            it is necessary to pass either the Labfront ID or the full ID.
        start_date : Union[str, datetime.date, datetime.datetime], optional
            Start date from which data should be retrieved, by default None
        end_date : Union[str, datetime.date, datetime.datetime], optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing Garmin Connect pulse ox data.
        """
        return self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_DAILY_PULSE_OX_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=False,
        ).reset_index(drop=True)

    def load_garmin_connect_respiration(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.datetime] = None,
        end_date: Union[str, datetime.date, datetime.datetime] = None,
    ) -> pd.DataFrame:
        """Load Garmin Connect respiratory data.

        This function loads Garmin Connect respiratory samples from a given
        participant and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user for which data must be loaded. This can be the user ID,
            Labfront ID, or the full ID. If multiple users exist with the same user ID,
            then it is necessary to pass either the Labfront ID or the full ID.
        start_date : str or datetime.date or datetime.datetime, optional
            Start date from which data should be retrieved, by default None
        end_date : Union[str, datetime.date, datetime.datetime], optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
             Dataframe containing Garmin Connect respiration data.
        """
        return self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_RESPIRATION_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=False,
        ).reset_index(drop=True)

    def load_garmin_connect_sleep_pulse_ox(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.datetime] = None,
        end_date: Union[str, datetime.date, datetime.datetime] = None,
    ) -> pd.DataFrame:
        sleep_data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_SLEEP_PULSE_OX_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=False,
        ).reset_index(drop=True)
        if len(sleep_data) > 0:
            # We need to add calendarDate from sleep summaries
            sleep_summary = self._load_sleep_summary_raw(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                drop_duplicates=False,
            )
            sleep_summary = sleep_summary.loc[
                :, [lf_constants.CALENDAR_DATE_COL, lf_constants.SLEEP_SUMMARY_ID_COL]
            ]

            sleep_data = sleep_data.merge(
                right=sleep_summary,
                left_on=lf_constants.SLEEP_SUMMARY_ID_COL,
                right_on=lf_constants.SLEEP_SUMMARY_ID_COL,
                how="left",
            )

            sleep_data = sleep_data.drop([lf_constants.SLEEP_SUMMARY_ID_COL], axis=1)
            # Convert calendar date to pd.Timestamp
            # TODO remove since sleep_summary already has Timestamp on calendarDate
            sleep_data[lf_constants.CALENDAR_DATE_COL] = pd.to_datetime(
                sleep_data[lf_constants.CALENDAR_DATE_COL]
            )
        return sleep_data

    def load_garmin_connect_sleep_respiration(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.datetime] = None,
        end_date: Union[str, datetime.date, datetime.datetime] = None,
    ) -> pd.DataFrame:
        sleep_data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_SLEEP_RESPIRATION_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=False,
        ).reset_index(drop=True)
        if len(sleep_data) > 0:
            # We need to add calendarDate from sleep summaries
            sleep_summary = self._load_sleep_summary_raw(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                drop_duplicates=False,
            )
            sleep_summary = sleep_summary.loc[
                :, [lf_constants.CALENDAR_DATE_COL, lf_constants.SLEEP_SUMMARY_ID_COL]
            ]

            sleep_data = sleep_data.merge(
                right=sleep_summary,
                left_on=lf_constants.SLEEP_SUMMARY_ID_COL,
                right_on=lf_constants.SLEEP_SUMMARY_ID_COL,
                how="left",
            )

            sleep_data = sleep_data.drop([lf_constants.SLEEP_SUMMARY_ID_COL], axis=1)
            # Convert calendar date to pd.Timestamp
            sleep_data[lf_constants.CALENDAR_DATE_COL] = pd.to_datetime(
                sleep_data[lf_constants.CALENDAR_DATE_COL]
            )
        return sleep_data

    def load_garmin_connect_stress(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin Connect stress data.

        This function loads stress data from a given
        user and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        :class:`pandas.DataFrame`
            Dataframe containing stress data.
        """
        data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_STRESS_FOLDER,
            start_date=start_date,
            end_date=end_date,
        )
        return data

    def load_garmin_connect_user_metrics(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin Connect user metrics data.

        This function loads user metrics data from a given
        user and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        :class:`pandas.DataFrame`
            Dataframe containing user metrics data.
        """
        data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_USER_METRICS_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=True,
        )
        return data

    def load_garmin_device_heart_rate(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin Device (SDK) heart rate data.

        This function loads heart rate data from a given
        user and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing heart rate data.
        """
        data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_DEVICE_HEART_RATE_FOLDER,
            start_date=start_date,
            end_date=end_date,
        )
        return data

    def load_garmin_device_pulse_ox(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin Device (SDK) pulse ox data.

        This function loads pulse ox data from a given
        user and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing pulse ox data.
        """
        return self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_DEVICE_PULSE_OX_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=False,
        )

    def load_garmin_device_respiration(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin Device respiration data.

        This function loads respiratory data from a given
        user and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing respiration data.
        """
        return self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_DEVICE_RESPIRATION_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=False,
        )

    def load_garmin_device_step(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin Device (SDK) steps data.

        This function loads steps data from a given
        user and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing steps data.
        """
        data = self.get_data_from_datetime(
            user_id,
            lf_constants.GARMIN_DEVICE_STEP_FOLDER,
            start_date,
            end_date,
        )
        return data

    def load_garmin_device_stress(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin device (SDK) stress data.

        This function loads stress data from a given
        user and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing BBI data.
        """
        data = self.get_data_from_datetime(
            user_id,
            lf_constants.GARMIN_DEVICE_STRESS_FOLDER,
            start_date,
            end_date,
        )
        return data

    def load_bbi(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load BBI data.

        This function loads Beat to Beat Intervals (BBI) data from a given
        user and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing BBI data.
        """
        data = self.get_data_from_datetime(
            user_id,
            lf_constants.GARMIN_DEVICE_BBI_FOLDER,
            start_date,
            end_date,
        )
        if data.empty:
            return schema.empty_bbi_frame(["source", "timezone_name"])
        data = data.copy()
        data[schema.LOCAL_TIME_COL] = pd.to_datetime(data[lf_constants.ISODATE_COL])
        data[schema.UTC_TIME_COL] = pd.to_datetime(
            data[lf_constants.UNIXTIMESTAMP_IN_MS_COL], unit="ms", utc=True
        ).dt.tz_localize(None)
        if "timezone" in data.columns:
            data = data.rename(columns={"timezone": "timezone_name"})
        data[schema.TIMEZONE_COL] = (
            data[schema.LOCAL_TIME_COL] - data[schema.UTC_TIME_COL]
        )
        data[schema.BBI_COL] = pd.to_numeric(data[schema.BBI_COL], errors="coerce")
        data["source"] = "garmin_device_bbi"
        return schema.prepare_bbi(data)

    def load_daily_summary(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin Connect daily summary data.

        This function loads Garmin Connect daily summary data from a given
        participant and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing Garmin Connect daily summary data.
        """
        data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_DAILY_SUMMARY_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=True,
        )
        return data

    def load_garmin_connect_epoch(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin Connect epoch data.

        This function loads Garmin Connect epoch data from a given
        participant and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing Garmin Connect epoch data.
        """
        data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_EPOCH_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=False,
        )
        return data

    def load_garmin_connect_step(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """
        Load Garmin Connect steps data.

        This function loads Garmin Connect steps data from a given participant and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing Garmin Connect steps data.
        """

        df = self.load_garmin_connect_epoch(user_id, start_date, end_date)
        steps_df = (
            df.groupby(lf_constants.ISODATE_COL)[lf_constants.ACTIVITY_STEPS_COL]
            .sum()
            .reset_index()
        )
        # add back the most frequent timezone offset
        steps_df[lf_constants.TIMEZONEOFFSET_IN_MS_COL] = (
            df.groupby(lf_constants.ISODATE_COL)[lf_constants.TIMEZONEOFFSET_IN_MS_COL]
            .mean()
            .reset_index()[lf_constants.TIMEZONEOFFSET_IN_MS_COL]
        )
        # add the unix timestamp column
        steps_df[lf_constants.UNIXTIMESTAMP_IN_MS_COL] = steps_df[
            lf_constants.ISODATE_COL
        ].apply(lambda x: int(x.timestamp() * 1000))
        # add sampling frequency column durationInMs
        steps_df[lf_constants.DURATION_IN_MS_COL] = 15 * 60 * 1000
        # add a column for calendar date
        steps_df[lf_constants.CALENDAR_DATE_COL] = steps_df[lf_constants.ISODATE_COL].dt.date
        # add a column for cumulative steps at a daily level
        steps_df[lf_constants.ACTIVITY_TOTAL_STEPS_COL] = steps_df.groupby(
            lf_constants.CALENDAR_DATE_COL
        )[lf_constants.ACTIVITY_STEPS_COL].cumsum()

        return steps_df

    def load_garmin_metric(
        self,
        metric: str,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
        filter_on_calendar_date: bool = False,
    ):
        try:
            labfront_metric = _LABFRONT_METRICS_DICT[metric][source]
        except KeyError:
            raise (f"Could not find available metric {metric} with source {source}")
        return self.get_data_from_datetime(
            user_id=user_id,
            metric=labfront_metric,
            start_date=start_date,
            end_date=end_date,
            is_questionnaire=False,
            is_todo=False,
            filter_on_calendar_date=filter_on_calendar_date,
        )

    def load_heart_rate(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
        time_col: str = "local_time",
    ) -> pd.DataFrame:
        schema.validate_heart_rate_time_col(time_col)
        requested_start = schema.normalize_time_bound(start_date, time_col)
        requested_end = schema.normalize_time_bound(end_date, time_col)
        data = self.load_garmin_metric(
            metric=lf_constants.METRIC_HEART_RATE,
            user_id=user_id,
            # Labfront's file index and row filter use local ISO timestamps.
            # Read a wider interval, then apply the requested clock exactly.
            start_date=(
                None
                if requested_start is None
                else requested_start - pd.Timedelta(days=1)
            ),
            end_date=(
                None if requested_end is None else requested_end + pd.Timedelta(days=1)
            ),
            source=source,
            filter_on_calendar_date=False,
        )
        if data.empty:
            return schema.empty_heart_rate_frame(
                ["source", schema.TIMESTAMP_CONFLICT_COL]
            )

        data = data.copy()
        data["local_time"] = pd.to_datetime(data[lf_constants.ISODATE_COL])
        data["utc_time"] = pd.to_datetime(
            data[lf_constants.UNIXTIMESTAMP_IN_MS_COL], unit="ms", utc=True
        ).dt.tz_localize(None)
        if "timezone" in data.columns:
            data = data.rename(columns={"timezone": "timezone_name"})
        data["timezone"] = data["local_time"] - data["utc_time"]
        data["heart_rate"] = pd.to_numeric(
            data["beatsPerMinute"], errors="coerce"
        )
        data = data[data["heart_rate"] > 0]
        data["source"] = source
        data[schema.TIMESTAMP_CONFLICT_COL] = (
            data.groupby("utc_time")["heart_rate"].transform("nunique") > 1
        )
        data = data.drop_duplicates(
            subset=["utc_time", "heart_rate"], keep="last"
        )
        data = schema.prepare_heart_rate_samples(data)
        return (
            schema.filter_heart_rate_by_time(
                data, requested_start, requested_end, time_col
            )
            .sort_values("utc_time")
            .reset_index(drop=True)
        )

    @staticmethod
    def _prepare_respiratory_samples(
        data: pd.DataFrame,
        provider_value_col: str,
        canonical_value_col: str,
    ) -> pd.DataFrame:
        """Translate one Labfront stream into the shared sample schema."""
        data = data.copy()
        if (
            schema.LOCAL_TIME_COL not in data.columns
            and lf_constants.ISODATE_COL in data.columns
        ):
            data[schema.LOCAL_TIME_COL] = data[lf_constants.ISODATE_COL]
        if (
            schema.UTC_TIME_COL not in data.columns
            and lf_constants.UNIXTIMESTAMP_IN_MS_COL in data.columns
        ):
            data[schema.UTC_TIME_COL] = pd.to_datetime(
                data[lf_constants.UNIXTIMESTAMP_IN_MS_COL],
                unit="ms",
                errors="coerce",
                utc=True,
            ).dt.tz_localize(None)
        if schema.TIMEZONE_COL in data.columns and not pd.api.types.is_timedelta64_dtype(
            data[schema.TIMEZONE_COL]
        ):
            data = data.rename(columns={schema.TIMEZONE_COL: "timezone_name"})
        if (
            schema.TIMEZONE_COL not in data.columns
            and lf_constants.TIMEZONEOFFSET_IN_MS_COL in data.columns
        ):
            data[schema.TIMEZONE_COL] = pd.to_timedelta(
                pd.to_numeric(
                    data[lf_constants.TIMEZONEOFFSET_IN_MS_COL], errors="coerce"
                ),
                unit="ms",
            )
        if (
            canonical_value_col not in data.columns
            and provider_value_col in data.columns
        ):
            data[canonical_value_col] = data[provider_value_col]
        if (
            schema.IS_SLEEPING_COL not in data.columns
            and lf_constants.IS_SLEEPING_COL in data.columns
        ):
            data[schema.IS_SLEEPING_COL] = data[lf_constants.IS_SLEEPING_COL]
        if (
            schema.CALENDAR_DATE_COL not in data.columns
            and lf_constants.CALENDAR_DATE_COL in data.columns
        ):
            data[schema.CALENDAR_DATE_COL] = data[lf_constants.CALENDAR_DATE_COL]
        return data

    def load_pulse_ox(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
    ) -> pd.DataFrame:
        if source == lf_constants.DATA_SOURCE_GARMIN_HEALTH_API:
            data = self.load_garmin_connect_pulse_ox(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            data = self.load_garmin_device_pulse_ox(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
        data = data.copy()
        data["source"] = source
        data = self._prepare_respiratory_samples(
            data,
            lf_constants.SPO2_SPO2_COL,
            schema.SPO2_COL,
        )
        return schema.prepare_spo2_samples(data)

    def load_questionnaire(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
        questionnaire_name: str = None,
    ) -> pd.DataFrame:
        """Load questionnaire data.

        This function loads questionnaire data from a given
        participant and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user for which questionnaire data have to be loaded.
            This can be the user ID, Labfront ID, or
            the full ID. If multiple users exist with the same user ID, then
            it is necessary to pass either the Labfront ID or the full ID.
        start_date : str or datetime.date or datetime.date, optional
            Start data for data loading, by default None
        end_date : str or datetime.date or datetime.date, optional
            End data for data loading, by default None
        questionnaire_name : str, optional
            Name of the questionnaire to be loaded, by default None

        Returns
        -------
        pd.DataFrame
            Questionnaire data.
        """
        data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.QUESTIONNAIRE_FOLDER,
            start_date=start_date,
            end_date=end_date,
            is_questionnaire=True,
            task_name=questionnaire_name,
            filter_on_calendar_date=False,
        )
        return data

    def load_respiration(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
    ) -> pd.DataFrame:
        if source == lf_constants.DATA_SOURCE_GARMIN_HEALTH_API:
            data = self.load_garmin_connect_respiration(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            data = self.load_garmin_device_respiration(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
        data = data.copy()
        data["source"] = source
        data = self._prepare_respiratory_samples(
            data,
            lf_constants.RESPIRATION_BREATHS_PER_MINUTE_COL,
            schema.RESPIRATORY_RATE_COL,
        )
        return schema.prepare_respiration_samples(data)

    def _load_sleep_stage_raw(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date, None] = None,
        end_date: Union[str, datetime.date, datetime.date, None] = None,
        on_conflict: Union[str, None] = None,
    ) -> pd.DataFrame:
        """Load Garmin Connect sleep stage data.

        This function loads sleep stage data from a given
        user and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            Unique identifier for the user.
        start_date : :class:`str` or :class:`datetime.datetime` or :class:`datetime.date` or None, optional
            Start date from which data should be retrieved, by default None
        end_date : :class:`str` or :class:`datetime.datetime` or :class:`datetime.date` or None, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe with sleep stages data.
        """
        conflict_policy = (
            self.sleep_stage_on_conflict
            if on_conflict is None
            else self._validate_sleep_stage_conflict_policy(on_conflict)
        )
        requested_start = pd.Timestamp(start_date) if start_date is not None else None
        requested_end = pd.Timestamp(end_date) if end_date is not None else None
        data = self.get_data_from_datetime(
            user_id,
            lf_constants.GARMIN_CONNECT_SLEEP_STAGE_FOLDER,
            start_date,
            end_date,
            filter_on_calendar_date=False,
            include_source_file=True,
            filter_rows=False,
        )

        if len(data) > 0:
            summary_start = requested_start - pd.Timedelta(days=2) if requested_start is not None else None
            summary_end = requested_end + pd.Timedelta(days=2) if requested_end is not None else None
            summaries = self._load_sleep_summary_raw(
                user_id, summary_start, summary_end, drop_duplicates=False
            )
            data = self._reconcile_sleep_stage_snapshots(
                data,
                summaries,
                on_conflict=conflict_policy,
            )
            # Change sleep stage values
            data[lf_constants.SLEEP_STAGE_SLEEP_TYPE_COL] = data[
                lf_constants.SLEEP_STAGE_SLEEP_TYPE_COL
            ].map(
                {
                    lf_constants.SLEEP_STAGE_LIGHT_STAGE_VALUE: lf_constants.SLEEP_STAGE_LIGHT_STAGE_VALUE,
                    lf_constants.SLEEP_STAGE_DEEP_STAGE_VALUE: lf_constants.SLEEP_STAGE_DEEP_STAGE_VALUE,
                    lf_constants.SLEEP_STAGE_REM_STAGE_VALUE: lf_constants.SLEEP_STAGE_REM_STAGE_VALUE,
                    lf_constants.SLEEP_STAGE_AWAKE_STAGE_VALUE: lf_constants.SLEEP_STAGE_AWAKE_STAGE_VALUE,
                    lf_constants.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE: lf_constants.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE,
                }
            )

            if requested_start is not None:
                data = data[data[lf_constants.ISODATE_COL] >= requested_start]
            if requested_end is not None:
                data = data[data[lf_constants.ISODATE_COL] < requested_end]

        return data.reset_index(drop=True)

    def load_sleep_stage(
        self,
        user_id: str,
        start_date=None,
        end_date=None,
        on_conflict: Union[str, None] = None,
    ) -> pd.DataFrame:
        """Return Garmin sleep stages in the shared snake_case schema.

        ``on_conflict`` overrides the loader-level policy for unresolved
        repeated export snapshots. Valid values are ``"drop_night"``,
        ``"raise"``, and ``"last"``. ``"drop_night"`` removes only the
        ambiguous stage rows; the separate sleep-summary row remains
        available. ``"last"`` means the last snapshot in the deterministic
        Labfront file-index order (first timestamp, then last-edit timestamp).
        """
        return _normalize_garmin_sleep_columns(
            self._load_sleep_stage_raw(
                user_id, start_date, end_date, on_conflict=on_conflict
            )
        )

    @staticmethod
    def _reconcile_sleep_stage_snapshots(
        sleep_stages: pd.DataFrame,
        sleep_summaries: pd.DataFrame,
        on_conflict: str = "drop_night",
    ) -> pd.DataFrame:
        """Select one coherent stage export for each Garmin sleep summary.

        Labfront exports can repeat the same Garmin sleep stages in several
        CSV files, and a later file can contain a revised snapshot. File
        boundaries are therefore preserved temporarily. Candidate snapshots
        are matched to the selected summary by comparing the total duration of
        every stage type. Identical matching snapshots collapse to one;
        unresolved conflicts follow ``on_conflict``: drop their stage rows,
        raise an error, or retain the last loaded source snapshot. The
        temporary source-file column is removed from the result.
        """
        on_conflict = LabfrontLoader._validate_sleep_stage_conflict_policy(
            on_conflict
        )
        source_col = _LABFRONT_SOURCE_FILE_COL
        id_col = lf_constants.SLEEP_SUMMARY_ID_COL
        record_cols = [col for col in sleep_stages if col != source_col]
        if sleep_stages.empty:
            return sleep_stages.drop(columns=source_col, errors="ignore")

        dropped_ids = []

        def normalized(snapshot):
            return (
                snapshot[record_cols]
                .drop_duplicates()
                .sort_values(record_cols)
                .reset_index(drop=True)
            )

        def source_snapshots(rows):
            return [
                snapshot.drop_duplicates(subset=record_cols)
                for _, snapshot in rows.groupby(source_col, sort=False)
            ]

        def resolve_conflict(summary_id, snapshots, message):
            if on_conflict == "raise":
                raise ValueError(message)
            if on_conflict == "last":
                return snapshots[-1]
            dropped_ids.append(summary_id)
            return None

        def select_without_summary(summary_id, rows):
            snapshots = source_snapshots(rows)
            reference = normalized(snapshots[0])
            if any(not reference.equals(normalized(item)) for item in snapshots[1:]):
                return resolve_conflict(
                    summary_id,
                    snapshots,
                    "Conflicting Labfront sleep-stage snapshots without "
                    f"summary {summary_id}.",
                )
            return snapshots[0]

        if sleep_summaries.empty:
            summaries = None
        else:
            validation_rank = {
                "AUTO_TENTATIVE": 1,
                "AUTO_FINAL": 2,
                "ENHANCED_TENTATIVE": 3,
                "ENHANCED_FINAL": 4,
            }
            summaries = sleep_summaries.copy()
            summaries["__validation_rank"] = (
                summaries["validation"].map(validation_rank).fillna(0)
            )
            summaries = (
                summaries.sort_values(
                    [
                        id_col,
                        "__validation_rank",
                        lf_constants.SLEEP_SUMMARY_DURATION_IN_MS_COL,
                    ]
                )
                .groupby(id_col, as_index=False)
                .tail(1)
                .set_index(id_col)
            )
        stage_summary_cols = {
            lf_constants.SLEEP_STAGE_LIGHT_STAGE_VALUE: lf_constants.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL,
            lf_constants.SLEEP_STAGE_DEEP_STAGE_VALUE: lf_constants.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL,
            lf_constants.SLEEP_STAGE_REM_STAGE_VALUE: lf_constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL,
            lf_constants.SLEEP_STAGE_AWAKE_STAGE_VALUE: lf_constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL,
            lf_constants.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE: lf_constants.SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL,
        }

        def matches(snapshot, summary):
            actual = snapshot.groupby(lf_constants.SLEEP_STAGE_SLEEP_TYPE_COL)[
                lf_constants.SLEEP_STAGE_DURATION_IN_MS_COL
            ].sum()
            return all(
                pd.notna(summary[column]) and int(actual.get(stage, 0)) == int(summary[column])
                for stage, column in stage_summary_cols.items()
            )

        chosen = []
        for summary_id, rows in sleep_stages.groupby(id_col, sort=False):
            if summaries is None or summary_id not in summaries.index:
                selected = select_without_summary(summary_id, rows)
                if selected is not None:
                    chosen.append(selected)
                continue
            summary = summaries.loc[summary_id]
            candidates = []
            snapshots = source_snapshots(rows)
            for snapshot in snapshots:
                if matches(snapshot, summary):
                    candidates.append(snapshot)
            if not candidates:
                combined = rows.drop_duplicates(subset=record_cols)
                if matches(combined, summary):
                    selected = combined
                else:
                    selected = resolve_conflict(
                        summary_id,
                        snapshots,
                        "No coherent Labfront sleep-stage snapshot matches "
                        f"summary {summary_id}.",
                    )
            else:
                reference = normalized(candidates[0])
                if any(
                    not reference.equals(normalized(snapshot))
                    for snapshot in candidates[1:]
                ):
                    selected = resolve_conflict(
                        summary_id,
                        candidates,
                        "Conflicting Labfront sleep-stage snapshots match "
                        f"summary {summary_id}.",
                    )
                else:
                    selected = candidates[0]
            if selected is not None:
                chosen.append(selected)

        if dropped_ids:
            displayed_ids = ", ".join(map(str, dropped_ids[:10]))
            if len(dropped_ids) > 10:
                displayed_ids += ", ..."
            warnings.warn(
                "Dropped ambiguous Labfront sleep-stage snapshots for "
                f"{len(dropped_ids)} sleep summaries: {displayed_ids}.",
                UserWarning,
                stacklevel=2,
            )

        if not chosen:
            return sleep_stages.iloc[0:0].drop(columns=source_col).reset_index(drop=True)
        return (
            pd.concat(chosen, ignore_index=True)
            .drop(columns=source_col)
            .sort_values(lf_constants.ISODATE_COL)
            .reset_index(drop=True)
        )

    def _load_sleep_summary_raw(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        """Load Garmin Connect sleep summary data.

        This function loads sleep summary data from a given
        user and within a specified date and time range.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None
        drop_duplicates : bool, optional
            If true, retain the highest-ranked summary for each calendar day.
            If false, retain all exported summaries so callers can reconcile
            repeated or revised snapshots.

        Returns
        -------
        pd.DataFrame
            Dataframe containing sleep summary data.
        """
        data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_SLEEP_SUMMARY_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=True,
        )

        if len(data) > 0:
            data[lf_constants.CALENDAR_DATE_COL] = pd.to_datetime(
                data[lf_constants.CALENDAR_DATE_COL],
                format="%Y-%m-%d",
            ).dt.date

            if drop_duplicates:
                data["validationMap"] = data["validation"].map(
                    {
                        "AUTO_TENTATIVE": 1,
                        "AUTO_FINAL": 2,
                        "ENHANCED_TENTATIVE": 3,
                        "ENHANCED_FINAL": 4,
                    }
                )
                data = (
                    data.sort_values(
                        by=(
                            [
                                lf_constants.CALENDAR_DATE_COL,
                                "validationMap",
                                lf_constants.SLEEP_SUMMARY_DURATION_IN_MS_COL,
                            ]
                        ),
                        ascending=True,
                    )
                    .groupby(lf_constants.CALENDAR_DATE_COL)
                    .tail(1)
                )
                data = data.drop(["validationMap"], axis=1)

            # Rename columns based on pywearable specs
            data = data.rename(
                columns={
                    lf_constants.GARMIN_CONNECT_SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL: lf_constants.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL,
                    lf_constants.GARMIN_CONNECT_SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL: lf_constants.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL,
                    lf_constants.GARMIN_CONNECT_SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL: lf_constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL,
                    lf_constants.GARMIN_CONNECT_SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL: lf_constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL,
                    lf_constants.GARMIN_CONNECT_SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL: lf_constants.SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL,
                }
            )

            return data.reset_index(drop=True)
        else:
            return pd.DataFrame()

    def load_sleep_summary(
        self,
        user_id: str,
        start_date=None,
        end_date=None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        """Return Garmin sleep summaries in the shared snake_case schema."""
        return _normalize_garmin_sleep_columns(
            self._load_sleep_summary_raw(
                user_id, start_date, end_date, drop_duplicates=drop_duplicates
            )
        )

    def load_stress(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
    ) -> pd.DataFrame:
        return self.load_garmin_metric(
            metric=lf_constants.METRIC_STRESS,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            source=source,
        )

    def load_provider_feature(
        self,
        feature: str,
        user_id: str,
        start_date=None,
        end_date=None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
    ) -> pd.DataFrame:
        """Load one Garmin feature in the shared provider-feature schema.

        Stress and Body Battery samples are translated from Garmin's stress
        stream, including its negative sentinel values. Daily stress fields
        are translated from Garmin daily summaries, with durations converted
        from milliseconds to seconds. This method does not aggregate samples
        or calculate sleep recovery.
        """
        sample_columns = {
            "garmin_stress_level": lf_constants.STRESS_STRESS_LEVEL_COL,
            "garmin_body_battery": lf_constants.STRESS_BODY_BATTERY_COL,
        }
        daily_columns = {
            "garmin_daily_average_stress": (
                lf_constants.DAILY_SUMMARY_AVG_STRESS_IN_STRESS_LVL_COL
            ),
            "garmin_daily_max_stress": (
                lf_constants.DAILY_SUMMARY_MAX_STRESS_IN_STRESS_LVL_COL
            ),
            "garmin_daily_stress_duration_seconds": "stressDurationInMs",
            "garmin_daily_rest_stress_duration_seconds": (
                lf_constants.DAILY_SUMMARY_REST_STRESS_DURATION_IN_MS_COL
            ),
            "garmin_daily_activity_stress_duration_seconds": (
                lf_constants.DAILY_SUMMARY_UNRELIABLE_STRESS_DURATION_IN_MS_COL
            ),
            "garmin_daily_low_stress_duration_seconds": (
                lf_constants.DAILY_SUMMARY_LOW_STRESS_DURATION_IN_MS_COL
            ),
            "garmin_daily_medium_stress_duration_seconds": (
                lf_constants.DAILY_SUMMARY_MEDIUM_STRESS_DURATION_IN_MS_COL
            ),
            "garmin_daily_high_stress_duration_seconds": (
                lf_constants.DAILY_SUMMARY_HIGH_STRESS_DURATION_IN_MS_COL
            ),
            "garmin_daily_stress_qualifier": lf_constants.DAILY_SUMMARY_STRESS_SCORE_COL,
        }
        if feature in sample_columns:
            data = self.load_stress(
                user_id,
                start_date=start_date,
                end_date=end_date,
                source=source,
            ).copy()
            if data.empty:
                return schema.prepare_provider_feature_samples(
                    pd.DataFrame(columns=[feature]), feature
                )
            data[schema.LOCAL_TIME_COL] = pd.to_datetime(
                data[lf_constants.ISODATE_COL], errors="coerce"
            )
            data[schema.UTC_TIME_COL] = pd.to_datetime(
                data[lf_constants.UNIXTIMESTAMP_IN_MS_COL],
                unit="ms",
                errors="coerce",
                utc=True,
            ).dt.tz_localize(None)
            data[schema.TIMEZONE_COL] = pd.to_timedelta(
                pd.to_numeric(
                    data[lf_constants.TIMEZONEOFFSET_IN_MS_COL], errors="coerce"
                ),
                unit="ms",
            )
            data[feature] = pd.to_numeric(
                data[sample_columns[feature]], errors="coerce"
            )
            if feature == "garmin_stress_level":
                # Garmin uses -1 for unavailable stress and -2 for activity.
                data.loc[data[feature].lt(0), feature] = np.nan
            data.loc[data[feature].gt(100), feature] = np.nan
            if feature == "garmin_body_battery":
                data.loc[data[feature].lt(0), feature] = np.nan
            return schema.prepare_provider_feature_samples(data, feature)
        if feature not in daily_columns:
            return super().load_provider_feature(
                feature, user_id, start_date, end_date
            )
        data = self.load_daily_summary(user_id, start_date, end_date).copy()
        if data.empty:
            return schema.prepare_provider_feature_summary(
                pd.DataFrame(columns=[schema.CALENDAR_DATE_COL, feature]),
                feature,
                numeric=feature != "garmin_daily_stress_qualifier",
            )
        order_col = (
            lf_constants.ISODATE_COL
            if lf_constants.ISODATE_COL in data.columns
            else lf_constants.CALENDAR_DATE_COL
        )
        data = data.sort_values(order_col, kind="stable").drop_duplicates(
            lf_constants.CALENDAR_DATE_COL, keep="last"
        )
        result = pd.DataFrame(
            {
                schema.CALENDAR_DATE_COL: data[lf_constants.CALENDAR_DATE_COL],
                feature: data[daily_columns[feature]],
            }
        )
        numeric = feature != "garmin_daily_stress_qualifier"
        if numeric:
            result[feature] = pd.to_numeric(result[feature], errors="coerce")
            if feature.endswith("_seconds"):
                result[feature] = result[feature] / 1000
            else:
                result.loc[result[feature].lt(0), feature] = np.nan
        return schema.prepare_provider_feature_summary(
            result, feature, numeric=numeric
        )

    def load_steps(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
    ) -> pd.DataFrame:
        """
        Load steps data.

        This function loads steps data from a given user and within a specified date and time range.

        Parameters
        ----------
        user_id : str
            Unique user identifier for steps data retrieval.
        start_date : datetime.datetime or datetime.date or str or None, optional
            Start date for steps data retrieval, by default None
        end_date : datetime.datetime or datetime.date or str or None, optional
            End date for steps data retrieval, by default None
        source : str, optional
            Source of the data, by default lf_constants.DATA_SOURCE_GARMIN_HEALTH_API
        """

        if source == lf_constants.DATA_SOURCE_GARMIN_HEALTH_API:
            data = self.load_garmin_connect_step(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
        elif source == lf_constants.DATA_SOURCE_GARMIN_SDK:
            data = self.load_garmin_device_step(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            raise ValueError(f"Source {source} not supported.")

        return data

    def load_activity_samples(
        self,
        user_id: str,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """Return Garmin Health API epochs as canonical step intervals.

        Category rows sharing one epoch are collapsed by summing their
        incremental step counts while counting the epoch duration once.
        """
        epochs = self.load_garmin_connect_epoch(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        ).copy()
        if epochs.empty:
            return schema.empty_activity_frame(extra_columns=("source",))
        epochs[schema.LOCAL_TIME_COL] = pd.to_datetime(
            epochs[lf_constants.ISODATE_COL], errors="coerce"
        )
        epochs[schema.UTC_TIME_COL] = pd.to_datetime(
            epochs[lf_constants.UNIXTIMESTAMP_IN_MS_COL],
            unit="ms",
            errors="coerce",
            utc=True,
        ).dt.tz_localize(None)
        epochs[schema.TIMEZONE_COL] = pd.to_timedelta(
            pd.to_numeric(epochs[lf_constants.TIMEZONEOFFSET_IN_MS_COL], errors="coerce"),
            unit="ms",
        )
        epochs[schema.DURATION_IN_MS_COL] = epochs[lf_constants.DURATION_IN_MS_COL]
        epochs[schema.ACTIVITY_STEPS_COL] = epochs[lf_constants.ACTIVITY_STEPS_COL]
        if "activityType" in epochs:
            epochs = epochs.rename(columns={"activityType": "activity_type"})
        return prepare_epoch_activity_samples(epochs)


__all__ = ["LabfrontLoader"]
