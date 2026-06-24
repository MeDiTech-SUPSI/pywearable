"""
This module contains all the functions related to the loading of data from Labfront.

"""

import datetime
import os
import re
from pathlib import Path
from typing import Tuple, Union

import dateutil.parser
import numpy as np
import pandas as pd

from ... import constants, sleep, utils
from ..base import BaseLoader
from . import constants as lf_constants

# TODO some Labfront folders have midnigt in end-date, other in start-date...

_LABFRONT_METRICS_DICT = {
    constants.METRIC_HEART_RATE: {
        lf_constants.DATA_SOURCE_GARMIN_HEALTH_API: lf_constants.GARMIN_CONNECT_HEART_RATE_FOLDER,
        lf_constants.DATA_SOURCE_GARMIN_SDK: lf_constants.GARMIN_DEVICE_HEART_RATE_FOLDER,
    },
    constants.METRIC_STRESS: {
        lf_constants.DATA_SOURCE_GARMIN_HEALTH_API: lf_constants.GARMIN_CONNECT_STRESS_FOLDER,
        lf_constants.DATA_SOURCE_GARMIN_SDK: lf_constants.GARMIN_DEVICE_HEART_RATE_FOLDER,
    },
    constants.METRIC_PULSE_OX: {
        lf_constants.DATA_SOURCE_GARMIN_HEALTH_API: lf_constants.GARMIN_CONNECT_DAILY_PULSE_OX_FOLDER,
        lf_constants.DATA_SOURCE_GARMIN_SDK: lf_constants.GARMIN_DEVICE_PULSE_OX_FOLDER,
    },
    constants.METRIC_RESPIRATION: {
        lf_constants.DATA_SOURCE_GARMIN_HEALTH_API: lf_constants.GARMIN_CONNECT_RESPIRATION_FOLDER,
        lf_constants.DATA_SOURCE_GARMIN_SDK: lf_constants.GARMIN_DEVICE_RESPIRATION_FOLDER,
    },
}

LABFRONT_FILE_INDEX_NAME = ".pywearable_file_index.csv"


class LabfrontLoader(BaseLoader):
    """Loader for Labfront data.

    This is the main object for loading and analyzing data collected
    with Labfront.

    Parameters
    ----------
        data_path: str or Path
            Path to folder containing Labfront data.
    """

    def __init__(self, data_paths: Union[str, Path, list], reindex_files: bool = False):
        """Constructor method"""
        self.set_path(data_paths, reindex_files=reindex_files)

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

        # self.tasks_dict = self.get_available_questionnaires(
        #    return_dict=True
        # ) | self.get_available_todos(return_dict=True)

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
        self, user_id: Union[str, list] = "all", return_dict: bool = False
    ) -> list:
        """Get the list of available questionnaires.

        Parameters
        ----------
            user_id: str or list, optional
                IDs of the users for which to return available questionnaires, by default "all".
            return_dict: bool, optional
                Whether to return a dictionary of the name of the questionnaires and their full ids,
                or simply a sorted list of the available questionnaires, by default False.

        Returns
        -------
            list
                alphabetically sorted names of the questionnaires for the user(s).
        """
        if lf_constants.FILE_INDEX_TASK_NAME_KEY in self.file_index.columns:
            return (
                self.file_index.loc[
                    self.file_index[lf_constants.FILE_INDEX_IS_QUESTIONNAIRE_KEY]
                    == True,
                    lf_constants.FILE_INDEX_TASK_NAME_KEY,
                ]
                .unique()
                .tolist()
            )
        else:
            return []

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
        if constants.CALENDAR_DATE_COL in temp_df:
            column = constants.CALENDAR_DATE_COL
            temp_df[column] = pd.to_datetime(
                temp_df[column], format="%Y-%m-%d"
            ).dt.tz_localize(None)
            first_date = temp_df[column].min()
            last_date = temp_df[column].max()
        elif constants.ISODATE_COL in temp_df:
            column = constants.ISODATE_COL
            try:
                temp_df[column] = pd.to_datetime(
                    temp_df[constants.ISODATE_COL].str[:-6],
                    utc=True,
                ).dt.tz_localize(None)
            except:
                print(path_to_file)
                raise
            first_date = temp_df[column].min().normalize()
            last_date = temp_df[column].max().normalize()
        elif constants.UNIXTIMESTAMP_IN_MS_COL in temp_df:
            column = constants.UNIXTIMESTAMP_IN_MS_COL
            temp_df[column] = pd.to_datetime(
                temp_df[column], unit="ms", utc=True
            ).dt.tz_localize(None)
            first_date = temp_df[column].min() - pd.Timedelta(hours=14)
            last_date = temp_df[column].max() + pd.Timedelta(hours=14)
        else:
            raise ValueError(
                f"Could not find column {constants.ISODATE_COL} or {constants.CALENDAR_DATE_COL} or {constants.UNIXTIMESTAMP_IN_MS_COL} in file {path_to_file}"
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
        # TODO midnight is present in the previous file for some files
        # We need to floor start_date to the previous day
        # since Labfront files contain the midnight in the
        # file belonging to the previous day
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
                df_list.append(pd.read_csv(Path(f), skiprows=n_rows_to_skip))
            except pd.errors.EmptyDataError:
                # We may have empty CSV files
                continue
        if len(df_list) > 0:
            data = pd.concat(df_list, ignore_index=True, axis=0)
        else:
            return pd.DataFrame()

        # Convert columns to datetime
        cols_to_sort = []
        if constants.ISODATE_COL in data.columns:
            data[constants.ISODATE_COL] = pd.to_datetime(
                data[constants.ISODATE_COL].str[:-6],
                utc=True,
            ).dt.tz_localize(tz=None)

        if constants.CALENDAR_DATE_COL in data.columns:
            data[constants.CALENDAR_DATE_COL] = pd.to_datetime(
                data[constants.CALENDAR_DATE_COL], format="%Y-%m-%d"
            )
            cols_to_sort.append(constants.CALENDAR_DATE_COL)
        data = data.sort_values(by=cols_to_sort, ignore_index=True)
        # Get data only from given start and end dates
        if (start_date is None) and (end_date is None):
            return data
        elif not (
            constants.ISODATE_COL in data.columns
            or constants.CALENDAR_DATE_COL in data.columns
        ):
            return data
        else:
            date_filter_col = constants.ISODATE_COL
            if filter_on_calendar_date:
                start_date = (
                    None
                    if (start_date is None)
                    else pd.Timestamp(start_date).normalize()
                )
                end_date = (
                    None if (end_date is None) else pd.Timestamp(end_date).normalize()
                )
                date_filter_col = constants.CALENDAR_DATE_COL
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
        participant_ids = utils.get_user_ids(self, user_id)
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
        no_user_id = True
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
            regex_match = re.search("option\dName", col)
            if regex_match:
                option_cols.append(col)
        for idx, row in questions_df.iterrows():
            question_id = f"{row['sectionIndex']}_{row['questionIndex']}"
            questions_dict[question_id] = {
                "type": row["questionType"],
                "description": row["questionDescription"],
            }
            if row["questionType"] == constants.QUESTIONNAIRE_QUESTION_TYPE_TEXT_VALUE:
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
        # We need to load both sleep and daily pulse ox
        daily_data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_DAILY_PULSE_OX_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=False,
        ).reset_index(drop=True)
        # Load sleep pulse ox data
        sleep_data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_SLEEP_PULSE_OX_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=False,
        ).reset_index(drop=True)

        if len(sleep_data) > 0:
            # Set information about sleep
            # We need to add information about sleep summaries
            sleep_summary = self.load_sleep_summary(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                same_day_filter=False,
            )
            sleep_summary = sleep_summary.loc[
                :, [constants.CALENDAR_DATE_COL, constants.SLEEP_SUMMARY_ID_COL]
            ]

            sleep_data = sleep_data.merge(
                right=sleep_summary,
                left_on=constants.SLEEP_SUMMARY_ID_COL,
                right_on=constants.SLEEP_SUMMARY_ID_COL,
                how="left",
            )

            sleep_data[constants.IS_SLEEPING_COL] = True

            # Merge dataframes
            # We need to merge the dataframes because the daily_data already contain sleep_data
            if len(daily_data) > 0:
                sleep_data = sleep_data.drop(
                    [
                        x
                        for x in sleep_data.columns
                        if (
                            not x
                            in (
                                [
                                    constants.ISODATE_COL,
                                    constants.CALENDAR_DATE_COL,
                                    constants.IS_SLEEPING_COL,
                                ]
                            )
                        )
                    ],
                    axis=1,
                )
                merged_data = daily_data.merge(
                    sleep_data, on=constants.ISODATE_COL, how="left"
                )
                merged_data[constants.IS_SLEEPING_COL] = merged_data[
                    constants.IS_SLEEPING_COL
                ].fillna(value=False)
                return merged_data
            else:
                return sleep_data
        else:
            daily_data[daily_data[constants.IS_SLEEPING_COL]] = False
            return daily_data

    def load_garmin_connect_respiration(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.datetime] = None,
        end_date: Union[str, datetime.date, datetime.datetime] = None,
    ) -> pd.DataFrame:
        """Load Garmin Connect respiratory data.

        This function loads Garmin Connect respiratory data from a given
        participant and within a specified date and time range. This
        function loads both daily and sleep respiratory data. The
        resulting data frame contains an additional column named 'sleep',
        equal to 1 for respiratory data acquired during sleep.

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
        # We need to load both sleep and daily respiration
        daily_data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_RESPIRATION_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=False,
        ).reset_index(drop=True)

        # Get sleep data
        sleep_data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_SLEEP_RESPIRATION_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=False,
        ).reset_index(drop=True)

        # Add calendar date from sleep summary
        sleep_summary = self.load_sleep_summary(
            user_id, start_date, end_date, same_day_filter=True
        ).reset_index(drop=True)

        if len(sleep_summary) > 0:
            sleep_summary = sleep_summary.loc[
                :,
                [
                    constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL,
                    constants.CALENDAR_DATE_COL,
                ],
            ]

        if len(sleep_data) > 0:
            sleep_data = sleep_data[
                sleep_data[constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL].isin(
                    sleep_summary[constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL].unique()
                )
            ].reset_index(drop=True)
            sleep_data = pd.merge(
                left=sleep_data,
                right=sleep_summary,
                on=constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL,
                how="outer",
            )

            sleep_data.loc[:, constants.IS_SLEEPING_COL] = True
            sleep_data = sleep_data.drop(
                [
                    x
                    for x in sleep_data.columns
                    if (
                        not x
                        in (
                            [
                                constants.ISODATE_COL,
                                constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL,
                                constants.CALENDAR_DATE_COL,
                                constants.IS_SLEEPING_COL,
                            ]
                        )
                    )
                ],
                axis=1,
            )
            # Merge dataframes
            # We need to merge the dataframes because the daily_data already contain sleep_data
            merged_data = daily_data.merge(
                sleep_data, on=constants.ISODATE_COL, how="left"
            )
            merged_data.loc[
                merged_data[constants.IS_SLEEPING_COL] != True,
                constants.IS_SLEEPING_COL,
            ] = False
            merged_data[constants.CALENDAR_DATE_COL] = pd.to_datetime(
                merged_data[constants.CALENDAR_DATE_COL]
            )
            return merged_data
        else:
            daily_data[constants.CALENDAR_DATE_COL] = pd.NaT
            daily_data[constants.IS_SLEEPING_COL] = False
            return daily_data

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
            sleep_summary = self.load_sleep_summary(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                same_day_filter=False,
            )
            sleep_summary = sleep_summary.loc[
                :, [constants.CALENDAR_DATE_COL, constants.SLEEP_SUMMARY_ID_COL]
            ]

            sleep_data = sleep_data.merge(
                right=sleep_summary,
                left_on=constants.SLEEP_SUMMARY_ID_COL,
                right_on=constants.SLEEP_SUMMARY_ID_COL,
                how="left",
            )

            sleep_data = sleep_data.drop([constants.SLEEP_SUMMARY_ID_COL], axis=1)
            # Convert calendar date to pd.Timestamp
            # TODO remove since sleep_summary already has Timestamp on calendarDate
            sleep_data[constants.CALENDAR_DATE_COL] = pd.to_datetime(
                sleep_data[constants.CALENDAR_DATE_COL]
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
            sleep_summary = self.load_sleep_summary(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                same_day_filter=False,
            )
            sleep_summary = sleep_summary.loc[
                :, [constants.CALENDAR_DATE_COL, constants.SLEEP_SUMMARY_ID_COL]
            ]

            sleep_data = sleep_data.merge(
                right=sleep_summary,
                left_on=constants.SLEEP_SUMMARY_ID_COL,
                right_on=constants.SLEEP_SUMMARY_ID_COL,
                how="left",
            )

            sleep_data = sleep_data.drop([constants.SLEEP_SUMMARY_ID_COL], axis=1)
            # Convert calendar date to pd.Timestamp
            sleep_data[constants.CALENDAR_DATE_COL] = pd.to_datetime(
                sleep_data[constants.CALENDAR_DATE_COL]
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

    def _load_garmin_device_all_pulse_ox(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_DEVICE_PULSE_OX_FOLDER,
            start_date=start_date,
            end_date=end_date,
            filter_on_calendar_date=False,
        )
        if len(data) > 0:
            # We need to add calendarDate from sleep summaries
            # TODO probably better not to depend on sleep module here
            sleep_timestamps = sleep.get_sleep_timestamps(
                self,
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
            # Prepare columns for sleep information
            data[constants.IS_SLEEPING_COL] = False
            data[constants.CALENDAR_DATE_COL] = pd.NaT
            if type(sleep_timestamps) == dict:
                if user_id in sleep_timestamps.keys():
                    sleep_timestamps = sleep_timestamps[user_id]
                    if sleep_timestamps is not None:
                        # We have a dict with user data
                        for calendar_date in sleep_timestamps.keys():
                            bedtime = sleep_timestamps[calendar_date][0]
                            wakeup = sleep_timestamps[calendar_date][1]
                            data.loc[
                                (data[constants.ISODATE_COL] >= bedtime)
                                & (data[constants.ISODATE_COL] <= wakeup),
                                constants.CALENDAR_DATE_COL,
                            ] = pd.Timestamp(
                                calendar_date.year,
                                calendar_date.month,
                                calendar_date.day,
                            )
                            data.loc[
                                (data[constants.ISODATE_COL] >= bedtime)
                                & (data[constants.ISODATE_COL] <= wakeup),
                                constants.IS_SLEEPING_COL,
                            ] = True
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
        data = self._load_garmin_device_all_pulse_ox(
            user_id=user_id, start_date=start_date, end_date=end_date
        )
        # Keep isSleeping information and remove other columns
        if constants.CALENDAR_DATE_COL in data.columns:
            data = data.drop(constants.CALENDAR_DATE_COL, axis=1)
        return data

    def load_garmin_device_sleep_pulse_ox(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin Device (SDK) sleep pulse ox data.

        This function loads pulse ox data from a given
        user and within a specified date and time range,
        and uses information from :func:`load_sleep_summary`
        to return values only collected during sleep periods.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None.
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None.

        Returns
        -------
        pd.DataFrame
            Dataframe containing sleep pulse ox data.
        """
        data = self._load_garmin_device_all_pulse_ox(
            user_id=user_id, start_date=start_date, end_date=end_date
        )
        if constants.IS_SLEEPING_COL in data.columns:
            data = data[data[constants.IS_SLEEPING_COL] == True]
            data = data.drop(constants.IS_SLEEPING_COL, axis=1)
        return data

    def _load_garmin_device_all_respiration(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin Device respiration data."""
        data = self.get_data_from_datetime(
            user_id,
            lf_constants.GARMIN_DEVICE_RESPIRATION_FOLDER,
            start_date,
            end_date,
        )
        # We need to add information about sleep
        if len(data) > 0:
            # Let's prepare the columns for sleep information
            data[constants.IS_SLEEPING_COL] = False
            data[constants.CALENDAR_DATE_COL] = pd.NaT
            # Let's load sleep timestamps
            # We need to add calendarDate from sleep summaries
            sleep_timestamps = sleep.get_sleep_timestamps(
                self,
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
            if type(sleep_timestamps) == dict:
                if user_id in sleep_timestamps.keys():
                    sleep_timestamps = sleep_timestamps[user_id]
                    # We have a dict with user data
                    if not (sleep_timestamps is None):
                        for calendar_date in sleep_timestamps.keys():
                            bedtime = sleep_timestamps[calendar_date][0]
                            wakeup = sleep_timestamps[calendar_date][1]
                            data.loc[
                                (data[constants.ISODATE_COL] >= bedtime)
                                & (data[constants.ISODATE_COL] <= wakeup),
                                constants.CALENDAR_DATE_COL,
                            ] = pd.Timestamp(
                                calendar_date.year,
                                calendar_date.month,
                                calendar_date.day,
                            )
                            data.loc[
                                (data[constants.ISODATE_COL] >= bedtime)
                                & (data[constants.ISODATE_COL] <= wakeup),
                                constants.IS_SLEEPING_COL,
                            ] = True
        return data

    def load_garmin_device_respiration(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin Device respiration data.

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
            Dataframe containing respiration data.
        """
        data = self._load_garmin_device_all_respiration(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        # We need to remove calendarDate
        if len(data) > 0:
            if constants.CALENDAR_DATE_COL in data.columns:
                data = data.drop(constants.CALENDAR_DATE_COL, axis=1)
        return data

    def load_garmin_device_sleep_respiration(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        """Load Garmin Device (SDK) sleep respiration data.

        This function loads respiration data from a given
        user and within a specified date and time range,
        and uses information from :func:`load_sleep_summary`
        to return values only collected during sleep periods.

        Parameters
        ----------
        user_id : :class:`str`
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None.
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None.

        Returns
        -------
        pd.DataFrame
            Dataframe containing sleep pulse ox data.
        """
        data = self._load_garmin_device_all_respiration(
            user_id=user_id, start_date=start_date, end_date=end_date
        )
        if len(data) > 0:
            if constants.IS_SLEEPING_COL in data.columns:
                data = data[data[constants.IS_SLEEPING_COL] == True]
                data = data.drop(constants.IS_SLEEPING_COL, axis=1)
        return data

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
        return data

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
            df.groupby(constants.ISODATE_COL)[lf_constants.ACTIVITY_STEPS_COL]
            .sum()
            .reset_index()
        )
        # add back the most frequent timezone offset
        steps_df[constants.TIMEZONEOFFSET_IN_MS_COL] = (
            df.groupby(constants.ISODATE_COL)[constants.TIMEZONEOFFSET_IN_MS_COL]
            .mean()
            .reset_index()[constants.TIMEZONEOFFSET_IN_MS_COL]
        )
        # add the unix timestamp column
        steps_df[constants.UNIXTIMESTAMP_IN_MS_COL] = steps_df[
            constants.ISODATE_COL
        ].apply(lambda x: int(x.timestamp() * 1000))
        # add sampling frequency column durationInMs
        steps_df[constants.DURATION_IN_MS_COL] = 15 * 60 * 1000
        # add a column for calendar date
        steps_df[constants.CALENDAR_DATE_COL] = steps_df[constants.ISODATE_COL].dt.date
        # add a column for cumulative steps at a daily level
        steps_df[lf_constants.ACTIVITY_TOTAL_STEPS_COL] = steps_df.groupby(
            constants.CALENDAR_DATE_COL
        )[lf_constants.ACTIVITY_STEPS_COL].cumsum()

        return steps_df

    def load_hypnogram(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.datetime],
        end_date: Union[str, datetime.date, datetime.datetime, None],
        resolution: float = 1,
        map_hypnogram: bool = True,
    ) -> pd.DataFrame:
        """Load hypnogram for given user for a given day.

        This function allows to load sleep hypnograms starting from sleep data
        with a given resolution in minutes. The function returns a
        dictionary, with each key being a :class:`datetime.date`, and each value
        a :class:`numpy.array` that contains the hypnogram values.
        The hypnogram can be kept with standard Garmin values for sleep stages,
        or these values can be mapped to number according to the following
        convention:

            - unmeasurable -> -1
            - awake -> 0
            - light -> 1
            - deep -> 3
            - rem -> 4

        Parameters
        ----------
        user_id : :class:`str`
            Unique identifier of the user.
        start_date : str or datetime.date or datetime.date or None
            Start date for hypnogram loading.
        end_date : str or datetime.date or datetime.date or None
            End date for hypnogram loading.
        resolution : int, optional
            Desired resolution (in minutes) requested for the hypnogram, by default 1

        Returns
        -------
        dict
            Dictionary with hypnogram values, one per each day.

        Raises
        ------
        ValueError
            If date is passed as str and cannot be parsed.
        """

        # Load sleep summary and sleep stages data
        sleep_summaries = self.load_sleep_summary(
            user_id=user_id, start_date=start_date, end_date=end_date
        )
        if len(sleep_summaries) == 0:
            return {}

        sleep_start_time = sleep_summaries.iloc[0][constants.ISODATE_COL]

        sleep_end_time = pd.to_datetime(
            (
                sleep_summaries.iloc[-1][constants.UNIXTIMESTAMP_IN_MS_COL]
                + sleep_summaries.iloc[-1][constants.TIMEZONEOFFSET_IN_MS_COL]
                + sleep_summaries.iloc[-1][constants.SLEEP_SUMMARY_DURATION_IN_MS_COL]
                + sleep_summaries.iloc[-1][
                    constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL
                ]
            ),
            unit="ms",
            utc=True,
        ).tz_localize(None)

        sleep_start_time = sleep_start_time.to_pydatetime()
        sleep_end_time = sleep_end_time.to_pydatetime()

        sleep_stages = self.load_sleep_stage(
            user_id=user_id,
            start_date=sleep_start_time,
            end_date=sleep_end_time,
        )

        # Keep only sleep stages with correct sleep summaries
        sleep_stages = (
            sleep_stages[
                sleep_stages[constants.SLEEP_STAGE_SLEEP_SUMMARY_ID_COL].isin(
                    sleep_summaries[
                        constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL
                    ].unique()
                )
            ]
            .sort_values(by=[constants.UNIXTIMESTAMP_IN_MS_COL])
            .reset_index(drop=True)
        )
        # set index on sleep summaries
        sleep_summaries = sleep_summaries.set_index(
            constants.SLEEP_SUMMARY_SLEEP_SUMMARY_ID_COL
        )
        hypnograms = {}
        for sleep_summary_id, sleep_summary in sleep_summaries.iterrows():
            calendar_day = sleep_summary[constants.CALENDAR_DATE_COL]
            sleep_start_time = sleep_summary[constants.ISODATE_COL]
            sleep_end_time = pd.to_datetime(
                (
                    sleep_summary[constants.UNIXTIMESTAMP_IN_MS_COL]
                    + sleep_summary[
                        lf_constants.GARMIN_CONNECT_TIMEZONEOFFSET_IN_MS_COL
                    ]
                    + sleep_summary[constants.SLEEP_SUMMARY_DURATION_IN_MS_COL]
                    + sleep_summary[constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL]
                ),
                unit="ms",
                utc=True,
            ).tz_localize(None)

            intervals = int(
                divmod(
                    (sleep_end_time - sleep_start_time).total_seconds(), 60 * resolution
                )[0]
            )
            time_delta_intervals = [
                sleep_start_time + i * datetime.timedelta(minutes=resolution)
                for i in range(intervals)
            ]

            daily_sleep_stages = sleep_stages.loc[
                sleep_stages[constants.SLEEP_STAGE_SLEEP_SUMMARY_ID_COL]
                == sleep_summary_id
            ]

            hypnogram = pd.DataFrame(data={constants.ISODATE_COL: time_delta_intervals})

            hypnogram = hypnogram.merge(
                daily_sleep_stages.loc[
                    :,
                    [
                        constants.ISODATE_COL,
                        constants.SLEEP_STAGE_SLEEP_TYPE_COL,
                    ],
                ],
                how="left",
                on=constants.ISODATE_COL,
            )

            hypnogram[constants.SLEEP_STAGE_SLEEP_TYPE_COL] = hypnogram.loc[
                :, constants.SLEEP_STAGE_SLEEP_TYPE_COL
            ].ffill()
            if map_hypnogram:
                hypnogram[constants.SLEEP_STAGE_SLEEP_TYPE_COL] = hypnogram[
                    constants.SLEEP_STAGE_SLEEP_TYPE_COL
                ].map(
                    {
                        constants.SLEEP_STAGE_REM_STAGE_VALUE: constants.SLEEP_STAGE_REM_STAGE_MAPPED_VALUE,
                        constants.SLEEP_STAGE_AWAKE_STAGE_VALUE: constants.SLEEP_STAGE_AWAKE_STAGE_MAPPED_VALUE,
                        constants.SLEEP_STAGE_DEEP_STAGE_VALUE: constants.SLEEP_STAGE_DEEP_STAGE_MAPPED_VALUE,
                        constants.SLEEP_STAGE_LIGHT_STAGE_VALUE: constants.SLEEP_STAGE_LIGHT_STAGE_MAPPED_VALUE,
                        constants.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE: constants.SLEEP_STAGE_UNMEASURABLE_STAGE_MAPPED_VALUE,
                    }
                )
            hypnograms[calendar_day] = {}
            hypnograms[calendar_day]["start_time"] = sleep_start_time.to_pydatetime()
            hypnograms[calendar_day]["end_time"] = sleep_end_time.to_pydatetime()
            hypnograms[calendar_day]["values"] = hypnogram[
                constants.SLEEP_STAGE_SLEEP_TYPE_COL
            ].values

        return hypnograms

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
    ) -> pd.DataFrame:
        return self.load_garmin_metric(
            metric=constants.METRIC_HEART_RATE,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            source=source,
            filter_on_calendar_date=False,
        )

    def load_pulse_ox(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
    ) -> pd.DataFrame:
        if source == lf_constants.DATA_SOURCE_GARMIN_HEALTH_API:
            return self.load_garmin_connect_pulse_ox(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            return self.load_garmin_device_pulse_ox(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )

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
            return self.load_garmin_connect_respiration(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            return self.load_garmin_device_respiration(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )

    def load_sleep_pulse_ox(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
    ) -> pd.DataFrame:
        if source == lf_constants.DATA_SOURCE_GARMIN_HEALTH_API:
            return self.load_garmin_connect_sleep_pulse_ox(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            return self.load_garmin_device_sleep_pulse_ox(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )

    def load_sleep_respiration(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
    ) -> pd.DataFrame:
        if source == lf_constants.DATA_SOURCE_GARMIN_HEALTH_API:
            return self.load_garmin_connect_sleep_respiration(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            return self.load_garmin_device_sleep_respiration(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )

    def load_sleep_stage(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date, None] = None,
        end_date: Union[str, datetime.date, datetime.date, None] = None,
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
        data = self.get_data_from_datetime(
            user_id,
            lf_constants.GARMIN_CONNECT_SLEEP_STAGE_FOLDER,
            start_date,
            end_date,
            filter_on_calendar_date=False,
        )

        if len(data) > 0:
            # Change sleep stage values
            data[constants.SLEEP_STAGE_SLEEP_TYPE_COL] = data[
                constants.SLEEP_STAGE_SLEEP_TYPE_COL
            ].map(
                {
                    lf_constants.SLEEP_STAGE_LIGHT_STAGE_VALUE: constants.SLEEP_STAGE_LIGHT_STAGE_VALUE,
                    lf_constants.SLEEP_STAGE_DEEP_STAGE_VALUE: constants.SLEEP_STAGE_DEEP_STAGE_VALUE,
                    lf_constants.SLEEP_STAGE_REM_STAGE_VALUE: constants.SLEEP_STAGE_REM_STAGE_VALUE,
                    lf_constants.SLEEP_STAGE_AWAKE_STAGE_VALUE: constants.SLEEP_STAGE_AWAKE_STAGE_VALUE,
                    lf_constants.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE: constants.SLEEP_STAGE_UNMEASURABLE_STAGE_VALUE,
                }
            )

        return data

    def load_sleep_summary(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
        same_day_filter: bool = True,
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
            data[constants.CALENDAR_DATE_COL] = pd.to_datetime(
                data[constants.CALENDAR_DATE_COL],
                format="%Y-%m-%d",
            ).dt.date

            if same_day_filter:
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
                                constants.CALENDAR_DATE_COL,
                                "validationMap",
                                constants.SLEEP_SUMMARY_DURATION_IN_MS_COL,
                            ]
                        ),
                        ascending=True,
                    )
                    .groupby(constants.CALENDAR_DATE_COL)
                    .tail(1)
                )
                data = data.drop(["validationMap"], axis=1)

            # Rename columns based on pywearable specs
            data = data.rename(
                columns={
                    lf_constants.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL: constants.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL,
                    lf_constants.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL: constants.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL,
                    lf_constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL: constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL,
                    lf_constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL: constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL,
                    lf_constants.SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL: constants.SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL,
                }
            )

            return data.reset_index(drop=True)
        else:
            return pd.DataFrame()

    def load_stress(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
    ) -> pd.DataFrame:
        return self.load_garmin_metric(
            metric=constants.METRIC_STRESS,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            source=source,
        )

    def load_recovery(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
    ):
        """Load recovery data.

        This function returns Garmin Body Battery Recovery data from the
        given ``user_id`` with the time window defined by ``start_date``
        and ``end_date``. Recovery is computed considering body battery
        values at wake up and bedtime, and by comparing the total charge
        with respect to the available charge.

        Parameters
        ----------
        user_id : :class:`str`
            Unique user identifier for recovery data retrieval.
        start_date : :class:`datetime.date` or :class:`datetime.datetime` or :class:`str` or None
            Start date for recovery data retrieval.
        end_date : :class:`datetime.date` or :class:`datetime.datetime` or :class:`str` or None
            End date for recovery data retrieval.

        Returns
        -------
        :class:`pd.DataFrame`
            Formatted :class:`pd.DataFrame` with recovery data.
        """
        sleep_summary = self.load_sleep_summary(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            same_day_filter=True,
        )
        if len(sleep_summary) > 0:
            # We need to get body battery data for each sleep
            # Let's retrieve sleep timestamps
            sleep_summary["wakeup"] = pd.to_datetime(
                sleep_summary[constants.UNIXTIMESTAMP_IN_MS_COL]
                + sleep_summary[constants.TIMEZONEOFFSET_IN_MS_COL]
                + sleep_summary[constants.DURATION_IN_MS_COL]
                + sleep_summary[constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL],
                utc=True,
                unit="ms",
            ).dt.tz_localize(None)
            sleep_timestamps = sleep_summary.loc[
                :, [constants.CALENDAR_DATE_COL, constants.ISODATE_COL, "wakeup"]
            ]
            # Load body battery for all sleep sessions with 5 minute tolerance
            min_time = sleep_timestamps[constants.ISODATE_COL].min() - pd.Timedelta(
                minutes=5
            )
            max_time = sleep_timestamps["wakeup"].max() + pd.Timedelta(minutes=5)
            body_battery = self.load_stress(
                user_id=user_id,
                start_date=min_time.to_pydatetime(),
                end_date=max_time.to_pydatetime(),
                source=source,
            )
            if len(body_battery) > 0:
                if constants.STRESS_BODY_BATTERY_COL in body_battery.columns:
                    body_battery = body_battery.loc[
                        :,
                        [
                            constants.ISODATE_COL,
                            constants.STRESS_BODY_BATTERY_COL,
                        ],
                    ]
                    # Labfront has several duplicates, we take the mean by isoDate and make sure isoDate is monotonic increasing
                    body_battery = (
                        body_battery.groupby(constants.ISODATE_COL)[
                            constants.STRESS_BODY_BATTERY_COL
                        ]
                        .mean()
                        .sort_index()
                        .reset_index()
                    )
                    body_battery_bedtime = pd.merge_asof(
                        sleep_timestamps,
                        body_battery,
                        on=constants.ISODATE_COL,
                        direction="backward",
                        tolerance=pd.Timedelta(minutes=5),
                    ).loc[
                        :,
                        [
                            constants.CALENDAR_DATE_COL,
                            constants.STRESS_BODY_BATTERY_COL,
                        ],
                    ]
                    body_battery_wakeup = pd.merge_asof(
                        sleep_timestamps,
                        body_battery,
                        left_on="wakeup",
                        right_on=constants.ISODATE_COL,
                        direction="forward",
                        tolerance=pd.Timedelta(minutes=5),
                    ).loc[
                        :,
                        [
                            constants.CALENDAR_DATE_COL,
                            constants.STRESS_BODY_BATTERY_COL,
                        ],
                    ]
                    sleep_timestamps = pd.merge(
                        left=body_battery_bedtime,
                        right=body_battery_wakeup,
                        how="inner",
                        on=constants.CALENDAR_DATE_COL,
                        suffixes=["Bedtime", "Wakeup"],
                    )
                    sleep_timestamps[constants.RECOVERY_RECOVERY_COL] = (
                        sleep_timestamps[constants.STRESS_BODY_BATTERY_WAKING_COL]
                        - sleep_timestamps[constants.STRESS_BODY_BATTERY_BEDTIME_COL]
                    ) / (
                        100
                        - sleep_timestamps[constants.STRESS_BODY_BATTERY_BEDTIME_COL]
                    )
                    # make sure recovery isn't negative, set it to zero if it is
                    sleep_timestamps[constants.RECOVERY_RECOVERY_COL] = np.maximum(
                        sleep_timestamps[constants.RECOVERY_RECOVERY_COL], 0
                    )
                    sleep_timestamps = sleep_timestamps.reset_index(drop=True)
                    return sleep_timestamps.loc[
                        :,
                        [
                            constants.CALENDAR_DATE_COL,
                            constants.RECOVERY_RECOVERY_COL,
                            constants.STRESS_BODY_BATTERY_BEDTIME_COL,
                            constants.STRESS_BODY_BATTERY_WAKING_COL,
                        ],
                    ]
        return pd.DataFrame()

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
