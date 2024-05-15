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


class LabfrontLoader(BaseLoader):
    """Loader for Labfront data.

    This is the main object for loading and analyzing data collected
    with Labfront.

    Parameters
    ----------
        data_path: str or Path
            Path to folder containing Labfront data.
    """

    def __init__(self, data_path: Union[str, Path], force_reload: bool = False):
        """Constructor method"""
        self.set_path(data_path, force_reload=force_reload)

    def set_path(self, data_path: Union[str, Path], force_reload: bool = False):
        """Set path to folder containing Labfront data.

        This function allows to change the path of Labfront data
        to be loaded.

        Parameters
        ----------
        data_path : str or Path
            Path to folder containing Labfront data.
        """
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        self.data_path = data_path
        self.file_index = self.get_file_index(force_reload=force_reload)
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
        elif id in self.file_index[lf_constants.FILE_INDEX_PARTICIPANT_ID_KEY].unique():
            # This is a user ID
            # Check if we have more than one participant with the same user id
            if (
                len(
                    self.file_index.loc[
                        (slice(None), id, slice(None), slice(None)), :
                    ].index.get_level_values(
                        lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY
                    )
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
            in self.file_index[
                lf_constants.FILE_INDEX_PARTICIPANT_LABFRONT_ID_KEY
            ].unique()
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
        if lf_constants.FILE_INDEX_QUESTIONNAIRE_NAME_KEY in self.file_index:
            return self.file_index[
                lf_constants.FILE_INDEX_QUESTIONNAIRE_NAME_KEY
            ].unique()
        else:
            return []

    def get_available_todos(
        self, user_id: Union[str, list] = "all", return_dict: bool = False
    ) -> list:
        # TODO get the name of the todos from file_index
        """Get available todos for given users.

        Parameters
        ----------
        user_id : str or list, optional
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
        if lf_constants.FILE_INDEX_TODO_NAME_KEY in self.file_index:
            return self.file_index[lf_constants.FILE_INDEX_TODO_NAME_KEY].unique()
        else:
            return []

    def get_file_index(self, force_reload: bool = False) -> pd.DataFrame:
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
        if not self.data_path.exists():
            raise FileNotFoundError
        file_index = pd.DataFrame()
        for participant_folder in self.data_path.iterdir():
            # For each participant
            if participant_folder.is_dir():
                participant_dict = {}
                for participant_metric_folder in participant_folder.iterdir():
                    # For each metric
                    if participant_metric_folder.is_dir():
                        # Get the number of files inside folder
                        n_files = len(os.listdir(participant_metric_folder))
                        for metric_csv_file in participant_metric_folder.iterdir():
                            # If it is a folder, then we need to read the csv files
                            # and get first and last unix times for each CSV file
                            if metric_csv_file.is_file() and str(
                                metric_csv_file
                            ).endswith("csv"):
                                first_date = datetime.datetime.strptime(
                                    metric_csv_file.name[:6], "%y%m%d"
                                )
                                # If we have a single file, start is from file name, end is today
                                if n_files == 1:
                                    last_date = np.nan
                                else:
                                    last_date = first_date + datetime.timedelta(days=1)
                                participant_dict[metric_csv_file.name] = {
                                    lf_constants.FILE_INDEX_FIRST_TIMESTAMP_KEY: first_date,
                                    lf_constants.FILE_INDEX_LAST_TIMESTAMP_KEY: last_date,
                                    lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY: participant_folder.name,
                                    lf_constants.FILE_INDEX_METRIC_KEY: participant_metric_folder.name,
                                    lf_constants.FILE_INDEX_IS_QUESTIONNAIRE_KEY: False,
                                    lf_constants.FILE_INDEX_IS_TODO_KEY: False,
                                }
                            else:
                                # We don't have a file, we expect a questionnaire/task folder
                                if not metric_csv_file.is_dir():
                                    continue
                                # For each questionnaire/task folder
                                for csv_file in metric_csv_file.iterdir():
                                    if lf_constants.TODO_FOLDER == metric_csv_file.name:
                                        # We have a task
                                        is_todo = True
                                        is_questionnaire = False
                                    elif (
                                        lf_constants.QUESTIONNAIRE_FOLDER
                                        == metric_csv_file.name
                                    ):
                                        # We have a questionnaire
                                        is_todo = False
                                        is_questionnaire = True
                                    else:
                                        continue
                                    if csv_file.is_file() and str(csv_file).endswith(
                                        "csv"
                                    ):
                                        # We need to get the name of the questionnaire/task and its id
                                        # Get first date from filename
                                        first_date = datetime.datetime.strptime(
                                            csv_file.name[:6], "%y%m%d"
                                        )
                                        # If we have a single file, start is from file name, end is today
                                        if n_files == 1:
                                            last_date = np.nan
                                        else:
                                            last_date = first_date + datetime.timedelta(
                                                days=1
                                            )

                                        participant_dict[metric_csv_file.name] = {
                                            lf_constants.FILE_INDEX_FIRST_TIMESTAMP_KEY: first_date,
                                            lf_constants.FILE_INDEX_LAST_TIMESTAMP_KEY: last_date,
                                            lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY: participant_folder.name,
                                            lf_constants.FILE_INDEX_METRIC_KEY: participant_metric_folder.name,
                                            lf_constants.FILE_INDEX_IS_QUESTIONNAIRE_KEY: is_questionnaire,
                                            lf_constants.FILE_INDEX_IS_TODO_KEY: is_todo,
                                        }

                                        if is_questionnaire:
                                            (
                                                questionnaire_name,
                                                questionnaire_id,
                                                task_id,
                                            ) = self._get_questionnaire_info(csv_file)
                                            participant_dict[metric_csv_file.name][
                                                lf_constants.FILE_INDEX_QUESTIONNAIRE_NAME_KEY
                                            ] = questionnaire_name
                                            participant_dict[metric_csv_file.name][
                                                lf_constants.FILE_INDEX_QUESTIONNAIRE_ID_KEY
                                            ] = questionnaire_id
                                            participant_dict[metric_csv_file.name][
                                                lf_constants.FILE_INDEX_TASK_ID_KEY
                                            ] = task_id
                                        elif is_todo:
                                            todo_name, todo_id, task_id = (
                                                self._get_todo_info(csv_file)
                                            )
                                            participant_dict[metric_csv_file.name][
                                                lf_constants.TODO_NAME_COL
                                            ] = todo_name
                                            participant_dict[metric_csv_file.name][
                                                lf_constants.TODO_ID_COL
                                            ] = todo_id
                                            participant_dict[metric_csv_file.name][
                                                lf_constants.TASK_ID_COL
                                            ] = task_id
                if len(participant_dict) > 0:
                    if len(file_index) > 0:
                        file_index = pd.concat(
                            (
                                file_index,
                                pd.DataFrame.from_dict(
                                    participant_dict, orient="index"
                                ),
                            ),
                            axis=0,
                            ignore_index=True,
                        )
                    else:
                        file_index = pd.DataFrame.from_dict(
                            participant_dict, orient="index"
                        )
        if lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY in file_index.columns:
            # Create two separate columns for IDs
            file_index[
                [
                    lf_constants.FILE_INDEX_PARTICIPANT_ID_KEY,
                    lf_constants.FILE_INDEX_PARTICIPANT_LABFRONT_ID_KEY,
                ]
            ] = file_index[lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY].str.split(
                "_", expand=True
            )
        # Reset index
        file_index = file_index.reset_index(
            names=lf_constants.FILE_INDEX_FILENAME_KEY
        ).set_index(
            [
                lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY,
                lf_constants.FILE_INDEX_PARTICIPANT_ID_KEY,
                lf_constants.FILE_INDEX_PARTICIPANT_LABFRONT_ID_KEY,
                lf_constants.FILE_INDEX_METRIC_KEY,
            ]
        )

        return file_index

    def _get_questionnaire_info(path_to_file) -> Tuple[str, str]:
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

    def _get_todo_info(path_to_file) -> Tuple[str, str]:
        todo_info_df = pd.read_csv(
            path_to_file, nrows=1, skiprows=lf_constants.CSV_STATS_SKIP_ROWS
        )
        todo_name = todo_info_df.loc[0, lf_constants.TODO_NAME_COL]
        todo_id = todo_info_df.loc[0, lf_constants.TODO_ID_COL]
        task_id = todo_info_df.loc[0, lf_constants.TASK_ID_COL]
        return (todo_name, todo_id, task_id)

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
        user_id : str
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
        if not (
            metric
            in self.file_index.loc[(participant_id)].index.get_level_values(
                lf_constants.FILE_INDEX_METRIC_KEY
            )
        ):
            # Participant does not have files for this metric, return empty list
            return []
        # Reset index and fill last timestamps with end date
        participant_files = (
            self.file_index.loc[(participant_id, slice(None), slice(None), metric), :]
            .reset_index(drop=True)
            .fillna(end_date)
        )
        # We need to floor start_date to the previous day
        # since Labfront files contain the midnight in the
        # file belonging to the previous day
        start_date = pd.Timestamp(
            start_date - datetime.timedelta(microseconds=1)
        ).floor("1d")
        # We need to ceil end_date to the next date to
        # handle end-date non inclusiveness with dates
        end_date = pd.Timestamp(end_date).ceil("1d")
        if len(participant_files) == 1:
            # We only have one file
            if (
                start_date
                >= participant_files.iloc[0].loc[
                    lf_constants.FILE_INDEX_FIRST_TIMESTAMP_KEY
                ]
            ) or (
                end_date
                <= participant_files.iloc[0].loc[
                    lf_constants.FILE_INDEX_LAST_TIMESTAMP_KEY
                ]
            ):
                # The file contains either start date or end date
                return participant_files[lf_constants.FILE_INDEX_FILENAME_KEY].values
            else:
                return []
        else:
            # Return only files that contain start and end dates
            participant_files = (
                participant_files.set_index(lf_constants.FILE_INDEX_FIRST_TIMESTAMP_KEY)
                .truncate(before=start_date, after=end_date)
                .loc[:, lf_constants.FILE_INDEX_FILENAME_KEY]
                .values
            )
            return participant_files

    def get_data_from_datetime(
        self,
        user_id: str,
        metric: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        is_questionnaire: bool = False,
        is_todo: bool = False,
        task_name: str = None,
    ) -> pd.DataFrame:
        """Load data from a given user in a given time frame.

        This function allows to load data of a given metric from a specified user
        only within a given timeframe. If `start_date` and `end_date` are set to None, then
        all data available for the user of interest are returned.
        If no data are available, then an empty :class:`pandas.DataFrame` is returned.

        Parameters
        ----------
        user_id : str
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
        if not (
            (type(start_date) == datetime.datetime)
            or (type(start_date) == datetime.date)
            or (start_date is None)
        ):
            start_date = dateutil.parser.parse(start_date)
        elif type(start_date) == datetime.date:
            start_date = datetime.datetime.combine(start_date, datetime.time())

        if not (
            (type(end_date) == datetime.datetime)
            or (type(end_date) == datetime.date or (end_date is None))
        ):
            end_date = dateutil.parser.parse(end_date)
        elif type(end_date) == datetime.date:
            end_date = datetime.datetime.combine(end_date, datetime.time())

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
        if is_questionnaire:
            path_to_folder = (
                self.data_path
                / participant_id
                / lf_constants.QUESTIONNAIRE_FOLDER
                / task_name
            )
        elif is_todo:
            path_to_folder = (
                self.data_path
                / participant_id
                / lf_constants.QUESTIONNAIRE_FOLDER
                / task_name
            )
        else:
            path_to_folder = self.data_path / participant_id / metric

        n_rows_to_skip = self.get_header_length(path_to_folder / files[0])
        if is_questionnaire:
            n_rows_to_skip += self.get_key_length(path_to_folder / files[0]) + 1
        # Load data from first file
        df_list = []
        for f in files:
            try:
                df_list.append(pd.read_csv(path_to_folder / f, skiprows=n_rows_to_skip))
            except pd.errors.EmptyDataError:
                # We may have empty CSV files
                continue
        if len(df_list) > 0:
            data = pd.concat(df_list, ignore_index=True)
        else:
            return pd.DataFrame()

        # Convert to datetime according to isoformat
        data[constants.ISODATE_COL] = pd.to_datetime(
            data[constants.ISODATE_COL]
        ).dt.tz_localize(None)
        # Get data only from given start and end dates
        if (start_date is None) and (not end_date is None):
            return data[(data[constants.ISODATE_COL] < end_date)].reset_index(drop=True)
        elif (not start_date is None) and (end_date is None):
            return data[(data[constants.ISODATE_COL] >= start_date)].reset_index(
                drop=True
            )
        elif (not start_date is None) and (not end_date is None):
            return data[
                (data[constants.ISODATE_COL] >= start_date)
                & (data[constants.ISODATE_COL] < end_date)
            ].reset_index(drop=True)
        else:
            return data.reset_index(drop=True)

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
        user_id : str or list, optional
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

        return self.tasks_dict[task_id.lower()]

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
        for user_id in self.data_dictionary.keys():
            if (
                lf_constants.QUESTIONNAIRE_FOLDER
                in self.data_dictionary[user_id].keys()
            ):
                if (
                    full_task_id
                    in self.data_dictionary[user_id][
                        lf_constants.QUESTIONNAIRE_FOLDER
                    ].keys()
                ):
                    no_user_id = False
                    participant_id = user_id
                    break
        if no_user_id:
            raise ValueError(
                f"Could not find questions for questionnaire {questionnaire_name}"
            )

        files = self.get_files_from_timerange(
            participant_id,
            lf_constants.QUESTIONNAIRE_FOLDER,
            start_date=None,
            end_date=None,
            is_questionnaire=True,
            is_todo=False,
            task_name=questionnaire_name,
        )

        if len(files) == 0:
            return {}

        path_to_folder = (
            self.data_path
            / participant_id
            / lf_constants.QUESTIONNAIRE_FOLDER
            / full_task_id
        )

        questionnaire_info_file = path_to_folder / files[0]
        header_length = self.get_header_length(questionnaire_info_file)
        key_length = self.get_key_length(questionnaire_info_file)

        questions_df = pd.read_csv(
            questionnaire_info_file, skiprows=header_length + 2, nrows=key_length - 2
        )

        questions_dict = {}
        # For each row, we need to get the question and all the options
        option_cols = []
        for col in questions_df.columns:
            regex_match = re.search(constants.QUESTIONNAIRE_QUESTION_NAME_REGEX, col)
            if regex_match:
                option_cols.append(col)
        for idx, row in questions_df.iterrows():
            question_id = f"{row[constants.QUESTIONNAIRE_SECTION_INDEX_COL]}_{row[constants.QUESTIONNAIRE_QUESTION_INDEX_COL]}"
            questions_dict[question_id] = {
                "type": row[constants.QUESTIONNAIRE_QUESTION_TYPE_COL],
                "description": row[constants.QUESTIONNAIRE_QUESTION_DESCRIPTION_COL],
            }
            if (
                row[constants.QUESTIONNAIRE_QUESTION_TYPE_COL]
                == constants.QUESTIONNAIRE_QUESTION_TYPE_TEXT_VALUE
            ):
                continue
            question_options = []
            for col in option_cols:
                if not pd.isna(row[col]):
                    question_options.append(row[col])
            questions_dict[question_id]["options"] = question_options
        return questions_dict

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
        user_id : str
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
            user_id,
            lf_constants.GARMIN_CONNECT_HEART_RATE_FOLDER,
            start_date,
            end_date,
        )
        return data

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

    def load_garmin_connect_pulse_ox(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.datetime] = None,
        end_date: Union[str, datetime.date, datetime.datetime] = None,
    ) -> pd.DataFrame:
        """Load Garmin Connect pulse ox data.

        Parameters
        ----------
        user_id : str
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
        ).reset_index(drop=True)
        # Load sleep pulse ox data
        sleep_data = self.get_data_from_datetime(
            user_id=user_id,
            metric=lf_constants.GARMIN_CONNECT_SLEEP_PULSE_OX_FOLDER,
            start_date=start_date,
            end_date=end_date,
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
        user_id : str
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
            user_id,
            lf_constants.GARMIN_CONNECT_RESPIRATION_FOLDER,
            start_date,
            end_date,
        ).reset_index(drop=True)

        # Get sleep data
        sleep_data = self.get_data_from_datetime(
            user_id,
            lf_constants.GARMIN_CONNECT_SLEEP_RESPIRATION_FOLDER,
            start_date,
            end_date,
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
        )

        if len(data) > 0:
            # Change sleep stage values
            data[constants.SLEEP_STAGE_SLEEP_TYPE_COL] = data[
                constants.SLEEP_STAGE_SLEEP_TYPE_COL
            ].map(
                {
                    lf_constants.SLEEP_STAGE_LIGHT_STAGE_VALUE: constants.SLEEP_STAGE_N1_STAGE_VALUE,
                    lf_constants.SLEEP_STAGE_DEEP_STAGE_VALUE: constants.SLEEP_STAGE_N3_STAGE_VALUE,
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
        user_id : str
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
            user_id,
            lf_constants.GARMIN_CONNECT_SLEEP_SUMMARY_FOLDER,
            start_date,
            end_date,
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
                    lf_constants.SLEEP_SUMMARY_LIGHT_SLEEP_DURATION_IN_MS_COL: constants.SLEEP_SUMMARY_N1_SLEEP_DURATION_IN_MS_COL,
                    lf_constants.SLEEP_SUMMARY_DEEP_SLEEP_DURATION_IN_MS_COL: constants.SLEEP_SUMMARY_N3_SLEEP_DURATION_IN_MS_COL,
                    lf_constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL: constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL,
                    lf_constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL: constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL,
                    lf_constants.SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL: constants.SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL,
                }
            )

            data[constants.SLEEP_SUMMARY_N2_SLEEP_DURATION_IN_MS_COL] = np.nan

            data = data.loc[
                :,
                [
                    constants.SLEEP_SUMMARY_ID_COL,
                    constants.TIMEZONEOFFSET_IN_MS_COL,
                    constants.UNIXTIMESTAMP_IN_MS_COL,
                    constants.ISODATE_COL,
                    constants.CALENDAR_DATE_COL,
                    constants.DURATION_IN_MS_COL,
                    constants.SLEEP_SUMMARY_N1_SLEEP_DURATION_IN_MS_COL,
                    constants.SLEEP_SUMMARY_N2_SLEEP_DURATION_IN_MS_COL,
                    constants.SLEEP_SUMMARY_N3_SLEEP_DURATION_IN_MS_COL,
                    constants.SLEEP_SUMMARY_REM_SLEEP_DURATION_IN_MS_COL,
                    constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL,
                    constants.SLEEP_SUMMARY_UNMEASURABLE_SLEEP_DURATION_IN_MS_COL,
                    constants.SLEEP_SUMMARY_OVERALL_SLEEP_SCORE_COL,
                ],
            ]

            if (start_date is None) and (end_date is None):
                return data
            else:
                start_date = None if (start_date is None) else start_date.date()
                end_date = None if (end_date is None) else end_date.date()
                if (start_date is None) and (not (end_date is None)):
                    return data[(data[constants.CALENDAR_DATE_COL] <= end_date)]
                elif (not (start_date is None)) and (end_date is None):
                    return data[(data[constants.CALENDAR_DATE_COL] >= start_date)]
                else:
                    return data[
                        (data[constants.CALENDAR_DATE_COL] >= start_date)
                        & (data[constants.CALENDAR_DATE_COL] <= end_date)
                    ]
        else:
            return pd.DataFrame()

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
        user_id : str
            ID of the user.
        start_date : str or datetime.date or datetime.date, optional
            Start date from which data should be retrieved, by default None
        end_date : str or datetime.date or datetime.date, optional
            End date from which data should be retrieved, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing stress data.
        """
        data = self.get_data_from_datetime(
            user_id,
            lf_constants.GARMIN_CONNECT_STRESS_FOLDER,
            start_date,
            end_date,
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
        user_id : str
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
            user_id,
            lf_constants.GARMIN_DEVICE_HEART_RATE_FOLDER,
            start_date,
            end_date,
        )
        return data

    def _load_garmin_device_all_pulse_ox(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
    ) -> pd.DataFrame:
        data = self.get_data_from_datetime(
            user_id,
            lf_constants.GARMIN_DEVICE_PULSE_OX_FOLDER,
            start_date,
            end_date,
        )
        if len(data) > 0:
            # We need to add calendarDate from sleep summaries
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
        user_id : str
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
        user_id : str
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
        user_id : str
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
        user_id : str
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
        user_id : str
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
        user_id : str
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
        user_id : str
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
        user_id : str
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
        user_id : str
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
        start_date = utils.check_date(start_date)
        end_date = utils.check_date(end_date)
        if not start_date is None:
            new_start_date = start_date - datetime.timedelta(days=1)
            start_date_date = datetime.datetime.combine(
                start_date.date(), datetime.time(0, 0, 0)
            )
        else:
            new_start_date = None
        if not end_date is None:
            new_end_date = end_date + datetime.timedelta(days=1)
            end_date_date = datetime.datetime.combine(
                end_date.date(), datetime.time(0, 0, 0)
            )
        else:
            new_end_date = None
        data = self.get_data_from_datetime(
            user_id,
            lf_constants.GARMIN_CONNECT_DAILY_SUMMARY_FOLDER,
            new_start_date,
            new_end_date,
        )
        if len(data) > 0:
            data[constants.CALENDAR_DATE_COL] = pd.to_datetime(
                data[constants.CALENDAR_DATE_COL],
                format="%Y-%m-%d",
            )
            if (start_date is None) and (not (end_date is None)):
                return data[data[constants.CALENDAR_DATE_COL] <= end_date_date]
            elif (not (start_date is None)) and (end_date is None):
                return data[data[constants.CALENDAR_DATE_COL] >= start_date_date]
            elif (not (start_date is None)) and (not (end_date is None)):
                return data[
                    (data[constants.CALENDAR_DATE_COL] >= start_date_date)
                    & (data[constants.CALENDAR_DATE_COL] <= end_date_date)
                ]
            else:
                return data
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
        user_id : str
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
            user_id,
            lf_constants.GARMIN_CONNECT_EPOCH_FOLDER,
            start_date,
            end_date,
        )
        return data

    def load_todo(
        self,
        user_id: str,
        start_date: Union[str, datetime.date, datetime.date] = None,
        end_date: Union[str, datetime.date, datetime.date] = None,
        todo_name: str = None,
    ):
        """Load todo data.

        This function loads todo data from a given
        participant and within a specified date and time range.

        Parameters
        ----------
        user_id : str
            ID of the user for which todo data have to be loaded.
            This can be the user ID, Labfront ID, or
            the full ID. If multiple users exist with the same user ID, then
            it is necessary to pass either the Labfront ID or the full ID.
        start_date : str or datetime.date or datetime.date, optional
            Start data for data loading, by default None
        end_date : str or datetime.date or datetime.date, optional
            End data for data loading, by default None
        todo_name : str, optional
            Name of the todo to be loaded, by default None

        Returns
        -------
        pd.DataFrame
            Todo data.
        """
        data = self.get_data_from_datetime(
            user_id,
            lf_constants.TODO_FOLDER,
            start_date,
            end_date,
            is_todo=True,
            task_name=todo_name,
        )
        return data

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
        user_id : str
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
            user_id,
            metric=lf_constants.QUESTIONNAIRE_FOLDER,
            start_date=start_date,
            end_date=end_date,
            is_questionnaire=True,
            task_name=questionnaire_name,
        )
        return data

    def load_metric(
        self,
        metric: str,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
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
        )

    def load_heart_rate(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
    ) -> pd.DataFrame:
        return self.load_metric(
            metric=constants.METRIC_HEART_RATE,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            source=source,
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

    def load_stress(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        source=lf_constants.DATA_SOURCE_GARMIN_HEALTH_API,
    ) -> pd.DataFrame:
        return self.load_metric(
            metric=constants.METRIC_STRESS,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            source=source,
        )
