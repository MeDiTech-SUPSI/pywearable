import datetime
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd

from .. import constants, sleep_schema


class BaseLoader(ABC):
    """Abstract class for the implementation of loader classes."""

    heart_rate_time_columns = ("utc_time", "local_time")

    def __init__(self):
        pass

    # @abstractmethod
    # def load_daily_summary(
    #    self,
    #    user_id: str,
    #    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    #    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    # ) -> pd.DataFrame:
    #    pass

    @abstractmethod
    def load_sleep_summary(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_sleep_stage(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        pass

    def load_sleep_sessions(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        """Return known sleep sessions in the shared session schema.

        Loaders without a separate all-session source expose their nightly
        summaries as main-sleep sessions. Provider implementations can
        override this method to include naps and other non-main sessions.
        """
        summaries = self.load_sleep_summary(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            drop_duplicates=True,
        )
        columns = [
            "id",
            "calendar_date",
            "type",
            "main_sleep",
            "local_time",
            "local_end_time",
            "time_in_bed_seconds",
        ]
        if summaries.empty:
            return pd.DataFrame(columns=columns)

        summaries = sleep_schema.prepare_sleep_summary(summaries)
        reconstructed_end = pd.to_datetime(
            summaries[sleep_schema.ISODATE_COL]
        ) + pd.to_timedelta(
            pd.to_numeric(
                summaries[sleep_schema.SLEEP_SUMMARY_DURATION_IN_MS_COL],
                errors="coerce",
            ).fillna(0)
            + pd.to_numeric(
                summaries[sleep_schema.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL],
                errors="coerce",
            ).fillna(0),
            unit="ms",
        )
        if sleep_schema.LOCAL_END_TIME_COL in summaries:
            local_end = pd.to_datetime(
                summaries[sleep_schema.LOCAL_END_TIME_COL]
            ).fillna(reconstructed_end)
        else:
            local_end = reconstructed_end
        result = pd.DataFrame(
            {
                "id": summaries[sleep_schema.SLEEP_SUMMARY_ID_COL],
                "calendar_date": pd.to_datetime(
                    summaries[sleep_schema.CALENDAR_DATE_COL]
                ).dt.normalize(),
                "type": "main_sleep",
                "main_sleep": True,
                "local_time": pd.to_datetime(summaries[sleep_schema.ISODATE_COL]),
                "local_end_time": local_end,
                "time_in_bed_seconds": (
                    local_end
                    - pd.to_datetime(summaries[sleep_schema.ISODATE_COL])
                ).dt.total_seconds(),
            }
        )
        return result[columns].sort_values("local_time").reset_index(drop=True)

    def load_sleep_session_stages(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        """Return stages for all available sessions.

        The default is the loader's main-sleep stage trajectory. Loaders with
        staged naps or other non-main sessions override this method.
        """
        return self.load_sleep_stage(user_id, start_date, end_date)

    @abstractmethod
    def load_heart_rate(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        time_col: str = "local_time",
    ) -> pd.DataFrame:
        """Load heart-rate samples in the shared heart-rate schema.

        Implementations return ``utc_time``, ``local_time``, ``timezone``
        (a UTC offset as :class:`pandas.Timedelta`), and ``heart_rate``. A
        loader can advertise unavailable clocks through
        ``heart_rate_time_columns``; their canonical columns are retained but
        contain missing values. Bounds are inclusive/exclusive and are applied
        to ``time_col``. Direct calls default to local time. Feature extraction
        uses UTC for subdaily windows and local time for calendar windows unless
        explicitly overridden.
        """
        pass

    @abstractmethod
    def load_pulse_ox(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_respiration(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        """Load respiration data.

        This function loads respiration data from
        a given user and for a given time frame, from
        ``start_date`` to ``end_date``.

        Parameters
        ----------
        user_id : str
            User id for which data have to be loaded.
        start_date : datetime.datetime or datetime.date or str or None, optional
            Start date for data retrieval, by default None
        end_date : datetime.datetime or datetime.date or str or None, optional
            End date for data retrieval, by default None

        Returns
        -------
        :class:`pd.DataFrame`
            DataFrame with respiration data.

        Raises
        ------
        NotImplementedError
            Method must be implement by loaders.
        """
        raise NotImplementedError

    @abstractmethod
    def load_sleep_pulse_ox(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        """Load sleep pulse ox data.

        This function loads sleep pulse ox data from
        a given user and for a given time frame, from
        ``start_date`` to ``end_date``.

        Parameters
        ----------
        user_id : str
            User id for which data have to be loaded.
        start_date : datetime.datetime or datetime.date or str or None, optional
            Start date for data retrieval, by default None
        end_date : datetime.datetime or datetime.date or str or None, optional
            End date for data retrieval, by default None

        Returns
        -------
        :class:`pd.DataFrame`
            DataFrame with sleep pulse ox data.

        Raises
        ------
        NotImplementedError
            Method must be implement by loaders.
        """
        pass

    @abstractmethod
    def load_sleep_respiration(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        """Load sleep respiration data.

        This function loads sleep respiration data from
        a given user and for a given time frame, from
        ``start_date`` to ``end_date``.

        Parameters
        ----------
        user_id : str
            User id for which data have to be loaded.
        start_date : datetime.datetime or datetime.date or str or None, optional
            Start date for data retrieval, by default None
        end_date : datetime.datetime or datetime.date or str or None, optional
            End date for data retrieval, by default None

        Returns
        -------
        :class:`pd.DataFrame`
            DataFrame with sleep respiration data.

        Raises
        ------
        NotImplementedError
            Method must be implement by loaders.
        """
        pass

    @abstractmethod
    def load_bbi(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        pass

    def load_stress(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def load_epoch(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def load_bbi(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def load_hypnogram(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
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
        user_id : str
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
                    + sleep_summary[constants.TIMEZONEOFFSET_IN_MS_COL]
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

    @abstractmethod
    def get_user_ids(self) -> list:
        """Get list of available user IDs.

        This function is used in sub-modules when
        the ``user_id`` parameter is set to ``"all"``.

        Returns
        -------
        :class:`list`
            List of available user IDs.
        """
        pass
