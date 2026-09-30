import datetime
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd

from .. import schema


class BaseLoader(ABC):
    """Abstract class for the implementation of loader classes."""

    heart_rate_time_columns = schema.HEART_RATE_TIME_COLUMNS
    hrv_source = "bbi"
    hrv_time_columns = schema.BBI_TIME_COLUMNS
    respiratory_rate_source = "samples"
    respiratory_rate_time_columns = schema.RESPIRATION_TIME_COLUMNS
    spo2_source = "samples"
    spo2_time_columns = schema.RESPIRATION_TIME_COLUMNS
    activity_source = None
    activity_time_columns = schema.ACTIVITY_TIME_COLUMNS
    activity_primary_resolution = None
    provider_features = frozenset()
    provider_feature_time_columns = schema.HEART_RATE_TIME_COLUMNS

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

        summaries = schema.prepare_sleep_summary(summaries)
        reconstructed_end = pd.to_datetime(
            summaries[schema.LOCAL_TIME_COL]
        ) + pd.to_timedelta(
            pd.to_numeric(
                summaries[schema.DURATION_IN_MS_COL],
                errors="coerce",
            ).fillna(0)
            + pd.to_numeric(
                summaries[schema.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL],
                errors="coerce",
            ).fillna(0),
            unit="ms",
        )
        if schema.LOCAL_END_TIME_COL in summaries:
            local_end = pd.to_datetime(
                summaries[schema.LOCAL_END_TIME_COL]
            ).fillna(reconstructed_end)
        else:
            local_end = reconstructed_end
        result = pd.DataFrame(
            {
                "id": summaries[schema.SLEEP_SUMMARY_ID_COL],
                "calendar_date": pd.to_datetime(
                    summaries[schema.CALENDAR_DATE_COL]
                ).dt.normalize(),
                "type": "main_sleep",
                "main_sleep": True,
                "local_time": pd.to_datetime(summaries[schema.LOCAL_TIME_COL]),
                "local_end_time": local_end,
                "time_in_bed_seconds": (
                    local_end
                    - pd.to_datetime(summaries[schema.LOCAL_TIME_COL])
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

    def load_pulse_ox(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        """Load oxygen-saturation samples in the shared sample schema."""
        return schema.empty_spo2_frame()

    def load_activity_samples(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Load incremental step intervals in the shared activity schema."""
        return schema.empty_activity_frame()

    def load_activity_summary(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Load calendar-day step totals in the shared activity schema."""
        return schema.empty_activity_summary()

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

        """
        return schema.empty_respiration_frame()

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

        """
        return schema.empty_spo2_frame()

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

        """
        return schema.empty_respiration_frame()

    def load_respiration_summary(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        """Load provider-native nightly respiratory-rate summaries."""
        return pd.DataFrame(
            columns=[
                schema.CALENDAR_DATE_COL,
                schema.RESPIRATORY_RATE_MEAN_COL,
                schema.RESPIRATORY_RATE_STD_COL,
            ]
        )

    def load_spo2_summary(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        """Load provider-native nightly oxygen-saturation summaries."""
        return pd.DataFrame(
            columns=[
                schema.CALENDAR_DATE_COL,
                schema.SPO2_MEAN_COL,
                schema.SPO2_LOWER_BOUND_COL,
                schema.SPO2_UPPER_BOUND_COL,
            ]
        )

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

    def load_provider_feature(
        self,
        feature: str,
        user_id: str,
        start_date=None,
        end_date=None,
        **kwargs,
    ) -> pd.DataFrame:
        """Load one provider-native feature in its canonical row schema.

        Implementations translate vendor fields, units, and missing-value
        conventions only. Windowing, aggregation, and derived features belong
        to :mod:`pywearable.provider_features`.
        """
        raise ValueError(
            f"{type(self).__name__} does not provide feature {feature!r}"
        )

    def load_hrv_samples(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        """Load provider-computed HRV samples when the source supplies them."""
        return pd.DataFrame()

    def load_hrv_summary(
        self,
        user_id: str,
        start_date: Union[datetime.datetime, datetime.date, str, None] = None,
        end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ) -> pd.DataFrame:
        """Load provider-computed daily or nightly HRV summaries."""
        return pd.DataFrame()

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


__all__ = ["BaseLoader"]
