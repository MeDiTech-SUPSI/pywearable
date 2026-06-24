"""
This module offers methods to perform imputation of numerical
and categorical features on the dataset.
"""

# TODO check time scale and date_col
# TODO check sort dataset and set frequency
# TODO add imputation with other strategies

from typing import Literal, Union
import pandas as pd

MEAN_STRATEGY = "mean"
MEDIAN_STRATEGY = "median"
MEAN_DAYOFF_STRATEGY = "mean_dayoff"
MEDIAN_DAYOFF_STRATEGY = "median_dayoff"
MEAN_ROLLING_STRATEGY = "mean_rolling"
MEDIAN_ROLLING_STRATEGY = "median_rolling"

STRATEGIES_DICT = {
    MEAN_STRATEGY: {
        "transform": "mean",
        "rolling": False,
    },
    MEAN_ROLLING_STRATEGY: {
        "transform": "mean",
        "rolling": True,
    },
    MEDIAN_STRATEGY: {
        "transform": "median",
        "rolling": False,
    },
    MEDIAN_ROLLING_STRATEGY: {
        "transform": "median",
        "rolling": True,
    },
}


def impute_missing_values_with_personalized_mean(
    dataset: pd.DataFrame,
    on: Union[str, list, None],
    id_col: str,
    suffix: Union[str, None] = None,
) -> pd.DataFrame:
    """Perform imputation using personalized mean values.

    This function is just a wrapper for the main
    imputation function :py:func:`.impute_missing_values`, in
    which the imputation is performed with the
    subject-specific mean values.

    Parameters
    ----------
    dataset : :class:`pandas.DataFrame`
        Dataset in which imputation must be performed.
    on : :class:`str` or :class:`list` or ``None``
        Name of column(s) on which imputation must be performed.
    id_col: :class:`str`
        Name of column to use as group identifier.
    suffix : :class:`str` or ``None``, optional
        Suffix to be used for the columns with imputed values, by default ``None``.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame with ``min(0,len(on))`` new columns, each one named using the
        previous column names and the provided ``suffix``.
    """
    return impute_missing_values(
        dataset=dataset,
        on=on,
        strategy=MEAN_STRATEGY,
        suffix=suffix,
        use_population_values=False,
        id_col=id_col,
    )


def impute_missing_values_with_population_mean(
    dataset: pd.DataFrame,
    on: Union[str, list, None],
    suffix: Union[str, None] = None,
) -> pd.DataFrame:
    """Perform imputation using population mean values.

    This function is just a wrapper for the main
    imputation function :py:func:`.impute_missing_values`, in
    which the imputation is performed with the mean
    population values.

    Parameters
    ----------
    dataset : :class:`pandas.DataFrame`
        Dataset in which imputation must be performed.
    on : :class:`str` or :class:`list` or ``None``
        Name of column(s) on which imputation must be performed.
    suffix : :class:`str` or ``None``, optional
        Suffix to be used for the columns with imputed values, by default ``None``.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame with ``min(0,len(on))`` new columns, each one named using the
        previous column names and the provided ``suffix``.
    """
    return impute_missing_values(
        dataset=dataset,
        on=on,
        strategy=MEAN_STRATEGY,
        suffix=suffix,
        use_population_values=True,
    )


def impute_missing_values_with_personalized_median(
    dataset: pd.DataFrame,
    on: Union[str, list, None],
    id_col: str,
    suffix: Union[str, None] = None,
) -> pd.DataFrame:
    """Perform imputation using personalized median values.

    This function is just a wrapper for the main
    imputation function :py:func:`.impute_missing_values`, in
    which the imputation is performed with the
    subject-specific median values.

    Parameters
    ----------
    dataset : :class:`pandas.DataFrame`
        Dataset in which imputation must be performed.
    on : :class:`str` or :class:`list` or ``None``
        Name of column(s) on which imputation must be performed.
    id_col: :class:`str`
        Name of column to use as group identifier.
    suffix : :class:`str` or ``None``, optional
        Suffix to be used for the columns with imputed values, by default ``None``.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame with ``min(0,len(on))`` new columns, each one named using the
        previous column names and the provided ``suffix``.
    """
    return impute_missing_values(
        dataset=dataset,
        on=on,
        strategy=MEDIAN_STRATEGY,
        suffix=suffix,
        use_population_values=False,
        id_col=id_col,
    )


def impute_missing_values_with_population_median(
    dataset: pd.DataFrame,
    on: Union[str, list, None],
    suffix: Union[str, None] = None,
) -> pd.DataFrame:
    """Perform imputation using population median values.

    This function is just a wrapper for the main
    imputation function :py:func:`.impute_missing_values`, in
    which the imputation is performed with the median
    population values.

    Parameters
    ----------
    dataset : :class:`pandas.DataFrame`
        Dataset in which imputation must be performed.
    on : :class:`str` or :class:`list` or ``None``
        Name of column(s) on which imputation must be performed.
    suffix : :class:`str` or ``None``, optional
        Suffix to be used for the columns with imputed values, by default ``None``.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame with ``min(0,len(on))`` new columns, each one named using the
        previous column names and the provided ``suffix``.
    """
    return impute_missing_values(
        dataset=dataset,
        on=on,
        strategy=MEDIAN_STRATEGY,
        suffix=suffix,
        use_population_values=True,
    )


def impute_missing_values_with_rolling_mean(
    dataset: pd.DataFrame,
    on: Union[str, list, None],
    id_col: str,
    date_col: str,
    rolling_window: Union[int, pd.Timedelta],
    rolling_min_periods: int,
    suffix: Union[str, None] = None,
    force_impute: bool = True,
    use_population_values: bool = False,
) -> pd.DataFrame:
    """Perform imputation using rolling mean values.

    This function is just a wrapper for the main
    imputation function :py:func:`.impute_missing_values`, in
    which by default a rolling window with mean computation
    is used for the imputation of missing values.

    Parameters
    ----------
    dataset : :class:`pandas.DataFrame`
        Dataset in which imputation must be performed.
    on : :class:`str` or :class:`list` or ``None``
        Name of column(s) on which imputation must be performed.
    id_col: :class:`str`
        Name of column to use as group identifier.
    date_col: :class:`str`
        Name of column to use as time identifier in the rolling mean operation.
    suffix : :class:`str` or ``None``, optional
        Suffix to be used for the columns with imputed values, by default ``None``.
    use_population_values : :class:`bool`, optional
        Use population values when forcing imputation, instead of
        group specific values, by default ``False``. Its value
        is useless if ``force_impute`` is set to ``False``.
    force_impute : :class:`bool`, optional
        Whether to force impute values in case there are still missing values
        after imputing them, by default ``True``.
        If set to ``False``, then missing values will still be present in the days
        in which the rolling mean could not be computed.
    rolling_window : :class:`int` or :class:`pandas.Timedelta`
        Size of the window used in the rolling mean computation, it can be
        of any type compatible with the **window** parameter of
        :py:func:`pandas.core.groupby.DataFrameGroupBy.rolling`
    rolling_min_periods : :class:`int`
        Minimum number of observations to be used in the rolling mean computation.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame with ``min(0,len(on))`` new columns, each one named using the
        previous column names and the provided ``suffix``.
    """
    return impute_missing_values(
        dataset=dataset,
        on=on,
        strategy=MEAN_ROLLING_STRATEGY,
        suffix=suffix,
        use_population_values=use_population_values,
        force_impute=force_impute,
        rolling_min_periods=rolling_min_periods,
        rolling_window=rolling_window,
        id_col=id_col,
        date_col=date_col,
    )


def impute_missing_values_with_rolling_median(
    dataset: pd.DataFrame,
    on: Union[str, list, None],
    id_col: str,
    date_col: str,
    rolling_window: int,
    rolling_min_periods: int,
    suffix: Union[str, None] = None,
    use_population_values: bool = False,
    force_impute: bool = True,
) -> pd.DataFrame:
    """Perform imputation using rolling median values.

    This function is just a wrapper for the main
    imputation function :py:func:`.impute_missing_values`, in
    which by default a rolling window with median computation
    is used for the imputation of missing values.

    Parameters
    ----------
    dataset : :class:`pandas.DataFrame`
        Dataset in which imputation must be performed.
    on : :class:`str` or :class:`list` or ``None``
        Name of column(s) on which imputation must be performed.
    suffix : :class:`str` or ``None``, optional
        Suffix to be used for the columns with imputed values, by default ``None``.
    id_col: :class:`str`
        Name of column to use as group identifier.
    date_col: :class:`str`
        Name of column to use as time identifier in the rolling mean operation.
    suffix : :class:`str` or ``None``, optional
        Suffix to be used for the columns with imputed values, by default ``None``.
    use_population_values : :class:`bool`, optional
        Use population values when forcing imputation, instead of
        group specific values, by default ``False``. Its value
        is useless if ``force_impute`` is set to ``False``.
    force_impute : :class:`bool`, optional
        Whether to force impute values in case there are still missing values
        after imputing them, by default ``True``.
        If set to ``False``, then missing values will still be present in the days
        in which the rolling mean could not be computed.
    rolling_window : :class:`int` or :class:`pandas.Timedelta`
        Size of the window used in the rolling mean computation, it can be
        of any type compatible with the **window** parameter of
        :py:func:`pandas.core.groupby.DataFrameGroupBy.rolling`
    rolling_min_periods : :class:`int`
        Minimum number of observations to be used in the rolling mean computation.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame with ``min(0,len(on))`` new columns, each one named using the
        previous column names and the provided ``suffix``.
    """
    return impute_missing_values(
        dataset=dataset,
        on=on,
        strategy=MEDIAN_ROLLING_STRATEGY,
        suffix=suffix,
        use_population_values=use_population_values,
        force_impute=force_impute,
        rolling_min_periods=rolling_min_periods,
        rolling_window=rolling_window,
        id_col=id_col,
        date_col=date_col,
    )


def impute_missing_values(
    dataset: pd.DataFrame,
    on: Union[str, list, None],
    strategy: Literal[
        f"{MEAN_STRATEGY}",
        f"{MEDIAN_STRATEGY}",
        f"{MEAN_ROLLING_STRATEGY}",
        f"{MEDIAN_ROLLING_STRATEGY}",
    ],
    id_col: Union[str, None] = None,
    date_col: Union[str, None] = None,
    suffix: Union[str, None] = None,
    use_population_values: bool = False,
    force_impute: bool = True,
    rolling_window: Union[int, None] = None,
    rolling_min_periods: Union[int, None] = None,
) -> pd.DataFrame:
    """Impute missing values with a defined strategy.

    This function imputes the missing values among the
    columns of the dataset defined by ``feature_list``
    using the strategy defined by ``strategy``.
    By default, the imputation is done on a voucher-by-voucher basis.
    This behavior can be overriden by setting ``use_population_values``
    to ``True``, and in this case the imputation is done using
    the overall mean or median values for each given feature in
    the dataset.
    The ``use_population_values`` is applicable only for strategy ``"mean"`` and
    for strategy ``"median"``.
    The columns with imputed values are name based on the provided ``suffix``
    parameter. By default, its value corresponds to the name of the strategy,
    to which an underscore is prepended. So,
    for a feature named ``"steps"`` and a strategy ``"mean"``, the new column
    will be named ``"steps_mean"``. This behavior can be overriden by providing
    a custom suffix.

    .. warning ::
        This function creates new columns in the provided ``dataset``, and the
        same :class:`pandas.DataFrame` is returned. Be careful that, if the
        value of the ``suffix`` parameter is set to ``""``, then the column
        is replaced with the new values after imputation.

    Parameters
    ----------
    dataset : :class:`pandas.DataFrame`
        Dataset in which imputation must be performed.
    on : :class:`str` or :class:`list` or ``None``
        Name of column(s) on which imputation must be performed.
    strategy : :class:`str`
        Strategy to be used when performing imputation.
        Available options for the ``strategy`` parameter are:

            * "mean": perform imputation with the mean value of each feature
            * "median": perform imputation with the mean value of each feature
            * "rolling_mean": perform imputation with the mean value of each feature
            * "rolling_median": perform imputation with the mean value of each feature
    suffix : :class:`str` or ``None``, optional
        Suffix to be used for the columns with imputed values, by default ``None``.
    use_population_values : :class:`bool`, optional
        Use population values when performing imputation, instead of
        voucher specific values, by default ``False``.
    force_impute : :class:`bool`, optional
        Whether to force impute values in case there are still missing values
        after imputing them with either a "rolling_mean" or "rolling_median", by default ``True``.
        If set to ``False``, then missing values will still be present in the days
        in which the rolling mean or median could not be computed.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame with ``min(0,len(on))`` new columns, each one named using the
        previous column names and the provided ``suffix``.

    Raises
    ------
    ValueError
        If ``suffix`` is not provided and the strategy is not among the available ones.
    """
    # Check input
    if type(on) is str:
        on = [on]
    elif on is None:
        on = dataset.columns

    if suffix is None:
        # Check suffix
        if strategy in STRATEGIES_DICT.keys():
            suffix = "_" + strategy
        else:
            raise ValueError(
                f"suffix must be provided if strategy is not among {list(STRATEGIES_DICT.keys())}"
            )

    if rolling_window is None and STRATEGIES_DICT[strategy]["rolling"]:
        raise ValueError(
            f"rolling_window must be provided if strategy is of type 'rolling'"
        )

    if rolling_min_periods is None and STRATEGIES_DICT[strategy]["rolling"]:
        raise ValueError(
            f"rolling_min_periods must be provided if strategy is of type 'rolling'"
        )

    if STRATEGIES_DICT[strategy]["rolling"]:
        # We need to make sure the dataset is sorted
        dataset = dataset.sort_values(by=[id_col, date_col])
        # We also need to resample it based on the time_freq provided

    for col in on:
        # Perform imputation based on strategy
        if strategy in STRATEGIES_DICT.keys():
            if not STRATEGIES_DICT[strategy]["rolling"]:
                if use_population_values:
                    dataset[f"{col}{suffix}"] = dataset[col].fillna(
                        dataset[col].apply(STRATEGIES_DICT[strategy]["transform"])
                    )
                else:
                    dataset[f"{col}{suffix}"] = dataset[col].fillna(
                        dataset.groupby(id_col)[col].transform(
                            STRATEGIES_DICT[strategy]["transform"]
                        )
                    )
            else:
                # We need to perform a rolling mean/median on previous days

                if force_impute and (not use_population_values):
                    # Let's save mean or median values before-hand, in case
                    # suffix is empty and the values are replace
                    force_imputed_values = dataset.groupby(id_col)[col].transform(
                        "mean" if "mean" in strategy else "median"
                    )

                # Set up rolling operation
                rol = (
                    dataset.loc[:, [id_col, date_col, col]]
                    .groupby(id_col)
                    .rolling(
                        min_periods=rolling_min_periods,
                        window=rolling_window,
                        on=date_col,
                    )[col]
                )

                # Fill nan based on strategy
                dataset[f"{col}{suffix}"] = (
                    dataset[col].fillna(rol.mean().reset_index(drop=True))
                    if "mean" in strategy
                    else dataset[col].fillna(rol.median().reset_index(drop=True))
                )
                if force_impute:
                    if use_population_values:
                        value = dataset[col].apply(
                            "mean" if "mean" in strategy else "median"
                        )
                        dataset[f"{col}{suffix}"] = dataset[f"{col}{suffix}"].fillna(
                            value
                        )
                    else:
                        dataset[f"{col}{suffix}"] = dataset[f"{col}{suffix}"].fillna(
                            force_imputed_values
                        )
    return dataset


def impute_missing_categorical_values() -> pd.DataFrame:
    pass
