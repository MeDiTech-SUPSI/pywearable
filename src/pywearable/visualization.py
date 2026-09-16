"""
This module contains all the functions related to the visualization of data
"""

import datetime
import warnings
from pathlib import Path
from typing import Callable, Union, Optional, List, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import calendar
from datetime import date

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.dates import DateFormatter, DayLocator, MonthLocator, WeekdayLocator
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.pyplot import Axes
from matplotlib.ticker import (
    FuncFormatter,
    MultipleLocator,
    PercentFormatter,
    ScalarFormatter,
)

import pywearable.activity as activity
import pywearable.cardiac as cardiac
import pywearable.constants
import pywearable.respiration as respiration
import pywearable.sleep as sleep
import pywearable.stress as stress
import pywearable.utils as utils

from .loader.base import BaseLoader

date_form = DateFormatter("%m-%d")
cmaps_list = plt.colormaps()
cmaps_dict = dict(zip(cmaps_list, cmaps_list))


def _define_custom_colormap(
    base_colormap: Union[str, ListedColormap, LinearSegmentedColormap],
    missing_data_index: int = 0,
    missing_data_color: tuple = (0.8, 0.8, 0.8, 1.0),
    cmap_name: str = "custom_colormap",
):
    """Defines a custom colormap

    Parameters
    ----------
    base_colormap : :class:`str`, :class:`ListedColormap`, :class:`LinearSegmentedColormap`
        Colormap object from which to start the customization
    missing_data_index : int, optional
        The index within the colormap to associate with the missing_data_color, by default 0
    missing_data_color : tuple, optional
        The color for missing data, if not specified it's light grey, by default (0.8, 0.8, 0.8, 1.0)
    cmap_name : str, optional
        The name of the customized colormap, by default 'custom_colormap'

    Returns
    -------
    :class:`ListedColorMap`
        the customized colormap
    """

    if type(base_colormap) == str:
        cmap = plt.get_cmap(base_colormap)

    # ListedColormap with the same colors but extended to include a specific color for missing data
    cmap_list = [cmap(i) for i in range(cmap.N)]
    cmap_list[missing_data_index] = missing_data_color
    return ListedColormap(cmap_list, name=cmap_name)


def get_steps_line_graph_and_stats(
    loader: BaseLoader,
    user_id: str,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    verbose: bool = False,
    save_to: Union[str, None] = None,
    show: bool = True,
    steps_line_label: str = "steps",
    goal_line_label: str = "daily goal",
    ylabel: str = "Steps",
    title: Union[str, None] = None,
    figsize: tuple = (10, 6),
    fontsize: int = 15,
    locator_interval: int = 2,
    plot_guideline_zones: bool = False,
    zones_labels: list = [
        "Sedentary",
        "Low active",
        "Somewhat active",
        "Active",
        "Highly active",
    ],
    zones_colors: list = ["tomato", "orange", "yellow", "aquamarine", "forestgreen"],
    zones_alpha: float = 0.25,
) -> dict:
    """Generate line-plot of daily steps and goals.

    The function creates (and possibly saves) a graph of the activity of a selected ``user_id``
    for a time period between ``start_date`` and ``end_date``, showing his/her number of steps
    every day, his/her goal for that day and whether the goal was reached.
    Furthermore activity statistics are computed and returned.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader
    user_id : :class:`str`
        The id of the user of interest
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None.
    verbose : :class:`bool`, optional
        Whether to print out activity statistics, by default False
    save_to : :class:`str` or None, optional
        Path where to save the plot, by default None
    show : :class:`bool`, optional
        Whether to show the plot, by default True
    steps_line_label : :class:`str`, optional
        Label of the steps line plot, by default "steps"
    goal_line_label : :class:`str`, optional
        Label of the dashed goals line plot, by default "daily goal"
    ylabel : :class:`str`, optional
        Label of the y-axis, by default "Steps"
    plot_title : :class:`str` or None, optional
        Title of the plot, by default "Daily steps"
    figsize : :class:`tuple`, optional
        Size of the figure, by default (10,6)
    fontsize : :class:`int`, optional
        Font size of the graph, by default 15
    locator_interval : :class:`int`, optional
        Interval for the x-axis dates, by default 2
    plot_guideline_zones : :class:`bool`, optional
        Whether to plot shaded regions corresponding to general guidelines for daily steps as reported in [1], by default False
    zones_labels : :class:`str`, optional
        labels of the guideline ranges, by default ["Sedentary", "Low active", "Somewhat active", "Active", "Highly active"]
    zones_colors : :class:`list`, optional
        Colors of the guideline ranges, by default ["tomato", "orange", "yellow", "aquamarine", "forestgreen"]
    zones_alpha : :class:`float`, optional
        Alphas for the guideline ranges, by default 0.25

    Returns
    -------
    :class:`dict`
        dictionary of daily activity statistics
        (Mean daily steps, Mean daily distance, Percentage goal completion)

    References
    ----------
    [1] Tudor-Locke, Catrine, Yoshiro Hatano, Robert Pangrazi, and Minsoo Kang. “Revisiting How Many Steps Are Enough?” Medicine and Science in Sports and Exercise 40 (July 1, 2008): S537-43. https://doi.org/10.1249/MSS.0b013e31817c7133.
    """

    # get dates,steps,goals,compare steps to goal to get goal completion
    dates, steps = zip(
        *activity.get_daily_steps(loader, user_id, start_date, end_date)[
            user_id
        ].items()
    )
    goals = list(
        activity.get_daily_steps_goal(loader, user_id, start_date, end_date)[
            user_id
        ].values()
    )
    col = np.where(np.array(steps) > np.array(goals), "g", "r")
    # get stats from the series
    mean_steps = activity.get_daily_steps(
        loader, user_id, start_date, end_date, average=True
    )[user_id]
    mean_distance = activity.get_daily_distance(
        loader, user_id, start_date, end_date, average=True
    )[user_id]
    goal_reached = np.sum(np.array(steps) > np.array(goals))
    number_of_days = len(dates)
    percentage_goal = round(goal_reached / number_of_days * 100, 1)
    stats_dict = {
        "Mean daily steps": mean_steps,
        "Mean daily distance": mean_distance,
        "goal_reached": goal_reached,
        "number_of_days": number_of_days,
        "Percentage goal completion": f"{goal_reached}/{number_of_days} {percentage_goal}%",
    }

    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(dates, steps, label=steps_line_label, c="k")
        ax.plot(dates, goals, linestyle="--", c="g", label=goal_line_label)
        ax.scatter(dates, steps, c=col, s=100)

        ax.xaxis.set_major_formatter(date_form)
        locator = DayLocator(interval=locator_interval)
        ax.xaxis.set_major_locator(locator)

        plt.xticks(rotation=45, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.grid("on")

        plt.ylim([max(min(steps) - 500, 0), max(steps) + 2000])
        plt.xlim(
            [
                min(dates) - datetime.timedelta(hours=6),
                max(dates) + datetime.timedelta(hours=6),
            ]
        )

        plt.ylabel(ylabel, fontsize=fontsize)

        if title:
            plt.title(title, fontsize=fontsize + 2)

        if plot_guideline_zones:
            min_date = dates[0]
            max_date = dates[-1]
            # plot coloring by fill_between different y-ranges
            # highly active
            ax.fill_between(
                [min_date, max_date],
                [12500 for i in range(2)],
                [max(12500, max(steps) + 2000) for i in range(2)],
                color=zones_colors[4],
                alpha=zones_alpha,
                label=zones_labels[4],
            )
            # active
            ax.fill_between(
                [min_date, max_date],
                [10000 for i in range(2)],
                [12499 for i in range(2)],
                color=zones_colors[3],
                alpha=zones_alpha,
                label=zones_labels[3],
            )
            # somewhat active
            ax.fill_between(
                [min_date, max_date],
                [7500 for i in range(2)],
                [9999 for i in range(2)],
                color=zones_colors[2],
                alpha=zones_alpha,
                label=zones_labels[2],
            )
            # low active
            ax.fill_between(
                [min_date, max_date],
                [5000 for i in range(2)],
                [7499 for i in range(2)],
                color=zones_colors[1],
                alpha=zones_alpha,
                label=zones_labels[1],
            )
            # sedentary
            ax.fill_between(
                [min_date, max_date],
                [0 for i in range(2)],
                [4999 for i in range(2)],
                color=zones_colors[0],
                alpha=zones_alpha,
                label=zones_labels[0],
            )

        plt.legend(fontsize=fontsize - 1, loc="center left", bbox_to_anchor=(1, 0.5))

    if save_to:
        plt.savefig(save_to, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    # print out stats
    if verbose:
        print(f"Mean daily steps: {mean_steps}")
        print(f"Mean daily distance travelled: {mean_distance}")
        print(f"Goal reached: {goal_reached}/{number_of_days} ({percentage_goal}%)")

    return stats_dict


def get_cardiac_line_graph_and_stats(
    loader: BaseLoader,
    user_id: str,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    save_to: Union[str, None] = None,
    show: bool = True,
    resting_hr_label: str = "resting heart rate",
    maximum_hr_label: str = "maximum heart rate",
    ylabel: str = "Heart rate [beats/min]",
    title: Union[str, None] = None,
    figsize: tuple = (10, 6),
    fontsize: int = 15,
) -> dict:
    """Generate graph of cardiac activity

    This function generate a graph of cardiac data of `user_id`
    for a period of interest between `start_date` and `end_date`.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str`
        The id of the user of interest
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    save_to : :class:`str` or None, optional
        Path where to save the plot, by default None
    show : :class:`bool`, optional
        Whether to show the plot, by default True
    resting_hr_label : :class:`str`, optional
        Label for the resting HR line plot, by default "resting heart rate"
    maximum_hr_label : :class:`str`, optional
        Label for the maximum HR line plot, by default "maximum heart rate"
    ylabel : :class:`str`, optional
        Label for the y-axis, by default "Heart rate [beats/min]"
    title : :class:`str` or None, optional
        Title of the plot, by default None
    figsize : :class:`tuple`, optional
        Size of the figure, by default (10,6)
    fontsize : :class:`int`, optional
        Font size of the plot, by default 15
    """

    # get time series
    dates, rest_hr = zip(
        *cardiac.get_rest_heart_rate(
            loader=loader,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            return_df=False,
        )[user_id].items()
    )
    max_hr = list(
        cardiac.get_max_heart_rate(
            loader=loader,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            return_df=False,
        )[user_id].values()
    )

    # if there's no data at all raise error
    if np.isnan(rest_hr).all() or np.isnan(max_hr).all():
        raise ValueError(
            f"Cardiac data not available between {start_date} and {end_date}"
        )

    # plotting
    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=figsize)
        ax.xaxis.set_major_formatter(date_form)
        ax.plot(
            dates,
            rest_hr,
            label=resting_hr_label,
            c="k",
            linewidth=1.5,
            marker="o",
            markersize=4,
        )
        # ax.plot(dates, avg_hr, label="average heart rate", c="g",linewidth=1.5,marker="o",markersize=4)
        ax.plot(
            dates,
            max_hr,
            label=maximum_hr_label,
            c="r",
            linewidth=1.5,
            marker="o",
            markersize=4,
        )
        # ax.set_title(title,fontsize=18)
        ax.set_ylabel(ylabel, fontsize=fontsize + 1)
        plt.xticks(rotation=45, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(loc="upper right", fontsize=fontsize - 1)
        plt.grid("both")
        plt.ylim([min(30, min(rest_hr)), max(200, max(max_hr) + 30)])
        if title:
            plt.title(title, fontsize=fontsize + 2)
        if save_to:
            plt.savefig(save_to, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def get_rest_spo2_graph(
    loader: BaseLoader,
    user_id: str,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    save_to: Union[str, None] = None,
    show: bool = True,
    zones_labels: list = ["Normal", "Low", "Concerning", "Critical"],
    zones_colors: list = ["g", "yellow", "orange", "tomato"],
    zones_alpha: float = 0.25,
    title: str = r"Rest SpO$_2$",
    ylabel: str = r"SpO$_2$",
    figsize: tuple = (14, 6),
    fontsize: int = 18,
):
    """Generate spO2 night graph

    This function creates (and possibly save if `save_to` = True),
    a graph showing all night data of SpO2 for `user_id`,
    in the period of interest starting from `start_date` and ending at `end_date`

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str`
        _The id of the user
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None], optional
        End date for data retrieval, by default None
    save_to : :class:`str` or None, optional
        Path where to save the plot, by default None
    show : :class:`bool`, optional
        Whether to show the plot, by default True
    zones_labels : :class:`list`, optional
        labels of the risk ranges, by default ["Normal", "Low", "Concerning", "Critical"]
    zones_colors : :class:`list`, optional
        Colors of the risk ranges, by default ["g", "yellow", "orange", "tomato"]
    zones_alpha : :class:`float`, optional
        Alphas for the risk ranges, by default 0.25
    title : :class:`str`, optional
        Title of the plot, by default r"Rest SpO$"
    ylabel : :class:`str`, optional
        Label of the y-axis, by default r"SpO$"
    figsize : :class:`tuple`, optional
        Size of the figure, by default (14, 6)
    fontsize : :class:`int`, optional
        Font size for the plot, by default 18
    """

    timedelta = datetime.timedelta(
        hours=12
    )  # this assumes that at the last day a person wakes up before midday...
    spo2_df = loader.load_pulse_ox(user_id, start_date, end_date + timedelta)
    sleep_spo2_df = spo2_df[spo2_df.isSleeping == 1].loc[:, ["isoDate", "spo2"]]
    unique_dates = pd.to_datetime(sleep_spo2_df.isoDate.dt.date.unique())
    # in order to avoid plotting lines between nights, we need to plot separately each sleep occurrence
    # unfortunately this takes some time (~6-7s/month), to find appropriate night for every row, maybe try to improve? #TODO
    sleep_spo2_df["date"] = sleep_spo2_df.isoDate.apply(
        lambda x: utils.find_nearest_timestamp(x, unique_dates)
    )

    fig, ax = plt.subplots(figsize=figsize)
    # nights are plotted individually
    for date in unique_dates:
        relevant_data = sleep_spo2_df[sleep_spo2_df.date == date]
        ax.plot(relevant_data.isoDate, relevant_data.spo2, color="gray")

    # get extremes of x-axis
    min_date = sleep_spo2_df.isoDate.min() - timedelta
    max_date = sleep_spo2_df.isoDate.max() + timedelta

    # plot coloring by fill_between different y-ranges
    # normal zone
    ax.fill_between(
        [min_date, max_date],
        [90 for i in range(2)],
        [100 for i in range(2)],
        color=zones_colors[0],
        alpha=zones_alpha,
        label=zones_labels[0],
    )
    # low
    ax.fill_between(
        [min_date, max_date],
        [80 for i in range(2)],
        [90 for i in range(2)],
        color=zones_colors[1],
        alpha=zones_alpha,
        label=zones_labels[1],
    )
    # concerning
    ax.fill_between(
        [min_date, max_date],
        [70 for i in range(2)],
        [80 for i in range(2)],
        color=zones_colors[2],
        alpha=zones_alpha,
        label=zones_labels[2],
    )
    # critical
    ax.fill_between(
        [min_date, max_date],
        [0 for i in range(2)],
        [70 for i in range(2)],
        color=zones_colors[3],
        alpha=zones_alpha,
        label=zones_labels[3],
    )

    # graph params
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 2)

    ax.xaxis.grid(True, color="#CCCCCC")
    ax.xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=60, fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    plt.ylim([min(50, min(sleep_spo2_df.spo2)), 100])
    plt.legend(loc="best", fontsize=fontsize - 2)
    plt.xlim([min_date, max_date])
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def get_respiration_line_graph_and_stats(
    loader: BaseLoader,
    user_id: str,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    verbose: bool = False,
    save_to: Union[str, None] = None,
    show: bool = True,
    figsize: tuple = (10, 6),
    rest_line_label: str = "Sleep Avg",
    awake_line_label: str = "Awake Avg",
    title: Union[str, None] = None,
    xlabel: str = "Date",
    ylabel: str = "Breaths per minute",
    fontsize: int = 15,
) -> dict:
    """Generate a line-plot of daily and night average daily respiration rates

    Parameters
    ----------
    loader : :class:`pylabfront.loader.LabfrontLoader`
        An instance of a data loader
    user_id : :class:`str`
        The id of the user
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    verbose : :class:`bool`, optional
        Whether to print out respiration stats, by default False
    save_to : class`str` or None, optional
        Path where to save the plot, by default None
    show : :class:`bool`, optional
        Whether to show the plot, by default True
    figsize: :class:`tuple`, optional
        Size of the figure, by default (10,6)
    rest_line_label : :class:`str`, optional
        Label of the line relative to night respiration data, by default "Sleep Avg"
    awake_line_label : :class:`str`, optional
        Label of the line relative to daily respiration data, by default "Awake Avg"
    title : :class:`str` or None, optional
        Title of the plot, by default None
    xlabel : :class:`str`, optional
        Label of the x-axis, by default "Date"
    ylabel : :class:`str`, optional
        Label of the y-axis, by default "Breaths per minute"
    fontsize : :class:`int`, optional
        Fontsize for the plot, by default 15

    Returns
    -------
    :class:`dict`
        Dictionary reporting average breaths per minute during the day and the night
    """

    # get series, note that we're inclusive wrt the whole last day
    rest_dates, rest_resp = zip(
        *respiration.get_mean_rest_breaths_per_minute(
            loader, user_id, start_date, end_date, return_df=False, remove_zero=True
        )[user_id].items()
    )
    waking_dates, waking_resp = zip(
        *respiration.get_mean_awake_breaths_per_minute(
            loader,
            user_id,
            start_date,
            end_date,
            return_df=False,
            remove_zero=True,
        )[user_id].items()
    )
    combined_dates = sorted(
        list(set(rest_dates + waking_dates))
    )  # not always we have waking/rest data, but need consistent x labeling
    # get stats
    avg_sleeping_breaths = np.nanmean(rest_resp)
    avg_waking_breaths = np.nanmean(waking_resp)

    avg_sleeping_breaths = (
        float(round(avg_sleeping_breaths, 2))
        if not np.isnan(avg_sleeping_breaths)
        else None
    )
    avg_waking_breaths = (
        float(round(avg_waking_breaths, 2))
        if not np.isnan(avg_waking_breaths)
        else None
    )

    stats_dict = {
        "Average sleep breaths": avg_sleeping_breaths,
        "Average waking breaths": avg_waking_breaths,
    }

    # plotting
    dates_format = [date.strftime("%d-%m") for date in combined_dates]
    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(rest_dates, rest_resp, marker="o", label=rest_line_label)
        ax.plot(waking_dates, waking_resp, marker="o", label=awake_line_label)
        ax.legend(loc="best", fontsize=fontsize - 1)
        # ax.set_title(title,fontsize=15)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        plt.ylim(
            [min(8, min(rest_resp + waking_resp)), max(rest_resp + waking_resp) + 2.5]
        )
        plt.xticks(
            combined_dates[::2], dates_format[::2], rotation=45, fontsize=fontsize
        )
        plt.yticks(fontsize=fontsize)
        if save_to:
            plt.savefig(save_to, bbox_inches="tight")
        if title:
            plt.title(fontsize=fontsize + 2)
    if show:
        plt.show()
    else:
        plt.close()

    if verbose:
        print(f"Avg sleep: {avg_sleeping_breaths}")
        print(f"Avg awake: {avg_waking_breaths}")

    return stats_dict


class CalendarHeatmap:

    def __init__(
        self,
        dates: List[Union[str, datetime.date, datetime.datetime]],
        data: List[float],
        cmap: Union[str, LinearSegmentedColormap, ListedColormap] = "Greens",
        value_label: bool = False,
        weekday_label: bool = True,
        month_label: bool = True,
        year_label: bool = True,
        month_grid: bool = False,
        month_grid_color: str = "black",
        colorbar: bool = False,
        frame_on: bool = False,
        value_format: str = "int",
        title: Optional[str] = None,
        cmin: Optional[int] = None,
        cmax: Optional[int] = None,
        cbar_label_format: Optional[str] = None,
        figsize: tuple = (12, 5),
        fontsize: int = 14,
    ):
        self.dates = dates
        self.data = data
        self.cmap = cmap
        self.value_label = value_label
        self.weekday_label = weekday_label
        self.month_label = month_label
        self.year_label = year_label
        self.month_grid = month_grid
        self.month_grid_color = month_grid_color
        self.colorbar = colorbar
        self.frame_on = frame_on
        self.value_format = value_format
        self.title = title
        self.cmin = cmin
        self.cmax = cmax
        self.cbar_label_format = cbar_label_format
        self.figsize = figsize
        self.fontsize = fontsize

    def generate(self):
        cal = self._date_grid(self.dates, self.data)
        ax = self._cal_heatmap(
            cal=cal,
            dates=self.dates,
            cmap=self.cmap,
            value_label=self.value_label,
            weekday_label=self.weekday_label,
            month_label=self.month_label,
            year_label=self.year_label,
            month_grid=self.month_grid,
            month_grid_color=self.month_grid_color,
            colorbar=self.colorbar,
            frame_on=self.frame_on,
            value_format=self.value_format,
            title=self.title,
            cmin=self.cmin,
            cmax=self.cmax,
            cbar_label_format=self.cbar_label_format,
        )
        self.ax = ax

    def plot(self):
        plt.show()

    def save(self, path: str):
        self.ax.get_figure().savefig(path, bbox_inches="tight")

    def _date_grid(self, dates: List[date], data: List[Any], dtype: str = "float64"):
        iso_dates = np.array([day.isocalendar() for day in dates])
        unique_weeks = sorted(list(set([tuple(row) for row in iso_dates[:, :2]])))

        weeknum2idx = {week: i for i, week in enumerate(unique_weeks)}
        week_coords = np.array([weeknum2idx[tuple(week)] for week in iso_dates[:, :2]])
        day_coords = iso_dates[:, 2] - 1

        n_weeks = len(unique_weeks)
        n_days = 7

        grid = np.empty((n_weeks, n_days), dtype=dtype)
        grid = np.nan * grid if dtype == "float64" else grid
        grid[week_coords, day_coords] = data

        return grid.T

    def _cal_heatmap(
        self,
        cal: np.ndarray,
        dates: List[date],
        cmap: Union[str, LinearSegmentedColormap, ListedColormap] = "Greens",
        value_label: bool = False,
        weekday_label: bool = True,
        month_label: bool = True,
        year_label: bool = True,
        month_grid: bool = False,
        month_grid_color: str = "black",
        colorbar: bool = False,
        frame_on: bool = False,
        value_format: str = "int",
        title: Optional[str] = None,
        cmin: Optional[int] = None,
        cmax: Optional[int] = None,
        cbar_label_format: Optional[str] = None,
        ax: Optional[plt.axes] = None,
    ):
        if not ax:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=100)
        else:
            fig = ax.get_figure()

        if isinstance(cmap, str):
            cmap = cmaps_dict[cmap]

        pc = ax.pcolormesh(
            cal, edgecolors=ax.get_facecolor(), linewidth=0.25, cmap=cmap
        )
        pc.set_clim(cmin or np.nanmin(cal), cmax or np.nanmax(cal))
        ax.invert_yaxis()
        ax.set_aspect("equal")
        bbox = ax.get_position()

        if value_label:
            self._add_value_label(ax, cal, value_format)
        else:
            ax.set_xticklabels("")
        if weekday_label:
            self._add_weekday_label(ax)
        if month_label:
            self._add_month_label(ax, dates)
        if year_label:
            self._add_year_label(ax, dates)
        if month_grid:
            self._add_month_grid(ax, dates, cal, month_grid_color)
        if colorbar:
            self._add_colorbar(pc, fig, ax, bbox, cbar_label_format)
        if title:
            ax.set_title(title)

        ax.set_frame_on(frame_on)
        return ax

    def _get_month_outline(self, dates, month_grid, month):
        day_grid = self._date_grid(dates, dates, dtype="object").T
        month_grid = month_grid.T

        nrows, ncols = month_grid.shape
        coords_list = []

        for y in range(nrows):
            for x in range(ncols):
                if np.isfinite(month_grid[y, x]) and day_grid[y, x] is not None:
                    if day_grid[y, x].month == month:
                        coords_list.append((x, y))

        sorted_coords = np.array(coords_list)
        min_y = sorted_coords[:, 1].min()
        max_y = sorted_coords[:, 1].max()
        upper_left = sorted_coords[0]
        upper_right = np.array([7, min_y])
        lower_right = np.array([7, max_y])
        lower_right2 = sorted_coords[-1] + np.array([1, 1])

        lower_right1 = (
            lower_right2
            if np.array_equal(lower_right, lower_right2)
            else lower_right2 - np.array([0, 1])
        )
        lower_left = np.array([0, sorted_coords[:, 1].max() + 1])
        corner_last = upper_left + np.array([0, 1])
        second_last = np.copy(corner_last)
        second_last[0] = 0

        coords = np.array(
            [
                upper_left,
                upper_right,
                lower_right,
                lower_right1,
                lower_right2,
                lower_left,
                second_last,
                corner_last,
                upper_left,
            ]
        )

        return coords[:, [1, 0]]

    def _add_value_label(self, ax, cal, value_format):
        if value_format == "int":
            val_format = "{:0.0f}"
        elif value_format == "decimal":
            val_format = "{:0.1f}"
        else:
            raise ValueError(
                "Argument 'value_format' must be equal to either "
                f"'int' or 'float'. Got: {value_format}."
            )

        for (i, j), z in np.ndenumerate(cal):
            if np.isfinite(z):
                ax.text(
                    j + 0.5, i + 0.5, val_format.format(z), ha="center", va="center"
                )

    def _add_weekday_label(self, ax):
        ax.tick_params(axis="y", which="major", pad=8)
        ax.set_yticks([x + 0.5 for x in range(0, 7)])
        ax.set_yticklabels(
            calendar.weekheader(width=1).split(" "),
        )

    def _add_month_label(self, ax, dates):
        month_years = [(day.year, day.month) for day in dates]
        month_years_str = list(map(str, month_years))
        month_year_grid = self._date_grid(dates, month_years_str, dtype="object")
        unique_month_years = sorted(set(month_years))

        month_locs = {}
        for month in unique_month_years:
            yy, xx = np.nonzero(month_year_grid == str(month))
            month_locs[month] = (xx.max() + 1 + xx.min()) / 2

        month_labels = [calendar.month_abbr[x[1]] for x in month_locs.keys()]

        ax.set_xticks([*month_locs.values()])
        ax.set_xticklabels(month_labels, ha="center")

    def _add_year_label(self, ax, dates):
        years = [day.year for day in dates]
        year_grid = self._date_grid(dates, years)
        unique_years = sorted(set(years))

        year_locs = {}
        for year in unique_years:
            yy, xx = np.nonzero(year_grid == year)
            year_locs[year] = (xx.max() + 1 + xx.min()) / 2

        for year, loc in year_locs.items():
            ax.annotate(
                year,
                (loc / year_grid.shape[1], 1),
                (0, 12),
                xycoords="axes fraction",
                textcoords="offset points",
                fontsize="large",
                va="center",
                ha="center",
            )

    def _add_colorbar(self, pc, fig, ax, bbox, cbar_label_format):
        adj_bbox = ax.get_position()
        height_diff = adj_bbox.height - bbox.height
        cax = fig.add_axes(
            [
                bbox.x1 + 0.015,
                adj_bbox.y0 + height_diff / 2,
                0.015,
                bbox.height,
            ]
        )
        cbar_label_format = cbar_label_format or ScalarFormatter()
        plt.colorbar(pc, cax=cax, format=cbar_label_format)

    def _add_month_grid(self, ax, dates, month_grid, color):
        months = set([d.month for d in dates])
        for month in months:
            coords = self._get_month_outline(dates, month_grid, month=month)
            ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1)

        ax.set_xlim(ax.get_xlim()[0] - 0.1, ax.get_xlim()[1] + 0.1)
        ax.set_ylim(ax.get_ylim()[0] + 0.1, ax.get_ylim()[1] - 0.1)

        fig = ax.get_figure()

        for pos in ["top", "bottom", "right", "left"]:
            ax.spines[pos].set_edgecolor(fig.get_facecolor())
            return ax


def get_metric_heatmap(
    loader: BaseLoader,
    user_id: str,
    metric_fn: Callable,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    period_length: Union[int, None] = None,
    save_to: Union[str, None] = None,
    show: bool = True,
    kwargs_heatmap: dict = {},
):
    """
    Generate a github-like grid plot of a metric, user, and period of interest

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str`
        The id of the user for which the heatmap must be generated.
    metric_fn : :class:`callable`
        A pywearable function to calculate the metric of interest for which the heatmap must be generated.
        The function is expected to return a dictionary for the user, such that the primary key
        is the user's id, the secondary keys are the dates and the values are the metric observations.
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    period_length : :class:`int` or None, optional
        Duration of the period to be shown (in days) in the heatmap.
        If not specified the period coincides with the dates where data is available, by default None
    save_to : :class:`str` or None, optional
        Path where to save the grid plot, by default None
    show : :class:`str`, optional
        Whether to show the plot, by default True
    kwargs_heatmap : :class:`dict`, optional
        Additional keyword arguments for the july's heatmap function, by default {}
    """

    assert type(period_length) == int or period_length is None

    dates, scores = zip(
        *metric_fn(loader, user_id, start_date, end_date)[user_id].items()
    )

    # Get start and end days for the heatmap, based on data and desired backward extension
    start_date = (
        dates[0]
        if period_length is None
        else dates[-1] - datetime.timedelta(days=period_length - 1)
    )
    end_date = dates[-1]
    time_delta_intervals = [
        start_date + i * datetime.timedelta(days=1)
        for i in range((end_date - start_date).days + 1)
    ]

    data_df = pd.DataFrame({"date": dates, "metric": scores})

    extended_df = pd.DataFrame({"date": time_delta_intervals})

    extended_df = pd.merge(left=extended_df, right=data_df, on="date", how="left")
    extended_df.loc[extended_df.metric.isna(), "metric"] = 0

    heatmap = CalendarHeatmap(
        dates=extended_df.date,
        data=extended_df.metric,
        **kwargs_heatmap,
    )
    heatmap.generate()

    if show:
        heatmap.plot()

    if save_to:
        heatmap.save(save_to)


def get_sleep_performance_heatmap(
    loader: BaseLoader,
    user_id: str,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    save_to: Union[str, None] = None,
    show: bool = True,
    period_length: int = 365,
    cmin: Union[int, None] = None,
    cmax: Union[int, None] = None,
    title: Union[str, None] = "Sleep Performance",
):
    """Generate a github-like heatmap plot of daily sleep scores

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader
    user_id : :class:`str`
        The id of the user for which the sleep performance heatmap must be generated.
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    save_to : :class:`str` or None, optional
        Path where to save the plot, by default None
    show : :class:`bool`, optional
        Whether to show the plot, by default True
    period_length : :class:`int`, optional
        Period length (in days) to show, a year is shown is shown if not chosen otherwise, by default 365
    cmin : :class:`int` or None, optional
        Minimum value for the colorbar. If not specified the minimum value of the data is used, by default None
    cmax : :class:`int` or None, optional
        Maximum value for the colorbar. If not specified the maximum value of the data is used, by default None
    """

    custom_cmap = _define_custom_colormap(
        base_colormap="BuGn",
        missing_data_index=0,
        missing_data_color=(0.9, 0.9, 0.9, 1.0),
    )

    get_metric_heatmap(
        loader=loader,
        user_id=user_id,
        metric_fn=sleep.get_sleep_score,
        start_date=start_date,
        end_date=end_date,
        period_length=period_length,
        save_to=save_to,
        show=show,
        kwargs_heatmap={
            "cmap": custom_cmap,
            "title": title,
            "fontsize": 14,
            "frame_on": True,
            "month_grid": True,
            "colorbar": True,
            "year_label": period_length >= 365,
            "cmin": cmin,
            "cmax": cmax,
        },
    )


def get_stress_heatmap(
    loader: BaseLoader,
    user_id: str,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    save_to: Union[str, None] = None,
    show: bool = True,
    period_length: int = 365,
    cmin: Union[int, None] = None,
    cmax: Union[int, None] = None,
    title: Union[str, None] = "Stress scores",
):
    """Generate a github-like heatmap plot of daily stress scores

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader
    user_id : :class:`str`
        The id of the user for which the stress scores heatmap must be generated.
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    save_to : :class:`str` or None, optional
        Path where to save the plot, by default None
    show : :class:`bool`, optional
        Whether to show the plot, by default True
    period_length : :class:`int`, optional
        Period length (in days) to show, a year is shown is shown if not chosen otherwise, by default 365
    cmin : :class:`int` or None, optional
        Minimum value for the colorbar. If not specified the minimum value of the data is used, by default None
    cmax : :class:`int` or None, optional
        Maximum value for the colorbar. If not specified the maximum value of the data is used, by default None
    """

    custom_cmap = _define_custom_colormap(
        base_colormap="YlOrBr",
        missing_data_index=0,
        missing_data_color=(0.9, 0.9, 0.9, 1.0),
    )

    get_metric_heatmap(
        loader=loader,
        user_id=user_id,
        metric_fn=stress.get_daily_stress_score,
        start_date=start_date,
        end_date=end_date,
        period_length=period_length,
        save_to=save_to,
        show=show,
        kwargs_heatmap={
            "cmap": custom_cmap,
            "title": title,
            "fontsize": 14,
            "frame_on": True,
            "month_grid": True,
            "colorbar": True,
            "year_label": period_length >= 365,
            "cmin": cmin,
            "cmax": cmax,
        },
    )


def get_recovery_heatmap(
    loader: BaseLoader,
    user_id: str,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    save_to: Union[str, None] = None,
    show: bool = True,
    period_length: int = 365,
    cmin: Union[int, None] = None,
    cmax: Union[int, None] = None,
    title: Union[str, None] = "Recovery",
):
    """Generate a github-like heatmap plot of night recovery

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader
    user_id : :class:`str`
        The id of the user for which the night recovery heatmap must be generated.
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    save_to : :class:`str` or None, optional
        Path where to save the plot, by default None
    show : :class:`bool`, optional
        Whether to show the plot, by default True
    period_length : :class:`int`, optional
        Period length (in days) to show, a year is shown is shown if not chosen otherwise, by default 365
    cmin : :class:`int` or None, optional
        Minimum value for the colorbar. If not specified the minimum value of the data is used, by default None
    cmax : :class:`int` or None, optional
        Maximum value for the colorbar. If not specified the maximum value of the data is used, by default None
    """

    custom_cmap = _define_custom_colormap(
        base_colormap="RdYlGn",
        missing_data_index=0,
        missing_data_color=(0.9, 0.9, 0.9, 1.0),
    )

    get_metric_heatmap(
        loader=loader,
        user_id=user_id,
        metric_fn=stress.get_recovery_percentage,
        start_date=start_date,
        end_date=end_date,
        period_length=period_length,
        save_to=save_to,
        show=show,
        kwargs_heatmap={
            "cmap": custom_cmap,
            "title": title,
            "fontsize": 14,
            "frame_on": True,
            "month_grid": True,
            "colorbar": True,
            "year_label": period_length >= 365,
            "cmin": cmin,
            "cmax": cmax,
        },
    )


def get_sleep_summary_graph(
    loader: BaseLoader,
    user_id: str,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    save_to: Union[str, None] = None,
    show: bool = True,
    alpha: float = 0.25,
    title: str = "Sleep stages and score",
    xlabel: str = "Sleep time [hours]",
    ylabel: str = "Date",
    legend_title: str = "Sleep stages",
    legend_labels: str = ["Deep", "Light", "REM", "Awake"],
    colorbar_title: str = "Sleep Score",
    colorbar_labels: str = ["poor", "fair", "good", "excellent"],
    figsize: tuple = (15, 30),
    bottom_offset: int = 500,
    vertical_offset: float = -0.0,
    show_chronotype: bool = False,
    chronotype_sleep_start: Union[str, None] = None,
    chronotype_sleep_end: Union[str, None] = None,
    consistency_metric: Union[str, None] = None,
    legend_fontsize: int = 14,
    axis_fontsize: int = 14,
    title_fontsize: int = 20,
    score_fontsize: int = 15,
    kwargs_trend_analysis: dict = {},
):
    """
    Generates a graph of all hypnograms of main sleeps of `user_id` for the period of interest

    Parameters
    ----------
    loader : :class:`pylabfront.loader.LabfrontLoader`
        An instance of a data loader
    user_id : :class:`str`
        The id of the user
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    save_to : :class:`str`, optional
        Path where to save the plot, by default None
    show : class:`bool`, optional
        Whether to show the plot, by default True
    alpha : :class`float`, optional
        Alpha for elements in transparency in the plot, by default 0.25
    title : :class:`str`, optional
        Title of the plot, by default "Sleep stages and score"
    xlabel : :class:`str`, optional
        label of the x-axis, by default "Sleep time [hours]"
    ylabel : :class:`str`, optional
        label of the y-axis, by default "Date"
    legend_title : :class:`str`, optional
        title of the sleep stages legend, by default "Sleep stages"
    legend_labels : :class:`str`, optional
        labels for the sleep stages legend, by default ["Deep", "Light", "REM", "Awake"]
    colorbar_title : :class:`str`, optional
        title of the colorbar, by default "Sleep Score"
    colorbar_labels : :class:`str`, optional
        labels of the colorbar, by default ["poor", "fair", "good", "excellent"]
    figsize : :class:`tuple`, optional
        size of the figure, by default (15,30)
    bottom_offset : :class:`int`, optional
        distance of the scores from the bottom of the hypnograms, by default 500
    vertical_offset : :class:`float`, optional
        vertical offset of the scores at the bottom of the hypnograms, by default -0.
    show_chronotype : :class:`bool`, optional
        whether to show chronotype dashed vertical lines over the hypnograms, by default False
    chronotype_sleep_start : :class:`str` or None, optional
        usual sleeping time for `user_id` in format HH:MM, by default None
    chronotype_sleep_end : :class:`str` or None, optional
        usual waking time for `user_id` in format HH:MM, by default None
    consistency_metric : :class:`str` or None, optional
        metric used for circadian variability ("midpoint" or "duration"), by default None
    legend_fontsize: :class:`int`, optional
        fontsize of the sleep stages legend, by default 14
    axis_fontsize: :class:`int`, optional
        fontsize of the axis ticks, by default 14
    title_fontsize: :class:`int`, optional
        fontsize of the title, by default 20
    score_fontsize: :class:`int`, optional
        fontsize of the score on the hypnograms, by default
    kwargs_trend_analysis : :class:`dict`, optional
        kwargs for the trend analysis of consistency metrics, by default {}
    """

    # Define parameters for plotting
    ALPHA = alpha
    POSITION = 1.3

    # Get sleep summaries so that it is easier to get info
    sleep_summaries = loader.load_sleep_summary(user_id, start_date, end_date)
    # no data
    if len(sleep_summaries) == 0:
        raise ValueError(f"No sleep data available between {start_date} and {end_date}")
    sleep_min_time = datetime.time(15, 0)
    # Check for start and end dates and convert them appropriately
    start_date = utils.check_date(start_date)
    end_date = utils.check_date(end_date)

    # check chronotype and infer it from data if missing
    if consistency_metric is not None or show_chronotype is not None:
        if chronotype_sleep_start is None or chronotype_sleep_end is None:
            chronotype_sleep_start = utils.mean_time(
                sleep.get_bedtime(
                    loader=loader,
                    user_id=user_id,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
            chronotype_sleep_end = utils.mean_time(
                sleep.get_wakeup_time(
                    loader=loader,
                    user_id=user_id,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        else:
            assertion_msg = "Must specify chronotype in format 'HH:MM' when plotting circadian measures"
            assert (
                type(chronotype_sleep_start) == str
                and type(chronotype_sleep_end) == str
            ), assertion_msg

    sleep_summaries["isoDate-Min"] = pd.to_datetime(
        sleep_summaries["calendarDate"].apply(
            lambda x: datetime.datetime.combine(
                x - datetime.timedelta(days=1), sleep_min_time
            )
        )
    )

    sleep_summaries["endIsoDate"] = pd.to_datetime(
        (
            sleep_summaries[pywearable.constants.UNIXTIMESTAMP_IN_MS_COL]
            + sleep_summaries[pywearable.constants.TIMEZONEOFFSET_IN_MS_COL]
            + sleep_summaries[pywearable.constants.SLEEP_SUMMARY_DURATION_IN_MS_COL]
            + sleep_summaries[
                pywearable.constants.SLEEP_SUMMARY_AWAKE_DURATION_IN_MS_COL
            ]
        ),
        unit="ms",
        utc=True,
    ).dt.tz_localize(None)

    sleep_summaries["secondsDiff"] = (
        sleep_summaries["isoDate"] - sleep_summaries["isoDate-Min"]
    ).dt.total_seconds()

    sleep_summaries["endSecondsDiff"] = (
        sleep_summaries["endIsoDate"] - sleep_summaries["isoDate-Min"]
    ).dt.total_seconds()

    time_period = pd.date_range(
        start_date,
        periods=(end_date - start_date).days,
        freq="D",
    )

    time_period = pd.Series(time_period).dt.date

    # define color-coding based on garmin connect visuals
    color_dict = {
        1: "royalblue",
        3: "darkblue",
        4: "darkmagenta",
        0: "hotpink",
        -1: "gray",
    }

    # get relevant scores
    scores_series = sleep.get_sleep_score(
        loader=loader, start_date=start_date, end_date=end_date, user_id=user_id
    )[user_id]

    # setup an internal fn to get appropriate sleep score colors:
    def get_color(score):
        # if it's null/nan hide it out, this happens for manual input by the user
        if score == "" or np.isnan(score):
            return "white"
        else:
            if score >= 90:
                return "forestgreen"
            elif score >= 80:
                return "limegreen"
            elif score >= 60:
                return "darkorange"
            else:
                return "firebrick"

    hypnograms = loader.load_hypnogram(user_id, start_date, end_date, resolution=1)
    if consistency_metric is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif consistency_metric == "midpoint" or consistency_metric == "duration":
        fig, (ax, ax2) = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": [8, 2]}, figsize=figsize
        )
    elif consistency_metric == "both":
        fig, (ax, ax2, ax3) = plt.subplots(
            1, 3, gridspec_kw={"width_ratios": [8, 1, 1]}, figsize=figsize
        )
    else:
        raise KeyError("sleep metric specified for variability isn't valid.")

    # for every day in the period of interest, we plot the hypnogram
    for j, night in enumerate(time_period):
        try:  # this takes care of days where data is missing, skipping days through the plot y-axis
            # get the hypnogram
            hypnogram = hypnograms[night]
            # create dataframe for hypnogram
            hypnogram = pd.DataFrame(
                {
                    "stage": hypnogram["values"],
                    "isoDate": [
                        hypnogram["start_time"] + i * datetime.timedelta(minutes=1)
                        for i in range(
                            int(
                                (
                                    hypnogram["end_time"] - hypnogram["start_time"]
                                ).total_seconds()
                                / 60
                            )
                        )
                    ],
                }
            )
            # determine when there're stage variations
            hypnogram["stages_diff"] = (
                np.concatenate(
                    [
                        [0],
                        hypnogram.iloc[1:, :]["stage"].values
                        - hypnogram.iloc[:-1, :]["stage"].values,
                    ]
                )
                != 0
            )
            # get rows relative to initial/final timestamp of sleep or to variations in stage
            relevant_rows = hypnogram.loc[
                np.logical_or(
                    hypnogram.index.isin([0, len(hypnogram) - 1]), hypnogram.stages_diff
                ),
                ["isoDate", "stage"],
            ]

            # create a row that indicate the duration of permanence in the stages
            # (in hours, they will be the height of the sections of the stacked barplot)
            stages_duration = []
            for i in range(len(relevant_rows) - 1):
                stages_duration.append(
                    (
                        pd.to_datetime(relevant_rows.iloc[i + 1].isoDate)
                        - pd.to_datetime(relevant_rows.iloc[i].isoDate)
                    ).seconds
                )
            stages_duration.append(0)  # the last row is not starting a new stage
            relevant_rows["stage duration"] = stages_duration
            # get array of the heights for the sections of the stacked barplot
            stage_duration_array = relevant_rows["stage duration"].values
            colors = [color_dict[stage] for stage in relevant_rows.stage.values]

            # keep track on the position where to begin the bar section
            sleep_summary_row = sleep_summaries[
                (sleep_summaries["calendarDate"] == night)
            ]
            seconds_diff = sleep_summary_row["secondsDiff"]

            bottom = seconds_diff

            # look through the stages
            for i in range(len(stage_duration_array)):
                # only Deep and REM sleep get full alpha, light, awake, and unmeasurable are in transparence
                appropriate_alpha = (
                    ALPHA
                    if (
                        colors[i] == "royalblue"
                        or colors[i] == "hotpink"
                        or colors[i] == "gray"
                    )
                    else 1
                )
                ax.barh(
                    j * POSITION,
                    stage_duration_array[i],
                    left=bottom,
                    color=colors[i],
                    alpha=appropriate_alpha,
                )
                # Update the bottom coordinates for the next section
                bottom += stage_duration_array[i]

            # annotate on top of the daily bar with the daily sleep score, color-coded appropriately
            score = scores_series.get(pd.Timestamp(night))
            appropriate_color = get_color(score)
            # not sure why but with horizontal bars the annotation has to be manually adjusted in position
            ax.annotate(
                str(score),
                xy=(bottom + bottom_offset, j * POSITION + vertical_offset),
                color=appropriate_color,
                fontsize=score_fontsize,
            )
        except Exception as e:  # skip missing dates
            continue

    # Set limits to be an hour lower than lowest difference
    # and one hour more than longest and latest sleep
    ax.set_xlim(
        [
            sleep_summaries["secondsDiff"].min() - 3600,
            sleep_summaries["endSecondsDiff"].max() + 3600,
        ]
    )

    # graph and legend params
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False, left=False)

    def format_func(x, pos):
        hours = sleep_min_time.hour + int(x // 3600)
        if hours >= 24:
            hours -= 24
        minutes = int((x % 3600) // 60)

        return "{:02d}".format(hours)

    formatter = FuncFormatter(format_func)

    ax.xaxis.set_major_formatter(formatter)
    # this locates x-ticks at the hours
    ax.xaxis.set_major_locator(MultipleLocator(base=3600))
    ax.tick_params(axis="x", which="major", labelsize=axis_fontsize)

    # graph params
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#EEEEEE")
    ax.yaxis.grid(False)
    ax.set_ylabel(ylabel, labelpad=15, color="#333333", fontsize=axis_fontsize + 2)
    ax.set_xlabel(xlabel, labelpad=15, color="#333333", fontsize=axis_fontsize + 2)
    ax.set_yticks([i * POSITION for i in range(len(time_period))])
    ax.set_yticklabels(labels=[date.strftime("%d/%m") for date in time_period])
    ax.tick_params(labelrotation=0, labelsize=axis_fontsize)
    ax.set_title(title, pad=25, color="#333333", weight="bold", fontsize=title_fontsize)
    # ordinarly the yaxis starts from below, but it's better to visualize earlier dates on top instead
    ax.set_ylim([-POSITION, (len(time_period)) * POSITION])
    ax.invert_yaxis()

    ## Legend
    alphas = [1, ALPHA, 1, ALPHA]
    colors = ["darkblue", "royalblue", "darkmagenta", "hotpink"]
    lgd = fig.legend(
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.97, 0.45),
        fontsize=legend_fontsize,
    )
    # take care of opacity of of the colors selected
    for i, lh in enumerate(lgd.legend_handles):
        lh.set_color(colors[i])
        lh.set_alpha(alphas[i])
    lgd.get_frame().set_alpha(0.0)
    lgd.set_title(legend_title, prop={"size": legend_fontsize + 1})

    # habitual sleep times lines
    if show_chronotype:
        sleep_time_hour = int(chronotype_sleep_start.split(":")[0])
        sleep_time_mins = int(chronotype_sleep_start.split(":")[1])
        wake_time_hour = int(chronotype_sleep_end.split(":")[0])
        wake_time_mins = int(chronotype_sleep_end.split(":")[1])
        # this conversion takes into consideration the min_sleep_time = 15
        # and the change of day (assuming wake is before 12)
        converted_sleep_time = (
            sleep_time_hour + 24 * (0 <= sleep_time_hour <= 12) - 15
        ) * 60 * 60 + sleep_time_mins * 60
        converted_wake_time = (
            wake_time_hour + 24 * (0 <= wake_time_hour <= 12) - 15
        ) * 60 * 60 + wake_time_mins * 60
        ax.axvline(converted_sleep_time, linestyle="--", zorder=10)
        ax.axvline(converted_wake_time, linestyle="--", zorder=11)

    # CONSISTENCY SUBPLOTS
    if consistency_metric is not None:
        # determine which metrics are needed in the subplot(s)
        if consistency_metric == "duration":
            consistency_fns = [sleep.get_cpd_duration]
            axes = [ax2]
            axes_titles = [consistency_metric]
        elif consistency_metric == "midpoint":
            consistency_fns = [sleep.get_cpd_midpoint]
            axes = [ax2]
            axes_titles = [consistency_metric]
        elif consistency_metric == "both":
            consistency_fns = [sleep.get_cpd_midpoint, sleep.get_cpd_duration]
            axes = [ax2, ax3]
            axes_titles = ["midpoint", "duration"]
        else:
            raise ValueError(
                f"Warning: consistency sleep metric {consistency_metric} isn't valid."
            )
        # and populate a subplot which each one
        for k in range(len(consistency_fns)):
            current_ax = axes[k]
            consistency_fn = consistency_fns[k]
            # we get data for cpd antecedent to the period of interest so that we may have a NR already set in some cases
            cpd_dict = consistency_fn(
                loader,
                user_id,
                start_date - datetime.timedelta(days=30),
                end_date,
                chronotype_dict={
                    user_id: (chronotype_sleep_start, chronotype_sleep_end)
                },
            ).xs(user_id, level="user_id")
            cpd_trend = utils.trend_analysis(
                {
                    pd.Timestamp(calendar_date).date(): value
                    for calendar_date, value in cpd_dict.items()
                },
                **kwargs_trend_analysis,
            )
            # filter out to keep appropriate period
            cpd_trend = cpd_trend[cpd_trend.index.isin(time_period)]

            dates = cpd_trend.index
            # need to fillna to avoid skipping plotting some days at the start and end of the period
            metric = cpd_trend.metric.fillna(0)
            baseline = cpd_trend.BASELINE
            LB = cpd_trend.NR_LOWER_BOUND
            LB[LB < 0] = 0  # can't have a negative lower bound
            UB = cpd_trend.NR_UPPER_BOUND

            current_ax.barh(dates, metric, color="gray", alpha=0.7)
            current_ax.plot(baseline, dates, linestyle="-", linewidth=1, color="red")
            current_ax.fill_betweenx(dates, LB, UB, alpha=0.25, color="green")
            current_ax.grid("on")
            current_ax.set_title(f"CPD ({axes_titles[k]})", fontsize=10)
            current_ax.set_yticks([])

            # subplot params
            current_ax.set_axisbelow(True)
            current_ax.xaxis.grid(True, color="#EEEEEE")
            current_ax.yaxis.grid(False)
            current_ax.spines["top"].set_visible(False)
            current_ax.spines["right"].set_visible(False)
            current_ax.spines["left"].set_color("#DDDDDD")
            # as before invert to keep earlier dates in the upper part of the plot
            current_ax.set_ylim(
                [
                    time_period.iloc[0] - datetime.timedelta(days=1),
                    time_period.iloc[-1] + datetime.timedelta(days=1),
                ]
            )
            current_ax.invert_yaxis()

    # COLORBAR
    bins = [40, 60, 80, 90, 100]
    midpoints = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    colors = ["firebrick", "darkorange", "limegreen", "forestgreen", "forestgreen"]
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(bins, cmap.N)
    # Create an additional axis for the colorbar
    cax = fig.add_axes([0.935, 0.55, 0.03, 0.2])
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=midpoints, pad=0.10
    )
    cbar.ax.set_yticklabels(colorbar_labels, fontsize=legend_fontsize)
    plt.annotate(
        colorbar_title,
        xy=(1.2, 1.1),
        xycoords="axes fraction",
        ha="center",
        fontsize=legend_fontsize + 1,
    )

    if save_to:
        plt.savefig(save_to, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def get_errorbar_graph(
    quest_df: pd.DataFrame,
    questionnaire_dict: dict,
    variable_of_interest: str,
    answer_of_interest: str,
    title: str = "",
    ylabel: str = "",
    answer_categories: Union[list, None] = None,
    save_to: Union[str, None] = None,
    show: bool = True,
):
    """Plots an errorbar graph with respect to the ``variable_of_interest`` for a specific question in a questionnaire

    For every category of answer given

    Parameters
    ----------
    quest_df : :class:`pandas.DataFrame`
        DataFrame obtained from the pylabfront.questionnaire.process_questionnaire function for the questionnaire of interest,
        filtered by user of interest, and possibly processed in some other way to obtain additional column(s) of interest,
        used as ``variable_of_interest``.
    questionnaire_dict : :class:`dict`
        Dictionary obtained from the pylabfront.loader.LabfrontLoader.get_questionnare_questions method for the questionnaire of interest
    variable_of_interest : :class:`str`
        name of the column of ``quest_df`` against which to calculate stats
    answer_of_interest : :class:`str`
        number of the answer in the questionaire in the format reported by Labfront
    title : :class:`str`, optional
        title of the plot. Defaults to empty string.
    ylabel : :class:`str`, optional
        label for the y-axis in the plot. Defaults to empty string.
    answer_categories : :class:`list`, optional
        list of the answers, in the order desired for visaulization, by default None
        if this is left as default, the order is defined by the order proposed in the Labfront questionnaire
    """
    # if a specific order isn't explicitely given, get the order of the answers in the questionnaire
    if answer_categories is None:
        answer_categories = questionnaire_dict[answer_of_interest]["options"]
    # get the description and the string of the full answer id, useful to get the appropriate columns in the dataframe
    answer_descr = questionnaire_dict[answer_of_interest]["description"]
    answer_string = answer_of_interest + "-" + answer_descr

    # get only the relevant columns, drop the rows where there's incomplete data
    tmp_df = quest_df.loc[:, [answer_string, variable_of_interest]].copy().dropna()
    # Get mean and std wrt the variable of interest for every answer level (if std is None (single obs), put it to 0)
    bar_df = (
        tmp_df.loc[:, [answer_string, variable_of_interest]]
        .dropna()
        .groupby(answer_string)
        .agg(Mean=(variable_of_interest, np.mean), Std=(variable_of_interest, np.std))
        .fillna(0)
    )
    # a bit of a hack, but we need the indexes in the right order for appropriate plotting
    bar_df = bar_df.reindex(answer_categories).dropna()
    # plot bars with height equal to the mean and error reference of std
    ax = bar_df.plot(kind="bar", y="Mean", legend=False, alpha=0.25, figsize=(12, 6))
    ax.errorbar(
        bar_df.index,
        bar_df["Mean"],
        yerr=bar_df["Std"],
        linewidth=1.5,
        color="black",
        alpha=0.75,
        capsize=10,
        marker="o",
    )
    # graph params
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EEEEEE")
    ax.xaxis.grid(False)
    ax.set_ylabel(ylabel, labelpad=15, color="#333333", fontsize=15)
    ax.set_xlabel("")
    ax.set_title(title, pad=25, color="#333333", weight="bold", fontsize=20)
    # to avoid overlapping of xlabels, we add newlines every jumprow words
    # decide how often to insert a newline depending on the number of bars
    jumprow = 3 if len(bar_df) < 5 else 2
    labels = bar_df.index.unique().values
    new_labels = []
    for label in labels:
        label = label.split()
        new_label = []
        for i in range(len(label)):
            new_label.append(label[i])
            if (i + 1) % jumprow == 0:
                new_label.append("\n")
        new_labels.append(" ".join(new_label))
    plt.xticks(labels, new_labels, rotation=0, fontsize=15)
    plt.yticks(fontsize=15)

    if save_to:
        plt.savefig(save_to, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def compare_against_group(
    user_data: Union[int, float],
    comparison_data: list,
    show: bool = True,
    save_to: Union[str, None] = None,
    bins: int = 10,
    title: str = "",
    ylabel: str = "% users",
    xlabel: str = "",
    fontsize: int = 16,
    shaded_regions: bool = False,
    regions_cutoffs: Union[list, None] = None,
    regions_colors: Union[list, None] = None,
    alpha: float = 0.25,
    xlim: Union[list, None] = None,
):
    """Plots a histogram of the distribution of a desired metric, specifying where an user stands within the distribution

    This functions compare an user against a chosen population with respect to a specific metric.
    The metric should be evaluated for both the user and the comparison group prior to the calling of the function.

    Parameters
    ----------
    user_data : :class:`float` or :class:`int`
        The value of the user of interest for the metric of interest
    comparison_data : :class:`list`
        List of values for the metric of interest for the comparison group (including the user of interest)
    show : :class:`bool`, optional
        Whether to show the plot, by default True
    save_to : :class:`str` or None, optional
        Path where to save the plot, by default None
    bins : :class:`int`, optional
        Number of bins for the comparison histogram, by default 10
    title : :class:`str`, optional
        Title of the plot, by default ""
    ylabel : :class:`str`, optional
        Label of the y-axis, by default "% users"
    xlabel : :class:`str`, optional
        Label of the x-axis, by default ""
    fontsize : :class:`int`, optional
        Font size for the plot, by default 16
    shaded_regions : :class:`bool`, optional
        Whether to enable in the plot the use of colored regions, by default False
    regions_cutoffs : :class:`list`, optional
        Values of the cuttoff points for the shaded regions, by default None
    regions_colors : :class:`list`, optional
        Colors of the shaded regions, len(regions_colors) is expected to be len(regions_cutoffs)-1, by default None
    alpha : :class:`float`, optional
        Alpha of the shaded regions, by default 0.25
    xlim : :class:`list`, optional
        List of the leftmost and rightmost x-values for the plot, by dafault None

    Returns
    -------
    :class:`float`
        Percentile standing of the user among the comparison group considered
    """

    # note that the following is not strict percentile
    # this is good to say you were above x% of the others..
    # should we instead show the strict percentile??
    percentile_standing = np.round(
        np.sum(np.array(comparison_data) <= user_data) / len(comparison_data) * 100, 0
    )

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="w")
    cnts, values, bars = ax.hist(
        comparison_data,
        bins=bins,
        rwidth=0.95,
        weights=np.ones(len(comparison_data)) / len(comparison_data),
        zorder=2,
    )

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    if shaded_regions:
        for i in range(len(regions_colors)):
            start_point = regions_cutoffs[i]
            end_point = regions_cutoffs[i + 1]
            ax.axvspan(start_point, end_point, alpha=alpha, color=regions_colors[i])

    for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
        if i == len(cnts) or values[i] <= user_data <= values[i + 1]:
            bar.set_facecolor("darkorange")
            break
    ax.set_title(title, fontsize=fontsize + 2)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    if xlim:
        ax.set_xlim(xlim)
    if save_to:
        plt.savefig(save_to, bbox_inches="tight")
    if show:
        plt.show()

    return percentile_standing


def plot_trend_analysis(
    loader: BaseLoader,
    user_id: str,
    metric_fn: Callable,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    ax: Union[None, plt.axes] = None,
    ma_type: str = "MA",
    baseline_periods: int = 7,
    min_periods_baseline: int = 3,
    normal_range_periods: int = 30,
    min_periods_normal_range: int = 20,
    std_multiplier: float = 1.0,
    save_to: Union[str, None] = None,
    show: bool = True,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    fontsize: int = 12,
    xticks_frequency: int = 3,
    xticks_rotation: int = 45,
    ylim: Union[tuple, None] = None,
    figsize: tuple = (10, 6),
    show_legend: bool = False,
    fixed_normal_range: tuple = None,
    metric_fn_kwargs: dict = {},
    barplot_kwargs: dict = {},
    baseline_kwargs: dict = {},
    normal_range_kwargs: dict = {},
):
    """Plots a trend analysis graph

    This function plots a trend analysis graph for a given metric, a given user, and a period of interest,
    showing the short term variations (daily values), mid-term variations (the baseline),
    and long term variations (the normal range).

    Parameters
    ----------
    loader : :class:`pylabfront.loader.BaseLoader`
        Loader object to load the data
    user_id : :class:`str`
        ID of the user for which to plot the trend analysis
    metric_fn : :class:`Callable`
        Pywearable function to calculate the metric of interest
    start_date : :class:`datetime.datetime`, :class:`datetime.date`, :class:`str`, optional
        Start date of the period of interest, by default None
    end_date : :class:`datetime.datetime`, :class:`datetime.date`, :class:`str`, optional
        End date of the period of interest, by default None
    ax : None or class:`matplotlib.axes`, optional
        Axes where to build the figure, by default None
    ma_type : :class:`str`, optional
        Type of moving average to use, can be either "MA" or "EMA", by default "MA"
    baseline_periods : :class:`int`, optional
        Number of periods to consider for the baseline, by default 7
    min_periods_baseline : :class:`int`, optional
        Minimum number of periods to consider for the baseline, by default 3
    normal_range_periods : :class:`int`, optional
        Number of periods to consider for the normal range, by default 30
    min_periods_normal_range : :class:`int`, optional
        Minimum number of periods to consider for the normal range, by default 20
    std_multiplier : :class:`float`, optional
        Multiplier for the standard deviation to consider for the normal range, by default 1.0
    save_to : :class:`str`, optional
        Path where to save the plot, by default None
    show : :class:`bool`, optional
        Whether to show the plot, by default True
    xlabel : :class:`str`, optional
        Label of the x-axis, by default ""
    ylabel : :class:`str`, optional
        Label of the y-axis, by default ""
    title : :class:`str`, optional
        Title of the plot, by default ""
    fontsize : :class:`int`, optional
        Font size for the plot, by default 12
    xticks_frequency : :class:`int`, optional
        Frequency of visualization of x-axis ticks, by default 3
    xticks_rotation : :class:`int`, optional
        Rotation of ticks on the x-axis, by default 45
    ylim : :class:`tuple`, optional
        Limits of the y-axis, by default None
    figsize : :class:`tuple`, optional
        Size of the figure, by default (10,6)
    show_legend : :class:`bool`, optional
        Whether to show the legend of the plot, by default False
    fixed_normal_range : :class:`tuple`, optional
        Start and end of a fixed range (based on norm values) instead of a trend NR, by default None
    metric_fn_kwargs : :class:`dict`, optional
        Additional kwargs for the metric function, by default {}
    barplot_kwargs : :class:`dict`, optional
        Additional kwargs for the barplot, by default {}
    baseline_kwargs : :class:`dict`, optional
        Additional kwargs for the baseline plot, by default {}
    normal_range_kwargs : :class:`dict`, optional
        Additional kwargs for the normal range plot, by default {}

    Returns
    -------
    :class:`matplotlib.axes._subplots.AxesSubplot`
        Axes object of the plot
    """
    data_dict = metric_fn(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        **metric_fn_kwargs,
    )[user_id]

    df = utils.trend_analysis(
        data_dict=data_dict,
        ma_type=ma_type,
        baseline_periods=baseline_periods,
        min_periods_baseline=min_periods_baseline,
        normal_range_periods=normal_range_periods,
        min_periods_normal_range=min_periods_normal_range,
        std_multiplier=std_multiplier,
    )

    # default kwargs for normal range in case they're not provided
    if "alpha" not in normal_range_kwargs:
        normal_range_kwargs["alpha"] = 0.5
    if "color" not in normal_range_kwargs:
        normal_range_kwargs["color"] = "green"

    # default kwargs for baseline in case they're not provided
    if "color" not in baseline_kwargs:
        baseline_kwargs["color"] = "red"
    if "linestyle" not in baseline_kwargs:
        baseline_kwargs["linestyle"] = "-"
    if "linewidth" not in baseline_kwargs:
        baseline_kwargs["linewidth"] = 2

    # restrict viz to period with available data
    df = df.dropna(how="all")
    dates = df.index
    metric = df.metric
    baseline = df.BASELINE
    LB = df.NR_LOWER_BOUND
    UB = df.NR_UPPER_BOUND
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.bar(dates, metric, label="Daily metric", **barplot_kwargs)
    ax.plot(baseline, label="Baseline", **baseline_kwargs)
    if fixed_normal_range is not None:
        assert type(fixed_normal_range) == tuple and len(fixed_normal_range) == 2
        LB = fixed_normal_range[0]
        UB = fixed_normal_range[1]
    ax.fill_between(
        dates, LB, UB, label="Normal range", **normal_range_kwargs, zorder=-1
    )
    ax.grid("on")
    ax.set_xticks(dates[::xticks_frequency])
    ax.tick_params(axis="x", labelrotation=xticks_rotation)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 2)

    if ylim:
        ax.set_ylim(ylim)
    if show_legend:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, bbox_inches="tight")
    if show:
        plt.show()
    return ax


class RadarAxes(PolarAxes):
    name = "radar"

    def __init__(self, *args, **kwargs):
        super(RadarAxes, self).__init__(*args, **kwargs)
        self.set_theta_zero_location("N")
        self.theta = []

    def plot(self, *args, **kwargs):
        lines = super(RadarAxes, self).plot(*args, **kwargs)
        for line in lines:
            self._close_line(line)

    def _close_line(self, line):
        x, y = line.get_data()
        if x[0] != x[-1]:
            x = np.concatenate((x, [x[0]]))
            y = np.concatenate((y, [y[0]]))
            line.set_data(x, y)

    def set_varlabels(self, labels):
        self.set_thetagrids(np.degrees(self.theta), labels)

    def _gen_axes_patch(self):
        return plt.Circle((0.5, 0.5), 0.5)

    def _gen_axes_spines(self):
        return PolarAxes._gen_axes_spines(self)


def radar_chart(
    loader: BaseLoader,
    user_id: str,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    comparison_start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    comparison_end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: Union[str, None] = "mean",
    metrics: Union[list, None] = None,
    save_to: Union[str, None] = None,
    show: bool = True,
    title: Union[str, None] = None,
    reference_label: str = "Reference",
    comparison_label: str = "Comparison",
    legend: bool = True,
    colors: list = ["lightskyblue", "salmon"],
    kwargs_metric_fns={},
):
    """Plots a radar chart of the desired pywearable metrics.

    Plots a radar chart comparing the values between two periods for the desired pywearable metrics.
    The radar chart is not limited by the number of the metrics to be included in the chart;
    it dynamically adjusts itself to the number of compared metrics.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str`
        The id(s) for which the sleep statistic must be computed.
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    comparison_start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval of the comparison period, by default None
    comparison_end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval of the comparison period, by default None
    kind : :class:`str`, optional
        Summary statistic to calculate for the metrics to be displayed, by default "mean"
    metrics : :class:`list` or None, optional
        List of pywearable function for metrics calculation. The list must contain more than 1 metric, by default None
    save_to : :class:`str` or None, optional
        Path where to save the grid plot, by default None
    show : :class:`str`, optional
        Whether to show the plot, by default True
    title : :class:`str`, optional
        Title of the plot, by default None
    reference_label : :class:`str`, optional
        Plot label of the reference data, by default "Reference"
    comparison_label : :class:`str`, optional
        Plot label of the comparison data, by default "Comparison"
    legend: :class:`bool`, optional
        If true, add a legend with the computed results of the comparison to the plot, by default True
    colors: :class:`list`, optional
        Color for the radar chart areas, by default ['lightskyblue', 'salmon']
    kwargs_metric_fns : :class:`dict`, optional
        Keyword arguments for the computation of the pywearable functions, by default {}
    Raises
    -------
    TypeError
        If an error occurred during the computation of the metrics.
    TypeError
        if no selection of pywearable function metrics is provided.
    ValueError
        if less than 2 pywearable metric functions are provided.
    """

    unknown_metrics, ref_metrics, comp_metrics = [], {}, {}

    # check that the list of metrics is appropriate for radar chart plotting
    if metrics is None:
        raise TypeError(
            "No input list of metrics provided for 'metrics'. Please specify a list of the metric functions"
            "to be computed and compared."
        )
    elif len(metrics) <= 2:
        raise ValueError(
            "Not enough metrics selected for a radar chart. Please specify at least 3 pywearable metric functions."
        )

    # Register projection of custom RadarAxes class
    register_projection(RadarAxes)

    # Compute available metrics
    for metric_fn in metrics:
        try:
            ref_result = metric_fn(
                loader=loader,
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                kind=kind,
                return_df=True,
                **kwargs_metric_fns,
            )
            ref_metrics[ref_result.loc[user_id].index.values[0]] = ref_result.loc[
                user_id
            ].values[0]
            comp_result = metric_fn(
                loader=loader,
                user_id=user_id,
                start_date=comparison_start_date,
                end_date=comparison_end_date,
                kind=kind,
                return_df=True,
                **kwargs_metric_fns,
            )
            comp_metrics[comp_result.loc[user_id].index.values[0]] = comp_result.loc[
                user_id
            ].values[0]
        except:
            unknown_metrics.append(metric_fn)

    # Raise warning pointing out unknown parameters
    if unknown_metrics != []:
        warnings.warn(
            "Unknown parameters '%s' will not be computed." % unknown_metrics,
            stacklevel=2,
        )

    # Prepare plot
    if legend:
        fig, (ax_l, ax) = plt.subplots(
            1, 2, figsize=(12, 6), subplot_kw=dict(projection="radar")
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={"projection": "radar"})
    theta = np.linspace(0, 2 * np.pi, len(ref_metrics.keys()), endpoint=False)
    ax.theta = theta

    # Prepare plot data
    ax.set_varlabels([metric for metric in ref_metrics.keys()])
    max_vals = {
        metric: max(ref_metrics[metric], comp_metrics[metric])
        for metric in ref_metrics.keys()
    }
    ref_vals = [
        (ref_metrics[metric] / max_vals[metric]) * 100 for metric in ref_metrics.keys()
    ]
    com_vals = [
        (comp_metrics[metric] / max_vals[metric]) * 100 for metric in ref_metrics.keys()
    ]

    # Plot data
    for i, vals in enumerate([ref_vals, com_vals]):
        ax.plot(theta, vals, color=colors[i])
        ax.fill(theta, vals, color=colors[i], alpha=0.3)
    title = "Radar Chart\nReference (%s) vs. Comparison (%s)\n" % (colors[0], colors[1])

    # Add legend to second empty plot
    if legend:
        ax_l.set_title(title, horizontalalignment="center")
        legend = []

        # Helper function
        def _add_legend(label, fc="white"):
            return legend.append(mpl.patches.Patch(fc=fc, label="\n" + label))

        # Add list of computed parameters
        _add_legend(reference_label, colors[0])
        for metric in ref_metrics.keys():
            _add_legend("%s:" % metric)
        # Add list of comparison parameters
        _add_legend(comparison_label, colors[1])
        for metric in ref_metrics.keys():
            _add_legend("%.2f vs. %.2f" % (ref_metrics[metric], comp_metrics[metric]))
        # Add relative differences
        _add_legend("")
        for i, _ in enumerate(ref_metrics.keys()):
            rel_diff = (com_vals[i] - ref_vals[i]) / (ref_vals[i]) * 100
            _add_legend(
                "+%.2f%s" % (rel_diff, r"$\%$")
                if rel_diff > 0
                else "%.2f%s" % (rel_diff, r"$\%$")
            )
        ax_l.legend(handles=legend, ncol=3, frameon=False, loc=7)
        ax_l.axis("off")
    else:
        ax.set_title(title, horizontalalignment="center")
    if save_to:
        plt.savefig(save_to, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def period_to_period_comparison_chart(
    loader: BaseLoader,
    user_id: str,
    metric_fn: Callable,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    kind: str = "mean",
    period_interval: str = "month",
    show: bool = True,
    save_to: Union[str, None] = None,
    figsize: tuple = (12, 8),
    xlabel: Union[str, None] = None,
    ylabel: Union[str, None] = None,
    locator_interval: int = 1,
    title: str = "",
    fontsize: int = 14,
    metric_fn_kwargs: dict = {},
):
    """Generates a period-to-period comparison chart for a user and a metric of interest.

    This function is used to visualize period to period comparison of a metric. It can be used
    to show weekly variations or monthly variations over extended periods.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str`
        The id(s) for which the sleep statistic must be computed.
    metric_fn : :class:`callable`
        A pywearable function that computes the metric of interest.
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    kind : :class:`str`, optional
        Summary statistic to calculate for the metrics to be displayed, by default "mean"
    period_interval : :class:`str`, optional
        Interval of the period (month or week) to be considered for the comparison, by default "month"
    show : :class:`bool`, optional
        Whether to show the plot, by default True
    save_to : :class:`str` or None, optional
        Path where to save the grid plot, by default None
    figsize : :class:`tuple`, optional
        Size of the figure, by default (12, 8)
    xlabel : :class:`str` or None, optional
        Label of the x-axis, by default None
    ylabel : :class:`str` or None, optional
        Label of the y-axis, by default None
    locator_interval : :class:`int`, optional
        Interval of the locator, by default 1
    title : :class:`str`, optional
        Title of the plot, by default ""
    fontsize : :class:`int`, optional
        Font size for the plot, by default 14
    metric_fn_kwargs : :class:`dict`, optional
        Keyword arguments for the computation of the pywearable functions, by default {}
    """
    if period_interval not in ["month", "week"]:
        raise ValueError(f"period interval {period_interval} is not a valid interval.")

    fig, ax = plt.subplots(figsize=figsize)
    if period_interval == "month":
        month_loc = MonthLocator(interval=locator_interval)
        ax.xaxis.set_major_locator(month_loc)
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        date_aggr_fn = lambda x: x.month
    elif period_interval == "week":
        week_loc = DayLocator(interval=locator_interval)
        ax.xaxis.set_major_locator(week_loc)
        ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
        date_aggr_fn = lambda x: x.week

    metric_data = metric_fn(loader, user_id, start_date, end_date, **metric_fn_kwargs)[
        user_id
    ]
    metric_dates, metric_values = zip(*metric_data.items())

    # compute metric aggregate transformation (kind) during each period and its changes
    metric_df = pd.DataFrame({"Date": metric_dates, "metric": metric_values})
    metric_df["Date"] = pd.to_datetime(metric_df["Date"])
    metric_df["aggregate_over"] = metric_df["Date"].apply(date_aggr_fn)

    changes = pd.DataFrame(metric_df.groupby("aggregate_over")["metric"].apply(kind))
    abs_change = changes.diff()
    pct_change = changes.pct_change() * 100
    changes["abs_change"] = abs_change
    changes["pct_change"] = pct_change

    metric_df["aggr_metric"] = metric_df.groupby("aggregate_over")["metric"].transform(
        kind
    )

    ax.plot(
        metric_dates,
        metric_values,
        color="#25BCCA",
        alpha=0.2,
    )

    for period in metric_df["aggregate_over"].unique():
        relative_change = changes[changes.index == period]["pct_change"].values[0]
        ax.plot(
            np.array(metric_df[metric_df["aggregate_over"] == period]["Date"]),
            np.array(metric_df[metric_df["aggregate_over"] == period]["aggr_metric"]),
            (
                "#06B478"
                if relative_change >= 0
                else ("tab:orange" if not np.isnan(relative_change) else "#25BCCA")
            ),
            alpha=0.8,
        )
        ax.text(
            metric_df[metric_df["aggregate_over"] == period]["Date"].mean(),
            metric_df[metric_df["aggregate_over"] == period]["aggr_metric"].unique(),
            metric_df[metric_df["aggregate_over"] == period]["aggr_metric"]
            .unique()
            .round(1)[0],
            ha="center",
            va="bottom",
        )
        if not np.isnan(relative_change):
            # the following offset is an heuristic which seems to work well always, possibly to improve
            vertical_offset = (max(metric_values) - min(metric_values)) / 20
            ax.text(
                metric_df[metric_df["aggregate_over"] == period]["Date"].mean(),
                metric_df[metric_df["aggregate_over"] == period]["aggr_metric"].unique()
                - vertical_offset,
                f"{'+' if relative_change >= 0 else ''}{round(relative_change,1)}%",
                ha="center",
                va="bottom",
                color="#06B478" if relative_change >= 0 else "tab:orange",
            )

    ax.set_xlim([start_date, end_date])
    for label in ax.get_xticklabels(which="major"):
        label.set(rotation=30, horizontalalignment="right")
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 2)

    if save_to:
        plt.savefig(save_to, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def weekday_barplot(
    loader: BaseLoader,
    user_id: str,
    metric_fn: Callable,
    start_date: Union[datetime.datetime, datetime.date, str, None] = None,
    end_date: Union[datetime.datetime, datetime.date, str, None] = None,
    aggr_fn: Union[str, callable] = "mean",
    figsize: tuple = (12, 8),
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    save_to: Union[str, None] = None,
    show: bool = True,
    horizontal: bool = False,
    fontsize: int = 12,
    alpha: float = 0.7,
    color: Union[str, list] = "blue",
    errorbar: bool = False,
    error_range: str = "std",
    metric_lim: Union[list, None] = None,
    metric_fn_kwargs: dict = {},
    bar_kwargs: dict = {},
):
    """Generates a barplot showing how a metric changes over the week.

    For a metric of interest, aggregate statistics, determined by the `kind` parameter,
    are showed as averages throughout the days of the week.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str`
        The id for which the weekday barplot must be generated.
    metric_fn : :class:`Callable`
        A pywearable function that computes the metric of interest.
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        Start date for data retrieval, by default None
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str` or None, optional
        End date for data retrieval, by default None
    aggr_fn : :class:`str` or class:`Callable`, optional
        Summary statistic to calculate for the metrics to be displayed, by default "mean"
    figsize : :class:`tuple`, optional
        Size of the figure, by default (12, 8)
    title : :class:`str`, optional
        Title of the plot, by default ""
    xlabel : :class:`str`, optional
        Label of the x-axis, by default ""
    ylabel : :class:`str`, optional
        Label of the y-axis, by default ""
    save_to : :class:`str` or None, optional
        Path where to save the grid plot, by default None
    show : :class:`str`, optional
        Whether to show the plot, by default True
    horizontal : :class:`bool`, optional
        Whether to show the barplot horizontally, by default False
    fontsize : :class:`int`, optional
        Font size for the plot, by default 12
    alpha : :class:`float`, optional
        Alpha of the bars, by default 0.7
    color : :class:`str` or :class:`list`, optional
        Color(s) of the bars, by default "blue"
    errorbar : :class:`bool`, optional
        Whether to show error bars, by default False
    error_range : :class:`str`, optional
        Range of the error bars, can be either "std", "IQR" or "minmax", by default "std"
    metric_lim : :class:`list`, optional
        Limits of the metric in the plot, by default None
    metric_fn_kwargs : :class:`dict`, optional
        Keyword arguments for the computation of the pywearable functions, by default {}
    bar_kwargs : :class:`dict`, optional
        Additional keyword arguments for the bar plot, by default {}

    Raises
    -------
    ValueError
        If the error range is not a valid range.
    """
    metric_data = metric_fn(
        loader=loader,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        **metric_fn_kwargs,
    )[user_id]

    if isinstance(aggr_fn, str) and aggr_fn.lower() == "mean":
        aggr_fn = np.nanmean
    elif isinstance(aggr_fn, str) and aggr_fn.lower() == "median":
        aggr_fn = np.nanmedian

    if errorbar:
        if isinstance(error_range, str) and error_range.lower() == "std":
            error_fn = np.nanstd
        # note that the following assumes that aggr_fn computes a central statistic of the metric
        elif isinstance(error_range, str) and error_range.lower() == "iqr":
            error_fn = lambda x: (
                abs(aggr_fn(x) - np.nanpercentile(x, 25)),
                abs(aggr_fn(x) - np.nanpercentile(x, 75)),
            )
        elif isinstance(error_range, str) and error_range.lower() == "minmax":
            error_fn = lambda x: (
                abs(aggr_fn(x) - np.nanmin(x)),
                abs(aggr_fn(x) - np.nanmax(x)),
            )
        else:
            raise ValueError(f"Error range {error_range} is not a valid range.")

    weekday_metrics = [[] for _ in range(7)]

    for date, value in metric_data.items():
        weekday = date.weekday()
        weekday_metrics[weekday].append(value)

    mean_weekday_metrics = [aggr_fn(weekday) for weekday in weekday_metrics]
    if errorbar:
        error_weekday_metrics = np.array(
            [error_fn(weekday) for weekday in weekday_metrics]
        ).T

    weekdays_labels = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)

    if not horizontal:
        ax.bar(
            weekdays_labels,
            mean_weekday_metrics,
            color=color,
            alpha=alpha,
            yerr=error_weekday_metrics if errorbar else None,
            **bar_kwargs,
        )
        if metric_lim:
            ax.set_ylim(metric_lim)
    else:
        ax.barh(
            weekdays_labels,
            mean_weekday_metrics,
            color=color,
            alpha=alpha,
            xerr=error_weekday_metrics if errorbar else None,
            **bar_kwargs,
        )
        if metric_lim:
            ax.set_xlim(metric_lim)
        ax.invert_yaxis()

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 2)

    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, bbox_inches="tight")

    if show:
        plt.show()


def summary_longitudinal_chart(
    loader: BaseLoader,
    user_id: str,
    start_date: Union[datetime.datetime, datetime.date, str],
    end_date: Union[datetime.datetime, datetime.date, str],
    start_hour: int = 0,
    figsize: Union[tuple, None] = None,
    hypnogram_fn: Union[Callable, None] = None,
    cardiac_fn: Union[Callable, None] = None,
    activity_fn: Union[Callable, None] = None,
    show_missing: bool = True,
    resolution_sleep: Union[float, int] = 1.0,
    resolution_cardiac: Union[float, int] = 1.0,
    resolution_activity: Union[float, int] = 15.0,
    fontsize: int = 10,
    show: bool = True,
    save_to: str = None,
    title: str = "",
    show_dates: bool = False,
):
    """
    Generates a longitudinal chart for a user and period of interest

    The idea of this function is to show for a period of interest cardiac data,
    in terms of heart rate, activity data, in terms of steps, and sleep data, in
    terms of hypnograms. The data is showing on daily subplots, starting from the chosen
    start_time.

    Parameters
    ----------
    loader : :class:`pywearable.loader.base.BaseLoader`
        An instance of a data loader.
    user_id : :class:`str`
        The id for which the longitudinal chart must be generated.
    start_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str`
        Start date for data retrieval.
    end_date : :class:`datetime.datetime` or :class:`datetime.date` or :class:`str`
        End date for data retrieval.
    start_hour : :class:`int`, optional
        Start hour for the daily subplots, by default 0
    figsize : :class:`tuple`, optional
        Size of the figure, by default None
    hypnogram_fn : :class:`Callable`, optional
        Function to load the hypnogram data, by default None
    cardiac_fn : :class:`Callable`, optional
        Function to load the cardiac data, by default None
    activity_fn : :class:`Callable`, optional
        Function to load the activity data, by default None
    show_missing : :class:`bool`, optional
        Whether to show missing data, by default True
    resolution_sleep : :class:`float` or :class:`int`, optional
        Resolution of the sleep data in minutes, by default 1.
    resolution_cardiac : :class:`float` or :class:`int`, optional
        Resolution of the cardiac data in minutes, by default 1.
    resolution_activity : :class:`float` or :class:`int`, optional
        Resolution of the activity data in minutes, by default 15.
    fontsize : :class:`int`, optional
        Font size for the plot, by default 10
    show : :class:`bool`, optional
        Whether to show the plot, by default True
    save_to : :class:`str`, optional
        Path where to save the grid plot, by default None
    title : :class:`str`, optional
        Title of the plot, by default ""
    show_dates : :class:`bool`, optional
        Whether to show the dates on the x-axis, by default False

    Returns
    -------
    :class:`list[matplotlib.axes._subplots.AxesSubplot]`
        Axes objects of the plot
    """

    if figsize is None:
        figsize = (10, (end_date - start_date).days * 2)
    if hypnogram_fn is None:
        hypnogram_fn = loader.load_hypnogram
    if cardiac_fn is None:
        cardiac_fn = loader.load_heart_rate
    if activity_fn is None:
        activity_fn = loader.load_steps

    # load all the data with some room for the next day due to start_hour
    hypnograms = hypnogram_fn(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date + datetime.timedelta(days=1),
    )
    cardiac_data = cardiac_fn(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date + datetime.timedelta(days=1),
    )
    activity_data = activity_fn(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date + datetime.timedelta(days=1),
    )

    plt.figure(figsize=figsize)
    axes = []

    for i in range((end_date - start_date).days):
        date = start_date + datetime.timedelta(days=i)
        start_date_hour = datetime.datetime.combine(date, datetime.time(start_hour))

        # get the hypnogram for the given date
        hypnogram = hypnograms.get(date, None)
        if hypnogram is not None:
            hypnogram_stages = hypnogram["values"]
            hypnogram_length = len(hypnogram_stages)
            hypnogram_start = hypnogram["start_time"]
            hypnogram_end = hypnogram["end_time"]
            hypnogram_time = pd.date_range(
                start=hypnogram_start,
                periods=hypnogram_length + 1,
                freq=f"{resolution_sleep}min",
            )

        # and for the next date
        next_hypnogram = hypnograms.get(date + datetime.timedelta(days=1), None)
        if next_hypnogram is not None:
            next_hypnogram_stages = next_hypnogram["values"]
            next_hypnogram_length = len(next_hypnogram_stages)
            next_hypnogram_start = next_hypnogram["start_time"]
            next_hypnogram_end = next_hypnogram["end_time"]
            next_hypnogram_time = pd.date_range(
                start=next_hypnogram_start,
                periods=next_hypnogram_length + 1,
                freq=f"{resolution_sleep}min",
            )

        full_day_range = pd.date_range(
            start=start_date_hour,
            periods=int(24 * ((60) / resolution_sleep) + 1),
            freq=f"{resolution_sleep}min",
        )
        full_day_hypnogram = np.full(
            len(full_day_range) - 1, 5
        )  # 5 (wake) is assumed as default value

        if hypnogram is not None or next_hypnogram is not None:
            if hypnogram is not None:
                diff_init = hypnogram_start - start_date_hour
                # in case the hypnogram starts before the start of the subplot
                # we need to see only the last part of the hypnogram
                if diff_init.days < 0:
                    diff_steps = abs(
                        int(diff_init.total_seconds() / (resolution_sleep * 60))
                    )
                    hypnogram_time = hypnogram_time[diff_steps:]
                    hypnogram_values = hypnogram_stages[diff_steps:]
                    full_day_hypnogram[: len(hypnogram_values)] = hypnogram_values
                else:  # otherwise the full hypnogram is present and we must add it to the full_day_hypnogram
                    diff_steps = abs(
                        int(diff_init.total_seconds() / (resolution_sleep * 60))
                    )
                    full_day_hypnogram[diff_steps : hypnogram_length + diff_steps] = (
                        hypnogram_stages
                    )
            if next_hypnogram is not None:
                diff_end = next_hypnogram_end - (
                    start_date_hour + datetime.timedelta(days=1)
                )
                # in case the hypnogram ends after the end of the subplot
                if diff_end.days >= 0:
                    diff_steps = abs(
                        int(diff_end.total_seconds() / (resolution_sleep * 60))
                    )
                    next_hypnogram_time = next_hypnogram_time[:-diff_steps]
                    next_hypnogram_values = next_hypnogram_stages[:-diff_steps]
                    # Note that if the hypnogram is completely in the next subplot, we must avoid the next part
                    # If it is not, we need to see only the first part of the hypnogram
                    if len(next_hypnogram_values) > 0:
                        full_day_hypnogram[-len(next_hypnogram_values) :] = (
                            next_hypnogram_values
                        )
                else:  # otherwise the full hypnogram is present and we must add it to the full_day_hypnogram
                    diff_steps = abs(
                        int(diff_end.total_seconds() / (resolution_sleep * 60))
                    )
                    full_day_hypnogram[
                        -next_hypnogram_length - diff_steps : -diff_steps
                    ] = next_hypnogram_stages
        # map 0 (previous encoding of wake) to 5 in full_day_hypnogram for visualization purposes
        full_day_hypnogram[full_day_hypnogram == 0] = 5

        # get cardiac data
        full_day_range_cardiac = pd.date_range(
            start=start_date_hour,
            periods=int(24 * ((60) / resolution_cardiac) + 1),
            freq=f"{resolution_cardiac}min",
        )
        date_cardiac_expanded = pd.merge(
            left=full_day_range_cardiac.to_series(
                name=pywearable.constants.ISODATE_COL
            ),
            right=cardiac_data,
            how="left",
        )
        # get activity data
        full_day_range_activity = pd.date_range(
            start=start_date_hour,
            periods=int(24 * ((60) / resolution_activity) + 1),
            freq=f"{resolution_activity}min",
        )
        date_activity_expanded = pd.merge(
            left=full_day_range_activity.to_series(
                name=pywearable.constants.ISODATE_COL
            ),
            right=activity_data,
            how="left",
        )

        ax = plt.subplot((end_date - start_date).days, 1, i + 1)
        date_times = pd.date_range(start=start_date_hour, periods=9, freq="3h")
        ax.set_xlim([date_times[0], date_times[-1]])

        # show times on x-axis top only in the first subplot
        if i == 0:
            ax.set_title(title, fontsize=fontsize + 4)
            ax.xaxis.tick_top()
            ax.set_xticks(date_times)
            ax.set_xticklabels(date_times.strftime("%H:%M"))
            ax.tick_params(axis="x", labelrotation=0, labelsize=fontsize)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.tick_params(axis="x", labelrotation=0, labelsize=fontsize)

        # plot activity data
        ax.bar(
            date_activity_expanded[pywearable.constants.ISODATE_COL],
            date_activity_expanded[pywearable.constants.ACTIVITY_STEPS_COL]
            / resolution_activity,
            color="green",
            alpha=0.5,
            width=0.01,
            edgecolor="black",
        )
        max_yaxis_activity = max(
            1000 / resolution_activity,
            max(
                date_activity_expanded[pywearable.constants.ACTIVITY_STEPS_COL]
                / resolution_activity
            )
            + 100 / resolution_activity,
        )
        ax.set_ylim([0, max_yaxis_activity])
        ax.tick_params(axis="y", colors="green")
        ax.set_ylabel("Steps per minute", color="green", fontsize=fontsize)
        if show_dates:
            ax.set_xlabel(date, fontsize=fontsize + 2)

        # plot cardiac data
        ax2 = ax.twinx()
        ax2.plot(
            date_cardiac_expanded[pywearable.constants.ISODATE_COL],
            date_cardiac_expanded[pywearable.constants.HR_COLUMN],
            color="red",
        )
        max_yaxis = max(175, max(date_cardiac_expanded[pywearable.constants.HR_COLUMN]))
        if show_missing:
            ax2.fill_between(
                date_cardiac_expanded[pywearable.constants.ISODATE_COL],
                max_yaxis - 10,
                where=date_cardiac_expanded.beatsPerMinute.isna(),
                facecolor="darkgrey",
                alpha=0.5,
            )
        ax2.set_ylim([35, max_yaxis])
        ax2.tick_params(axis="y", colors="red")
        ax2.set_ylabel("Heart rate (bpm)", color="red", fontsize=fontsize)

        # plot hypnogram(s)
        ax3 = ax.twinx()
        ax3.set_ylim([0, 5.5])
        ax3.stairs(full_day_hypnogram, full_day_range, baseline=4, color="darkblue")
        # Offset the third y-axis to the right
        ax3.spines["right"].set_position(("outward", 50))
        ax3.set_yticks(
            ticks=[1, 2, 3, 4, 5], labels=["Light", "", "Deep", "REM", "Wake"]
        )
        ax3.tick_params(axis="y", colors="darkblue")

        axes.append([ax, ax2, ax3])

    plt.tight_layout()

    if save_to:
        plt.savefig(save_to)
    if show:
        plt.show()

    return axes
