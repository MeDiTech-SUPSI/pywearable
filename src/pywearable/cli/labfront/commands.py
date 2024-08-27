import datetime
from pathlib import Path
from typing import Union

import click
import pandas as pd

from ... import constants
from ...loader.labfront import constants as lf_constants

LABFRONT_METRICS_SPLIT_BY_DAY_DICT = {
    lf_constants.GARMIN_CONNECT_ACTIVITY_DETAILS_LAP_FOLDER: False,
    lf_constants.GARMIN_CONNECT_ACTIVITY_DETAILS_SAMPLE_FOLDER: False,
    lf_constants.GARMIN_CONNECT_ACTIVITY_DETAILS_SUMMARY_FOLDER: False,
    lf_constants.GARMIN_CONNECT_ACTIVITY_MOVEIQ_SUMMARY_FOLDER: False,
    lf_constants.GARMIN_CONNECT_ACTIVITY_SUMMARY_FOLDER: False,
    lf_constants.GARMIN_CONNECT_HEART_RATE_FOLDER: True,
    lf_constants.GARMIN_CONNECT_DAILY_SUMMARY_FOLDER: False,
    lf_constants.GARMIN_CONNECT_EPOCH_FOLDER: False,
    lf_constants.GARMIN_CONNECT_RESPIRATION_FOLDER: True,
    lf_constants.GARMIN_CONNECT_SLEEP_PULSE_OX_FOLDER: True,
    lf_constants.GARMIN_CONNECT_SLEEP_RESPIRATION_FOLDER: True,
    lf_constants.GARMIN_CONNECT_SLEEP_STAGE_FOLDER: False,
    lf_constants.GARMIN_CONNECT_SLEEP_SUMMARY_FOLDER: False,
    lf_constants.GARMIN_CONNECT_STRESS_FOLDER: True,
    lf_constants.GARMIN_CONNECT_USER_METRICS_FOLDER: False,
    # TODO add actigraphy
    # lf_constants.GARMIN_DEVICE_ACTIGRAPHY_FOLDER: True,
    lf_constants.GARMIN_DEVICE_BBI_FOLDER: True,
    lf_constants.GARMIN_DEVICE_HEART_RATE_FOLDER: True,
    lf_constants.GARMIN_DEVICE_RESPIRATION_FOLDER: True,
    lf_constants.GARMIN_DEVICE_STEP_FOLDER: True,
    lf_constants.GARMIN_DEVICE_STRESS_FOLDER: True,
    lf_constants.QUESTIONNAIRE_FOLDER: False,
    lf_constants.TODO_FOLDER: False,
}


@click.group()
def labfront():
    pass


def get_header_length(path_to_csv_file: Union[str, Path]) -> int:
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
    with open(path_to_csv_file, "r", encoding="utf-8") as f:
        line = f.readline().split(",")
    header_length = int(line[1])
    return header_length


def read_csv_file_header(path_to_csv_file: Union[str, Path]) -> str:
    header_length = get_header_length(path_to_csv_file=path_to_csv_file)
    with open(path_to_csv_file) as input_file:
        head = [next(input_file) for _ in range(header_length)]
    return head


def process_csv_file_header(header: list):
    header_keys = header[-2].split(",")
    header_values = header[-1].split(",")
    header[-2] = ",".join(header_keys[:3])
    header[-2] += header_keys[3] + "\n"
    header[-1] = ",".join(header_values[:3])
    header[-1] += header_values[3] + "\n"
    return header


def get_csv_file_header(path_to_csv_file: Union[str, Path]) -> str:
    head = read_csv_file_header(path_to_csv_file=path_to_csv_file)
    processed_head = process_csv_file_header(header=head)
    return "".join(processed_head)


def read_csv_file_keys(path_to_csv_file: Union[str, Path]) -> pd.DataFrame:
    pass


def read_csv_data(path_to_csv_file: Union[str, Path]) -> pd.DataFrame:
    pass


def write_new_metric_csv_file(output_folder, participant_id, metric, old_csv_paths):
    # Read and process csv header
    header = get_csv_file_header(path_to_csv_file=old_csv_paths[0])
    # Read csv data
    data_list = [
        pd.read_csv(csv_file, skiprows=get_header_length(csv_file) + 1)
        for csv_file in old_csv_paths
    ]
    df = pd.concat(data_list, axis=0, ignore_index=True)
    if constants.ISODATE_COL in df.columns:
        df[constants.ISODATE_COL] = pd.to_datetime(
            df[constants.ISODATE_COL].str.rsplit("+", expand=True)[0], utc=True
        ).dt.tz_localize(tz=None)
        df[constants.CALENDAR_DATE_COL] = df[constants.ISODATE_COL].dt.normalize()
        for calendar_date in df[constants.CALENDAR_DATE_COL].unique():
            daily_data = df.loc[
                (df[constants.ISODATE_COL] > calendar_date)
                & (
                    (
                        df[constants.ISODATE_COL]
                        <= calendar_date + datetime.timedelta(days=1)
                    )
                ),
                :,
            ].drop(constants.CALENDAR_DATE_COL, axis=1)
            path_to_metric_folder = Path(output_folder) / participant_id / metric
            if not path_to_metric_folder.exists():
                path_to_metric_folder.mkdir(exist_ok=True, parents=True)
            path_to_file = (
                path_to_metric_folder
                / f"{calendar_date.strftime('%y%m%d')}_{metric}_{participant_id}.csv"
            )
            with open(str(path_to_file), "w") as f:
                f.write(header)
            daily_data.to_csv(path_to_file, mode="a", index=False)
    # If metric is split by day:
    # ... Split data by calendar date
    # ... For each day
    # ...... Write header without timestamps
    # ...... Write data
    pass


def write_new_task_csv_file(output_folder, participant_id, metric, old_csv):
    pass


@click.command()
@click.argument("input_folder")
@click.argument("output_folder")
def convert_folder(input_folder, output_folder):
    """
    This script converts Labfront data located
    in INPUT_FOLDER, and that use the 'old'
    Labfront file structure with CSV files named
    sequentially starting from 000000, to the
    new Labfront file structure, that use
    CSV files named based on the solar date of
    the data in the CSV file, and store the
    new formatted data into OUTPUT_FOLDER.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    if input_folder == output_folder:
        raise ValueError(f"Input folder must be different than output_folder")
    if not (input_folder).exists():
        raise FileNotFoundError(f"Input folder {input_folder} does not exist")
    if output_folder.exists():
        output_folder_check = True
        while output_folder_check:
            check = input(
                f"Output folder {output_folder} already exists. Do you wish to continue? [y]"
            )
            if check.lower() == "n":
                return
    # Go into input folder and convert CSV
    participants_folders = [x for x in input_folder.iterdir() if x.is_dir()]
    for folder in participants_folders:
        metrics_folders = [x for x in folder.iterdir() if x.is_dir()]
        for metric_folder in metrics_folders:
            csv_files = [x for x in metric_folder.iterdir() if x.name.endswith("csv")]
            metric_subfolders = [x for x in metric_folder.iterdir() if x.is_dir()]
            if len(csv_files) > 0:
                write_new_metric_csv_file(
                    output_folder=output_folder,
                    participant_id=folder.name,
                    metric=metric_folder.name,
                    old_csv_paths=csv_files,
                )
            for metric_subfolder in metric_subfolders:
                write_new_task_csv_file(
                    output_folder=output_folder,
                    participant_id=folder.name,
                    metric=metric_folder.name,
                    task=metric_subfolder.name,
                    old_csv_path=csv_file,
                )


labfront.add_command(convert_folder)
