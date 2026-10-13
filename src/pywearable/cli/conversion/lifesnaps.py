"""
This script converts LifeSnaps data, which are provided as MongoDB
dumps, into Parquet format.

In order to run this script, you need to specify the host for the MongoDB
server in which LifeSnaps data were imported.
"""

import warnings

import pandas as pd

try:
    from pymongo import MongoClient
except ImportError:
    warnings.warn(f"Package pymongo is needed to convert LifeSnaps")

import os
from pathlib import Path

import click as cli
import tqdm


@cli.command()
@cli.argument("output-folder")
@cli.option("--mongodb-host", default="localhost")
@cli.option("--mongodb-port", default=27017)
@cli.option(
    "--mongodb-db-name",
    default="rais_anonymized",
)
@cli.option("--scored-survey-folder", default=None)
def lifesnaps_to_parquet(
    output_folder: os.PathLike, host: str, port: int, db_name: str, scored_survey_folder
):
    """
    Convert LifeSnaps dataset from MongoDB to Parquet
    format for easier and faster access to data.

    This script takes data (FitBit, EMA, raw surveys)
    stored on the MongoDB
    server on host MONGODB-HOST with access on
    port MONGODB-PORT contained in the database
    MONGO-DB-NAME, and scored surveys stored in
    SCORED-SURVEY-FOLDER and exports them in the
    OUTPUT-FOLDER in parquet format.
    """
    # Setup
    client = MongoClient(f"mongodb://{host}:{str(port)}/")
    db = client[db_name]
    output_folder = Path(output_folder)

    collections = db.list_collection_names()
    if len(collections) == 0:
        raise ValueError(
            f"Couldn't find any collection at mongodb://{host}:{str(port)}/ with db {db_name}"
        )

    for coll_name in collections:
        print(f"--- Processing Collection: {coll_name} ---")
        coll = db[coll_name]

        # 1. Fitbit Data: Partition by User AND Type
        if "fitbit" in coll_name.lower():

            user_ids = coll.distinct("id")

            for user_id in tqdm.tqdm(user_ids, desc="Fitbit"):
                # Find all unique 'type' metrics for this specific user
                metrics = coll.distinct("type", {"id": user_id})

                user_folder = output_folder / str(user_id)
                user_folder.mkdir(parents=True, exist_ok=True)

                for metric in metrics:
                    # Query data for specific user and specific type
                    cursor = coll.find({"id": user_id, "type": metric}, {"_id": 0})
                    raw_data = list(cursor)

                    if raw_data:
                        # 1. Load the "envelope" into a DataFrame
                        df_envelope = pd.DataFrame(raw_data)

                        # 2. Extract and flatten the 'data' column
                        # This turns {'heart_rate': 70, 'val': 1} into separate columns
                        df_payload = pd.json_normalize(df_envelope["data"])

                        if metric == "sleep":
                            # List of sub-fields to extract and explode
                            sub_fields = {
                                "levels.data": "stages",
                                "levels.shortData": "short_stages",
                            }

                            for field, suffix in sub_fields.items():
                                if field in df_payload.columns:
                                    # We need the logId and the field to explode
                                    # 'logId' is the standard Fitbit unique identifier for a sleep session
                                    temp_df = df_payload[["logId", field]].dropna()

                                    # Explode the list of dictionaries into individual rows
                                    exploded = temp_df.explode(field)
                                    exploded = exploded.dropna(subset=field)

                                    # Flatten the dictionaries into columns
                                    sub_data = pd.json_normalize(exploded[field])

                                    # Re-attach the logId (reset index is crucial for the join)
                                    sub_data.index = exploded.index
                                    final_sub_df = pd.concat(
                                        [exploded[["logId"]], sub_data], axis=1
                                    )

                                    # Save as a separate file
                                    sub_file_path = (
                                        user_folder / f"sleep_{suffix}.parquet"
                                    )
                                    final_sub_df.to_parquet(sub_file_path, index=False)

                            # Remove the massive nested lists from the main summary file to keep it light
                            cols_to_drop = [
                                f for f in sub_fields.keys() if f in df_payload.columns
                            ]
                            df_payload = df_payload.drop(columns=cols_to_drop)
                            df_payload.columns = df_payload.columns.map(
                                lambda x: x.replace("levels.summary.", "").replace(
                                    ".", "_"
                                )
                            )
                        elif metric in [
                            "demographic_vo2_max",
                            "heart_rate",
                            "resting_heart_rate",
                        ]:
                            # These have "value" in column names
                            df_payload.columns = df_payload.columns.map(
                                lambda x: x.replace("value.", "")
                            )

                        elif metric == "time_in_heart_rate_zones":
                            # These have "value" in column names
                            df_payload.columns = df_payload.columns.map(
                                lambda x: x.replace("value.valuesInZones.", "")
                            )

                        # Clean filename and save
                        safe_metric_name = str(metric).replace(" ", "_").lower()
                        file_path = user_folder / f"{safe_metric_name}.parquet"

                        df_payload.to_parquet(file_path, index=False)

        # 2. Surveys/EMA: Save as single files
        elif "surveys" in coll_name.lower():
            cursor = coll.find({}, {"_id": 0})

            surveys = coll.distinct("type")
            for survey in surveys:
                cursor = coll.find({"type": survey}, {"_id": 0})
                df = pd.DataFrame(list(cursor))
                df["user_id"] = df["user_id"].astype(str)

                # 2. Extract and flatten the 'data' column
                # This turns {'heart_rate': 70, 'val': 1} into separate columns
                df_payload = pd.json_normalize(df["data"])

                df_final = pd.concat(
                    [
                        df.drop(columns=["data", "type"]),
                        df_payload.drop("id", axis=1),
                    ],
                    axis=1,
                )
                df_final.to_parquet(output_folder / f"{survey}.parquet", index=False)
        elif "sema" in coll_name.lower():
            cursor = coll.find({}, {"_id": 0})
            df = pd.DataFrame(list(cursor))

            df["user_id"] = df["user_id"].astype(str)

            # 2. Extract and flatten the 'data' column
            # This turns {'heart_rate': 70, 'val': 1} into separate columns
            df_payload = pd.json_normalize(df["data"]).drop(
                ["PARTICIPANT_ID", "STUDY_ID"], axis=1
            )
            df_final = pd.concat(
                [
                    df.drop(columns=["data"]),
                    df_payload,
                ],
                axis=1,
            )
            df_final = df_final.replace("<no-response>", None)
            df_final.to_parquet(output_folder / f"sema.parquet", index=False)
    if not (scored_survey_folder is None):
        # Let's read and convert each scored folder
        scored_surveys_files = [
            x for x in Path(scored_survey_folder).iterdir() if ".csv" in x.name
        ]
        for f in scored_surveys_files:
            df = pd.read_csv(f, index_col=0, parse_dates=["submitdate"])
            df.to_parquet(output_folder / f"{f.stem}_scored.parquet", index=False)
    print("\nLifeSnaps conversion finished!")
