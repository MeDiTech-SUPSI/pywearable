import os
import warnings

import click as cli

try:
    import duckdb
except ImportError:
    warnings.warn("\nPackage duckdb is needed for Tesserae conversion\n")

high_resolution_metrics_map = {
    "GarminHealthAPI": [
        "Activities",
        "ActivitySeries",
        "Epochs",
        "HR",
        "MoveIQ",
        "Sleep",
        "Stress",
    ],
    "GarminPA": [
        "BBI",
        "Calories",
        "Floors",
        "GarminBattery",
        "HR",
        "HRVSeries",
        "Intensity",
        "Steps",
    ],
    "Phone": ["Pedometer"],
    "Gimbal": ["BeaconSighting", "BeaconPlace"],
}

low_resolutions_metrics_map = {
    "GarminSummary": [
        "ActivitySummary",
        "AdjImputedSleep",
        "DailySummaries",
        "HRSummary",
        "HRVSummary",
        "StressSummary",
    ],
    "GimbalSummary": [
        "BeaconSummary",
        "InteractionSummary",
    ],
    "GroundTruthSurvey": ["DailiesScored", "IGTBDemographics", "IGTBScored"],
    "Mapping": ["Beacons", "ParticipantInfo"],
    "Mixed": ["HRVSummaryBaseline", "HRVWorkDeskHomeSummary", "PASummary"],
    "Contextual": ["NOAAStormEvents", "WeatherSummary"],
}


@cli.command()
@cli.argument("input_folder")
@cli.argument("output_folder")
def tesserae_to_parquet(input_folder: os.PathLike, output_folder: os.PathLike):
    """
    Convert Tesserae dataset from CSV files to Parquet
    format for easier and faster access to data.

    This script takes raw CSV files stored in
    INPUT_FOLDER and converts them into a new
    structure with Parquet files inside
    OUTPUT_FOLDER:
    """
    con = duckdb.connect()

    # 1. Get a list of all participant folders
    participants = [
        f
        for f in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, f))
    ]

    print(f"Found {len(participants)} participants. Starting conversion...")

    for p_id in participants:
        print(f"Processing Participant: {p_id}")
        p_input_path = os.path.join(input_folder, p_id)

        # 2. Iterate through data types (GarminHealthAPI, etc.)
        for d_type in high_resolution_metrics_map.keys():
            if d_type == "Gimbal":
                d_type_path = os.path.join(p_input_path, "data-pub-onreq", p_id, d_type)
            else:
                d_type_path = os.path.join(p_input_path, d_type)
            if not os.path.isdir(d_type_path):
                continue

            # Create corresponding output directory
            target_dir = os.path.join(output_folder, p_id, d_type)
            os.makedirs(target_dir, exist_ok=True)

            # 3. For each metric we want, find the matching CSV and convert
            for metric in high_resolution_metrics_map[d_type]:
                # Look for CSVs in this folder that contain the metric name
                # We use a wildcard * to handle the long timestamps in your filenames
                csv_pattern = os.path.join(d_type_path, f"*{metric}*.csv")
                output_file = os.path.join(target_dir, f"{metric}.parquet")

                # Check if any files actually match before running SQL
                # This avoids 'File not found' errors in DuckDB
                import glob

                if glob.glob(csv_pattern):
                    query = f"""
                        COPY (
                            SELECT * EXCLUDE (participantID)
                            FROM read_csv_auto(
                            '{csv_pattern}', 
                            union_by_name=True)
                        ) TO '{output_file}' (FORMAT PARQUET);
                    """
                    try:
                        con.execute(query)
                    except Exception as e:
                        print(f"  [!] Error on {p_id}/{metric}: {e}")

    for d_type in low_resolutions_metrics_map.keys():
        for metric in low_resolutions_metrics_map[d_type]:

            output_file = os.path.join(
                output_folder, "_global_metrics", d_type, f"{metric}.parquet"
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # We glob every participant's version of this small file
            input_pattern = os.path.join(input_folder, "**", d_type, f"*{metric}*.csv")

            query = f"""
                COPY (
                    SELECT *
                    FROM read_csv_auto('{input_pattern}', union_by_name=True)
                ) TO '{output_file}' (FORMAT PARQUET);
            """
            con.execute(query)
    print("\nTesserae Conversion finished!")
