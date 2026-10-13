from pathlib import Path

import numpy as np
import pandas as pd

from pywearable.loader import LabfrontLoader, LifeSnapsParquetLoader
from pywearable.loader.labfront import constants as lf_constants
from pywearable.loader.tesserae import TesseraeCSVLoader, TesseraeParquetLoader


def test_labfront_questionnaire_discovery_filters_participants():
    index = pd.MultiIndex.from_tuples(
        [
            ("alice_001", "alice", "001", "questionnaire"),
            ("alice_001", "alice", "001", "questionnaire"),
            ("bob_002", "bob", "002", "questionnaire"),
            ("bob_002", "bob", "002", "todo"),
        ],
        names=[
            lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY,
            lf_constants.FILE_INDEX_PARTICIPANT_ID_KEY,
            lf_constants.FILE_INDEX_PARTICIPANT_LABFRONT_ID_KEY,
            lf_constants.FILE_INDEX_METRIC_KEY,
        ],
    )
    loader = LabfrontLoader.__new__(LabfrontLoader)
    loader.file_index = pd.DataFrame(
        {
            lf_constants.FILE_INDEX_IS_QUESTIONNAIRE_KEY: [True, True, True, False],
            lf_constants.FILE_INDEX_TASK_NAME_KEY: [
                "Evening",
                "Morning",
                "Morning",
                "Setup",
            ],
            lf_constants.FILE_INDEX_TASK_FULL_ID_KEY: [
                "Evening_evening-id",
                "Morning_morning-id",
                "Morning_morning-id",
                "Setup_setup-id",
            ],
        },
        index=index,
    )

    assert loader.get_available_questionnaires("alice") == ["Evening", "Morning"]
    assert loader.get_available_questionnaires("002") == ["Morning"]


def test_labfront_questionnaire_metadata_preserves_option_numbers(
    tmp_path, monkeypatch
):
    participant = "alice_001"
    task_folder = "Morning_morning-id"
    folder = tmp_path / participant / "questionnaire" / task_folder
    folder.mkdir(parents=True)
    path = folder / "responses.csv"
    path.write_text(
        "metadata,ignored\n"
        "metadata,ignored\n"
        "sectionIndex,questionIndex,questionType,questionDescription,option1Name,option2Name\n"
        "1,1,radio,Sleep quality,Good,Poor\n"
        "1,2,text,Notes,,\n",
        encoding="utf-8",
    )
    index = pd.MultiIndex.from_tuples(
        [(participant, "alice", "001", "questionnaire")],
        names=[
            lf_constants.FILE_INDEX_PARTICIPANT_FULL_ID_KEY,
            lf_constants.FILE_INDEX_PARTICIPANT_ID_KEY,
            lf_constants.FILE_INDEX_PARTICIPANT_LABFRONT_ID_KEY,
            lf_constants.FILE_INDEX_METRIC_KEY,
        ],
    )
    loader = LabfrontLoader.__new__(LabfrontLoader)
    loader.file_index = pd.DataFrame(
        {
            lf_constants.FILE_INDEX_TASK_NAME_KEY: ["Morning"],
            lf_constants.FILE_INDEX_TASK_ID_KEY: ["morning-id"],
            lf_constants.FILE_INDEX_TASK_FULL_ID_KEY: [task_folder],
            lf_constants.FILE_INDEX_BASE_FOLDER_KEY: [tmp_path],
            lf_constants.FILE_INDEX_TASK_FOLDER: [task_folder],
            lf_constants.FILE_INDEX_FILENAME_KEY: [path.name],
        },
        index=index,
    )
    monkeypatch.setattr(loader, "get_header_length", lambda _: 0)
    monkeypatch.setattr(loader, "get_key_length", lambda _: 4)

    questions = loader.get_questionnaire_questions("Morning")

    assert questions == {
        "1_1": {
            "type": "radio",
            "description": "Sleep quality",
            "options": {1: "Good", 2: "Poor"},
        },
        "1_2": {"type": "text", "description": "Notes"},
    }


def test_labfront_load_questionnaire_returns_raw_responses(monkeypatch):
    expected = pd.DataFrame(
        {"isoDate": pd.to_datetime(["2024-01-01 08:00"]), "1_1": [2], "1_2": ["ok"]}
    )
    loader = LabfrontLoader.__new__(LabfrontLoader)
    calls = []

    def load_raw(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(loader, "get_data_from_datetime", load_raw)

    result = loader.load_questionnaire(
        "alice", "2024-01-01", "2024-01-02", "Morning"
    )

    pd.testing.assert_frame_equal(result, expected)
    assert calls == [
        {
            "user_id": "alice",
            "metric": lf_constants.QUESTIONNAIRE_FOLDER,
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "is_questionnaire": True,
            "task_name": "Morning",
            "filter_on_calendar_date": False,
        }
    ]


def test_lifesnaps_raw_survey_parses_dates_and_preserves_answers(
    tmp_path, monkeypatch
):
    (tmp_path / "person-1").mkdir()
    raw = pd.DataFrame(
        {
            "user_id": ["person-1", "person-1", "person-2"],
            "submitdate": [
                "31/5/2021 13:32",
                "1/6/2021 09:15",
                "1/6/2021 10:00",
            ],
            "startdate": [
                "31/5/2021 13:17",
                "1/6/2021 09:00",
                "1/6/2021 09:50",
            ],
            "datestamp": [
                "31/5/2021 13:32",
                "1/6/2021 09:15",
                "1/6/2021 10:00",
            ],
            "STAI[SQ001]": [1, 4, 3],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: raw.copy())

    result = LifeSnapsParquetLoader(tmp_path).load_survey(
        "stai",
        user_id="person-1",
        start_date="2021-06-01",
        end_date="2021-06-02",
    )

    assert result["STAI[SQ001]"].tolist() == [4]
    assert result.loc[0, "submitdate"] == pd.Timestamp("2021-06-01 09:15")
    assert pd.api.types.is_datetime64_any_dtype(result["startdate"])
    assert pd.api.types.is_datetime64_any_dtype(result["datestamp"])


def test_lifesnaps_scored_survey_is_loaded_without_raw_survey_join(
    tmp_path, monkeypatch
):
    (tmp_path / "person-1").mkdir()
    scored = pd.DataFrame(
        {
            "user_id": ["person-1", "person-1", "person-2"],
            "type": ["panas", "panas", "panas"],
            "submitdate": pd.to_datetime(
                ["2021-05-31", "2021-06-07", "2021-06-07"]
            ),
            "positive_affect_score": [25, 31, 17],
            "negative_affect_score": [10, 8, 20],
        }
    )
    paths = []

    def read_parquet(path):
        paths.append(Path(path).name)
        return scored.copy()

    monkeypatch.setattr(pd, "read_parquet", read_parquet)

    result = LifeSnapsParquetLoader(tmp_path).load_survey_scored(
        "panas",
        user_id="person-1",
        start_date="2021-06-01",
        end_date="2021-06-08",
    )

    assert paths == ["panas_scored.parquet"]
    assert result.to_dict("records") == [
        {
            "user_id": "person-1",
            "type": "panas",
            "submitdate": pd.Timestamp("2021-06-07"),
            "positive_affect_score": 31,
            "negative_affect_score": 8,
        }
    ]


def test_lifesnaps_ema_filters_on_creation_time(tmp_path, monkeypatch):
    (tmp_path / "person-1").mkdir()
    raw = pd.DataFrame(
        {
            "user_id": ["person-1", "person-1"],
            "CREATED_TS": ["2021-06-01T09:00:00", "2021-06-02T09:00:00"],
            "SCHEDULED_TS": ["2021-06-01T08:50:00", "2021-06-02T08:50:00"],
            "STARTED_TS": ["2021-06-01T09:00:00", None],
            "COMPLETED_TS": ["2021-06-01T09:02:00", None],
            "EXPIRED_TS": [None, "2021-06-02T10:00:00"],
            "UPLOADED_TS": ["2021-06-01T09:02:00", None],
            "MOOD": [4, np.nan],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: raw.copy())
    (tmp_path / "sema.parquet").touch()

    result = LifeSnapsParquetLoader(tmp_path).load_ema(
        "person-1", start_date="2021-06-02", end_date="2021-06-03"
    )

    assert len(result) == 1
    assert result.loc[0, "CREATED_TS"] == pd.Timestamp("2021-06-02 09:00")
    assert pd.isna(result.loc[0, "MOOD"])


def test_tesserae_csv_questionnaire_filters_are_optional(tmp_path):
    folder = tmp_path / "person-1" / "GroundTruthSurvey"
    folder.mkdir(parents=True)
    pd.DataFrame(
        {
            "ParticipantID": ["person-1", "person-1"],
            "Timestamp": [
                "2018-01-19T16:10:00-0500",
                "2018-01-20T16:10:00-0500",
            ],
            "timezone_change": [False, False],
            "date": ["2018-01-19", "2018-01-20"],
            "survey_name": ["daily", "daily"],
            "stress": [3.0, 5.0],
            "survey_sent_datetime": [
                "2018-01-19 16:00:00",
                "2018-01-20 16:00:00",
            ],
            "survey_start_datetime": [
                "2018-01-19 16:07:00",
                "2018-01-20 16:07:00",
            ],
            "survey_end_datetime": [
                "2018-01-19 16:10:00",
                "2018-01-20 16:10:00",
            ],
        }
    ).to_csv(folder / "DailiesScored.csv", index=False)

    loader = TesseraeCSVLoader(tmp_path)

    assert loader.load_daily_survey("person-1")["survey_stress"].tolist() == [
        3.0,
        5.0,
    ]
    filtered = loader.load_daily_survey(
        "person-1", start_date="2018-01-20", end_date="2018-01-21"
    )
    assert filtered["survey_stress"].tolist() == [5.0]


def test_tesserae_parquet_loads_daily_and_demographic_responses(
    tmp_path, monkeypatch
):
    (tmp_path / "person-1").mkdir()
    survey_folder = tmp_path / "_global_metrics" / "GroundTruthSurvey"
    survey_folder.mkdir(parents=True)
    for filename in (
        "DailiesScored.parquet",
        "IGTBDemographics.parquet",
        "IGTBScored.parquet",
    ):
        (survey_folder / filename).touch()

    tables = {
        "DailiesScored.parquet": pd.DataFrame(
            {
                "ParticipantID": ["person-1", "person-1", "person-2"],
                "Timestamp": pd.to_datetime(
                    [
                        "2018-01-19 21:00Z",
                        "2018-01-20 21:00Z",
                        "2018-01-20 21:00Z",
                    ]
                ),
                "timezone_change": ["False", "False", "False"],
                "date": ["2018-01-19", "2018-01-20", "2018-01-20"],
                "survey_name": ["daily", "daily", "daily"],
                "stress": ["3.0", "5.0", "2.0"],
                "sleep": ["NULL", "4.0", "3.0"],
                "survey_sent_datetime": pd.to_datetime(
                    [
                        "2018-01-19 16:00",
                        "2018-01-20 16:00",
                        "2018-01-20 16:00",
                    ]
                ),
                "survey_start_datetime": pd.to_datetime(
                    [
                        "2018-01-19 16:07",
                        "2018-01-20 16:07",
                        "2018-01-20 16:07",
                    ]
                ),
                "survey_end_datetime": pd.to_datetime(
                    [
                        "2018-01-19 16:10",
                        "2018-01-20 16:10",
                        "2018-01-20 16:10",
                    ]
                ),
            }
        ),
        "IGTBDemographics.parquet": pd.DataFrame(
            {
                "ParticipantID": ["person-1"],
                "Timestamp": pd.to_datetime(["2018-01-18 23:11Z"]),
                "timezone_change": ["False"],
                "igtb.datetime": pd.to_datetime(["2018-01-18 18:11"]),
                "gender": ["Female"],
                "age_range": ["25-34"],
            }
        ),
        "IGTBScored.parquet": pd.DataFrame(
            {
                "ParticipantID": ["person-1"],
                "Timestamp": pd.to_datetime(["2018-01-18 23:11Z"]),
                "timezone_change": ["False"],
                "igtb.datetime": pd.to_datetime(["2018-01-18 18:11"]),
                "shipley.abs": [11],
                "psqi": ["15.0"],
            }
        ),
    }
    monkeypatch.setattr(
        pd, "read_parquet", lambda path: tables[Path(path).name].copy()
    )
    loader = TesseraeParquetLoader(tmp_path)

    csv_root = tmp_path / "csv"
    csv_folder = csv_root / "person-1" / "GroundTruthSurvey"
    csv_folder.mkdir(parents=True)
    daily_csv = tables["DailiesScored.parquet"].iloc[:2].copy()
    daily_csv["Timestamp"] = [
        "2018-01-19T16:00:00-0500",
        "2018-01-20T16:00:00-0500",
    ]
    daily_csv.to_csv(csv_folder / "DailiesScored.csv", index=False)
    raw_csv = tables["IGTBDemographics.parquet"].copy()
    raw_csv["Timestamp"] = ["2018-01-18T18:11:00-0500"]
    raw_csv.to_csv(csv_folder / "IGTBDemographics.csv", index=False)
    scored_csv = tables["IGTBScored.parquet"].copy()
    scored_csv["Timestamp"] = ["2018-01-18T18:11:00-0500"]
    scored_csv.to_csv(csv_folder / "IGTBScored.csv", index=False)
    csv_loader = TesseraeCSVLoader(csv_root)

    daily = loader.load_daily_survey(
        "person-1", start_date="2018-01-20", end_date="2018-01-21"
    )
    raw = loader.load_demographics_raw("person-1")
    scored = loader.load_demographics_scored("person-1")

    assert daily["survey_stress"].tolist() == [5.0]
    assert daily["survey_sleep"].tolist() == [4.0]
    assert daily["timezoneOffsetInMs"].tolist() == [-18_000_000.0]
    assert raw["demographics_gender"].tolist() == ["Female"]
    assert raw["demographics_age_range"].tolist() == ["25-34"]
    assert scored["demographics_shipley.abs"].tolist() == [11]
    assert scored["demographics_psqi"].tolist() == [15.0]
    assert scored["timezoneOffsetInMs"].tolist() == [-18_000_000.0]
    pd.testing.assert_frame_equal(
        csv_loader.load_daily_survey(
            "person-1", start_date="2018-01-20", end_date="2018-01-21"
        ),
        daily,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        csv_loader.load_demographics_raw("person-1"), raw, check_dtype=False
    )
    pd.testing.assert_frame_equal(
        csv_loader.load_demographics_scored("person-1"),
        scored,
        check_dtype=False,
    )
