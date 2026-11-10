import pandas as pd
import pytest

from src import config, data_loader


@pytest.fixture(scope="module")
def raw():
    return data_loader.load_raw()


def test_raw_file_exists():
    assert config.RAW_FILE.exists()


def test_schema_and_size(raw):
    data_loader.validate_schema(raw)
    assert raw.shape == (119_390, 32)


def test_schema_validation_rejects_bad_frame(raw):
    with pytest.raises(data_loader.DataValidationError):
        data_loader.validate_schema(raw.drop(columns=["adr"]))


def test_arrival_dates_cover_expected_span(raw):
    d = data_loader.arrival_dates(raw)
    assert d.min() == pd.Timestamp("2015-07-01")
    assert d.max() == pd.Timestamp("2017-08-31")
    assert d.nunique() == (d.max() - d.min()).days + 1  # no missing days


def test_load_raw_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        data_loader.load_raw(tmp_path / "nope.csv")
