import pytest

from src import config
from src.feature_engineering import FeatureBuilder
from src.predict import Forecaster
from src.preprocessing import load_clean


@pytest.fixture(scope="session")
def clean():
    return load_clean()


@pytest.fixture(scope="session")
def builder(clean):
    return FeatureBuilder(clean)


@pytest.fixture(scope="session")
def rows(builder):
    return builder.all_rows()


@pytest.fixture(scope="session")
def forecaster():
    assert config.MODEL_FILE.exists(), "Run `python -m src.train` before the test-suite"
    return Forecaster.load()
