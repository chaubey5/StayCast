import numpy as np
import pandas as pd
import pytest

from src import config, data_loader, preprocessing as pp


@pytest.fixture(scope="module")
def raw():
    return data_loader.load_raw()


@pytest.fixture(scope="module")
def cleaned(raw):
    return pp.clean_bookings(raw)


def test_clean_does_not_mutate_raw(raw):
    before = raw.copy()
    pp.clean_bookings(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_standardise_handles_null_tokens(raw):
    s = pp.standardise(raw)
    assert s["children"].isna().sum() == 0
    assert (s["country"] == "NULL").sum() == 0 and (s["country"] == "UNK").sum() == 488
    assert s["agent"].dtype.kind == "f" and s["agent"].isna().sum() == 16340


def test_duplicates_and_invalid_rows_removed(cleaned):
    clean, report = cleaned
    assert report["exact_duplicates_dropped"] == 31_994
    assert (clean["nights"] > 0).all()
    assert ((clean["adults"] + clean["children"] + clean["babies"]) > 0).all()
    assert (clean["adr"] >= 0).all()
    assert len(clean) == report["clean_rows"]
    assert report["raw_rows"] - report["exact_duplicates_dropped"] - sum(report["invalid_rows_dropped"].values()) == len(clean)


def test_cancellations_are_kept(cleaned):
    clean, _ = cleaned
    assert 0.2 < clean["is_canceled"].mean() < 0.4   # 27.7% after de-duplication; not silently removed


def test_booking_date_reconstruction(cleaned):
    clean, _ = cleaned
    delta = (clean["arrival_date"] - clean["booking_date"]).dt.days
    assert (delta == clean["lead_time"]).all()


def test_room_categories_are_the_configured_set(cleaned):
    clean, _ = cleaned
    assert set(clean["room_category"]) == set(config.CATEGORIES)


def test_expand_to_stay_nights_row_count_and_dates(cleaned):
    clean, _ = cleaned
    sample = clean.iloc[:200]
    nights = pp.expand_to_stay_nights(sample)
    assert len(nights) == sample["nights"].sum()
    first = sample.iloc[0]
    got = nights[nights.index < first["nights"]].stay_date.tolist()[: int(first["nights"])]
    expected = [first["arrival_date"] + pd.Timedelta(days=i) for i in range(int(first["nights"]))]
    assert got == expected


def test_cancelled_nights_carry_cancel_date(cleaned):
    clean, _ = cleaned
    nights = pp.expand_to_stay_nights(clean.iloc[:2000])
    assert nights.loc[nights.is_canceled == 1, "cancel_date"].notna().all()
    assert nights.loc[nights.is_canceled == 0, "cancel_date"].isna().all()


def test_processed_file_roundtrip():
    c = pp.load_clean()
    assert {"arrival_date", "booking_date", "room_category"} <= set(c.columns)
    assert np.issubdtype(c["arrival_date"].dtype, np.datetime64)
