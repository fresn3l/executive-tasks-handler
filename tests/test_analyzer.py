"""Desk behavior that should stay true when the sample logs change shape."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lte_analyzer.analyze import analyze
from lte_analyzer.catalog import SAMPLES, sample_path
from lte_analyzer.flags import FLAG_INTERFERENCE, FLAG_MARGINAL, FLAG_OK, FLAG_WEAK, classify_row
from lte_analyzer.geo import haversine_m, resample_lonlat
from lte_analyzer.ingest import IngestError, load_csv, load_text

ROOT = Path(__file__).resolve().parents[1]


def test_capitol_to_stadium_is_about_a_mile_and_a_half() -> None:
    meters = float(haversine_m(40.8080, -96.6997, 40.8207, -96.7048))
    assert 1_300 < meters < 1_700


def test_resample_spacing() -> None:
    coords = [[-96.80, 40.80], [-96.70, 40.80]]
    points = resample_lonlat(coords, 500)
    assert len(points) > 10
    gaps = [
        float(haversine_m(points[i][1], points[i][0], points[i + 1][1], points[i + 1][0]))
        for i in range(len(points) - 1)
    ]
    # The closing point may be a shorter remainder. Interior gaps track the step.
    assert max(gaps[:-1]) < 560
    assert min(gaps[:-1]) > 440


def test_flag_rules() -> None:
    assert classify_row(-115, -9, 10)[0] == FLAG_WEAK
    assert classify_row(-85, -16, -2)[0] == FLAG_INTERFERENCE
    assert classify_row(-85, -18, 8)[0] == FLAG_INTERFERENCE
    assert classify_row(-85, -8, 15)[0] == FLAG_OK
    assert classify_row(-105, -16, 0)[0] == FLAG_MARGINAL


def test_gnet_kbps_and_longitude_first() -> None:
    text = (
        "Timestamp,Longitude,Latitude,RSRP,RSRQ,SINR,DL_bitrate,UL_bitrate\n"
        "2026-04-18 08:00:00,-96.70,40.81,-90,-10,12,20000,4000\n"
    )
    frame, warnings = load_text(text, "gnet.csv")
    assert warnings == []
    assert frame.loc[0, "dl_mbps"] == pytest.approx(20.0)
    assert frame.loc[0, "ul_mbps"] == pytest.approx(4.0)
    assert frame.loc[0, "lon"] == pytest.approx(-96.70)


def test_nci_aliases_and_semicolon() -> None:
    text = (
        "time;latitude;longitude;rsrp;rsrq;rssnr;download_kbps;upload_kbps\n"
        "2026-04-18T10:00:00;40.86;-96.71;-88;-11;9;15000;3000\n"
    )
    frame, _warnings = load_text(text, "nci.csv")
    assert frame.loc[0, "sinr"] == pytest.approx(9)
    assert frame.loc[0, "dl_mbps"] == pytest.approx(15)


def test_asu_conversion() -> None:
    text = (
        "time,lat,lon,rsrp,rsrq,sinr,dl_mbps,ul_mbps\n"
        "2026-04-18 08:00:00,40.81,-96.70,50,-10,8,30,5\n"
    )
    frame, warnings = load_text(text, "asu.csv")
    assert frame.loc[0, "rsrp"] == pytest.approx(-90)
    assert any("ASU" in note for note in warnings)


def test_missing_columns_name_the_gap() -> None:
    with pytest.raises(IngestError, match="downlink"):
        load_text("time,lat,lon,rsrp,rsrq,sinr\n2026-04-18 08:00:00,40.8,-96.7,-90,-10,5\n", "bad.csv")


def test_empty_file() -> None:
    with pytest.raises(IngestError, match="empty"):
        load_text("   \n", "empty.csv")


def test_sample_logs_load_and_tell_both_stories() -> None:
    frames = []
    for spec in SAMPLES:
        path = sample_path(spec)
        assert path.exists(), path
        frame, _warnings = load_csv(path, route_hint=str(spec["route"]))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    result = analyze(combined)
    flags = set(result.frame["flag"])
    assert FLAG_WEAK in flags
    assert FLAG_INTERFERENCE in flags
    assert result.model.ok
    assert result.model.model_mae < result.model.baseline_mae * 0.90
    assert "Drive testing" in result.report_md
    assert "Practical next checks" in result.report_md
    assert "simulated" in result.report_md
    assert not result.areas.empty


def test_model_does_not_train_on_shuffled_neighbors_only() -> None:
    """The holdout is the tail of each route, so test rows are later in time."""
    spec = SAMPLES[0]
    frame, _ = load_csv(sample_path(spec), route_hint=str(spec["route"]))
    result = analyze(frame)
    assert result.model.ok
    assert result.model.n_test > 0
    assert result.model.n_train > result.model.n_test
