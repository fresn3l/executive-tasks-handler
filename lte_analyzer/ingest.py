"""Read G-NetTrack and Network Cell Info style drive-test CSVs."""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

class IngestError(Exception):
    """The file is not a usable drive-test log."""


# Canonical field -> header names after normalization.
_ALIASES: dict[str, set[str]] = {
    "timestamp": {"timestamp", "time", "datetime", "date_time", "logged_at", "date"},
    "lat": {"latitude", "lat", "gps_lat"},
    "lon": {"longitude", "lon", "lng", "long", "gps_lon"},
    "rsrp": {"rsrp", "lte_rsrp", "ss_rsrp", "nr_ss_rsrp", "ssrsrp"},
    "rsrq": {"rsrq", "lte_rsrq", "ss_rsrq", "nr_ss_rsrq"},
    "sinr": {"sinr", "snr", "rssnr", "ss_sinr", "lte_sinr", "nr_ss_sinr"},
    "dl_mbps": {
        "dl_mbps",
        "download_mbps",
        "dl_throughput_mbps",
        "downlink_mbps",
        "dl_bitrate_mbps",
    },
    "dl_kbps": {
        "dl_bitrate",
        "dl_bitrate_kbps",
        "download_kbps",
        "dl_kbps",
        "downlink_kbps",
        "dl_throughput_kbps",
        "dl_throughput",
        "downlink",
        "download",
        "dl",
    },
    "ul_mbps": {
        "ul_mbps",
        "upload_mbps",
        "ul_throughput_mbps",
        "uplink_mbps",
        "ul_bitrate_mbps",
    },
    "ul_kbps": {
        "ul_bitrate",
        "ul_bitrate_kbps",
        "upload_kbps",
        "ul_kbps",
        "uplink_kbps",
        "ul_throughput_kbps",
        "ul_throughput",
        "uplink",
        "upload",
        "ul",
    },
    "pci": {"pci", "physical_cell_id", "serving_pci"},
    "rat": {"rat", "network_type", "network", "tech"},
    "band": {"band"},
    "earfcn": {"earfcn", "arfcn", "nrarfcn", "narfcn"},
    "bw_mhz": {"bw_mhz", "bandwidth_mhz", "bandwidth", "channel_bw", "dl_bw"},
    "speed_mph": {"speed_mph"},
    "speed": {"speed"},
    "cell_id": {"cellid", "cell_id", "cid", "eci", "nci"},
    "operator": {"operatorname", "operator", "carrier"},
    "route": {"route", "route_name", "logfile"},
}

_REQUIRED_RADIO = ("timestamp", "lat", "lon", "rsrp", "rsrq", "sinr")

EXPECTED_COLUMNS = (
    "timestamp, latitude, longitude, RSRP (dBm), RSRQ (dB), SINR (dB), "
    "downlink throughput, uplink throughput"
)


def _norm_header(name: str) -> str:
    text = str(name).replace("\ufeff", "").strip().lower()
    text = text.replace("(kbps)", "_kbps").replace("(mbps)", "_mbps")
    text = text.replace("[kbps]", "_kbps").replace("[mbps]", "_mbps")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def _detect_sep(sample: str) -> str:
    first = sample.splitlines()[0] if sample else ""
    return ";" if first.count(";") > first.count(",") else ","


def _map_headers(columns: list[str]) -> dict[str, str]:
    """Map original column name -> canonical field. First match wins."""
    used: set[str] = set()
    found: dict[str, str] = {}
    normalized = [(col, _norm_header(col)) for col in columns]
    for canonical, names in _ALIASES.items():
        for original, norm in normalized:
            if original in used:
                continue
            if norm in names or norm.startswith(tuple(n + "_" for n in names)):
                # Prefer an exact alias over a prefix collision.
                if norm in names:
                    found[canonical] = original
                    used.add(original)
                    break
        if canonical in found:
            continue
        for original, norm in normalized:
            if original in used:
                continue
            if any(norm == name or norm.startswith(name + "_") for name in names):
                found[canonical] = original
                used.add(original)
                break
    return found


def _series(frame: pd.DataFrame, mapping: dict[str, str], key: str) -> pd.Series | None:
    column = mapping.get(key)
    if column is None or column not in frame.columns:
        return None
    return frame[column]


def load_text(text: str, source_name: str, route_hint: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Parse one CSV document. Returns the frame and human-readable warnings."""
    if not text or not text.strip():
        raise IngestError(f"{source_name} is empty.")

    sep = _detect_sep(text)
    try:
        raw = pd.read_csv(io.StringIO(text), sep=sep)
    except pd.errors.ParserError as exc:
        raise IngestError(f"{source_name} is not a readable CSV ({exc}).") from exc

    if raw.empty or len(raw.columns) == 0:
        raise IngestError(f"{source_name} has headers but no measurement rows.")

    mapping = _map_headers([str(c) for c in raw.columns])
    missing = [name for name in _REQUIRED_RADIO if name not in mapping]
    has_dl = "dl_mbps" in mapping or "dl_kbps" in mapping
    if not has_dl:
        missing.append("downlink throughput")
    if missing:
        found = ", ".join(str(c) for c in raw.columns)
        raise IngestError(
            f"{source_name} is missing {', '.join(missing)}. "
            f"Expected {EXPECTED_COLUMNS}. Found columns: {found}."
        )

    warnings: list[str] = []
    if "ul_mbps" not in mapping and "ul_kbps" not in mapping:
        warnings.append(f"{source_name} has no uplink throughput column. Downlink is still used.")

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(_series(raw, mapping, "timestamp"), errors="coerce", format="mixed")
    out["lat"] = pd.to_numeric(_series(raw, mapping, "lat"), errors="coerce")
    out["lon"] = pd.to_numeric(_series(raw, mapping, "lon"), errors="coerce")
    out["rsrp"] = pd.to_numeric(_series(raw, mapping, "rsrp"), errors="coerce")
    out["rsrq"] = pd.to_numeric(_series(raw, mapping, "rsrq"), errors="coerce")
    out["sinr"] = pd.to_numeric(_series(raw, mapping, "sinr"), errors="coerce")

    if "dl_mbps" in mapping:
        out["dl_mbps"] = pd.to_numeric(_series(raw, mapping, "dl_mbps"), errors="coerce")
    else:
        kbps = pd.to_numeric(_series(raw, mapping, "dl_kbps"), errors="coerce")
        header = _norm_header(mapping["dl_kbps"])
        if "mbps" in header:
            out["dl_mbps"] = kbps
        elif "kbps" in header or header in {"dl_bitrate", "dl_throughput", "downlink"}:
            out["dl_mbps"] = kbps / 1000.0
        else:
            # Ambiguous magnitude: phone exports in kbps are typically hundreds or more.
            median = float(np.nanmedian(kbps.to_numpy(dtype=float))) if kbps.notna().any() else 0.0
            if median > 300:
                out["dl_mbps"] = kbps / 1000.0
                warnings.append(
                    f"{source_name}: treated {mapping['dl_kbps']} as kilobits per second "
                    "because the typical value is above 300."
                )
            else:
                out["dl_mbps"] = kbps

    if "ul_mbps" in mapping:
        out["ul_mbps"] = pd.to_numeric(_series(raw, mapping, "ul_mbps"), errors="coerce")
    elif "ul_kbps" in mapping:
        kbps = pd.to_numeric(_series(raw, mapping, "ul_kbps"), errors="coerce")
        header = _norm_header(mapping["ul_kbps"])
        if "mbps" in header:
            out["ul_mbps"] = kbps
        elif "kbps" in header or header in {"ul_bitrate", "ul_throughput", "uplink"}:
            out["ul_mbps"] = kbps / 1000.0
        else:
            median = float(np.nanmedian(kbps.to_numpy(dtype=float))) if kbps.notna().any() else 0.0
            out["ul_mbps"] = kbps / 1000.0 if median > 300 else kbps
    else:
        out["ul_mbps"] = np.nan

    optional_numeric = {
        "pci": "pci",
        "earfcn": "earfcn",
        "cell_id": "cell_id",
        "bw_mhz": "bw_mhz",
    }
    for canonical, column in optional_numeric.items():
        series = _series(raw, mapping, canonical)
        if series is not None:
            out[column] = pd.to_numeric(series, errors="coerce")

    for canonical, column in (("rat", "rat"), ("band", "band"), ("operator", "operator")):
        series = _series(raw, mapping, canonical)
        if series is not None:
            out[column] = series.astype(str).replace({"nan": np.nan})

    if "speed_mph" in mapping:
        out["speed_mph"] = pd.to_numeric(_series(raw, mapping, "speed_mph"), errors="coerce")
    elif "speed" in mapping:
        out["speed_mph"] = pd.to_numeric(_series(raw, mapping, "speed"), errors="coerce")
        warnings.append(f"{source_name}: Speed has no unit in the header, so it is shown as mph.")

    route_series = _series(raw, mapping, "route")
    if route_series is not None:
        out["route"] = route_series.astype(str)
    else:
        out["route"] = route_hint or Path(source_name).stem

    out["source_file"] = source_name

    rsrp = out["rsrp"]
    if rsrp.notna().any() and float(rsrp.median()) > 0:
        if float(rsrp.min()) >= 0 and float(rsrp.max()) <= 97:
            out["rsrp"] = rsrp - 140.0
            warnings.append(
                f"{source_name}: RSRP looked like LTE ASU (0–97), so it was converted with RSRP = ASU − 140."
            )
        else:
            raise IngestError(
                f"{source_name}: RSRP values are positive and not in the LTE ASU range 0–97. "
                "This desk expects dBm, usually between -140 and -50."
            )

    before = len(out)
    valid = (
        out["timestamp"].notna()
        & out["lat"].between(-90, 90)
        & out["lon"].between(-180, 180)
        & out["rsrp"].notna()
    )
    out = out.loc[valid].copy()
    dropped = before - len(out)
    if out.empty:
        raise IngestError(
            f"{source_name}: no rows survived parsing. Check that latitude/longitude are degrees "
            "and that the timestamp column is a date-time."
        )
    if dropped:
        warnings.append(f"{source_name}: dropped {dropped} row(s) with a bad timestamp or coordinate.")

    out = out.sort_values("timestamp").reset_index(drop=True)
    return out, warnings


def load_csv(path: Path, route_hint: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise IngestError(f"Could not read {path.name}: {exc}") from exc
    return load_text(text, path.name, route_hint=route_hint or path.stem)


def load_bytes(payload: bytes, source_name: str, route_hint: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestError(f"{source_name} is not UTF-8 text.") from exc
    return load_text(text, source_name, route_hint=route_hint)


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["timestamp", "route"]).reset_index(drop=True)
