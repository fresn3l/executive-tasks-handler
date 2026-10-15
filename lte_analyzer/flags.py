"""Separate weak coverage from interference.

Rules of thumb used by this desk, not a carrier acceptance test and not a
3GPP pass/fail limit:

- Weak coverage: RSRP worse than -110 dBm. The serving cell is too quiet.
  A poor SINR in that state is mostly noise.
- Interference: RSRP of -100 dBm or better, and SINR below 3 dB or RSRQ
  below -15 dB. The phone hears the cell, and the link is still dirty.
- Marginal: RSRP from -110 dBm up to but not including -100 dBm. Thin
  coverage. Not labeled interference, because the level is already below
  the bar this desk uses for a usable signal.
"""

from __future__ import annotations

import pandas as pd

from lte_analyzer.catalog import LANDMARKS
from lte_analyzer.geo import bearing_deg, compass, haversine_m

WEAK_RSRP_DBM = -110.0
USABLE_RSRP_DBM = -100.0
POOR_SINR_DB = 3.0
POOR_RSRQ_DB = -15.0

FLAG_WEAK = "weak_coverage"
FLAG_INTERFERENCE = "interference"
FLAG_MARGINAL = "marginal"
FLAG_OK = "ok"

FLAG_LABELS = {
    FLAG_WEAK: "Weak coverage",
    FLAG_INTERFERENCE: "Interference",
    FLAG_MARGINAL: "Marginal RSRP",
    FLAG_OK: "OK",
}

FLAG_ORDER = [FLAG_WEAK, FLAG_INTERFERENCE, FLAG_MARGINAL, FLAG_OK]


def classify_row(rsrp: float, rsrq: float, sinr: float) -> tuple[str, str]:
    if pd.isna(rsrp):
        return FLAG_OK, "RSRP is missing, so this row was not flagged."
    if rsrp < WEAK_RSRP_DBM:
        return (
            FLAG_WEAK,
            f"RSRP {rsrp:.0f} dBm is below {WEAK_RSRP_DBM:.0f} dBm.",
        )
    sinr_poor = pd.notna(sinr) and sinr < POOR_SINR_DB
    rsrq_poor = pd.notna(rsrq) and rsrq < POOR_RSRQ_DB
    if rsrp >= USABLE_RSRP_DBM and (sinr_poor or rsrq_poor):
        bits: list[str] = []
        if sinr_poor:
            bits.append(f"SINR {sinr:.1f} dB")
        if rsrq_poor:
            bits.append(f"RSRQ {rsrq:.1f} dB")
        joined = " and ".join(bits)
        verb = "is" if len(bits) == 1 else "are"
        return (
            FLAG_INTERFERENCE,
            f"RSRP {rsrp:.0f} dBm is usable, but {joined} {verb} poor.",
        )
    if rsrp < USABLE_RSRP_DBM:
        return (
            FLAG_MARGINAL,
            f"RSRP {rsrp:.0f} dBm is between {WEAK_RSRP_DBM:.0f} and {USABLE_RSRP_DBM:.0f} dBm.",
        )
    return FLAG_OK, "Inside the rules of thumb used here."


def classify(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with `flag` and `reason` columns."""
    out = frame.copy()
    flags: list[str] = []
    reasons: list[str] = []
    for rsrp, rsrq, sinr in zip(out["rsrp"], out["rsrq"], out["sinr"], strict=False):
        flag, reason = classify_row(float(rsrp) if pd.notna(rsrp) else float("nan"), rsrq, sinr)
        flags.append(flag)
        reasons.append(reason)
    out["flag"] = flags
    out["reason"] = reasons
    out["flag_label"] = out["flag"].map(FLAG_LABELS)
    return out


def _nearest_place(lat: float, lon: float) -> tuple[str, float]:
    best_name = "this stretch"
    best_m = float("inf")
    for name, plat, plon in LANDMARKS:
        meters = float(haversine_m(lat, lon, plat, plon))
        if meters < best_m:
            best_name = name
            best_m = meters
            best_lat, best_lon = plat, plon
    if best_m > 3_500:
        return f"{lat:.3f}, {lon:.3f}", best_m
    return best_name, best_m


def _next_check(flag: str, pci: float | None) -> str:
    pci_bit = ""
    if pci is not None and pd.notna(pci):
        pci_int = int(pci)
        pci_bit = f" Serving PCI {pci_int} (mod 3 = {pci_int % 3})."
    if flag == FLAG_WEAK:
        return (
            "Re-drive the worst 100 m slowly, outdoors, on the same phone. "
            "Check whether a low-band neighbor is missing and whether this street sits outside the serving azimuth."
            + pci_bit
        )
    if flag == FLAG_INTERFERENCE:
        return (
            "RSRP is healthy and the link is dirty. A phone CSV usually has no neighbor list, "
            "so the next measurement is a scanner (or a barred-PCI test) to name the interferer. "
            "Also compare quiet-hour and busy-hour RSRQ: load can look like interference."
            + pci_bit
        )
    if flag == FLAG_MARGINAL:
        return (
            "Coverage is thin. Confirm the phone is not camping indoors-in-a-car on a high band, "
            "then check low-band availability before calling it interference."
            + pci_bit
        )
    return "No radio flag on this patch."


def cluster_areas(frame: pd.DataFrame, radius_m: float = 350.0, min_samples: int = 4) -> pd.DataFrame:
    """Group flagged samples that are close in space into named areas.

    Points are considered in time order. A point joins the current cluster
    when it has the same flag and is within `radius_m` of that cluster's
    centroid. OK samples are not clustered.
    """
    columns = [
        "area",
        "flag",
        "flag_label",
        "n",
        "lat",
        "lon",
        "median_rsrp",
        "median_rsrq",
        "median_sinr",
        "median_dl",
        "median_ul",
        "top_pci",
        "second_pci",
        "second_share",
        "place",
        "next_check",
    ]
    flagged = frame[frame["flag"].isin([FLAG_WEAK, FLAG_INTERFERENCE, FLAG_MARGINAL])].copy()
    if flagged.empty:
        return pd.DataFrame(columns=columns)

    clusters: list[dict] = []
    for route, group in flagged.groupby("route", sort=False):
        ordered = group.sort_values("timestamp")
        current: list[int] = []
        current_flag: str | None = None

        def flush() -> None:
            if len(current) < min_samples or current_flag is None:
                current.clear()
                return
            chunk = ordered.loc[current]
            clusters.append(_summarize_cluster(chunk, current_flag))
            current.clear()

        for idx, row in ordered.iterrows():
            if not current:
                current = [idx]
                current_flag = str(row["flag"])
                continue
            same = row["flag"] == current_flag
            # Chain along the drive. A centroid test splits a long bad stretch into
            # crumbs once the patch is longer than the radius.
            last = ordered.loc[current[-1]]
            near = float(haversine_m(row["lat"], row["lon"], last["lat"], last["lon"])) <= radius_m
            if same and near:
                current.append(idx)
            else:
                flush()
                current = [idx]
                current_flag = str(row["flag"])
        flush()

    if not clusters:
        return pd.DataFrame(columns=columns)

    areas = pd.DataFrame(clusters)
    areas["area"] = _unique_area_names(areas)
    areas = areas.sort_values(
        by=["flag", "n"],
        key=lambda col: col.map({f: i for i, f in enumerate(FLAG_ORDER)}) if col.name == "flag" else col,
        ascending=[True, False],
    ).reset_index(drop=True)
    return areas


def _summarize_cluster(chunk: pd.DataFrame, flag: str) -> dict:
    lat = float(chunk["lat"].mean())
    lon = float(chunk["lon"].mean())
    place, _meters = _nearest_place(lat, lon)
    top_pci = None
    second_pci = None
    second_share = 0.0
    if "pci" in chunk.columns and chunk["pci"].notna().any():
        counts = chunk["pci"].dropna().astype(int).value_counts()
        top_pci = int(counts.index[0])
        if len(counts) > 1:
            second_pci = int(counts.index[1])
            second_share = float(counts.iloc[1] / counts.sum())
    return {
        "flag": flag,
        "flag_label": FLAG_LABELS[flag],
        "n": int(len(chunk)),
        "lat": lat,
        "lon": lon,
        "median_rsrp": float(chunk["rsrp"].median()),
        "median_rsrq": float(chunk["rsrq"].median()),
        "median_sinr": float(chunk["sinr"].median()),
        "median_dl": float(chunk["dl_mbps"].median()) if chunk["dl_mbps"].notna().any() else float("nan"),
        "median_ul": float(chunk["ul_mbps"].median()) if "ul_mbps" in chunk and chunk["ul_mbps"].notna().any() else float("nan"),
        "top_pci": top_pci,
        "second_pci": second_pci,
        "second_share": second_share,
        "place": place,
        "next_check": _next_check(flag, top_pci),
        "route": str(chunk["route"].iloc[0]),
    }


def _unique_area_names(areas: pd.DataFrame) -> list[str]:
    base = [
        f"{FLAG_LABELS[flag]} near {place}"
        for flag, place in zip(areas["flag"], areas["place"], strict=False)
    ]
    counts: dict[str, int] = {}
    for name in base:
        counts[name] = counts.get(name, 0) + 1
    seen: dict[str, int] = {}
    names: list[str] = []
    for name, (_, row) in zip(base, areas.iterrows(), strict=False):
        if counts[name] == 1:
            names.append(name)
            continue
        # Point the duplicate at the landmark's bearing so two patches can share a neighborhood.
        landmark = next((item for item in LANDMARKS if item[0] == row["place"]), None)
        if landmark is None:
            suffix = row["route"]
        else:
            direction = compass(bearing_deg(landmark[1], landmark[2], row["lat"], row["lon"]))
            suffix = f"{direction} side"
        seen[name] = seen.get(name, 0) + 1
        candidate = f"{name}, {suffix}"
        if candidate in names:
            candidate = f"{candidate} ({seen[name]})"
        names.append(candidate)
    return names
