"""Describe how signal strength and throughput move together on one log."""

from __future__ import annotations

import math

import pandas as pd

from lte_analyzer.flags import FLAG_INTERFERENCE, USABLE_RSRP_DBM


def _corr(frame: pd.DataFrame, column: str) -> float | None:
    pair = frame[[column, "dl_mbps"]].dropna()
    if len(pair) < 20:
        return None
    if pair[column].nunique() < 2 or pair["dl_mbps"].nunique() < 2:
        return None
    value = float(pair[column].corr(pair["dl_mbps"]))
    if math.isnan(value):
        return None
    return value


def binned_median(frame: pd.DataFrame, column: str, edges: list[float]) -> pd.DataFrame:
    work = frame[[column, "dl_mbps"]].dropna().copy()
    if work.empty:
        return pd.DataFrame(columns=["bin", "n", "median_dl"])
    work["bin"] = pd.cut(work[column], bins=edges, include_lowest=True)
    grouped = (
        work.groupby("bin", observed=True)["dl_mbps"]
        .agg(n="size", median_dl="median")
        .reset_index()
    )
    grouped["bin"] = grouped["bin"].astype(str)
    return grouped


def relationship_text(frame: pd.DataFrame) -> str:
    """Plain-language read of RSRP, SINR, and downlink speed for this log only."""
    usable = frame.dropna(subset=["rsrp", "sinr", "dl_mbps"])
    if len(usable) < 30:
        return (
            "There are not enough samples with both radio KPIs and a downlink speed "
            "to describe a relationship. A few dozen points will draw a chart; a claim needs more."
        )

    c_sinr = _corr(usable, "sinr")
    c_rsrp = _corr(usable, "rsrp")
    c_rsrq = _corr(usable, "rsrq")

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    paragraphs = [
        (
            f"On this log, downlink throughput lines up with SINR at r = {fmt(c_sinr)}, "
            f"with RSRP at r = {fmt(c_rsrp)}, and with RSRQ at r = {fmt(c_rsrq)}. "
            "These are Pearson correlations on the samples you loaded, not a general radio law."
        )
    ]

    if c_sinr is not None and c_rsrp is not None and c_sinr >= c_rsrp + 0.1:
        paragraphs.append(
            "Speed follows cleanliness more than loudness. That is what an interference-limited "
            "stretch looks like: the serving cell can still be strong while the modulation is stuck "
            "in the mud. RSRP alone would miss those points."
        )
    elif c_rsrp is not None and c_sinr is not None and c_rsrp >= c_sinr + 0.1:
        paragraphs.append(
            "Speed follows RSRP more than SINR on this particular drive. That fits a coverage-limited "
            "route, where the phone is simply far from the serving cell and interference is not the main story."
        )
    else:
        paragraphs.append(
            "RSRP and SINR move with throughput by similar amounts here, so this drive does not "
            "cleanly separate a coverage problem from an interference problem on the correlation alone. "
            "The flags further down the page use the thresholds instead of the correlation."
        )

    loud = usable[usable["rsrp"] >= USABLE_RSRP_DBM]
    if len(loud) >= 40 and loud["sinr"].nunique() > 3:
        low_cut, high_cut = loud["sinr"].quantile([0.25, 0.75])
        slow = float(loud.loc[loud["sinr"] <= low_cut, "dl_mbps"].median())
        fast = float(loud.loc[loud["sinr"] >= high_cut, "dl_mbps"].median())
        paragraphs.append(
            f"Among samples with usable RSRP (at least {USABLE_RSRP_DBM:.0f} dBm), "
            f"the dirtier quarter of SINR (at or below {low_cut:.1f} dB) has a median downlink "
            f"of {slow:.0f} Mbps. The cleaner quarter (at or above {high_cut:.1f} dB) has "
            f"{fast:.0f} Mbps. Same kind of signal level, different speed."
        )

    dirty = usable[usable["flag"] == FLAG_INTERFERENCE] if "flag" in usable.columns else usable.iloc[0:0]
    if len(dirty) >= 8:
        paragraphs.append(
            f"The interference flag marks {len(dirty)} samples. Their median downlink is "
            f"{dirty['dl_mbps'].median():.0f} Mbps, against {usable['dl_mbps'].median():.0f} Mbps "
            "for the log as a whole. If a flagged patch is also slow at a quiet hour, treat it as RF. "
            "If it is only slow when the cell is busy, RSRQ can sag from load, and this export "
            "does not include PRB utilization to tell those apart."
        )

    paragraphs.append(
        "A strong SINR with a weak speed test is a different failure: backhaul, scheduler load, "
        "the test server, or a phone that never got the carrier aggregation the site could offer. "
        "Do not retune antennas for that."
    )
    return "\n\n".join(paragraphs)
