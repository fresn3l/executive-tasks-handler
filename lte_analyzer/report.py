"""Troubleshooting report rendered from the loaded log."""

from __future__ import annotations

import pandas as pd

from lte_analyzer.flags import (
    FLAG_INTERFERENCE,
    FLAG_MARGINAL,
    FLAG_WEAK,
    POOR_RSRQ_DB,
    POOR_SINR_DB,
    USABLE_RSRP_DBM,
    WEAK_RSRP_DBM,
)
from lte_analyzer.geo import path_length_km
from lte_analyzer.model import ModelResult


def _place_phrase(frame: pd.DataFrame) -> str:
    lat = float(frame["lat"].median())
    lon = float(frame["lon"].median())
    if 40.65 <= lat <= 41.05 and -96.95 <= lon <= -96.45:
        return "Lincoln, Nebraska"
    return f"around {lat:.3f} N, {lon:.3f} E"


def _pct_table(frame: pd.DataFrame) -> str:
    rows = []
    specs = [
        ("RSRP (dBm)", "rsrp"),
        ("RSRQ (dB)", "rsrq"),
        ("SINR (dB)", "sinr"),
        ("Downlink (Mbps)", "dl_mbps"),
        ("Uplink (Mbps)", "ul_mbps"),
    ]
    for label, column in specs:
        if column not in frame or frame[column].notna().sum() == 0:
            continue
        series = frame[column].dropna()
        qs = series.quantile([0.1, 0.5, 0.9])
        rows.append(
            f"| {label} | {series.min():.1f} | {qs.loc[0.1]:.1f} | {qs.loc[0.5]:.1f} | {qs.loc[0.9]:.1f} | {series.max():.1f} |"
        )
    header = "| KPI | Min | P10 | Median | P90 | Max |\n| --- | ---: | ---: | ---: | ---: | ---: |"
    return header + "\n" + "\n".join(rows)


def _area_block(row: pd.Series) -> str:
    pci_line = ""
    if pd.notna(row.get("top_pci")):
        pci_line = f" Most common serving PCI: {int(row['top_pci'])}."
        if pd.notna(row.get("second_pci")) and float(row["second_share"]) >= 0.25:
            pci_line += (
                f" PCI {int(row['second_pci'])} is also common "
                f"({float(row['second_share']):.0%} of the patch). "
                "The phone is changing serving cells inside a small area, which fits overlap or a short handover loop."
            )
        elif row["flag"] == FLAG_INTERFERENCE:
            pci_line += " One PCI dominates, so this looks less like a handover flap and more like a neighbor the phone can hear but will not name."
    return (
        f"### {row['area']}\n\n"
        f"{int(row['n'])} samples on {row['route']}. "
        f"Median RSRP {row['median_rsrp']:.0f} dBm, RSRQ {row['median_rsrq']:.1f} dB, "
        f"SINR {row['median_sinr']:.1f} dB, downlink {row['median_dl']:.0f} Mbps."
        f"{pci_line}\n\n"
        f"Next check: {row['next_check']}\n"
    )


def _sample_notice(frame: pd.DataFrame) -> str:
    if "operator" not in frame.columns:
        return ""
    operators = {str(value) for value in frame["operator"].dropna().unique()}
    if operators and operators <= {"SampleNet"}:
        return (
            "The operator on this file is SampleNet, the bundled Lincoln sample. "
            "The streets are real driving paths. RSRP, RSRQ, SINR, throughput, and the cell identities are simulated. "
            "Do not treat a colored stretch as measured coverage from a carrier."
        )
    return ""


def _kpi_read(frame: pd.DataFrame) -> str:
    med_rsrp = float(frame["rsrp"].median())
    med_sinr = float(frame["sinr"].median())
    med_rsrq = float(frame["rsrq"].median())
    bits = []
    if med_rsrp < WEAK_RSRP_DBM:
        bits.append(
            f"The median RSRP is {med_rsrp:.0f} dBm, which is already in weak coverage. "
            "The drive spent its typical minute below the level this desk treats as a reliable outdoor signal."
        )
    elif med_rsrp < USABLE_RSRP_DBM:
        bits.append(
            f"The median RSRP is {med_rsrp:.0f} dBm. That is marginal on this desk: audible, not comfortable."
        )
    else:
        bits.append(
            f"The median RSRP is {med_rsrp:.0f} dBm, on the usable side of {USABLE_RSRP_DBM:.0f} dBm. "
            "A coverage problem, if there is one, is a hole rather than the whole route."
        )
    if med_sinr < POOR_SINR_DB or med_rsrq < POOR_RSRQ_DB:
        bits.append(
            f"Median SINR is {med_sinr:.1f} dB and median RSRQ is {med_rsrq:.1f} dB. "
            "The typical sample is already on the dirty side of the interference rule, so look at overlap and load before you hunt for a single dead site."
        )
    else:
        bits.append(
            f"Median SINR is {med_sinr:.1f} dB and median RSRQ is {med_rsrq:.1f} dB. "
            "The typical sample is clean enough. Dirty samples are the exception, which is why the map matters more than the median."
        )
    return " ".join(bits)


def build_report(
    frame: pd.DataFrame,
    areas: pd.DataFrame,
    relation: str,
    model: ModelResult,
) -> str:
    place = _place_phrase(frame)
    t0 = frame["timestamp"].min()
    t1 = frame["timestamp"].max()
    routes = sorted(frame["route"].dropna().unique())
    km = path_length_km(frame)
    n = len(frame)
    counts = frame["flag"].value_counts()
    n_weak = int(counts.get(FLAG_WEAK, 0))
    n_interf = int(counts.get(FLAG_INTERFERENCE, 0))
    n_marg = int(counts.get(FLAG_MARGINAL, 0))

    rats = ""
    if "rat" in frame.columns and frame["rat"].notna().any():
        mix = frame["rat"].value_counts(normalize=True)
        rats = " Radio access on the log: " + ", ".join(f"{idx} {share:.0%}" for idx, share in mix.items()) + "."
    bands = ""
    if "band" in frame.columns and frame["band"].notna().any():
        listed = ", ".join(str(b) for b in frame["band"].dropna().unique())
        bands = f" Bands seen: {listed}."

    lines = [
        f"# Drive-test troubleshooting report",
        "",
        f"This note is computed from the log currently loaded. It describes {place}. "
        "It is a field read of these samples, not a citywide coverage claim and not a forecast.",
        "",
        _sample_notice(frame),
        "## Drive testing",
        "",
        "A drive test is a receiver that moves and writes down where it was. "
        "Each row here is one sample: time, latitude, longitude, RSRP, RSRQ, SINR, and throughput. "
        "RSRP is how loud the serving cell is. RSRQ and SINR are how clean that loudness is. "
        "Throughput is what the pipe did at that second, which also depends on load, bandwidth, and whether the test was actually filling the link.",
        "",
        f"The loaded selection has **{n} samples** across **{len(routes)} route(s)** "
        f"({', '.join(routes)}), from {t0:%Y-%m-%d %H:%M} to {t1:%Y-%m-%d %H:%M}. "
        f"The path length, ignoring gaps over 2 km, is about **{km:.1f} km**.{rats}{bands}",
        "",
        "Phone exports from G-NetTrack and Network Cell Info are enough for this desk. "
        "They are not enough to name an interferer. They rarely include neighbor RSRP, PRB utilization, or a locked band. "
        "When the flag says interference, the next instrument is a scanner or a controlled re-drive, not another copy of the same CSV.",
        "",
        "Collect a log you want to trust in the open, with the phone in the same vehicle position each pass, "
        "about one sample a second, and write down whether the speed test was running. A parked phone at a red light is still useful; "
        "just do not read a cluster of identical points as extra evidence.",
        "",
        "## KPI analysis",
        "",
        _kpi_read(frame),
        "",
        "Rules used to flag this log:",
        "",
        f"- Weak coverage: RSRP below {WEAK_RSRP_DBM:.0f} dBm.",
        f"- Interference: RSRP at or above {USABLE_RSRP_DBM:.0f} dBm, and SINR below {POOR_SINR_DB:.0f} dB or RSRQ below {POOR_RSRQ_DB:.0f} dB.",
        f"- Marginal: RSRP from {WEAK_RSRP_DBM:.0f} dBm up to {USABLE_RSRP_DBM:.0f} dBm. Not called interference.",
        "",
        "Counts: "
        f"**{n_weak}** weak ({n_weak / n:.0%}), "
        f"**{n_interf}** interference ({n_interf / n:.0%}), "
        f"**{n_marg}** marginal ({n_marg / n:.0%}), "
        f"the rest OK.",
        "",
        _pct_table(frame),
        "",
        "P10 is a better 'bad minute' than the minimum. One sample behind a truck will set the minimum and should not set the work plan.",
        "",
        "## SINR and throughput",
        "",
        relation,
        "",
        "## Flagged areas",
        "",
    ]

    if areas.empty:
        lines.append(
            "No patch of at least four nearby flagged samples turned up. "
            "Either this drive is clean under the rules above, or the bad samples are scattered single points. "
            "Scattered points are worth a glance on the map and usually are not worth a site visit."
        )
    else:
        lines.append(
            f"{len(areas)} patch(es) have at least four nearby flagged samples. "
            "Isolated points are left off this list so a single deep fade does not look like a project."
        )
        lines.append("")
        # Weak and interference first; marginal is context.
        focus = areas[areas["flag"].isin([FLAG_WEAK, FLAG_INTERFERENCE])]
        rest = areas[areas["flag"] == FLAG_MARGINAL]
        show = focus if not focus.empty else areas
        for _, row in show.iterrows():
            lines.append(_area_block(row))
        if not rest.empty and not focus.empty:
            lines.append(
                f"{len(rest)} more patch(es) are only marginal RSRP. "
                "They are on the map and in the table. They are a watch item, not the first truck roll."
            )

    lines.extend(
        [
            "",
            "## Throughput model",
            "",
        ]
    )
    if not model.ok:
        lines.append(model.reason)
    else:
        lines.append(model.split_note)
        lines.append("")
        lines.append(model.note)
        lines.append("")
        lines.append(
            f"Test size {model.n_test}, training size {model.n_train}. "
            f"Baseline MAE {model.baseline_mae:.1f} Mbps, RMSE {model.baseline_rmse:.1f}, R² {model.baseline_r2:.2f}. "
            f"Linear regression MAE {model.model_mae:.1f} Mbps, RMSE {model.model_rmse:.1f}, R² {model.model_r2:.2f}."
        )

    lines.extend(
        [
            "",
            "## Practical next checks",
            "",
            "Work the list from the radio evidence, not from the model score.",
            "",
        ]
    )
    steps = []
    if n_weak:
        steps.append(
            f"1. Walk the weak patches ({n_weak} samples). Confirm the serving PCI on site, then check low-band neighbors and antenna azimuth before you assume a hardware fault. Trees, underpasses, and a phone on the seat instead of the dash all imitate a coverage hole."
        )
    else:
        steps.append("1. No weak-coverage patch met the bar. Do not open a coverage ticket on this log alone.")
    if n_interf:
        steps.append(
            f"2. For the interference patches ({n_interf} samples), compare serving PCI mod-3 with whatever neighbors a scanner can see. Re-test the worst patch at a quiet hour. If it cleans up, you are looking at load. If it stays dirty while RSRP stays healthy, you are looking at overlap."
        )
    else:
        steps.append("2. No interference patch met the bar. Poor speed, if you still have it, is more likely load, backhaul, or the test itself.")
    steps.append(
        "3. Where SINR is healthy and the speed test is still poor, stop looking at antennas. Check that the test was saturating the link, which band and bandwidth the phone actually held, and whether the cell was busy."
    )
    steps.append(
        "4. Treat the linear model as a second opinion on this file only. If a future drive's error is far worse than the number above, the city or the band changed and the old fit should be thrown out, not tuned."
    )
    lines.extend(steps)
    lines.append("")
    return "\n".join(lines)
