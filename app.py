"""Drive-test KPI desk. Streamlit UI over the lte_analyzer package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from lte_analyzer.analyze import Analysis, analyze
from lte_analyzer.catalog import SAMPLES, list_inbox, sample_path
from lte_analyzer.charts import (
    coefficient_bars,
    coverage_map,
    flag_map,
    histogram,
    model_bars,
    pred_vs_actual,
    scatter_kpi,
)
from lte_analyzer.flags import (
    FLAG_INTERFERENCE,
    FLAG_MARGINAL,
    FLAG_WEAK,
    POOR_RSRQ_DB,
    POOR_SINR_DB,
    USABLE_RSRP_DBM,
    WEAK_RSRP_DBM,
)
from lte_analyzer.ingest import IngestError, combine, load_bytes, load_csv

st.set_page_config(
    page_title="Drive-test KPI desk",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.1rem; padding-bottom: 2.4rem; max-width: 1180px;}
    div[data-testid="stMetric"] {
        background: #121a2b;
        border: 1px solid #243044;
        padding: 0.65rem 0.75rem;
        border-radius: 12px;
    }
    div[data-testid="stCaption"] {color: #b6c3d6;}
    @media (max-width: 700px) {
        .block-container {padding-left: 0.7rem; padding-right: 0.7rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _empty_state() -> None:
    st.subheader("No log is loaded")
    st.write(
        "This desk reads a phone drive-test export and separates weak coverage from interference. "
        "It does not connect to a network, and it does not need an account."
    )
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**Use the Lincoln samples**  \n"
            "Four routes are already in `data/samples`. Select them in the sidebar. "
            "They follow real streets. The radio numbers are simulated, so you can click around with no file of your own."
        )
    with right:
        st.markdown(
            "**Or bring a CSV**  \n"
            "Upload one in the sidebar, or drop it in `data/inbox` and refresh. "
            "G-NetTrack and Network Cell Info headers both work. A semicolon-separated file works too."
        )
    with st.expander("Columns the desk looks for"):
        st.markdown(
            "- Time: `Timestamp`, `time`, or `datetime`\n"
            "- Position: `Latitude` and `Longitude` (either order)\n"
            "- Radio: `RSRP`, `RSRQ`, and `SINR` (also `SNR` or `RSSNR`)\n"
            "- Speed of the pipe: `DL_bitrate` in kbps, or `download_kbps`, or a column already in Mbps\n"
            "- Uplink is optional. If the header says kbps, values are converted to Mbps."
        )


def _collect() -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Load whatever the sidebar selected. Returns frame, warnings, errors."""
    warnings: list[str] = []
    errors: list[str] = []
    frames: list[pd.DataFrame] = []

    chosen = st.session_state.get("sample_routes", [spec["route"] for spec in SAMPLES])
    for spec in SAMPLES:
        if spec["route"] not in chosen:
            continue
        path = sample_path(spec)
        if not path.exists():
            errors.append(f"{spec['filename']} is missing from data/samples.")
            continue
        try:
            frame, notes = load_csv(path, route_hint=str(spec["route"]))
        except IngestError as exc:
            errors.append(str(exc))
            continue
        frames.append(frame)
        warnings.extend(notes)

    if st.session_state.get("use_inbox", True):
        for path in list_inbox():
            try:
                frame, notes = load_csv(path)
            except IngestError as exc:
                errors.append(str(exc))
                continue
            frames.append(frame)
            warnings.extend(notes)

    uploads = st.session_state.get("uploads") or []
    for upload in uploads:
        try:
            frame, notes = load_bytes(upload.getvalue(), upload.name, route_hint=Path(upload.name).stem)
        except IngestError as exc:
            errors.append(str(exc))
            continue
        frames.append(frame)
        warnings.extend(notes)

    if not frames:
        return None, warnings, errors
    return combine(frames), warnings, errors


def _metrics(result: Analysis) -> None:
    frame = result.frame
    counts = frame["flag"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", f"{len(frame):,}")
    c2.metric("Median RSRP", f"{frame['rsrp'].median():.0f} dBm")
    c3.metric("Median SINR", f"{frame['sinr'].median():.1f} dB")
    c4, c5, c6 = st.columns(3)
    c4.metric("Median downlink", f"{frame['dl_mbps'].median():.0f} Mbps")
    c5.metric("Weak coverage", f"{int(counts.get(FLAG_WEAK, 0)):,}")
    c6.metric("Interference", f"{int(counts.get(FLAG_INTERFERENCE, 0)):,}")


def _overview(result: Analysis) -> None:
    frame = result.frame
    t0, t1 = frame["timestamp"].min(), frame["timestamp"].max()
    routes = ", ".join(sorted(frame["route"].unique()))
    sample = ""
    if "operator" in frame.columns and set(frame["operator"].dropna().astype(str).unique()) <= {"SampleNet"}:
        sample = (
            " This is the bundled Lincoln sample: real streets, simulated radio, operator SampleNet. "
            "It is not a carrier measurement."
        )
    st.write(
        f"This selection is **{len(frame):,} samples** on {routes}, "
        f"from {t0:%b %d %H:%M} to {t1:%b %d %H:%M}. "
        "RSRP says how loud the serving cell is. RSRQ and SINR say whether that loudness is usable. "
        "Downlink is what the pipe did, which is not the same question."
        + sample
    )
    _metrics(result)
    st.markdown("##### Where the minutes went")
    percentiles = pd.DataFrame(
        {
            "P10": frame[["rsrp", "rsrq", "sinr", "dl_mbps", "ul_mbps"]].quantile(0.1),
            "Median": frame[["rsrp", "rsrq", "sinr", "dl_mbps", "ul_mbps"]].median(),
            "P90": frame[["rsrp", "rsrq", "sinr", "dl_mbps", "ul_mbps"]].quantile(0.9),
        }
    )
    percentiles.index = ["RSRP (dBm)", "RSRQ (dB)", "SINR (dB)", "Downlink (Mbps)", "Uplink (Mbps)"]
    st.dataframe(percentiles.style.format("{:.1f}"), width="stretch")
    if result.areas.empty:
        st.info("No cluster of four or more flagged samples. The map is the right place to see isolated dips.")
    else:
        st.markdown("##### Patches worth a second pass")
        focus = result.areas[result.areas["flag"].isin([FLAG_WEAK, FLAG_INTERFERENCE])]
        show = focus if not focus.empty else result.areas
        for _, row in show.head(3).iterrows():
            st.markdown(
                f"**{row['area']}** — {int(row['n'])} samples, "
                f"RSRP {row['median_rsrp']:.0f} dBm, SINR {row['median_sinr']:.1f} dB, "
                f"downlink {row['median_dl']:.0f} Mbps."
            )


def _coverage(result: Analysis) -> None:
    st.write(
        "Each dot is one sample, colored by the KPI you pick. Red is the poor end and green is the healthy end. "
        "This is the drive, not a raster of the whole city: streets you did not survey stay blank."
    )
    kpi = st.radio(
        "Coverage KPI",
        options=["rsrp", "rsrq", "sinr"],
        format_func=lambda key: {"rsrp": "RSRP", "rsrq": "RSRQ", "sinr": "SINR"}[key],
        horizontal=True,
    )
    st.plotly_chart(coverage_map(result.frame, kpi), width="stretch", config={"scrollZoom": True, "displayModeBar": False})
    st.caption("OpenStreetMap tiles draw the streets. Zoom with the scroll wheel. Hover a point for the rest of the row.")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.plotly_chart(histogram(result.frame, "rsrp"), width="stretch", config={"displayModeBar": False})
    with h2:
        st.plotly_chart(histogram(result.frame, "rsrq"), width="stretch", config={"displayModeBar": False})
    with h3:
        st.plotly_chart(histogram(result.frame, "sinr"), width="stretch", config={"displayModeBar": False})
    st.caption(
        f"Dashed lines are the desk rules: weak RSRP below {WEAK_RSRP_DBM:.0f} dBm, "
        f"usable RSRP at {USABLE_RSRP_DBM:.0f} dBm, poor RSRQ below {POOR_RSRQ_DB:.0f} dB, "
        f"poor SINR below {POOR_SINR_DB:.0f} dB."
    )


def _flags(result: Analysis) -> None:
    frame = result.frame
    counts = frame["flag"].value_counts()
    st.write(
        "Weak coverage means the serving cell is too quiet. Interference means the cell is loud enough "
        "and the link is still dirty. Marginal RSRP sits between those bars and is not called interference."
    )
    st.caption("Red is weak coverage. Amber is interference. Purple is marginal RSRP. Gray is inside the rules.")
    st.plotly_chart(flag_map(frame), width="stretch", config={"scrollZoom": True, "displayModeBar": False})
    c1, c2, c3 = st.columns(3)
    c1.metric("Weak coverage", int(counts.get(FLAG_WEAK, 0)))
    c2.metric("Interference", int(counts.get(FLAG_INTERFERENCE, 0)))
    c3.metric("Marginal RSRP", int(counts.get(FLAG_MARGINAL, 0)))

    st.markdown("##### Patches, not single fades")
    if result.areas.empty:
        st.info("Nothing clustered. Four nearby samples of the same flag are required before a patch is named.")
    else:
        view = result.areas[
            ["area", "flag_label", "n", "median_rsrp", "median_rsrq", "median_sinr", "median_dl", "next_check"]
        ].rename(
            columns={
                "area": "Patch",
                "flag_label": "Flag",
                "n": "Samples",
                "median_rsrp": "RSRP",
                "median_rsrq": "RSRQ",
                "median_sinr": "SINR",
                "median_dl": "DL Mbps",
                "next_check": "Next check",
            }
        )
        st.dataframe(
            view,
            width="stretch",
            hide_index=True,
            column_config={
                "RSRP": st.column_config.NumberColumn(format="%.0f"),
                "RSRQ": st.column_config.NumberColumn(format="%.1f"),
                "SINR": st.column_config.NumberColumn(format="%.1f"),
                "DL Mbps": st.column_config.NumberColumn(format="%.0f"),
                "Next check": st.column_config.TextColumn(width="large"),
            },
        )

    flagged = frame[frame["flag"].isin([FLAG_WEAK, FLAG_INTERFERENCE, FLAG_MARGINAL])]
    st.markdown("##### Flagged samples")
    if flagged.empty:
        st.info("No row tripped a flag.")
        return
    detail_cols = ["timestamp", "route", "lat", "lon", "flag_label", "rsrp", "rsrq", "sinr", "dl_mbps", "ul_mbps", "reason"]
    if "pci" in flagged.columns:
        detail_cols.insert(5, "pci")
    detail = flagged[detail_cols].rename(
        columns={
            "timestamp": "Time",
            "route": "Route",
            "lat": "Lat",
            "lon": "Lon",
            "flag_label": "Flag",
            "pci": "PCI",
            "rsrp": "RSRP",
            "rsrq": "RSRQ",
            "sinr": "SINR",
            "dl_mbps": "DL Mbps",
            "ul_mbps": "UL Mbps",
            "reason": "Why",
        }
    )
    st.dataframe(detail, width="stretch", hide_index=True, height=360)
    st.download_button(
        "Download flagged rows",
        data=detail.to_csv(index=False).encode("utf-8"),
        file_name="flagged_samples.csv",
        mime="text/csv",
    )


def _throughput(result: Analysis) -> None:
    frame = result.frame
    st.write(
        "The pale line on each chart is the median downlink inside a bin. "
        "It is easier to read than the cloud, and it is still just this drive."
    )
    st.plotly_chart(
        scatter_kpi(frame, "rsrp", "sinr", "Downlink against RSRP, colored by SINR"),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        scatter_kpi(frame, "sinr", "rsrp", "Downlink against SINR, colored by RSRP"),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.markdown("##### What the cloud is saying")
    st.write(result.relation)


def _model(result: Analysis) -> None:
    model = result.model
    st.write(
        "One model, on purpose. Linear regression of downlink Mbps on RSRP, RSRQ, and SINR, "
        "plus channel bandwidth when the log has it. The baseline guesses the training-set average for every test minute."
    )
    if not model.ok:
        st.warning(model.reason)
        return
    st.caption(model.split_note)
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline MAE", f"{model.baseline_mae:.1f} Mbps")
    c2.metric("Linear MAE", f"{model.model_mae:.1f} Mbps")
    c3.metric("Test R²", f"{model.model_r2:.2f}")
    table = pd.DataFrame(
        [
            {"Fit": "Guess the training average", "MAE": model.baseline_mae, "RMSE": model.baseline_rmse, "R²": model.baseline_r2},
            {"Fit": "Linear regression", "MAE": model.model_mae, "RMSE": model.model_rmse, "R²": model.model_r2},
        ]
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "RMSE": st.column_config.NumberColumn(format="%.2f"),
            "R²": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(model_bars(model.baseline_mae, model.model_mae), width="stretch", config={"displayModeBar": False})
    with right:
        st.plotly_chart(coefficient_bars(model.coefficients), width="stretch", config={"displayModeBar": False})
    st.plotly_chart(pred_vs_actual(model.y_test, model.pred_test), width="stretch", config={"displayModeBar": False})
    coef_table = pd.DataFrame(model.coefficients)[["feature", "unit", "coef", "per_std"]]
    coef_table.columns = ["KPI", "Unit", "Mbps per unit", "Mbps per 1 SD"]
    st.dataframe(
        coef_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Mbps per unit": st.column_config.NumberColumn(format="%.2f"),
            "Mbps per 1 SD": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(f"Intercept: {model.intercept:.1f} Mbps. Training R²: {model.train_r2:.2f}. Test rows: {model.n_test}. Training rows: {model.n_train}.")
    st.markdown("##### How to read the score")
    st.write(model.note)


def _report(result: Analysis) -> None:
    st.caption("Written from the log on screen. Download it if you want the same words in a file.")
    st.markdown(result.report_md)
    st.download_button(
        "Download report",
        data=result.report_md.encode("utf-8"),
        file_name="drive_test_report.md",
        mime="text/markdown",
    )


def main() -> None:
    st.title("Drive-test KPI desk")
    st.caption(
        "LTE and 5G phone logs, read for coverage, interference, and whether the radio KPIs explain downlink speed."
    )

    with st.sidebar:
        st.header("Logs")
        st.caption(
            "The Lincoln files are simulated radio on real streets, so the desk runs offline. "
            "Your own export replaces them the moment you clear the sample list."
        )
        st.multiselect(
            "Lincoln sample routes",
            options=[str(spec["route"]) for spec in SAMPLES],
            default=[str(spec["route"]) for spec in SAMPLES],
            key="sample_routes",
            help="Clear every route to see the empty state, or to review an upload on its own.",
        )
        for spec in SAMPLES:
            st.caption(f"{spec['route']}: {spec['blurb']}")
        st.checkbox("Include CSVs in data/inbox", value=True, key="use_inbox")
        st.file_uploader(
            "Upload a drive-test CSV",
            type=["csv"],
            accept_multiple_files=True,
            key="uploads",
            help="Headers from G-NetTrack or Network Cell Info. UTF-8.",
        )
        st.divider()
        st.markdown("**Flag rules**")
        st.caption(f"Weak: RSRP below {WEAK_RSRP_DBM:.0f} dBm.")
        st.caption(
            f"Interference: RSRP at or above {USABLE_RSRP_DBM:.0f} dBm, "
            f"with SINR below {POOR_SINR_DB:.0f} dB or RSRQ below {POOR_RSRQ_DB:.0f} dB."
        )
        st.caption(f"Marginal: RSRP from {WEAK_RSRP_DBM:.0f} up to {USABLE_RSRP_DBM:.0f} dBm.")
        st.caption("These are field rules of thumb, not a 3GPP limit and not a carrier spec.")

    try:
        with st.spinner("Reading the log, flagging patches, and fitting the throughput line…"):
            frame, warnings, errors = _collect()
    except Exception as exc:  # unexpected parse or IO failure, shown as an error state
        st.error("The log could not be analyzed. The file may be truncated or the columns may not be numeric.")
        with st.expander("What went wrong"):
            st.code(f"{type(exc).__name__}: {exc}")
        return

    for message in errors:
        st.error(message)
    for message in warnings:
        st.warning(message)

    if frame is None:
        if not errors:
            _empty_state()
        else:
            st.info("Nothing usable was loaded. Fix the file named above, or turn the Lincoln samples back on.")
        return

    try:
        with st.spinner("Reading the log, flagging patches, and fitting the throughput line…"):
            result = analyze(frame)
    except Exception as exc:
        st.error("The samples loaded, but the analysis stopped. Check for a column that is entirely blank.")
        with st.expander("What went wrong"):
            st.code(f"{type(exc).__name__}: {exc}")
        return

    overview, coverage, flags, throughput, model, report = st.tabs(
        ["Overview", "Coverage", "Flags", "Throughput", "Model", "Report"]
    )
    with overview:
        _overview(result)
    with coverage:
        _coverage(result)
    with flags:
        _flags(result)
    with throughput:
        _throughput(result)
    with model:
        _model(result)
    with report:
        _report(result)


if __name__ == "__main__":
    main()
else:
    main()
