# Drive-test KPI desk

A small LTE/5G drive-test reviewer for phone logs. Load a CSV, look at coverage along the route, and separate **weak coverage** (the serving cell is too quiet) from **interference** (the cell is loud enough and the link is still dirty). Charts show how RSRP and SINR sit next to downlink speed. A linear model tries to predict that speed from the radio KPIs and is scored against a naive baseline.

The desk is a local Streamlit app. No account, no database, no API key.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8747
```

Open `http://127.0.0.1:8747`. The app binds to all interfaces on port **8747**.

Tests:

```bash
pytest
```

Regenerate the sample CSVs (optional; the files in `data/samples` are already written):

```bash
python -m lte_analyzer.generate
```

## What the sample data is

Four logs in `data/samples` follow real driving paths around **Lincoln, Nebraska**:

| File | Route | Column style |
| --- | --- | --- |
| `lincoln_haymarket_unl.csv` | Haymarket, UNL campus, O Street, the capitol | G-NetTrack (`DL_bitrate` in kbps, longitude before latitude) |
| `lincoln_west_o.csv` | West O Street, from the west edge toward downtown | G-NetTrack |
| `lincoln_hwy2_wilderness.csv` | Highway 2 from SouthPointe past Wilderness Park | G-NetTrack |
| `lincoln_i80.csv` | I-80 across north Lincoln, airport side toward Waverly | Network Cell Info (`rssnr`, `download_kbps`) |

Street geometry came from OpenStreetMap driving routes. **The radio values are simulated** so the app runs with no download and no credentials. Serving cells, PCIs, and the operator name `SampleNet` (MCC 999 / MNC 99) are fictional. Do not read the samples as a measurement of any carrier in Lincoln, and do not treat a colored street as a coverage guarantee for an address you did not drive.

The shapes are still the ones the importer is meant to see: a weak stretch where RSRP falls apart (far West O, the woods at Wilderness Park), and interference where RSRP stays usable while SINR and/or RSRQ do not (co-channel overlap on the interstate and on West O, plus a campus patch where RSRQ sags).

## Add your own log

Any of these works:

1. Select nothing in the sidebar and upload a CSV.
2. Drop a `.csv` into `data/inbox` and refresh. Inbox files are included by default alongside the samples; clear the sample list if you want the inbox file alone.
3. Leave the samples selected and upload a file if you want them on one map.

The desk accepts comma or semicolon separators and a UTF-8 BOM.

### Columns

Required, under any of these header names (case and extra punctuation do not matter):

| Meaning | Names the desk recognizes |
| --- | --- |
| Time | `Timestamp`, `time`, `datetime` |
| Latitude | `Latitude`, `lat` |
| Longitude | `Longitude`, `lon`, `lng` |
| RSRP | `RSRP`, `SS-RSRP` (dBm). Values in the LTE ASU range 0–97 are converted with RSRP = ASU − 140. |
| RSRQ | `RSRQ` (dB) |
| SINR | `SINR`, `SNR`, `RSSNR` |
| Downlink | `DL_bitrate` or `download_kbps` (treated as kbps), or `dl_mbps` |
| Bandwidth (optional) | `BW_MHz`, `bandwidth`. Used by the throughput model when more than one width is present. |

Uplink (`UL_bitrate`, `upload_kbps`, `ul_mbps`) is used in the tables when it is present. A missing uplink column is a warning, not a failure.

Optional columns that show up when they exist: `PCI`, `RAT` / `network_type`, `Band`, `ARFCN` / `EARFCN`, `Route`.

G-NetTrack’s `DL_bitrate` is kilobits per second. If a downlink header does not say kbps or Mbps, values whose typical size is above 300 are treated as kbps.

## How the flags are decided

These are engineering rules of thumb, not a 3GPP pass/fail and not a carrier acceptance spec.

- **Weak coverage:** RSRP below −110 dBm. Poor SINR here is treated as noise, not as a separate interference ticket.
- **Interference:** RSRP at or above −100 dBm, and SINR below 3 dB or RSRQ below −15 dB.
- **Marginal:** RSRP from −110 dBm up to −100 dBm. Shown on the map so it is not silently called OK, and not called interference.

A “patch” in the report is at least four flagged samples of the same kind within about 350 m. One deep fade does not become a project.

## The throughput model

Downlink Mbps is fit with ordinary linear regression on RSRP, RSRQ, and SINR. If the log includes channel bandwidth and it is not all the same width, that column is a fourth input. The baseline predicts the training-set mean for every test row. The split is a **time holdout**: the last 25% of each route is the test set. A random split would leak neighboring samples and print a flattering score.

Read the MAE, RMSE, and R² on the Model tab as a description of **this file**. Training and test rows are later and earlier minutes of the same drive. The model does not know bandwidth, MIMO rank, cell load, or whether the speed test was actually filling the link. If SINR and RSRQ move together, their separate coefficients are unstable; the tab says so when that happens.

## What is in the app

- **Overview** — counts, percentiles, the first patches to look at.
- **Coverage** — RSRP, RSRQ, and SINR along the route, plus the distributions.
- **Flags** — weak vs interference on the map, a patch table, and the flagged rows.
- **Throughput** — signal against downlink, with a short read of the relationship.
- **Model** — baseline vs linear regression.
- **Report** — drive testing, KPI analysis, SINR and throughput, flagged areas, and practical next checks. You can download it.

Empty selection, a blank file, and a CSV that is missing columns each stop with an explanation instead of a stack trace. A long file shows a spinner while the log is read and the line is fit.
