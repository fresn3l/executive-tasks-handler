"""Build the offline Lincoln sample logs.

The street geometry in data/routes comes from OpenStreetMap driving paths
(retrieved once through the public OSRM service). The radio values are
simulated. They are not a carrier drive test and not surveyed cell sites.

Run: python -m lte_analyzer.generate
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lte_analyzer.analyze import analyze
from lte_analyzer.catalog import ROUTES_DIR, SAMPLES, SAMPLES_DIR, sample_path
from lte_analyzer.geo import haversine_m, resample_lonlat

# ref is RSRP at 100 m before the site eirp offset. exp is the slope per decade
# of distance beyond 100 m. Open ground (the interstate) falls off slower than
# downtown streets.
ENV = {
    "urban": {"ref": -74.0, "exp": 35.0},
    "suburban": {"ref": -68.0, "exp": 32.0},
    "open": {"ref": -58.0, "exp": 26.0},
}

# Chosen so a clean -90 dBm cell lands near 16 dB SINR, not on the UE's 30 dB ceiling.
NOISE_DBM = -106.0
HYSTERESIS_DB = 4.0
MASTER_SEED = 7

# Extra co-channel energy along a fraction of a route: (start, end, dB below serving).
# This is the overshoot / reuse pattern the two-site geometry only produces in a thin ribbon.
INTERFERENCE_SECTORS: dict[str, list[tuple[float, float, float]]] = {
    "haymarket": [(0.18, 0.36, 1.0)],
    "west_o": [(0.64, 0.82, 0.5)],
    "i80": [(0.26, 0.46, 0.0), (0.58, 0.76, 1.0)],
}


@dataclass
class Site:
    name: str
    lat: float
    lon: float
    pci: int
    earfcn: int
    band: str
    rat: str
    bw_mhz: float
    eirp_db: float
    mimo: float
    env: str


def _point(lats: np.ndarray, lons: np.ndarray, frac: float, north_m: float = 120.0) -> tuple[float, float]:
    idx = int(round(frac * (len(lats) - 1)))
    idx = min(max(idx, 0), len(lats) - 1)
    # Shift the site off the roadway so the trace is not sitting on the antenna.
    return float(lats[idx] + north_m / 111_320.0), float(lons[idx])


def _sites(key: str, lats: np.ndarray, lons: np.ndarray) -> list[Site]:
    if key == "haymarket":
        a = _point(lats, lons, 0.06)
        b = _point(lats, lons, 0.42)
        c = _point(lats, lons, 0.58, north_m=40)
        d = _point(lats, lons, 0.78)
        return [
            Site("Haymarket macro", *a, 101, 675, "B2", "LTE", 20, 1.5, 1.5, "urban"),
            Site("Campus macro", *b, 204, 675, "B2", "LTE", 20, 2.0, 1.6, "urban"),
            Site("O Street small cell", *c, 501, 646666, "n77", "NR", 40, -1.0, 2.0, "urban"),
            Site("Capitol macro", *d, 88, 66786, "B66", "LTE", 10, 1.0, 1.3, "urban"),
        ]
    if key == "west_o":
        a = _point(lats, lons, 0.40)
        b = _point(lats, lons, 0.62)
        c = _point(lats, lons, 0.80)
        return [
            Site("West O low band", *a, 318, 5035, "B12", "LTE", 10, 6.0, 1.3, "suburban"),
            Site("Midtown B2 west", *b, 156, 900, "B2", "LTE", 20, 1.0, 1.5, "suburban"),
            Site("Midtown B2 east", *c, 159, 900, "B2", "LTE", 20, 1.0, 1.5, "suburban"),
        ]
    if key == "hwy2":
        start = _point(lats, lons, 0.08)
        mid = _point(lats, lons, 0.30)
        # A weak low-band site east of the park, so the woods are a hole rather than a total outage.
        park_lat, park_lon = 40.7862, -96.7243
        distances = haversine_m(lats, lons, park_lat, park_lon)
        near = int(np.argmin(distances))
        weak = (float(lats[near]), float(lons[near] + 1_300.0 / 84_300.0))
        north = _point(lats, lons, 0.93)
        return [
            Site("SouthPointe", *start, 240, 2100, "B4", "LTE", 15, 2.0, 1.4, "suburban"),
            Site("Highway 2", *mid, 244, 2250, "B4", "LTE", 15, 1.0, 1.4, "suburban"),
            Site("Wilderness edge", *weak, 271, 68686, "B71", "LTE", 10, 0.0, 1.2, "suburban"),
            Site("Near South", *north, 280, 66786, "B66", "LTE", 10, 1.0, 1.3, "suburban"),
        ]
    if key == "i80":
        fracs = (0.14, 0.36, 0.60, 0.84)
        pcis = (410, 411, 418, 419)
        # Two co-channel pairs. Each pair overlaps between its sites; the pairs do not share an EARFCN.
        earfcns = (5030, 5030, 5100, 5100)
        sites = []
        for frac, pci, earfcn in zip(fracs, pcis, earfcns, strict=True):
            lat, lon = _point(lats, lons, frac, north_m=180)
            sites.append(Site(f"I-80 PCI {pci}", lat, lon, pci, earfcn, "B2", "LTE", 20, 5.0, 1.8, "open"))
        return sites
    raise KeyError(key)


def _ar1(n: int, sigma: float, rho: float, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(n)
    out[0] = rng.normal(0.0, sigma)
    scale = sigma * math.sqrt(max(1.0 - rho * rho, 1e-6))
    for i in range(1, n):
        out[i] = rho * out[i - 1] + rng.normal(0.0, scale)
    return out


def _receiver_clutter(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Extra loss at the receiver. Wilderness Park is the deliberate hole."""
    extra = np.zeros(len(lats))
    park = haversine_m(lats, lons, 40.7862, -96.7243)
    park_mask = park < 900
    extra[park_mask] += 14.0 * (1.0 - park[park_mask] / 900.0)
    capitol = haversine_m(lats, lons, 40.8080, -96.6997)
    cap_mask = capitol < 220
    extra[cap_mask] += 5.0 * (1.0 - capitol[cap_mask] / 220.0)
    return extra


def _rsrp_from_site(site: Site, lats: np.ndarray, lons: np.ndarray, shadow: np.ndarray, clutter: np.ndarray) -> np.ndarray:
    dist = np.maximum(haversine_m(lats, lons, site.lat, site.lon), 30.0)
    env = ENV[site.env]
    rsrp = env["ref"] + site.eirp_db - env["exp"] * np.log10(dist / 100.0) - clutter + shadow
    return rsrp


def _serving(rsrp: np.ndarray) -> np.ndarray:
    n, _k = rsrp.shape
    serv = int(np.argmax(rsrp[0]))
    chosen = np.empty(n, dtype=int)
    for i in range(n):
        best = int(np.argmax(rsrp[i]))
        if rsrp[i, best] > rsrp[i, serv] + HYSTERESIS_DB:
            serv = best
        chosen[i] = serv
    return chosen


def _sinr(rsrp: np.ndarray, serving: np.ndarray, sites: list[Site]) -> np.ndarray:
    mw = 10.0 ** (rsrp / 10.0)
    noise = 10.0 ** (NOISE_DBM / 10.0)
    earfcn = np.array([site.earfcn for site in sites])
    rat = np.array([site.rat for site in sites])
    n = len(serving)
    sinr = np.empty(n)
    for i in range(n):
        s = int(serving[i])
        mask = (earfcn == earfcn[s]) & (rat == rat[s])
        mask[s] = False
        interf = float(mw[i, mask].sum()) if mask.any() else 0.0
        sinr[i] = 10.0 * math.log10(mw[i, s] / (interf + noise))
    return sinr


def _apply_sectors(key: str, sinr: np.ndarray, serv_rsrp: np.ndarray) -> np.ndarray:
    """Pour extra interference into a few stretches so overlap is a patch, not a pinprick."""
    out = sinr.copy()
    n = len(sinr)
    for start, end, below_db in INTERFERENCE_SECTORS.get(key, []):
        i0 = int(start * n)
        i1 = max(int(end * n), i0 + 1)
        span = i1 - i0
        for i in range(i0, i1):
            edge = min(i - i0, i1 - 1 - i) / span
            weight = float(np.clip(edge / 0.15, 0.0, 1.0))
            if weight <= 0:
                continue
            serv_mw = 10.0 ** (serv_rsrp[i] / 10.0)
            extra_mw = serv_mw * (10.0 ** (-below_db / 10.0)) * weight
            unwanted = serv_mw / (10.0 ** (sinr[i] / 10.0))
            out[i] = 10.0 * math.log10(serv_mw / (unwanted + extra_mw))
    return out


def _throughput(rsrp: np.ndarray, sinr: np.ndarray, sites: list[Site], serving: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    sinr_c = np.clip(sinr, -10, 28)
    se = np.log2(1.0 + 10.0 ** (sinr_c / 10.0))
    se = np.minimum(se, 6.2)
    bw = np.array([sites[s].bw_mhz for s in serving])
    mimo = np.array([sites[s].mimo for s in serving])
    layers = np.where(sinr >= 12, mimo, np.where(sinr >= 4, np.minimum(mimo, 1.6), 1.0))
    coverage = 1.0 / (1.0 + np.exp(-(rsrp + 112.0) / 1.6))
    dl = bw * se * 0.68 * layers * coverage * rng.lognormal(0.0, 0.12, size=len(rsrp))
    ceiling = np.array([260.0 if sites[s].rat == "NR" else 140.0 for s in serving])
    dl = np.clip(dl, 0.05, ceiling)
    ul_cov = 1.0 / (1.0 + np.exp(-(rsrp + 105.0) / 3.0))
    ul = np.clip(dl * 0.2 * ul_cov * rng.lognormal(0.0, 0.14, size=len(rsrp)), 0.02, None)
    return dl, ul


def _rsrq(sinr: np.ndarray, lats: np.ndarray, lons: np.ndarray, campus: tuple[float, float] | None, rng: np.random.Generator) -> np.ndarray:
    rsrq = -6.8 - 0.50 * (14.0 - sinr)
    if campus is not None:
        dist = haversine_m(lats, lons, campus[0], campus[1])
        penalty = (dist < 280) & (sinr > 2)
        rsrq = rsrq - np.where(penalty, 10.0, 0.0)
    rsrq = rsrq + rng.normal(0.0, 0.65, size=len(sinr))
    return np.clip(rsrq, -20, -4)


def simulate(spec: dict[str, object], rng: np.random.Generator) -> pd.DataFrame:
    geometry = json.loads((ROUTES_DIR / str(spec["geometry"])).read_text())
    samples = resample_lonlat(geometry, float(spec["step_m"]))
    lons = np.array([p[0] for p in samples])
    lats = np.array([p[1] for p in samples])
    sites = _sites(str(spec["key"]), lats, lons)
    clutter = _receiver_clutter(lats, lons)
    columns = []
    for site in sites:
        sigma = {"urban": 5.4, "suburban": 4.6, "open": 3.6}[site.env]
        columns.append(_rsrp_from_site(site, lats, lons, _ar1(len(lats), sigma, 0.93, rng), clutter))
    rsrp_matrix = np.column_stack(columns)
    serving = _serving(rsrp_matrix)
    sinr_true = _sinr(rsrp_matrix, serving, sites)
    serv_rsrp = rsrp_matrix[np.arange(len(lats)), serving]
    sinr_true = _apply_sectors(str(spec["key"]), sinr_true, serv_rsrp)
    dl, ul = _throughput(serv_rsrp, sinr_true, sites, serving, rng)

    rsrp_rep = np.clip(serv_rsrp + rng.normal(0.0, 0.7, size=len(lats)), -140, -55)
    sinr_rep = np.clip(sinr_true + rng.normal(0.0, 0.55, size=len(lats)), -15, 30)
    campus = None
    if spec["key"] == "haymarket":
        campus = (sites[1].lat, sites[1].lon)
    rsrq_rep = _rsrq(sinr_rep, lats, lons, campus, rng)

    speed = np.clip(float(spec["speed_mph"]) + rng.normal(0.0, 2.5, size=len(lats)), 4, None)
    step_m = float(spec["step_m"])
    dt = step_m / (speed * 0.44704)
    clock = pd.Timestamp(str(spec["start"])) + pd.to_timedelta(np.cumsum(dt), unit="s")

    chosen = [sites[i] for i in serving]
    frame = pd.DataFrame(
        {
            "timestamp": clock,
            "lon": np.round(lons, 6),
            "lat": np.round(lats, 6),
            "speed_mph": np.round(speed).astype(int),
            "accuracy_m": rng.integers(4, 12, size=len(lats)),
            "operator": "SampleNet",
            "mcc": 999,
            "mnc": 99,
            "tac": 14021,
            "cell_id": [site.pci * 256 + 17 for site in chosen],
            "rat": [site.rat for site in chosen],
            "band": [site.band for site in chosen],
            "earfcn": [site.earfcn for site in chosen],
            "pci": [site.pci for site in chosen],
            "bw_mhz": [site.bw_mhz for site in chosen],
            "rsrp": np.round(rsrp_rep, 1),
            "rsrq": np.round(rsrq_rep, 1),
            "sinr": np.round(sinr_rep, 1),
            "dl_kbps": np.round(dl * 1000).astype(int),
            "ul_kbps": np.round(ul * 1000).astype(int),
            "route": spec["route"],
        }
    )
    return frame


def _write_gnet(frame: pd.DataFrame, path) -> None:
    export = pd.DataFrame(
        {
            "Timestamp": frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Longitude": frame["lon"],
            "Latitude": frame["lat"],
            "Speed_mph": frame["speed_mph"],
            "Accuracy_m": frame["accuracy_m"],
            "Operatorname": frame["operator"],
            "MCC": frame["mcc"],
            "MNC": frame["mnc"],
            "LAC": frame["tac"],
            "CID": frame["cell_id"],
            "RAT": frame["rat"],
            "Band": frame["band"],
            "ARFCN": frame["earfcn"],
            "PCI": frame["pci"],
            "BW_MHz": frame["bw_mhz"],
            "RSRP": frame["rsrp"],
            "RSRQ": frame["rsrq"],
            "SINR": frame["sinr"],
            "DL_bitrate": frame["dl_kbps"],
            "UL_bitrate": frame["ul_kbps"],
            "Route": frame["route"],
        }
    )
    export.to_csv(path, index=False)


def _write_nci(frame: pd.DataFrame, path) -> None:
    export = pd.DataFrame(
        {
            "time": frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "latitude": frame["lat"],
            "longitude": frame["lon"],
            "accuracy": frame["accuracy_m"],
            "network_type": frame["rat"],
            "pci": frame["pci"],
            "earfcn": frame["earfcn"],
            "band": frame["band"],
            "bw_mhz": frame["bw_mhz"],
            "rsrp": frame["rsrp"],
            "rsrq": frame["rsrq"],
            "rssnr": frame["sinr"],
            "download_kbps": frame["dl_kbps"],
            "upload_kbps": frame["ul_kbps"],
            "Route": frame["route"],
        }
    )
    export.to_csv(path, index=False)


def write_samples() -> list[pd.DataFrame]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(MASTER_SEED)
    frames = []
    for spec in SAMPLES:
        frame = simulate(spec, rng)
        path = sample_path(spec)
        if spec["style"] == "nci":
            _write_nci(frame, path)
        else:
            _write_gnet(frame, path)
        frames.append(frame)
        print(
            f"{spec['filename']}: n={len(frame)} "
            f"rsrp_med={frame['rsrp'].median():.1f} sinr_med={frame['sinr'].median():.1f} "
            f"dl_med={frame['dl_kbps'].median() / 1000:.1f}"
        )
    return frames


def _qc(frames: list[pd.DataFrame]) -> None:
    from lte_analyzer.ingest import load_csv

    loaded = []
    for spec, _frame in zip(SAMPLES, frames, strict=True):
        parsed, warnings = load_csv(sample_path(spec), route_hint=str(spec["route"]))
        loaded.append(parsed)
        if warnings:
            print(" warnings:", "; ".join(warnings))
    combined = pd.concat(loaded, ignore_index=True)
    result = analyze(combined)
    counts = result.frame["flag"].value_counts()
    print("flags:", counts.to_dict())
    print("areas:", len(result.areas))
    if not result.areas.empty:
        print(result.areas[["area", "n", "median_rsrp", "median_sinr", "median_dl"]].to_string(index=False))
    print("model ok:", result.model.ok, "mae", result.model.model_mae, "base", result.model.baseline_mae, "r2", result.model.model_r2)
    usable = result.frame
    print(
        "corr sinr",
        round(usable["sinr"].corr(usable["dl_mbps"]), 3),
        "rsrp",
        round(usable["rsrp"].corr(usable["dl_mbps"]), 3),
    )


def main() -> None:
    frames = write_samples()
    _qc(frames)


if __name__ == "__main__":
    main()
