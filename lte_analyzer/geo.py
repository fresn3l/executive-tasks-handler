"""Distance and polyline helpers. No network access."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

EARTH_M = 6_371_000.0


def haversine_m(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Great-circle distance in meters. Inputs are degrees and may be arrays."""
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def resample_lonlat(coords: list[list[float]], step_m: float) -> list[tuple[float, float]]:
    """Walk a [lon, lat] polyline and emit points about `step_m` apart.

    Gaps in the source line are filled along the straight chord. That is
    accurate enough for highway segments a few hundred meters long.
    """
    if len(coords) == 0:
        return []
    if len(coords) == 1:
        return [(float(coords[0][0]), float(coords[0][1]))]

    lons = np.array([float(c[0]) for c in coords], dtype=float)
    lats = np.array([float(c[1]) for c in coords], dtype=float)
    out: list[tuple[float, float]] = [(lons[0], lats[0])]
    carry = 0.0
    step = float(step_m)
    for i in range(len(coords) - 1):
        seg = float(haversine_m(lats[i], lons[i], lats[i + 1], lons[i + 1]))
        if seg < 0.5:
            continue
        dist_into = step - carry
        while dist_into <= seg:
            t = dist_into / seg
            out.append(
                (
                    lons[i] + t * (lons[i + 1] - lons[i]),
                    lats[i] + t * (lats[i + 1] - lats[i]),
                )
            )
            dist_into += step
        carry = seg - (dist_into - step)
    end = (float(lons[-1]), float(lats[-1]))
    if haversine_m(out[-1][1], out[-1][0], end[1], end[0]) > step * 0.25:
        out.append(end)
    return out


def path_length_km(df: pd.DataFrame, max_jump_m: float = 2_000.0) -> float:
    """Sum consecutive spacing inside each route, ignoring teleports."""
    total = 0.0
    if df.empty or "route" not in df.columns:
        return 0.0
    for _, group in df.groupby("route", sort=False):
        ordered = group.sort_values("timestamp")
        lat = ordered["lat"].to_numpy()
        lon = ordered["lon"].to_numpy()
        if len(lat) < 2:
            continue
        steps = np.asarray(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]), dtype=float)
        total += float(steps[steps <= max_jump_m].sum())
    return total / 1000.0


def compass(bearing_deg: float) -> str:
    names = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    idx = int((bearing_deg + 22.5) // 45) % 8
    return names[idx]


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
