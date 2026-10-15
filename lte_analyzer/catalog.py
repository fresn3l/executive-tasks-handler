"""Sample-log catalog and on-disk locations."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "data" / "samples"
ROUTES_DIR = ROOT / "data" / "routes"
INBOX_DIR = ROOT / "data" / "inbox"

# Landmarks used only to name clusters in plain language. Coordinates are
# public places in Lincoln, Nebraska, not cell sites.
LANDMARKS: list[tuple[str, float, float]] = [
    ("Haymarket", 40.8164, -96.7086),
    ("Pinnacle Bank Arena", 40.8179, -96.7125),
    ("Memorial Stadium", 40.8207, -96.7048),
    ("UNL City Campus", 40.8185, -96.7015),
    ("Nebraska State Capitol", 40.8080, -96.6997),
    ("Downtown O Street", 40.8136, -96.7030),
    ("West O Street", 40.8136, -96.7550),
    ("O Street west of downtown", 40.8136, -96.7280),
    ("far West O", 40.8132, -96.7950),
    ("SouthPointe", 40.7419, -96.6776),
    ("Highway 2", 40.7560, -96.7050),
    ("Wilderness Park", 40.7862, -96.7243),
    ("South Bottoms", 40.8050, -96.7080),
    ("Lincoln Airport", 40.8473, -96.7574),
    ("I-80 at I-180", 40.8625, -96.7156),
    ("I-80 north of downtown", 40.8800, -96.6900),
    ("I-80 east of 27th", 40.8966, -96.6500),
    ("I-80 toward Waverly", 40.8970, -96.5750),
]

SAMPLES: list[dict[str, object]] = [
    {
        "key": "haymarket",
        "filename": "lincoln_haymarket_unl.csv",
        "route": "Haymarket, UNL, and the Capitol",
        "geometry": "haymarket.json",
        "style": "gnet",
        "step_m": 32,
        "speed_mph": 18.0,
        "start": "2026-04-18 08:10:00",
        "blurb": "A morning loop from the arena through campus, O Street, and the capitol.",
    },
    {
        "key": "west_o",
        "filename": "lincoln_west_o.csv",
        "route": "West O Street",
        "geometry": "west_o.json",
        "style": "gnet",
        "step_m": 40,
        "speed_mph": 33.0,
        "start": "2026-04-18 09:05:00",
        "blurb": "Eastbound on West O, from the edge of town into downtown.",
    },
    {
        "key": "hwy2",
        "filename": "lincoln_hwy2_wilderness.csv",
        "route": "Highway 2 and Wilderness Park",
        "geometry": "hwy2.json",
        "style": "gnet",
        "step_m": 48,
        "speed_mph": 42.0,
        "start": "2026-04-18 09:40:00",
        "blurb": "SouthPointe to O Street, passing the wooded ground at Wilderness Park.",
    },
    {
        "key": "i80",
        "filename": "lincoln_i80.csv",
        "route": "I-80 across north Lincoln",
        "geometry": "i80.json",
        "style": "nci",
        "step_m": 70,
        "speed_mph": 70.0,
        "start": "2026-04-18 10:20:00",
        "blurb": "Interstate 80 from the airport side east toward Waverly.",
    },
]


def sample_path(spec: dict[str, object]) -> Path:
    return SAMPLES_DIR / str(spec["filename"])


def list_inbox() -> list[Path]:
    if not INBOX_DIR.exists():
        return []
    return sorted(p for p in INBOX_DIR.glob("*.csv") if p.is_file())
