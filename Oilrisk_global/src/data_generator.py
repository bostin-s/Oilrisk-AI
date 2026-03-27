"""
data_generator.py
=================
Generates a realistic 5000-row synthetic conflict event dataset
covering WORLDWIDE oil-supply-affecting conflict zones, with a focus
on the Israel–Iran theatre but also including all major oil-supply-risk
regions: Persian Gulf, Strait of Hormuz, Red Sea / Houthis, Russia–Ukraine,
Libya, Nigeria, Venezuela, Sudan, South China Sea, and more.

Each row represents one conflict event with:
  - Location (global coordinates)
  - Attacker / target country / group
  - Event type and target description
  - Casualties
  - Oil infrastructure impact
  - Derived oil_supply_risk label (LOW / MEDIUM / HIGH / CRITICAL)
  - Region and country fields for worldwide filtering
"""

import os
import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

LOCATIONS = [
    # Israel–Iran theatre
    ("Tehran",              35.6892,  51.3890, "Middle East",    "Iran"),
    ("Isfahan",             32.6546,  51.6680, "Middle East",    "Iran"),
    ("Natanz",              33.7256,  51.9261, "Middle East",    "Iran"),
    ("Bushehr",             28.9684,  50.8385, "Middle East",    "Iran"),
    ("Bandar Abbas",        27.1832,  56.2666, "Middle East",    "Iran"),
    ("Kharg Island",        29.2522,  50.3254, "Middle East",    "Iran"),
    ("Ahvaz",               31.3183,  48.6706, "Middle East",    "Iran"),
    ("Strait of Hormuz",    26.5667,  56.2500, "Middle East",    "International"),
    ("Haifa",               32.7940,  34.9896, "Middle East",    "Israel"),
    ("Tel Aviv",            32.0853,  34.7818, "Middle East",    "Israel"),
    ("Dimona",              31.0656,  35.0338, "Middle East",    "Israel"),
    # Saudi Arabia / Gulf
    ("Riyadh",              24.7136,  46.6753, "Middle East",    "Saudi Arabia"),
    ("Abqaiq",              25.9333,  49.6667, "Middle East",    "Saudi Arabia"),
    ("Ras Tanura",          26.6500,  50.1667, "Middle East",    "Saudi Arabia"),
    ("Khurais",             25.0833,  48.8333, "Middle East",    "Saudi Arabia"),
    ("Kuwait City",         29.3759,  47.9774, "Middle East",    "Kuwait"),
    ("Basra",               30.5085,  47.7804, "Middle East",    "Iraq"),
    ("Baghdad",             33.3152,  44.3661, "Middle East",    "Iraq"),
    ("Abu Dhabi",           24.4539,  54.3773, "Middle East",    "UAE"),
    ("Dubai",               25.2048,  55.2708, "Middle East",    "UAE"),
    # Red Sea / Houthi
    ("Sanaa",               15.3694,  44.1910, "Middle East",    "Yemen"),
    ("Hodeidah",            14.7978,  42.9550, "Middle East",    "Yemen"),
    ("Bab-el-Mandeb",       12.5833,  43.4167, "Red Sea",        "International"),
    ("Suez Canal",          30.5852,  32.2654, "Red Sea",        "Egypt"),
    ("Aden Gulf",           11.0000,  45.0000, "Red Sea",        "International"),
    # Russia–Ukraine
    ("Kyiv",                50.4501,  30.5234, "Europe",         "Ukraine"),
    ("Kharkiv",             49.9935,  36.2304, "Europe",         "Ukraine"),
    ("Zaporizhzhia",        47.8388,  35.1396, "Europe",         "Ukraine"),
    ("Moscow",              55.7558,  37.6173, "Europe",         "Russia"),
    ("Novorossiysk",        44.7238,  37.7685, "Europe",         "Russia"),
    ("Baltic Pipeline",     57.0000,  22.0000, "Europe",         "International"),
    # Libya
    ("Tripoli",             32.8872,  13.1913, "Africa",         "Libya"),
    ("Benghazi",            32.1194,  20.0868, "Africa",         "Libya"),
    ("Sirte Basin",         30.5000,  18.0000, "Africa",         "Libya"),
    # Nigeria
    ("Lagos",                6.5244,   3.3792, "Africa",         "Nigeria"),
    ("Port Harcourt",        4.8156,   7.0498, "Africa",         "Nigeria"),
    ("Niger Delta",          5.0000,   6.0000, "Africa",         "Nigeria"),
    # Venezuela
    ("Caracas",             10.4806, -66.9036, "Americas",       "Venezuela"),
    ("Maracaibo",            10.6312, -71.6407, "Americas",      "Venezuela"),
    ("Orinoco Belt",          8.5000, -63.5000, "Americas",      "Venezuela"),
    # India — oil infrastructure
    ("Jamnagar",            22.4700,  70.0600, "South Asia",     "India"),
    ("Mumbai Offshore",     19.0800,  71.5000, "South Asia",     "India"),
    ("Indian Ocean Route",   9.0000,  72.5000, "South Asia",     "International"),
    ("Kochi",                9.9300,  76.2700, "South Asia",     "India"),
    ("Vizag",               17.6900,  83.2200, "South Asia",     "India"),
    ("Paradip",             20.3200,  86.6000, "South Asia",     "India"),
    ("Chennai",             13.0000,  80.2800, "South Asia",     "India"),
    # South China Sea
    ("South China Sea",     12.0000, 114.0000, "Asia-Pacific",   "International"),
    ("Strait of Malacca",    2.5000, 101.5000, "Asia-Pacific",   "International"),
    ("Spratly Islands",      8.6500, 111.9200, "Asia-Pacific",   "International"),
    # Sudan / Horn of Africa
    ("Khartoum",            15.5007,  32.5599, "Africa",         "Sudan"),
    ("Port Sudan",          19.6180,  37.2164, "Africa",         "Sudan"),
    # Caucasus / Azerbaijan
    ("Baku",                40.4093,  49.8671, "Caucasus",       "Azerbaijan"),
    ("BTC Pipeline",        41.0000,  44.0000, "Caucasus",       "International"),
    # India — oil infrastructure & strategic chokepoints
    ("Mumbai",              19.0760,  72.8777, "South Asia",     "India"),
    ("Vadodara",            22.3072,  73.1812, "South Asia",     "India"),
    ("Jamnagar",            22.4707,  70.0577, "South Asia",     "India"),
    ("Vishakhapatnam",      17.6868,  83.2185, "South Asia",     "India"),
    ("Kochi",                9.9312,  76.2673, "South Asia",     "India"),
    ("Mangalore",           12.9141,  74.8560, "South Asia",     "India"),
    ("Paradip",             20.3167,  86.6000, "South Asia",     "India"),
    ("Indian Ocean Route",   8.0000,  77.0000, "South Asia",     "International"),
]

ACTORS_ATTACKER   = [
    "Israel", "Iran", "Hezbollah", "IRGC", "IDF",
    "Houthis", "Russia", "Ukraine", "Wagner_Group",
    "ISIS", "Al-Qaeda", "LNA_Libya", "MEND_Nigeria",
    "PLA_China", "Maduro_Regime", "Sudan_SAF", "RSF_Sudan",
    "US_Forces", "NATO", "Pakistan_Militants", "Naxalites"
]
ACTORS_TARGET     = [
    "Iran", "Israel", "US_Base", "Saudi_Arabia", "UAE",
    "Ukraine", "Russia", "NATO_Base", "Iraq",
    "Yemen_Rebels", "Libya_GNA", "Nigeria_Gov",
    "International_Shipping", "China_SCS", "EU_Pipeline",
    "Sudan_Rebels", "India", "India_Shipping"
]
EVENT_TYPES       = [
    "Airstrike", "Missile_Strike", "Drone_Attack",
    "Naval_Strike", "Cyber_Attack", "Ground_Assault",
    "Pipeline_Sabotage", "Mine_Attack", "Rocket_Barrage"
]
TARGET_DESCS      = [
    "Oil_Refinery", "Nuclear_Facility", "Military_Base",
    "Pipeline", "Port", "Power_Plant", "Residential",
    "Oil_Tanker", "LNG_Terminal", "Pumping_Station",
    "Offshore_Platform", "Strait_Blockade"
]
CASUALTY_CONF     = ["High", "Medium", "Low"]
GEOCODE_METHODS   = ["GPS", "Satellite", "OSINT", "Manual"]
SOURCES           = [
    "Reuters", "AP", "BBC", "Al_Jazeera", "IDF_Spokesperson",
    "IRNA", "Times_of_Israel", "Guardian", "TASS", "Kyiv_Post",
    "UN_OCHA", "EIA_Report"
]

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ── Risk labelling logic ─────────────────────────────────────────────────────

def _assign_risk(oil_hit: int, cas_min: float, event: str, target: str,
                 region: str) -> str:
    """
    Derives an oil_supply_risk label using a scoring heuristic that
    mirrors real-world risk factors including regional multipliers.
    """
    score = 0
    if oil_hit == 1:
        score += 2
    if cas_min > 30:
        score += 3
    elif cas_min > 15:
        score += 2
    elif cas_min > 5:
        score += 1
    if event in ("Missile_Strike", "Airstrike", "Naval_Strike"):
        score += 1
    if event in ("Pipeline_Sabotage", "Mine_Attack"):
        score += 2
    if event in ("Ground_Assault", "Rocket_Barrage"):
        score += 1
    if target in ("Oil_Refinery", "Oil_Tanker", "LNG_Terminal","Pumping_Station", "Offshore_Platform"):
        score += 3
    if target == "Pipeline":
        score += 1
    if target in ("Strait_Blockade", "Port"):
        score += 2
    if target == "Nuclear_Facility":
        score += 1
    # Regional chokepoint multiplier
    if region in ("Red Sea", "Middle East") and ("Strait" in target or "Port" in target):
        score += 2
    if region == "South Asia" and oil_hit == 1:
        score += 1  # India downstream risk from Hormuz closure
    if score >= 8:
        return "CRITICAL"
    elif score >= 5:
        return "HIGH"
    elif score >= 2:
        return "MEDIUM"
    return "LOW"


# ── Main generator ───────────────────────────────────────────────────────────

def generate_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate and return a synthetic global conflict-events DataFrame.
    """
    np.random.seed(seed)

    # Sample locations
    loc_idx    = np.random.randint(0, len(LOCATIONS), n)
    loc_names  = [LOCATIONS[i][0] for i in loc_idx]
    lats       = np.array([LOCATIONS[i][1] for i in loc_idx]) + np.random.uniform(-0.5, 0.5, n)
    lons       = np.array([LOCATIONS[i][2] for i in loc_idx]) + np.random.uniform(-0.5, 0.5, n)
    regions    = [LOCATIONS[i][3] for i in loc_idx]
    countries  = [LOCATIONS[i][4] for i in loc_idx]

    # Sample events
    event_arr    = np.random.choice(EVENT_TYPES,    n)
    attacker_arr = np.random.choice(ACTORS_ATTACKER, n)
    target_arr   = np.random.choice(ACTORS_TARGET,   n)
    tdesc_arr    = np.random.choice(TARGET_DESCS,    n)
    conf_arr     = np.random.choice(CASUALTY_CONF,   n, p=[0.40, 0.40, 0.20])
    geo_arr      = np.random.choice(GEOCODE_METHODS, n)
    src_arr      = np.random.choice(SOURCES,         n)

    # Oil infrastructure flag
    oil_hit = np.where(
        np.isin(tdesc_arr, ["Oil_Refinery", "Pipeline", "Port", "Oil_Tanker",
                             "LNG_Terminal", "Pumping_Station", "Offshore_Platform",
                             "Strait_Blockade"]), 1, 0
    )

    # Casualties
    cas_min = np.random.randint(0, 50, n).astype(float)
    cas_max = (cas_min + np.random.randint(0, 30, n)).astype(float)

    # Dates (2023-04-01 onwards, one event every ~2 h)
    dates = pd.date_range("2023-04-01", periods=n, freq="2h")

    # Risk labels
    risk_labels = [
        _assign_risk(oil_hit[i], cas_min[i], event_arr[i], tdesc_arr[i], regions[i])
        for i in range(n)
    ]

    df = pd.DataFrame({
        "id":                        range(1001, 1001 + n),
        "date_utc":                  dates.date,
        "time_utc":                  dates.time,
        "location_name":             loc_names,
        "region":                    regions,
        "country":                   countries,
        "latitude":                  np.round(lats, 4),
        "longitude":                 np.round(lons, 4),
        "actor_attacker":            attacker_arr,
        "actor_target":              target_arr,
        "event_type":                event_arr,
        "target_description":        tdesc_arr,
        "reported_casualties_min":   cas_min,
        "reported_casualties_max":   cas_max,
        "casualty_confidence":       conf_arr,
        "geocode_method":            geo_arr,
        "sources":                   src_arr,
        "oil_infrastructure_hit":    oil_hit,
        "oil_supply_risk":           risk_labels,
    })

    # Inject ~3% missing values in casualty columns for realism
    for col in ["reported_casualties_min", "reported_casualties_max", "casualty_confidence"]:
        mask = np.random.rand(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


def save_dataset(df: pd.DataFrame, out_dir: str = "data") -> str:
    """Save generated dataset CSV and return the file path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "Global_Oil_Risk_dataset.csv")
    df.to_csv(path, index=False)
    print(f"[data_generator] Dataset saved → {path}  ({len(df)} rows)")
    return path


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_dataset()
    save_dataset(df)
    print(df.head())
    print("\nClass distribution:")
    print(df["oil_supply_risk"].value_counts())
    print("\nRegion distribution:")
    print(df["region"].value_counts())
