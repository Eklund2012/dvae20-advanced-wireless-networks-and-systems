import numpy as np
import pandas as pd

# Set reproducible seed
np.random.seed(42)

# 1. Define nodes and geographic distribution
n_nodes = 12
nodes = []
zones = ["Industrial", "Commercial/Traffic", "Residential/Park"]

for i in range(1, n_nodes + 1):
  zone = zones[(i - 1) % 3]
  nodes.append({
      "node_id": f"NODE_{i:03d}",
      "zone": zone,
      "lat": round(59.37 + np.random.uniform(-0.03, 0.03), 4),
      "lon": round(13.50 + np.random.uniform(-0.03, 0.03), 4),
      # Aging node baseline voltages (nodes 11 and 12 simulated as low battery)
      "base_v": 2.92 if i >= 11 else round(np.random.uniform(3.25, 3.55), 2),
  })

# 2. Generate 48-hour time series (hourly interval)
timestamps = pd.date_range(
    start="2026-10-01 00:00", end="2026-10-02 23:00", freq="1h"
)
records = []

for t in timestamps:
  hr = t.hour
  # Rush hour traffic multipliers: 07:00-09:00 and 16:00-19:00
  is_peak = (7 <= hr <= 9) or (16 <= hr <= 19)
  traffic_mult = 1.65 if is_peak else (0.65 if 0 <= hr <= 5 else 1.0)

  for n in nodes:
    zone = n["zone"]
    zone_mult = (
        1.35
        if zone == "Industrial"
        else (1.20 if zone == "Commercial/Traffic" else 0.75)
    )

    # Environmental KPIs
    pm25 = max(4.0, np.random.normal(16 * traffic_mult * zone_mult, 2.8))
    pm10 = max(pm25 * 1.3, pm25 * np.random.uniform(1.4, 1.9))
    noise_base = 66 if (7 <= hr <= 21) else 48
    noise = max(
        38.0,
        np.random.normal(
            noise_base
            * (traffic_mult**0.5)
            * (1.1 if zone != "Residential/Park" else 0.9),
            3.5,
        ),
    )
    aqi = int(pm25 * 3.3 + np.random.uniform(-4, 4))

    # 5G Network KPIs (mMTC Slice)
    density_base = (
        320_000 if zone == "Commercial/Traffic" else (180_000 if zone == "Residential/Park" else 220_000)
    )
    conn_density = int(
        np.random.normal(density_base * (1.2 if is_peak else 0.9), 12_000)
    )

    # RF link: deeper attenuation in industrial canyons
    rsrp = int(np.random.normal(-102 if zone == "Industrial" else -91, 4))
    sinr = round(np.random.normal(8.0 if rsrp < -100 else 14.5, 2.5), 1)

    # Correlation: poor RF slightly degrades PDR and increases latency
    rf_degraded = rsrp < -104
    pdr = round(
        np.random.uniform(97.5, 98.8)
        if rf_degraded
        else np.random.uniform(99.1, 99.9),
        2,
    )
    latency = int(
        np.random.normal(280 if rf_degraded else 110, 25 if rf_degraded else 15)
    )

    # Fleet Maintenance KPIs
    voltage = round(n["base_v"] - (np.random.uniform(0.001, 0.003)), 2)
    proj_years = round(max(0.4, ((voltage - 2.8) / (3.6 - 2.8)) * 10), 1)

    # Injected anomaly: Node 007 intermittently loses connectivity on Day 2
    is_offline = (n["node_id"] == "NODE_007") and (t.day == 2 and 10 <= hr <= 15)
    uptime = 0.0 if is_offline else 100.0
    ota_stat = (
        0
        if (is_offline or voltage < 3.0)
        else (1 if np.random.rand() > 0.04 else 0)
    )

    records.append({
        "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
        "node_id": n["node_id"],
        "zone": zone,
        "latitude": n["lat"],
        "longitude": n["lon"],
        "aqi": aqi if not is_offline else np.nan,
        "pm2_5_ug_m3": round(pm25, 2) if not is_offline else np.nan,
        "pm10_ug_m3": round(pm10, 2) if not is_offline else np.nan,
        "noise_dba": round(noise, 1) if not is_offline else np.nan,
        "conn_density_km2": conn_density,
        "pdr_pct": pdr if not is_offline else 0.0,
        "uplink_latency_ms": latency if not is_offline else np.nan,
        "rsrp_dbm": rsrp,
        "sinr_db": sinr,
        "battery_voltage_v": voltage,
        "projected_life_yrs": proj_years,
        "uptime_pct": uptime,
        "ota_success": ota_stat,
    })

df = pd.DataFrame(records)
df.to_csv("urban_environmental_5g_data.csv", index=False)
print(f"Generated {len(df)} records across {n_nodes} nodes.")