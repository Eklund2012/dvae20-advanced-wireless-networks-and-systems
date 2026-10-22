import numpy as np
import pandas as pd

def generate_challenging_5g_dataset(
    num_samples=2000, 
    freq_minutes=5, 
    anomaly_ratio=0.10, 
    random_seed=42
):
    np.random.seed(random_seed)
    
    # 1. Timestamps
    timestamps = pd.date_range(start="2026-03-01 00:00", periods=num_samples, freq=f"{freq_minutes}min")
    
    hours = np.array(timestamps.hour + timestamps.minute / 60.0)
    day_of_week = np.array(timestamps.dayofweek)
    is_weekend = np.array((day_of_week >= 5).astype(int))
    
    # 2. Daily cycle
    peak_1 = np.exp(-((hours - 14.0) ** 2) / 12.0)
    peak_2 = np.exp(-((hours - 20.0) ** 2) / 8.0)
    night_dip = np.exp(-((hours - 4.0) ** 2) / 6.0)
    
    cycle = (peak_1 + peak_2 - 0.5 * night_dip)
    cycle = (cycle - cycle.min()) / (cycle.max() - cycle.min())
    
    # 3. Baselines with natural correlated noise
    active_ues = np.clip(100 + 400 * cycle * (1 - 0.2 * is_weekend) + np.random.normal(0, 20, num_samples), 20, 650)
    prb_usage = np.clip(15 + 65 * cycle + np.random.normal(0, 5, num_samples), 5, 95)
    throughput_mbps = np.clip(200 + 700 * (1 - cycle * 0.4) + np.random.normal(0, 45, num_samples), 50, 1200)
    latency_ms = np.clip(8 + 15 * (prb_usage / 100)**2 + np.random.normal(0, 2.5, num_samples), 4, 55)
    packet_loss_pct = np.clip(0.01 + 0.1 * (prb_usage / 100)**3 + np.random.exponential(0.03, num_samples), 0.0, 2.0)
    
    rsrp_dbm = np.clip(-85 + np.random.normal(0, 5, num_samples), -115, -60)
    sinr_db = np.clip(18 + np.random.normal(0, 4, num_samples), -2, 30)
    slice_bandwidth_gbps = np.clip(0.5 + 4.0 * (active_ues / 600) + np.random.normal(0, 0.25, num_samples), 0.1, 5.0)

    is_anomaly = np.zeros(num_samples, dtype=int)
    anomaly_type = ["normal"] * num_samples
    
    # 4. Inject Subtle / Contextual Anomalies
    total_target_anomalies = int(num_samples * anomaly_ratio)
    candidate_indices = np.random.choice(np.arange(50, num_samples - 20), size=20, replace=False)
    
    anomaly_counter = 0
    for idx in candidate_indices:
        if anomaly_counter >= total_target_anomalies:
            break
            
        atype = np.random.choice(["subtle_ddos", "slow_degradation", "off_peak_congestion", "intermittent_loss"])
        duration = np.random.randint(6, 15)  # 30-75 min bursts
        end_idx = min(idx + duration, num_samples)
        
        if atype == "subtle_ddos":
            # Latency and packet loss rise moderately—still within global max bounds, but abnormal for time of day
            latency_ms[idx:end_idx] += np.random.normal(25.0, 5.0, size=end_idx - idx)
            packet_loss_pct[idx:end_idx] += np.random.exponential(0.8, size=end_idx - idx)
            prb_usage[idx:end_idx] = np.clip(prb_usage[idx:end_idx] + 20, 0, 98)
            
        elif atype == "slow_degradation":
            # Gradual signal drop that blends into normal radio noise
            fade = np.linspace(0, -18, end_idx - idx)
            rsrp_dbm[idx:end_idx] += fade + np.random.normal(0, 2, size=end_idx - idx)
            throughput_mbps[idx:end_idx] *= np.linspace(1.0, 0.4, end_idx - idx)
            
        elif atype == "off_peak_congestion":
            # High load during late night/early morning hours (contextual anomaly)
            prb_usage[idx:end_idx] = np.clip(prb_usage[idx:end_idx] + 45, 0, 95)
            latency_ms[idx:end_idx] += np.random.normal(18.0, 4.0, size=end_idx - idx)
            
        elif atype == "intermittent_loss":
            # Short packet loss spikes without severe latency changes
            packet_loss_pct[idx:end_idx] += np.random.uniform(1.2, 3.5, size=end_idx - idx)
            
        is_anomaly[idx:end_idx] = 1
        for i in range(idx, end_idx):
            anomaly_type[i] = atype
            
        anomaly_counter += (end_idx - idx)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "active_ues": np.round(active_ues).astype(int),
        "prb_usage_pct": np.round(np.clip(prb_usage, 0, 100), 2),
        "throughput_mbps": np.round(np.clip(throughput_mbps, 0, None), 2),
        "latency_ms": np.round(np.clip(latency_ms, 0, None), 2),
        "packet_loss_pct": np.round(np.clip(packet_loss_pct, 0, 100), 4),
        "rsrp_dbm": np.round(np.clip(rsrp_dbm, -140, -40), 2),
        "sinr_db": np.round(sinr_db, 2),
        "slice_bandwidth_gbps": np.round(np.clip(slice_bandwidth_gbps, 0, None), 2),
        "is_weekend": is_weekend,
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type
    })
    
    return df

df = generate_challenging_5g_dataset(num_samples=2000, anomaly_ratio=0.10)
output_path = "/home/marcnord105/work/DVAE23/Dataset/5g_network_traffic_anomalies.csv"
df.to_csv(output_path, index=False)

print(f"Dataset saved successfully to: {output_path}")
print(f"Total rows: {len(df)}")
print(f"Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.2f}%)")
