import argparse
from pathlib import Path
import pandas as pd
import re

"""
Convert PeMS Station 5-Minute text files + Station Metadata into a raw CSV
with columns: timestamp, lat, lon, speed_kmh, flow_vpm.

Assumptions:
- Station 5-Minute files are comma-separated without header.
- First 6 tokens: timestamp, station_id, district, freeway, direction, type (ML/OR/FF/FR).
- Subsequent lane groups repeat as: samples, flow(veh/5-min), occupancy(% as fraction), speed(mph), status.
- We aggregate lane flow and speed using flow-weighted average; convert speed to km/h; flow is converted to vehicles/minute (divide by 5).
- Station Metadata is tab-separated with header containing Latitude/Longitude.

Notes:
- Lines with missing values are handled conservatively; only lanes with numeric flow and speed are used in aggregation.
- If no lane values are present, we attempt to use aggregate tokens (if available) at positions around index 7..11.
"""

MPH_TO_KMH = 1.60934


def read_metadata(meta_dir: Path) -> pd.DataFrame:
    files = sorted(meta_dir.glob('*.txt'))
    if not files:
        raise FileNotFoundError(f"No metadata .txt files found in {meta_dir}")
    dfs = []
    for p in files:
        df = pd.read_csv(p, sep='\t')
        dfs.append(df)
    meta = pd.concat(dfs, ignore_index=True)
    # Deduplicate by ID keeping first non-null lat/lon
    meta = meta.sort_values(['ID']).drop_duplicates(subset=['ID'], keep='first')
    meta = meta[['ID', 'Latitude', 'Longitude', 'Type', 'Lanes']]
    meta = meta.rename(columns={'ID': 'station_id', 'Latitude': 'lat', 'Longitude': 'lon'})
    return meta


def parse_station_row(tokens):
    # tokens: list of strings split by comma
    if len(tokens) < 12:
        return None
    ts = tokens[0].strip()
    try:
        station_id = int(tokens[1])
    except Exception:
        return None
    lane_tokens = tokens[12:]
    total_flow_5min = 0.0
    weighted_speed_mph = 0.0
    lanes_used = 0
    # Parse lane groups sized by 5
    for i in range(0, len(lane_tokens), 5):
        group = lane_tokens[i:i+5]
        if len(group) < 5:
            break
        samples, flow, occ, speed, status = group
        # Filter empty entries
        try:
            f = float(flow) if flow not in (None, '',) else None
            s = float(speed) if speed not in (None, '',) else None
        except Exception:
            f, s = None, None
        if f is not None and s is not None:
            total_flow_5min += f
            weighted_speed_mph += f * s
            lanes_used += 1
    if total_flow_5min > 0:
        avg_speed_mph = weighted_speed_mph / total_flow_5min
        # convert
        speed_kmh = avg_speed_mph * MPH_TO_KMH
        flow_vpm = total_flow_5min / 5.0
        return ts, station_id, speed_kmh, flow_vpm
    # Fallback: try aggregate fields around index positions 9 (flow), 11 (speed)
    try:
        flow_agg = float(tokens[9])
        speed_agg_mph = float(tokens[11])
        speed_kmh = speed_agg_mph * MPH_TO_KMH
        flow_vpm = flow_agg / 5.0
        return ts, station_id, speed_kmh, flow_vpm
    except Exception:
        return None


def read_station_files(station_dir: Path, limit_files: int = 0) -> pd.DataFrame:
    files = sorted(station_dir.glob('*.txt'))
    if limit_files and limit_files > 0:
        files = files[:limit_files]
    rows = []
    for p in files:
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip('\n')
                if not line:
                    continue
                tokens = line.split(',')
                parsed = parse_station_row(tokens)
                if parsed is not None:
                    ts, sid, speed_kmh, flow_vpm = parsed
                    rows.append({'timestamp': ts, 'station_id': sid, 'speed_kmh': speed_kmh, 'flow_vpm': flow_vpm})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Parsed station dataset is empty; check file format")
    return df


def main():
    ap = argparse.ArgumentParser(description='Convert PeMS Station 5-Minute + Metadata to raw_traffic.csv')
    ap.add_argument('--station_dir', type=str, required=True, help='Directory with Station 5-Minute .txt files')
    ap.add_argument('--metadata_dir', type=str, required=True, help='Directory with Station Metadata .txt files')
    ap.add_argument('--out_csv', type=str, default='D:/PGT/my_project/external/pems_station_5min/converted/raw_traffic.csv', help='Output raw traffic CSV path')
    ap.add_argument('--use_ml_only', action='store_true', help='Filter to Mainline (ML) stations only if desired')
    ap.add_argument('--limit_files', type=int, default=0, help='Limit number of station files to process (0 means all)')
    args = ap.parse_args()

    station_dir = Path(args.station_dir)
    metadata_dir = Path(args.metadata_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print('Reading metadata ...')
    meta = read_metadata(metadata_dir)
    print(f'Metadata rows: {len(meta)}')

    print('Reading station 5-minute files ...')
    df = read_station_files(station_dir, limit_files=args.limit_files)
    print(f'Station rows: {len(df)}')

    # Join lat/lon
    df = df.merge(meta[['station_id','lat','lon','Type']], on='station_id', how='left')
    if args.use_ml_only:
        df = df[df['Type'] == 'ML']
    df = df.dropna(subset=['lat','lon'])

    # Reorder columns
    df = df[['timestamp','lat','lon','speed_kmh','flow_vpm','station_id']]

    print('Writing raw_traffic.csv ...')
    df.to_csv(out_csv, index=False)
    print(f'Output -> {out_csv}')
    print('Next: run data_adapter.py to build project files.')

if __name__ == '__main__':
    main()