# scripts/fetch_opencellid_london.py
import os, time, io
import requests
import pandas as pd
from dotenv import load_dotenv

# Greater London bbox (approx)
LAT_MIN, LAT_MAX = 51.28, 51.70
LON_MIN, LON_MAX = -0.5103, 0.334

# tile sizes chosen to remain well under area limits per request
LAT_STEP = 0.018   # ~2.0 km
LON_STEP = 0.028   # ~1.9 km

BASE = "https://opencellid.org/cell"
LIMIT = 50         # API max rows per page (per docs)
RADIO = "LTE"      # set to None to fetch all radios (more volume)

def bbox_tiles(lat_min, lon_min, lat_max, lon_max, dlat, dlon):
    lat = lat_min
    while lat < lat_max:
        lon = lon_min
        lat_hi = min(lat + dlat, lat_max)
        while lon < lon_max:
            lon_hi = min(lon + dlon, lon_max)
            yield (lat, lon, lat_hi, lon_hi)
            lon = lon_hi
        lat = lat_hi

def get_count(session, token, bbox, radio=None):
    params = {"key": token, "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}", "format": "json"}
    if radio:
        params["radio"] = radio
    r = session.get(f"{BASE}/getInAreaSize", params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    return int(js.get("count", 0))

def fetch_page(session, token, bbox, offset, radio=None, fmt="csv"):
    params = {
        "key": token,
        "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "limit": LIMIT,
        "offset": offset,
        "format": fmt,
    }
    if radio:
        params["radio"] = radio
    r = session.get(f"{BASE}/getInArea", params=params, timeout=60)
    r.raise_for_status()
    return r

def main():
    load_dotenv()
    token = os.getenv("OPENCELLID_TOKEN")
    if not token:
        raise SystemExit("Missing OPENCELLID_TOKEN in .env")

    os.makedirs("data", exist_ok=True)
    session = requests.Session()
    frames = []

    for bbox in bbox_tiles(LAT_MIN, LON_MIN, LAT_MAX, LON_MAX, LAT_STEP, LON_STEP):
        try:
            cnt = get_count(session, token, bbox, radio=RADIO)
        except requests.HTTPError as e:
            print("Count error:", e, "bbox=", bbox)
            continue

        if cnt == 0:
            continue

        print(f"BBOX {bbox} → {cnt} cells")
        fetched = 0
        offset = 0
        while fetched < cnt:
            try:
                resp = fetch_page(session, token, bbox, offset, radio=RADIO, fmt="csv")
            except requests.HTTPError as e:
                print("Page error:", e, "offset=", offset, "bbox=", bbox)
                break

            chunk = pd.read_csv(io.StringIO(resp.text))
            keep = [c for c in ["lat","lon","radio","mcc","mnc","lac","cellid","tac","pci",
                                "averageSignalStrength","samples","range","changeable"]
                    if c in chunk.columns]
            if keep:
                chunk = chunk[keep]
            frames.append(chunk)

            got = len(chunk)
            fetched += got
            offset += LIMIT
            time.sleep(0.3)  # polite pacing

    if not frames:
        print("No data fetched.")
        return

    towers = pd.concat(frames, ignore_index=True).drop_duplicates()
    out = "data/opencellid_london.csv"
    towers.to_csv(out, index=False)
    print(f"Saved {len(towers):,} towers to {out}")

if __name__ == "__main__":
    main()
