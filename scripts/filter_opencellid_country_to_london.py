
import os, glob, gzip

import pandas as pd

# Accept both .csv.gz and .csv
PATTERNS = [r"data/opencellid_uk_*.csv.gz", r"data/opencellid_uk_*.csv"]
OUT_PATH = r"data/opencellid_london.csv"

# Greater London bbox (approx)
LAT_MIN, LAT_MAX = 51.28, 51.70
LON_MIN, LON_MAX = -0.5103, 0.334

CHUNKSIZE = 200_000

# Canonical column set used by OpenCelliD exports (some trailing cols may be missing)
COLS = [
    "radio","mcc","mnc","lac","cellid","unit","lon","lat","range","samples",
    "changeable","created","updated","averageSignalStrength"
]

REN = {
    "Latitude":"lat","LAT":"lat","latitude":"lat",
    "Longitude":"lon","LON":"lon","longitude":"lon"
}

def find_files():
    files = []
    for pat in PATTERNS:
        files.extend(glob.glob(pat))
    if not files:
        files = glob.glob(r"data/*.csv*")
    return sorted(files)

def detect_delim(path):
    """Return ',' or ';' based on first line."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        line = f.readline()
    # Count delimiters
    return ";" if line.count(";") > line.count(",") else ","

def read_headerless_chunks(path, sep):
    """Yield chunks reading as headerless with fixed column names."""
    # Use the fast C engine; headerless + fixed names avoids the previous engine issue
    return pd.read_csv(
        path,
        header=None,
        names=COLS,
        sep=sep,
        engine="c",
        compression="infer",
        chunksize=CHUNKSIZE
    )

def read_with_header_chunks(path, sep):
    """Yield chunks for a file that actually has headers with lat/lon column names."""
    return pd.read_csv(
        path,
        sep=sep,
        engine="c",
        compression="infer",
        chunksize=CHUNKSIZE
    )

def main():
    files = find_files()
    if not files:
        raise SystemExit("No UK CSV files found in /data. Put your downloads like data/opencellid_uk_234.csv.gz")

    # fresh output
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    print("Merging & filtering:", files)
    first = True
    total_out = 0

    for path in files:
        sep = detect_delim(path)
        # Peek the first row quickly to decide headerless vs headered
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()

        looks_headerless = False
        # Heuristic: headerless starts with a radio string like GSM/UMTS/LTE/NR or a number
        if first_line:
            token0 = first_line.split(sep)[0].strip().upper()
            looks_headerless = token0 in {"GSM","UMTS","LTE","NR","NBIOT","CDMA"} or token0.isdigit()

        print(f"\nFile: {path} | sep='{sep}' | headerless={looks_headerless}")

        if looks_headerless:
            chunk_iter = read_headerless_chunks(path, sep)
        else:
            chunk_iter = read_with_header_chunks(path, sep)

        for i, chunk in enumerate(chunk_iter, 1):
            # Normalize potential alternate lat/lon names if any header existed
            chunk = chunk.rename(columns=REN)

            # Ensure lat/lon present
            if "lat" not in chunk.columns or "lon" not in chunk.columns:
                # For headerless case, our fixed COLS include 'lat'/'lon'
                # but if the file had weird columns, skip this chunk
                continue

            # Coerce numerics
            for c in ["lat","lon","averageSignalStrength","samples","range","pci","tac","mcc","mnc","lac"]:
                if c in chunk.columns:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

            # Drop bad coords & clip to London bbox
            chunk = chunk.dropna(subset=["lat","lon"])
            chunk = chunk[chunk["lat"].between(LAT_MIN, LAT_MAX) & chunk["lon"].between(LON_MIN, LON_MAX)]
            if chunk.empty:
                continue

            # Append to output
            mode = "w" if first else "a"
            chunk.to_csv(OUT_PATH, index=False, mode=mode, header=first)
            first = False
            total_out += len(chunk)

            if i <= 2:
                # print a small sample to show it’s working
                print(f"  [chunk {i}] kept {len(chunk)} rows; lat range {chunk['lat'].min():.4f}-{chunk['lat'].max():.4f}")

    if total_out == 0:
        print("\nNo rows matched the London bbox. Double-check the files are UK and the bbox values.")
        print("Tip: UK MCCs are typically 234 and 235. Files like 730/748 belong to other countries.")
        return

    # Optional: de-duplicate
    df = pd.read_csv(OUT_PATH)
    before = len(df)
    subset = [c for c in ["radio","mcc","mnc","lac","cellid","tac","pci","lat","lon"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df):,} towers to {OUT_PATH} (deduped from {before:,})")
    print(df[["lat","lon"]].describe())

if __name__ == "__main__":
    main()
