
import os
import pandas as pd

# London bbox
LAT_MIN, LAT_MAX = 51.28, 51.70
LON_MIN, LON_MAX = -0.5103, 0.334

S3_PATH = "ookla-open-data/parquet/performance/type=mobile/year=2024/quarter=4/2024-10-01_performance_mobile_tiles.parquet"
COLS = ["tile","tile_x","tile_y","avg_d_kbps","avg_u_kbps","avg_lat_ms","tests","devices","quadkey"]

def try_fsspec_dataset():
    # s3fs for I/O
    import s3fs
    import pyarrow.dataset as ds
    from pyarrow.fs import PyFileSystem, FSSpecHandler

    fs_s3 = s3fs.S3FileSystem(anon=True)
    pa_fs = PyFileSystem(FSSpecHandler(fs_s3))  # wrap fsspec FS for pyarrow

    dataset = ds.dataset(S3_PATH, format="parquet", filesystem=pa_fs)
    filt = (
        (ds.field("tile_y") >= LAT_MIN) & (ds.field("tile_y") <= LAT_MAX) &
        (ds.field("tile_x") >= LON_MIN) & (ds.field("tile_x") <= LON_MAX)
    )
    table = dataset.to_table(filter=filt, columns=COLS)
    return table.to_pandas()

def try_pandas_s3fs():
    # Portable fallback: pandas -> pyarrow engine -> s3fs
    import pyarrow  
    return pd.read_parquet(
        f"s3://{S3_PATH}",
        engine="pyarrow",
        storage_options={"anon": True},
        columns=COLS,
    )

def main():
    os.makedirs("data", exist_ok=True)
    try:
        df = try_fsspec_dataset()
        print("Loaded via pyarrow.dataset with s3fs handler.")
    except Exception as e:
        print("FSSpec-backed dataset failed, falling back to pandas+s3fs:", e)
        df = try_pandas_s3fs()
        # apply bbox locally if fallback used
        df = df[(df.tile_y.between(LAT_MIN, LAT_MAX)) & (df.tile_x.between(LON_MIN, LON_MAX))]

    out = "data/ookla_mobile_london_2024q4.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} tiles to {out}")

if __name__ == "__main__":
    main()
