import argparse
import os

import pandas as pd

from TypeIntervalQueryUtil import TimestampUtil, TypeIntervalQueryUtil
from XcalLogReader import XcalLogReader


def append_fields_from_src_to_dst_by_ts_matching(
    src_df: pd.DataFrame,
    dst_df: pd.DataFrame,
    src_utc_ts_field: str,
    dst_utc_ts_field: str,
    src_data_field: str,
    dst_data_field: str | None = None,
    coherent_time_sec: int = 7 * 24 * 60 * 60,
):
    """
    Append the specified data field from src_df to dst_df by matching the timestamp.

    Args:
        src_df: the source dataframe
        dst_df: the destination dataframe
        src_utc_ts_field: timestamp field in src_df (UTC timestamp format)
        dst_utc_ts_field: timestamp field in dst_df (UTC timestamp format)
        src_data_field: the data field name in src_df
        dst_data_field: the expected data field name in dst_df
        coherent_time_sec: max time difference in seconds for matching
    """
    if dst_data_field is None:
        dst_data_field = src_data_field

    if src_df.empty or dst_df.empty:
        raise ValueError("either src_df or dst_df is empty")

    if (
        src_df[src_utc_ts_field].dtype != float
        or dst_df[dst_utc_ts_field].dtype != float
    ):
        raise ValueError(
            f"either {src_utc_ts_field} or {dst_utc_ts_field} is not float type, "
            "please convert them to utc timestamp"
        )

    src_sample = src_df[src_utc_ts_field].dropna().iloc[0]
    dst_sample = dst_df[dst_utc_ts_field].dropna().iloc[0]
    src_unit = TimestampUtil.get_timestamp_unit(src_sample)
    dst_unit = TimestampUtil.get_timestamp_unit(dst_sample)
    if src_unit != dst_unit:
        raise ValueError(
            f"timestamp precision mismatch: source is in {src_unit} "
            f"(e.g. {src_sample}), destination is in {dst_unit} "
            f"(e.g. {dst_sample}). Convert them to the same unit first."
        )

    timed_tuples = list(zip(src_df[src_utc_ts_field], src_df[src_data_field]))
    timed_value_query = TypeIntervalQueryUtil(
        timed_tuples, coherent_time_sec=coherent_time_sec
    )
    dst_df[dst_data_field] = dst_df[dst_utc_ts_field].apply(
        lambda ts: timed_value_query.query(ts)
    )
    return dst_df


def load_dataframe(
    filepath: str,
    ts_field: str,
    xcal_timezone: str | None = None,
) -> pd.DataFrame:
    """Load a CSV/XLSX file and ensure the timestamp field is float.

    Args:
        filepath: path to CSV or XLSX file
        ts_field: name of the timestamp column
        xcal_timezone: if set, treat the file as an XCAL log and parse
            timestamps from this timezone to UTC
    """
    if xcal_timezone:
        reader = XcalLogReader(
            base_dir=os.path.dirname(filepath) or ".",
            parsing_timezone=xcal_timezone,
        )
        df = reader.read_xcal_log(filepath)
    elif filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    df[ts_field] = df[ts_field].astype(float)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append a field from a source file to a destination file "
            "by nearest-timestamp matching."
        ),
    )
    parser.add_argument(
        "--src", required=True, help="Path to source CSV/XLSX file"
    )
    parser.add_argument(
        "--dst", required=True, help="Path to destination CSV/XLSX file"
    )
    parser.add_argument(
        "--src-ts-field",
        default="utc_ts",
        help="Timestamp column in source (default: utc_ts)",
    )
    parser.add_argument(
        "--dst-ts-field",
        default="utc_ts",
        help="Timestamp column in destination (default: utc_ts)",
    )
    parser.add_argument(
        "--src-data-field",
        required=True,
        help="Data column to copy from source",
    )
    parser.add_argument(
        "--dst-data-field",
        default=None,
        help="Column name in destination (default: same as --src-data-field)",
    )
    parser.add_argument(
        "--coherent-time-sec",
        type=float,
        default=7 * 24 * 60 * 60,
        help="Max time gap in seconds for matching (default: 604800 = 7 days)",
    )
    parser.add_argument(
        "--src-xcal-tz",
        default=None,
        help=(
            "If source is an XCAL log, specify its timezone "
            "(e.g. US/Eastern)"
        ),
    )
    parser.add_argument(
        "--dst-xcal-tz",
        default=None,
        help=(
            "If destination is an XCAL log, specify its timezone "
            "(e.g. US/Eastern)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: <dst>_with_<field>.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    src_df = load_dataframe(args.src, args.src_ts_field, args.src_xcal_tz)
    print(f"Loaded {src_df.shape[0]:,} rows from {args.src}")

    dst_df = load_dataframe(args.dst, args.dst_ts_field, args.dst_xcal_tz)
    print(f"Loaded {dst_df.shape[0]:,} rows from {args.dst}")

    dst_data_field = args.dst_data_field or args.src_data_field
    print(
        f"Appending '{args.src_data_field}' -> '{dst_data_field}' "
        f"(coherent_time={args.coherent_time_sec}s)..."
    )

    dst_df = append_fields_from_src_to_dst_by_ts_matching(
        src_df=src_df,
        dst_df=dst_df,
        src_utc_ts_field=args.src_ts_field,
        dst_utc_ts_field=args.dst_ts_field,
        src_data_field=args.src_data_field,
        dst_data_field=dst_data_field,
        coherent_time_sec=args.coherent_time_sec,
    )

    matched = dst_df[dst_data_field].notna().sum()
    print(f"Matched {matched:,} / {dst_df.shape[0]:,} rows")

    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.dst)
        output_path = f"{base}_with_{dst_data_field}.csv"

    dst_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()