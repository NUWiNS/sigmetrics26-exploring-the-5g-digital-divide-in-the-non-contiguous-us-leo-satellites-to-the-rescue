#!/usr/bin/env bash
# Example: reproduce the original Hawaii area-append workflow using the CLI.
#
# This appends the 'value' field from hawaii.area.csv (renamed to 'area')
# into the ATT XCAL log by nearest-timestamp matching.
#
# NOTE: The source CSV uses 10-digit (seconds) timestamps. If your source
# timestamps need conversion (e.g. seconds -> milliseconds), pre-process the
# file or use pandas before calling the CLI.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 "$SCRIPT_DIR/append_fields_by_time_matching.py" \
    --src ./data/alaska.area.csv \
    --dst ../alaska_ta_data/20240622_ATT_ALASKA_100MS.xlsx \
    --dst-xcal-tz "US/Eastern" \
    --src-data-field value \
    --dst-data-field area \
    -o ../alaska_ta_data/outputs/20240622_ATT_ALASKA_100MS.with_area.csv
