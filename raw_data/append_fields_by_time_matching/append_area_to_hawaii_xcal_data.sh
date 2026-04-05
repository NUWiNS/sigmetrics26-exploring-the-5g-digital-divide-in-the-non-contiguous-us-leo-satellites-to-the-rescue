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

python "$SCRIPT_DIR/append_fields_by_time_matching.py" \
    --src ./data/hawaii.area.csv \
    --dst ./data/20240816_ATT_HAWAII_100MS.xlsx \
    --dst-xcal-tz "US/Eastern" \
    --src-data-field value \
    --dst-data-field area \
    --coherent-time-sec 604800 \
    -o ./data/20240816_ATT_HAWAII_100MS_with_area.csv
