# Cellular Network Base Station Distance Analysis

This repository contains scripts and data for analyzing base station distances and signal strength (RSRP) across different geographical regions in the United States, including mainland, Hawaii, and Alaska.

## Repository Structure
```
.
├── plots/                  # Generated visualization outputs
├── processed_files/        # Processed data files (pickled dictionaries)
├── raw_data/              # Raw input data
│   ├── alaska_ta_data/
│   ├── hawaii_ta_data/
│   ├── template_data/
│   └── timing_advance_csv/
└── scripts/               # Data processing and visualization scripts
    ├── extract_bs_dist_hi_ak.py
    ├── extract_bs_dist_mainland_2024.py
    └── plot.py
```

## Contents

### Plots (`plots/`)

Generated boxplot visualizations showing:
- **Distance analysis**: Cell tower distance distributions for mainland, Alaska, and Hawaii
- **RSRP analysis**: Reference Signal Received Power distributions for mainland, Alaska, and Hawaii
- Separate visualizations for each geographical region (mainland, Alaska, Hawaii)

### Processed Files (`processed_files/`)

Pickled Python dictionaries containing processed network metrics:

**Hawaii and Alaska Data:**
- `hawaii_alaska_rsrp_distance_dict_*.pkl` - RSRP and distance measurements
- `hawaii_alaska_ta_dist_dict_*.pkl` - Timing Advance (TA) distance calculations

**Mainland Data:**
- `mainland_rsrp_distance_dict_*.pkl` - RSRP and distance measurements
- `mainland_ta_dist_dict_*.pkl` - Timing Advance (TA) distance calculations

Each dataset is broken down by area type:
- `*_all_areas.pkl` - Combined data from all area types
- `*_urban.pkl` - Urban area measurements
- `*_suburban.pkl` - Suburban area measurements
- `*_rural.pkl` - Rural area measurements

### Raw Data (`raw_data/`)

Contains input data directories:
- `alaska_ta_data/` - Raw timing advance data from Alaska measurements
- `hawaii_ta_data/` - Raw timing advance data from Hawaii measurements
- `template_data/` - Template files or reference data
- `timing_advance_csv/` - CSV files containing timing advance measurements

## Scripts

### Data Extraction Scripts

#### `extract_bs_dist_hi_ak.py`

Processes raw cellular network data for Hawaii and Alaska regions to extract base station distance and RSRP metrics.

**Input:** Raw data from `raw_data/hawaii_ta_data/` and `raw_data/alaska_ta_data/`

**Output:** Generates the following processed files in `processed_files/`:
- `hawaii_alaska_rsrp_distance_dict_*.pkl` (all_areas, urban, suburban, rural)
- `hawaii_alaska_ta_dist_dict_*.pkl` (all_areas, urban, suburban, rural)

**Processing:** 
- Analyzes timing advance measurements to estimate base station distances
- Categorizes measurements by area type (urban, suburban, rural)
- Calculates RSRP statistics across different geographical regions

#### `extract_bs_dist_mainland_2024.py`

Processes raw cellular network data for the mainland United States to extract base station distance and RSRP metrics.

**Input:** Raw data from `raw_data/timing_advance_csv/`

**Output:** Generates the following processed files in `processed_files/`:
- `mainland_rsrp_distance_dict_*.pkl` (all_areas, urban, suburban, rural)
- `mainland_ta_dist_dict_*.pkl` (all_areas, urban, suburban, rural)

**Processing:**
- Analyzes 2024 cellular network measurements from mainland US
- Categorizes measurements by area type
- Computes base station distance estimates using timing advance data

### Visualization Script

#### `plot.py`

Generates boxplot visualizations comparing base station distances and RSRP across different geographical regions.

**Input:** Processed pickle files from `processed_files/`

**Output:** Generates boxplots in `plots/`:
- `boxplots_mainland_distance_individual_v2.png` - Mainland distance distribution
- `boxplots_mainland_rsrp_individual_v2.png` - Mainland RSRP distribution
- `boxplots_alaska_distance_individual_v2.png` - Alaska distance distribution
- `boxplots_alaska_rsrp_individual_v2.png` - Alaska RSRP distribution
- `boxplots_hawaii_distance_individual_v2.png` - Hawaii distance distribution
- `boxplots_hawaii_rsrp_individual_v2.png` - Hawaii RSRP distribution

**Visualization Features:**
- Boxplots showing statistical distributions (quartiles, median, outliers)
- Individual plots for each geographical region
- Separate visualizations for distance and RSRP metrics

## Workflow

1. **Data Extraction:**
```bash
   python scripts/extract_bs_dist_hi_ak.py
   python scripts/extract_bs_dist_mainland_2024.py
```
   These scripts process raw data and generate pickled dictionaries in `processed_files/`

2. **Visualization:**
```bash
   python scripts/plot.py
```
   This script reads the processed files and generates boxplot visualizations in `plots/`

## Requirements

- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - matplotlib
  - pickle

## Data Analysis Metrics

- **RSRP (Reference Signal Received Power)**: Measures the signal strength from cellular base stations
- **Timing Advance**: Used to estimate the distance between user equipment and base stations
- **Area Classification**: Urban, suburban, and rural categorizations based on geographical characteristics

## Notes

- All distance measurements are derived from timing advance values
- Data is collected from cellular network measurements across the United States
- Area type classification helps identify coverage and performance patterns across different geographical contexts
