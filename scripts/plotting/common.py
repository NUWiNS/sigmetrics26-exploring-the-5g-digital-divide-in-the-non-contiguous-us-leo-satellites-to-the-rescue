import os
import pandas as pd
import sys
from typing import Dict, List, Tuple

from scripts.utilities.distance_utils import DistanceUtils

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from scripts.constants import XcalField, operator_color_map
from scripts.configs.alaska import ROOT_DIR as AL_DATASET_DIR
from scripts.configs.hawaii import ROOT_DIR as HI_DATASET_DIR
from scripts.configs.mainland import ROOT_DIR as ML_ROOT_DIR

cellular_location_conf = {
    'alaska': {
        'label': 'Alaska',
        'abbr': 'AK',
        'root_dir': AL_DATASET_DIR,
        'operators': ['verizon', 'att'],
        'order': 1,
        'linestyle': '-',
        'tcp_downlink': {
            'interval_x': 50,
            'max_xlim': 300,
        },
        'tcp_uplink': {
            'interval_x': 15,
            'max_xlim': 75,
        },
    },
    'hawaii': {
        'label': 'Hawaii',
        'abbr': 'HI',
        'root_dir': HI_DATASET_DIR,
        'operators': ['verizon', 'att', 'tmobile'],
        'order': 2,
        'linestyle': '--',
        'tcp_downlink': {
            'interval_x': 50,
            'max_xlim': 300,
        },
        'tcp_uplink': {
            'interval_x': 15,
            'max_xlim': 75,
        },
    },
    'la_to_omaha': {
        'label': 'Mainland US',
        'abbr': 'ML',
        'root_dir': ML_ROOT_DIR,
        'operators': ['verizon', 'att', 'tmobile'],
        'order': 4,
        'linestyle': ':',
        'tcp_downlink': {
            'interval_x': 400,
            'max_xlim': 2000,
        },
        'tcp_uplink': {
            'interval_x': 25,
            'max_xlim': 75,
        },
    }
}


location_conf = {
    'alaska': {
        'label': 'Alaska',
        'abbr': 'AK',
        'root_dir': AL_DATASET_DIR,
        'operators': ['starlink', 'verizon', 'att'],
        'op_pairs': [
            ('starlink', 'att'),
            ('starlink', 'verizon'),
            ('verizon', 'att'),
        ],
        'tcp_downlink': {
            'interval_x': 100,
            'max_xlim': 350,
        },
        'tcp_uplink': {
            'interval_x': 50,
            'max_xlim': 100,
        },
        'order': 1,
        'color': 'b',
        'linestyle': '-',
    },
    'hawaii': {
        'label': 'Hawaii',
        'abbr': 'HI',
        'root_dir': HI_DATASET_DIR,
        'operators': ['starlink', 'verizon', 'att', 'tmobile'],
        'op_pairs': [
            ('starlink', 'att'),
            ('starlink', 'tmobile'),
            ('starlink', 'verizon'),
            ('att', 'tmobile'),
            ('tmobile', 'verizon'),
            ('verizon', 'att'),
        ],
        'tcp_downlink': {
            'interval_x': 300,
            'max_xlim': 900,
        },
        'tcp_uplink': {
            'interval_x': 50,
            'max_xlim': 150,
        },
        'order': 2,
        'color': 'r',
        'linestyle': '--',
    },

    'la_to_omaha': {
        'label': 'Mainland US',
        'abbr': 'ML',
        'root_dir': ML_ROOT_DIR,
        'operators': ['starlink', 'verizon', 'att', 'tmobile'],
        'order': 4,
        'color': 'g',
        'linestyle': ':',
        'tcp_downlink': {
            'interval_x': 100,
            'max_xlim': 350,
        },
        'tcp_uplink': {
            'interval_x': 25,
            'max_xlim': 75,
        },
    }
}

cellular_operator_conf = {
    'att': {
        'label': 'AT&T',
        'abbr': 'AT',
        'order': 1,
        'color': operator_color_map['att'],
        'linestyle': '--'
    },
    'verizon': {
        'label': 'Verizon',
        'abbr': 'VZ',
        'order': 2,
        'color': operator_color_map['verizon'],
        'linestyle': ':'
    },
    'tmobile': {
        'label': 'T-Mobile',
        'abbr': 'TM',
        'order': 3,
        'color': operator_color_map['tmobile'],
        'linestyle': '-.'
    },
}


operator_conf = {
    **cellular_operator_conf,
    'starlink': {
        'label': 'Starlink',
        'abbr': 'SL',
        'order': 4,
        'color': operator_color_map['starlink'],
        'linestyle': '-',
    },
}


mptcp_operator_conf = {
    'starlink': {
        'label': 'SL',
        'color': operator_conf['starlink']['color'],
        'hatch': None,
    },
    'att': {
        'label': 'AT',
        'color': operator_conf['att']['color'],
        'hatch': None,
    },
    'verizon': {
        'label': 'VZ',
        'color': operator_conf['verizon']['color'],
        'hatch': None,
    },
    'tmobile': {
        'label': 'TM',
        'color': operator_conf['tmobile']['color'],
        'hatch': None,
    },
    'starlink_att': {
        'label': 'SL+AT',
        'color': operator_conf['att']['color'],
        'hatch': None,
        'order': 0, 
    },
    'starlink_verizon': {
        'label': 'SL+VZ',
        'color': operator_conf['verizon']['color'],
        'hatch': None,
        'order': 1,
    },
    'starlink_tmobile': {
        'label': 'SL+TM',
        'color': operator_conf['tmobile']['color'],
        'hatch': None,
        'order': 2,
    },
    'verizon_att': {
        'label': 'VZ+AT',
        'color': operator_conf['verizon']['color'],
        'hatch': '///',
        'order': 3,
    },
    'verizon_tmobile': {
        'label': 'TM+VZ',
        'color': operator_conf['tmobile']['color'],
        'hatch': '///',
        'order': 4,
    },
    'att_tmobile': {
        'label': 'AT+TM',
        'color': operator_conf['att']['color'],
        'hatch': '///',
        'order': 5,
    },
}

tech_conf = {
    'LTE': {
        'label': 'LTE',
        'color': '#326f21',
        'order': 1
    },
    'LTE-A': {
        'label': 'LTE-A',
        'color': '#86c84d',
        'order': 2
    },
    '5G-low': {
        'label': '5G Low',
        'color': '#ffd700',
        'order': 3
    },
    '5G-mid': {
        'label': '5G Mid',
        'color': '#ff9900',
        'order': 4
    },
    '5G-mmWave (28GHz)': {
        # 'label': '5G-mmWave (28GHz)',
        # 'label': '5G 28GHz',
        'label': '5G mmWave',
        'color': '#ff4500',
        'order': 5
    },
    '5G-mmWave (39GHz)': {
        # 'label': '5G-mmWave (39GHz)',
        # 'color': '#ba281c',
        'label': '5G mmWave',
        'color': '#ff4500',
        'order': 6
    },
    'NO SERVICE': {
        'label': 'No Service',
        'color': '#808080',
        'order': 7
    },
    'Unknown': {
        'label': 'Unknown',
        'color': '#000000',
        'order': 8
    },
}

area_conf = {
    "urban": {
        "label": "Urban",
        "color": "#ff9900",
        "order": 1,
        "linestyle": '-'
    },
    "rural": {
        "label": "Rural",
        "color": "#86c84d",
        "order": 2,
        "linestyle": '--'
    },
}

style_confs = {
    'starlink_att': {
        'color': operator_color_map['att'],
        'linestyle': '-',
        'label': 'SL-AT',
    },
    'starlink_tmobile': {
        'color': operator_color_map['tmobile'],
        'linestyle': '-',
        'label': 'SL-TM',
    },
    'starlink_verizon': {
        'color': operator_color_map['verizon'],
        'linestyle': '-',
        'label': 'SL-VZ',
    },
    'att_tmobile': {
        'color': operator_color_map['att'],
        'linestyle': ':',
        'label': 'AT-TM',
    },
    'tmobile_verizon': {
        'color': operator_color_map['tmobile'],
        'linestyle': ':',
        'label': 'TM-VZ',
    },
    'verizon_tmobile': {
        'color': operator_color_map['tmobile'],
        'linestyle': ':',
        'label': 'TM-VZ',
    },
    'verizon_att': {
        'color': operator_color_map['verizon'],
        'linestyle': ':',
        'label': 'VZ-AT',
    },
}

threshold_confs = {
    'tcp_downlink': {
        0: {
            'color': 'blue',
            'label': '0 Mbps',
            'linestyle': '-',   
        },
        10: {
            'color': 'green',
            'label': '10 Mbps',
            'linestyle': 'dashed',
        },
        50: {
            'color': 'red',
            'label': '50 Mbps',
            'linestyle': 'dotted',
        }
    },
    'tcp_uplink': {
        0: {
            'color': 'blue',
            'label': '0 Mbps',
            'linestyle': '-',
        },
        5: {
            'color': 'green',
            'label': '5 Mbps',
            'linestyle': 'dashed',
        },
        10: {
            'color': 'red',
            'label': '10 Mbps',
            'linestyle': 'dotted',
        }
    }
}
# Colors for different technologies - from grey (NO SERVICE) to rainbow gradient (green->yellow->orange->red) for increasing tech
colors = ['#808080', '#326f21', '#86c84d', '#ffd700', '#ff9900', '#ff4500', '#ba281c']  # Grey, green, light green, yellow, amber, orange, red
tech_order = ['Unknown', 'NO SERVICE', 'LTE', 'LTE-A', '5G-low', '5G-mid', '5G-mmWave (28GHz)', '5G-mmWave (39GHz)']


def read_xcal_tput_data(root_dir: str, operator: str, protocol: str = None, direction: str = None):
    input_csv_path = os.path.join(root_dir, 'xcal', f'{operator}_xcal_smart_tput.csv')
    df = pd.read_csv(input_csv_path)
    if protocol:
        df = df[df[XcalField.APP_TPUT_PROTOCOL] == protocol]
    if direction:
        df = df[df[XcalField.APP_TPUT_DIRECTION] == direction]
    return df

def aggregate_xcal_tput_data(
        root_dir: str, 
        operators: List[str], 
        protocol: str = None, 
        direction: str = None, 
    ):
    data = pd.DataFrame()
    for operator in operators:
        df = read_xcal_tput_data(
            root_dir=root_dir, 
            operator=operator, 
            protocol=protocol, 
            direction=direction, 
        )
        df['operator'] = operator
        data = pd.concat([data, df])
    return data

def aggregate_xcal_tput_data_by_location(
        locations: List[str], 
        location_conf: Dict[str, Dict],
        protocol: str = None, 
        direction: str = None, 
    ):
    data = pd.DataFrame()
    for location in locations:
        conf = location_conf[location]
        df = aggregate_xcal_tput_data(
            root_dir=os.path.join(conf['root_dir'], 'processed'), 
            operators=conf['operators'], 
            protocol=protocol, 
            direction=direction, 
        )
        df['location'] = location
        data = pd.concat([data, df])
    return data


def read_latency_data(root_dir: str, operator: str):
    latency_data_path = os.path.join(root_dir, 'ping', f'{operator}_ping.csv')
    if not os.path.exists(latency_data_path):
        raise FileNotFoundError(f'Latency data file not found: {latency_data_path}')
    return pd.read_csv(latency_data_path)

def aggregate_operator_latency_data(root_dir: str, operators: List[str]):
    df = pd.DataFrame()
    for operator in operators:
        latency_data = read_latency_data(root_dir, operator)
        latency_data['operator'] = operator
        df = pd.concat([df, latency_data], ignore_index=True)
    return df

def aggregate_latency_data_by_location(locations: List[str], location_conf: Dict[str, Dict]):
    combined_data = pd.DataFrame()
    for location in locations:
        conf = location_conf[location]
        latency_data = aggregate_operator_latency_data(os.path.join(conf['root_dir'], 'processed'), conf['operators'])
        latency_data['location'] = location
        combined_data = pd.concat([combined_data, latency_data], ignore_index=True)
    return combined_data


def filter_gps_outliers_by_speed(
    df: pd.DataFrame, 
    lat_field: str, 
    lon_field: str, 
    timestamp_field: str,
    max_speed_mph: float = 150.0,
    time_unit: str = 'ms',
    min_time_diff_sec: float = 0.01,
    handle_invalid_coords: str = 'remove',
    handle_nan: str = 'remove',
    validate_first_point: bool = False,
    bidirectional_filtering: bool = False,
) -> pd.DataFrame:
    """
    Filter GPS outliers based on speed between consecutive points.
    
    Args:
        df: DataFrame with GPS data
        lat_field: Column name for latitude
        lon_field: Column name for longitude  
        timestamp_field: Column name for timestamp (in ms unix epoch)
        max_speed_mph: Maximum reasonable speed in mph (default: 150.0 mph)
        time_unit: Unit of time for timestamp field (default: 'ms')
        min_time_diff_sec: Minimum time difference in seconds to avoid division by tiny values (default: 0.01s)
        handle_invalid_coords: How to handle invalid GPS coordinates ('remove', 'keep', 'error')
        handle_nan: How to handle NaN coordinates ('remove', 'keep', 'error')
        validate_first_point: Whether to validate the first point against the second (default: False)
        bidirectional_filtering: Apply filtering in both directions to reduce chain effects (default: False)
    Returns:
        Filtered DataFrame with outlier GPS points removed
    """
    if len(df) <= 1:
        return df.copy()
    
    # Create a copy to avoid modifying the original
    df_work = df.copy()
    
    # Handle NaN coordinates
    nan_mask = df_work[lat_field].isna() | df_work[lon_field].isna()
    if nan_mask.any():
        if handle_nan == 'error':
            raise ValueError(f"Found {nan_mask.sum()} NaN coordinates in GPS data")
        elif handle_nan == 'remove':
            df_work = df_work[~nan_mask].reset_index(drop=True)
            if len(df_work) <= 1:
                return df_work
        # If 'keep', continue with NaN values (will be handled in distance calculation)
    
    # Handle invalid GPS coordinates
    invalid_lat = (df_work[lat_field] < -90) | (df_work[lat_field] > 90)
    invalid_lon = (df_work[lon_field] < -180) | (df_work[lon_field] > 180)
    invalid_mask = invalid_lat | invalid_lon
    
    if invalid_mask.any():
        if handle_invalid_coords == 'error':
            invalid_count = invalid_mask.sum()
            raise ValueError(f"Found {invalid_count} invalid GPS coordinates (lat must be [-90,90], lon must be [-180,180])")
        elif handle_invalid_coords == 'remove':
            df_work = df_work[~invalid_mask].reset_index(drop=True)
            if len(df_work) <= 1:
                return df_work
        # If 'keep', continue with invalid coordinates
    
    # ensure the timestamp field is numeric
    if not pd.api.types.is_numeric_dtype(df_work[timestamp_field]):
        df_work[timestamp_field] = pd.to_datetime(df_work[timestamp_field], format='ISO8601').astype('int64') // 10**6  # convert to ms unix epoch
    # Sort by timestamp to ensure proper order
    df_sorted = df_work.sort_values(by=timestamp_field).reset_index(drop=True)
    
    if bidirectional_filtering:
        # Apply filtering in both directions and combine results
        forward_mask = _apply_speed_filtering_forward(
            df_sorted, lat_field, lon_field, timestamp_field, 
            max_speed_mph, time_unit, min_time_diff_sec, validate_first_point
        )
        backward_mask = _apply_speed_filtering_backward(
            df_sorted, lat_field, lon_field, timestamp_field, 
            max_speed_mph, time_unit, min_time_diff_sec
        )
        # Keep points that pass both forward and backward filtering
        keep_mask = [f and b for f, b in zip(forward_mask, backward_mask)]
    else:
        # Apply standard forward filtering
        keep_mask = _apply_speed_filtering_forward(
            df_sorted, lat_field, lon_field, timestamp_field, 
            max_speed_mph, time_unit, min_time_diff_sec, validate_first_point
        )
    
    # Return filtered dataframe
    filtered_df = df_sorted[keep_mask].reset_index(drop=True)
    return filtered_df


def _apply_speed_filtering_forward(
    df_sorted: pd.DataFrame,
    lat_field: str,
    lon_field: str, 
    timestamp_field: str,
    max_speed_mph: float,
    time_unit: str,
    min_time_diff_sec: float,
    validate_first_point: bool
) -> list[bool]:
    """Apply speed-based filtering in forward direction"""
    keep_mask = [True] * len(df_sorted)
    
    start_idx = 0 if validate_first_point else 1
    
    for i in range(start_idx, len(df_sorted)):
        if i == 0:
            # For first point validation, check against second point
            if len(df_sorted) < 2:
                continue
            prev_idx, curr_idx = 1, 0
        else:
            prev_idx, curr_idx = i-1, i
        
        # Skip if previous point was already filtered out
        if not keep_mask[prev_idx]:
            if i > 0:  # Find the last valid point for comparison
                for j in range(prev_idx - 1, -1, -1):
                    if keep_mask[j]:
                        prev_idx = j
                        break
                else:
                    # No valid previous point found
                    continue
        
        prev_row = df_sorted.iloc[prev_idx]
        curr_row = df_sorted.iloc[curr_idx]
        
        # Skip if either point has NaN coordinates
        if (pd.isna(prev_row[lat_field]) or pd.isna(prev_row[lon_field]) or
            pd.isna(curr_row[lat_field]) or pd.isna(curr_row[lon_field])):
            keep_mask[curr_idx] = False
            continue
        
        # Calculate time difference in seconds
        time_diff_ms = abs(curr_row[timestamp_field] - prev_row[timestamp_field])
        if time_unit == 'ms':
            time_diff_sec = time_diff_ms / 1000.0
        elif time_unit == 's':
            time_diff_sec = time_diff_ms
        else:
            raise ValueError(f"Invalid time unit: {time_unit}")
        
        # Skip if time difference is too small
        if time_diff_sec < min_time_diff_sec:
            keep_mask[curr_idx] = False
            continue
            
        # Calculate distance between points in miles
        try:
            distance_meter = DistanceUtils.haversine_distance(
                lon1=prev_row[lon_field], lat1=prev_row[lat_field],
                lon2=curr_row[lon_field], lat2=curr_row[lat_field]
            )
            distance_miles = DistanceUtils.meter_to_mile(distance_meter)
            # Calculate speed in mph
            speed_mph = distance_miles / (time_diff_sec / 3600.0)  # Convert seconds to hours
            
            # Filter out points that require unrealistic speed
            if speed_mph > max_speed_mph:
                keep_mask[curr_idx] = False
        except (ValueError, OverflowError):
            # Handle calculation errors (e.g., invalid coordinates)
            keep_mask[curr_idx] = False
    
    return keep_mask


def _apply_speed_filtering_backward(
    df_sorted: pd.DataFrame,
    lat_field: str,
    lon_field: str, 
    timestamp_field: str,
    max_speed_mph: float,
    time_unit: str,
    min_time_diff_sec: float
) -> list[bool]:
    """Apply speed-based filtering in backward direction"""
    keep_mask = [True] * len(df_sorted)
    
    for i in range(len(df_sorted) - 2, -1, -1):  # Go backwards from second-to-last
        next_idx, curr_idx = i+1, i
        
        # Skip if next point was already filtered out
        if not keep_mask[next_idx]:
            # Find the next valid point for comparison
            for j in range(next_idx + 1, len(df_sorted)):
                if keep_mask[j]:
                    next_idx = j
                    break
            else:
                # No valid next point found
                continue
        
        curr_row = df_sorted.iloc[curr_idx]
        next_row = df_sorted.iloc[next_idx]
        
        # Skip if either point has NaN coordinates
        if (pd.isna(curr_row[lat_field]) or pd.isna(curr_row[lon_field]) or
            pd.isna(next_row[lat_field]) or pd.isna(next_row[lon_field])):
            keep_mask[curr_idx] = False
            continue
        
        # Calculate time difference in seconds
        time_diff_ms = abs(next_row[timestamp_field] - curr_row[timestamp_field])
        if time_unit == 'ms':
            time_diff_sec = time_diff_ms / 1000.0
        elif time_unit == 's':
            time_diff_sec = time_diff_ms
        else:
            raise ValueError(f"Invalid time unit: {time_unit}")
        
        # Skip if time difference is too small
        if time_diff_sec < min_time_diff_sec:
            keep_mask[curr_idx] = False
            continue
            
        # Calculate distance between points in miles
        try:
            distance_meter = DistanceUtils.haversine_distance(
                lon1=curr_row[lon_field], lat1=curr_row[lat_field],
                lon2=next_row[lon_field], lat2=next_row[lat_field]
            )
            distance_miles = DistanceUtils.meter_to_mile(distance_meter)
            # Calculate speed in mph
            speed_mph = distance_miles / (time_diff_sec / 3600.0)  # Convert seconds to hours
            
            # Filter out points that require unrealistic speed
            if speed_mph > max_speed_mph:
                keep_mask[curr_idx] = False
        except (ValueError, OverflowError):
            # Handle calculation errors (e.g., invalid coordinates)
            keep_mask[curr_idx] = False
    
    return keep_mask


def get_recommended_gps_filter_params(use_case: str = 'default') -> dict:
    """
    Get recommended parameter configurations for GPS outlier filtering.
    
    Args:
        use_case: Type of GPS data being processed
            - 'default': Balanced settings for most use cases
            - 'high_frequency': For high-frequency GPS data (>1Hz)
            - 'urban': For urban environments with potential GPS errors
            - 'highway': For highway driving with higher speeds
            - 'stationary': For mostly stationary or low-speed scenarios
            - 'conservative': Very strict filtering
            - 'permissive': Lenient filtering that keeps more data
    
    Returns:
        Dictionary of recommended parameters for filter_gps_outliers_by_speed()
    """
    configs = {
        'default': {
            'max_speed_mph': 150.0,
            'min_time_diff_sec': 0.1,
            'handle_invalid_coords': 'remove',
            'handle_nan': 'remove',
            'validate_first_point': True,
            'bidirectional_filtering': True,
        },
        'high_frequency': {
            'max_speed_mph': 150.0,
            'min_time_diff_sec': 0.5,  # Longer minimum to handle high frequency
            'handle_invalid_coords': 'remove',
            'handle_nan': 'remove',
            'validate_first_point': True,
            'bidirectional_filtering': True,
        },
        'urban': {
            'max_speed_mph': 80.0,  # Lower speed limit for city driving
            'min_time_diff_sec': 0.2,
            'handle_invalid_coords': 'remove',
            'handle_nan': 'remove',
            'validate_first_point': True,
            'bidirectional_filtering': True,  # Important for urban GPS errors
        },
        'highway': {
            'max_speed_mph': 200.0,  # Higher speed limit for highway
            'min_time_diff_sec': 0.05,
            'handle_invalid_coords': 'remove',
            'handle_nan': 'remove',
            'validate_first_point': True,
            'bidirectional_filtering': False,  # Less needed on highway
        },
        'stationary': {
            'max_speed_mph': 30.0,  # Very low speed limit
            'min_time_diff_sec': 1.0,  # Longer time threshold
            'handle_invalid_coords': 'remove',
            'handle_nan': 'remove',
            'validate_first_point': True,
            'bidirectional_filtering': True,
        },
        'conservative': {
            'max_speed_mph': 100.0,  # Strict speed limit
            'min_time_diff_sec': 0.5,
            'handle_invalid_coords': 'error',  # Fail on invalid data
            'handle_nan': 'error',  # Fail on NaN data
            'validate_first_point': True,
            'bidirectional_filtering': True,
        },
        'permissive': {
            'max_speed_mph': 300.0,  # Very high speed limit
            'min_time_diff_sec': 0.01,
            'handle_invalid_coords': 'keep',  # Keep questionable data
            'handle_nan': 'keep',
            'validate_first_point': False,
            'bidirectional_filtering': False,
        }
    }
    
    if use_case not in configs:
        available = ', '.join(configs.keys())
        raise ValueError(f"Unknown use_case '{use_case}'. Available: {available}")
    
    return configs[use_case]


def calculate_tech_coverage_in_miles(
    grouped_df: pd.DataFrame, 
    lat_field: str, 
    lon_field: str, 
    timestamp_field: str = None, 
    max_speed_mph: float = 150.0, 
    time_unit: str = 'ms',
) -> Tuple[dict, float]:
    # Initialize mile fractions for each tech
    tech_distance_mile_map = {}
    for tech in tech_order:
        tech_distance_mile_map[tech] = 0

    filtered_count = 0

    # calculate cumulative distance for each segment
    for segment_id, segment_df in grouped_df:
        unique_techs = segment_df[XcalField.ACTUAL_TECH].unique()
        if len(unique_techs) > 1:
            raise ValueError(f"Segment ({segment_id}) should only have one tech: {unique_techs}")
        tech = unique_techs[0]

        # Apply GPS outlier filtering if timestamp field is provided
        if timestamp_field is not None and timestamp_field in segment_df.columns:
            filtered_segment_df = filter_gps_outliers_by_speed(
                df=segment_df, 
                lat_field=lat_field, 
                lon_field=lon_field, 
                timestamp_field=timestamp_field, 
                max_speed_mph=max_speed_mph, 
                time_unit=time_unit,
                # Use improved filtering parameters for better results
                min_time_diff_sec=0.1,  # Avoid division by very small time differences
                handle_invalid_coords='remove',  # Remove invalid GPS coordinates
                handle_nan='remove',  # Remove NaN coordinates
                validate_first_point=True,  # Check first point for outliers
                bidirectional_filtering=False,  # Reduce chain effects
            )
            filtered_count += len(segment_df) - len(filtered_segment_df)
        else:
            filtered_segment_df = segment_df

        tech_distance_miles = DistanceUtils.calculate_cumulative_miles(
            lats=filtered_segment_df[lat_field].tolist(), 
            lons=filtered_segment_df[lon_field].tolist()
        )
        if tech_distance_miles > 1:
            pass
        # add to total distance for this tech
        tech_distance_mile_map[tech] += tech_distance_miles

    total_distance_miles = sum(tech_distance_mile_map.values())
    if filtered_count > 0:
        print(f"Filtered {filtered_count} GPS outliers with max speed {max_speed_mph} mph")
    return tech_distance_mile_map, total_distance_miles

class LaToBosDataLoader:
    @staticmethod
    def get_xcal_filename(operator, trace_type, suffix):
        if suffix is None:
            return f'xcal_smart_tput.{trace_type}.{operator}.normal.csv'
        else:
            return f'xcal_smart_tput.{trace_type}.{operator}.normal.{suffix}.csv'

    @staticmethod
    def get_app_tput_filename(operator, trace_type, suffix):
        if suffix is None:
            return f'{operator}_{trace_type}.normal.csv'
        else:
            return f'{operator}_{trace_type}.normal.{suffix}.csv'

    @staticmethod
    def get_rtt_filename(operator, suffix):
        if suffix is None:
            return f'icmp_ping.{operator}.normal.csv'
        else:
            return f'icmp_ping.{operator}.normal.{suffix}.csv'