import re
import sys
import glob
import numpy as np 
import pandas as pd 
import pickle as pkl 
import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def forward_fill_with_time_limit(df, column, time_limit_ms=300):
    """
    Forward fill values but only if the time gap is within the specified limit
    
    Parameters:
    df: DataFrame sorted by timestamp
    column: Column name to forward fill
    time_limit_ms: Maximum time gap in milliseconds for forward filling
    
    Returns:
    Series with forward filled values within time limit
    """
    result = df[column].copy()
    last_valid_value = None
    last_valid_timestamp = None
    
    for idx in df.index:
        current_value = df.loc[idx, column]
        current_timestamp = df.loc[idx, 'Timestamp']
        
        if pd.notna(current_value):
            # Valid value found, update tracking variables
            last_valid_value = current_value
            last_valid_timestamp = current_timestamp
            result.iloc[idx] = current_value
        else:
            # NaN value, check if we can forward fill
            if last_valid_value is not None and last_valid_timestamp is not None:
                time_diff_ms = abs(current_timestamp - last_valid_timestamp) * 1000  # Convert to milliseconds
                
                if time_diff_ms <= time_limit_ms:
                    # Within time limit, forward fill
                    result.iloc[idx] = last_valid_value
                else:
                    # Exceeds time limit, keep as NaN
                    result.iloc[idx] = np.nan
            else:
                # No previous valid value, keep as NaN
                result.iloc[idx] = np.nan
    
    return result

def match_distances_with_rsrp_forward_fill(df, time_limit_ms=300):
    """
    Simple and efficient version using forward fill for RSRP values with time limit
    
    Parameters:
    df: DataFrame with distance and RSRP columns
    time_limit_ms: Maximum time gap in milliseconds for forward filling (default: 300ms)
    
    Returns:
    Dictionary containing 4 lists of (distance, RSRP) tuples
    """
    print(f"Starting forward fill distance-RSRP matching with {time_limit_ms}ms time limit...")
    
    # Sort DataFrame by timestamp
    df_sorted = df.sort_values('Timestamp').reset_index(drop=True).copy()
    
    # Forward fill the RSRP columns with time limit
    print("Forward filling RSRP values with time limit...")
    df_sorted['LTE_RSRP_filled'] = forward_fill_with_time_limit(
        df_sorted, 'LTE KPI PCell Serving RSRP [dBm]', time_limit_ms
    )
    if '5G KPI PCell RF Serving SS-RSRP [dBm]' in df_sorted.columns:
        df_sorted['5G_RSRP_filled'] = forward_fill_with_time_limit(
            df_sorted, '5G KPI PCell RF Serving SS-RSRP [dBm]', time_limit_ms
        )
    
    # Define the distance columns and their corresponding RSRP columns
    distance_mappings = {
        'lte_rach_distance_rsrp': {
            'distance_col': 'Random Access Procedure PCell Random Access Response(MSG2) Timing Advance [16Ts]',
            'rsrp_col': 'LTE_RSRP_filled'
        },
        'fiveg_rach_distance_rsrp': {
            'distance_col': 'Qualcomm 5G-NR MAC RACH Info RACH Attempt PCell RACH MSG2 TA Distance [meter]',
            'rsrp_col': '5G_RSRP_filled'
        },
        'lte_distance_rsrp': {
            'distance_col': 'Qualcomm Lte/LteAdv Timing Advance Info Timing Advance [us]',
            'rsrp_col': 'LTE_RSRP_filled'
        },
        'fiveg_distance_rsrp': {
            'distance_col': 'distance_fiveg',
            'rsrp_col': '5G_RSRP_filled'
        }
    }
    
    # Initialize result dictionary
    result_data = {}
    
    # Process each distance type
    for result_key, mapping in distance_mappings.items():
        distance_col = mapping['distance_col']
        rsrp_col = mapping['rsrp_col']
        
        if distance_col not in df_sorted.columns:
            print(f"Warning: Column '{distance_col}' not found, skipping {result_key}")
            result_data[result_key] = []
            continue
            
        print(f"Processing {result_key}...")
        
        # Get rows with valid distances
        valid_mask = df_sorted[distance_col].notna()
        valid_df = df_sorted[valid_mask]
        
        if len(valid_df) == 0:
            result_data[result_key] = []
            continue
        
        # Extract distance and corresponding RSRP values
        distances = valid_df[distance_col].values
        rsrp_values = valid_df[rsrp_col].values
        
        # Create tuples for valid pairs (both distance and RSRP are not NaN)
        valid_pairs_mask = ~pd.isna(distances) & ~pd.isna(rsrp_values)
        valid_distances = distances[valid_pairs_mask]
        valid_rsrp = rsrp_values[valid_pairs_mask]
        
        # Convert to list of tuples
        distance_rsrp_pairs = list(zip(valid_distances, valid_rsrp))
        
        result_data[result_key] = distance_rsrp_pairs
        print(f"Completed {result_key}: {len(distance_rsrp_pairs)} valid pairs")
    
    print("Forward fill matching completed!")
    return result_data

def change_numerology_to_integer(numerology):
    num_to_int_dict = {'15' : 0, '30' : 1, '60' : 2, '120' : 3, '240' : 4}
    for key in num_to_int_dict.keys():
        if pd.isnull(numerology):
            return np.nan 
        if key in numerology:
            return num_to_int_dict[key]
    return np.nan

def return_distance_from_lte_ta(T_a):
    if pd.isnull(T_a):
        return T_a 
    T_s = 1 / (2048 * 15000)
    N_T_a = 16 * T_a * T_s
    distance = (3 * 10 ** 8 * N_T_a) / 2
    return distance

def return_distance_from_lte_nta(N_T_a):
    if pd.isnull(N_T_a):
        return np.nan
    N_T_a = N_T_a * 10 ** -6
    distance = (3 * 10 ** 8 * N_T_a) / 2
    return distance 

def datetime_to_timestamp_original(datetime_str):
    from datetime import datetime
    if pd.isnull(datetime_str):
        return datetime_str
    date, time_all = datetime_str.split()
    temp_year = date.split("-")[0]
    temp_month = date.split("-")[1]
    temp_date = date.split("-")[2]
    datetime_string = temp_date + "." + temp_month + "." + temp_year + " " + time_all
    try:
        dt_obj = datetime.strptime(datetime_string, '%d.%m.%Y %H:%M:%S.%f')
    except:
        dt_obj = datetime.strptime(datetime_string, '%d.%m.%Y %H:%M:%S')
    sec = dt_obj.timestamp() 
    return sec

def datetime_to_timestamp(datetime_str):
    int(datetime_str.astimezone(datetime.timezone.utc).timestamp())
    return datetime_str.astimezone(datetime.timezone.utc).timestamp()


if 1:
    # Initialize dictionaries for all area types
    area_types = ['urban', 'suburban', 'rural']
    
    hawaii_alaska_ta_dist_dict_by_area = {area: {} for area in area_types}
    hawaii_alaska_rsrp_distance_dict_by_area = {area: {} for area in area_types}
    
    for path in ['hawaii_path', 'alaska_path']:
        if path == 'hawaii_path':
            path = '/mnt/nuwinsshared/moinak/sigmetrics26-exploring-the-5g-digital-divide-in-the-non-contiguous-us-leo-satellites-to-the-rescue/raw_data/hawaii_ta_data/'
            op_list = ['ATT', 'VERIZON', 'TMOBILE']
        else:
            path = '/mnt/nuwinsshared/moinak/sigmetrics26-exploring-the-5g-digital-divide-in-the-non-contiguous-us-leo-satellites-to-the-rescue/raw_data/alaska_ta_data/'
            op_list = ['ATT', 'VERIZON']

        # Initialize data structures for each area type
        for area in area_types:
            lte_op_rach_distances = {op: [] for op in op_list}
            lte_op_non_rach_distances = {op: [] for op in op_list}
            fiveg_op_rach_distances = {op: [] for op in op_list}
            fiveg_op_non_rach_distances = {op: [] for op in op_list}

            lte_op_rach_distances_rsrp = {op: [] for op in op_list}
            lte_op_non_rach_distances_rsrp = {op: [] for op in op_list}
            fiveg_op_rach_distances_rsrp = {op: [] for op in op_list}
            fiveg_op_non_rach_distances_rsrp = {op: [] for op in op_list}
            
            # Store in main dictionaries
            hawaii_alaska_ta_dist_dict_by_area[area][path] = [
                lte_op_rach_distances, lte_op_non_rach_distances, 
                fiveg_op_rach_distances, fiveg_op_non_rach_distances
            ]
            hawaii_alaska_rsrp_distance_dict_by_area[area][path] = [
                lte_op_rach_distances_rsrp, lte_op_non_rach_distances_rsrp, 
                fiveg_op_rach_distances_rsrp, fiveg_op_non_rach_distances_rsrp
            ]

        for op in op_list:
            for df_template in glob.glob(path + '*%s*_with_area.csv' % op):
                # load timing advance df 
                print("Processing file: ", df_template)
                tc = 5.086263020833334 * (10 ** -10)
                df_template = pd.read_csv(df_template)
                df_template.drop(df_template.tail(8).index, inplace=True)
                df_template['TIME_STAMP'] = df_template['TIME_STAMP'].apply(datetime_to_timestamp_original)
                df_template = df_template.rename(columns={'TIME_STAMP': 'Timestamp'})

                # Check if 'area' column exists
                if 'area' not in df_template.columns:
                    print(f"Warning: 'area' column not found in {df_template}")
                    continue

                # Get unique area types in this file
                file_area_types = df_template['area'].dropna().unique()
                print(f"Found area types: {file_area_types}")

                #######################################
                # LTE 
                df_template['Qualcomm Lte/LteAdv Timing Advance Info Timing Advance [us]'] = df_template['Qualcomm Lte/LteAdv Timing Advance Info Timing Advance [us]'].apply(return_distance_from_lte_nta)
                df_template['Random Access Procedure PCell Random Access Response(MSG2) Timing Advance [16Ts]'] = df_template['Random Access Procedure PCell Random Access Response(MSG2) Timing Advance [16Ts]'].apply(return_distance_from_lte_ta)

                # 5G processing
                import traceback
                
                required_columns = ['Timestamp', 'Qualcomm 5G-NR MAC CE Timing Advance Info PCell Timing Advance Command[Avg]', 
                                'Qualcomm 5G-NR MAC RACH Info RACH Attempt PCell RACH MSG2 TA Value', 
                                'Qualcomm 5G-NR MAC RACH Info RACH Attempt PCell RACH MSG2 TA Distance [meter]', 
                                'Qualcomm 5G-NR MAC RACH Info RACH Attempt PCell RACH MSG2 N_TA Value [Tc]', 
                                '5G KPI PCell RF Subcarrier Spacing']
                
                missing_columns = [col for col in required_columns if col not in df_template.columns]
                if missing_columns:
                    print(f"Warning: Missing columns: {missing_columns}")
                    print("Available columns:", df_template.columns.tolist())
                else:                    
                    df_template['5G KPI PCell RF Subcarrier Spacing'] = df_template['5G KPI PCell RF Subcarrier Spacing'].apply(change_numerology_to_integer)
                    fiveg_ta_sub_cols_df = df_template[required_columns]

                    non_empty_indices = fiveg_ta_sub_cols_df.index[fiveg_ta_sub_cols_df['Qualcomm 5G-NR MAC RACH Info RACH Attempt PCell RACH MSG2 N_TA Value [Tc]'].notna()].tolist()

                    sub_dfs = []
                    for i in range(len(non_empty_indices)):
                        start_idx = non_empty_indices[i]
                        end_idx = non_empty_indices[i + 1] if i + 1 < len(non_empty_indices) else len(fiveg_ta_sub_cols_df)
                        sub_dfs.append(fiveg_ta_sub_cols_df.iloc[start_idx:end_idx])

                    ts_distance_dict = {}
                    sub_df_idx = -1
                    for sub_df in sub_dfs:
                        sub_df_idx += 1
                        numerology = None 
                        try:
                            numerology = max(list(sub_df['5G KPI PCell RF Subcarrier Spacing'].dropna()), key=list(sub_df['5G KPI PCell RF Subcarrier Spacing'].dropna()).count)
                        except:
                            # error in fetching current numerology
                            numerology_pre = None 
                            numerology_post = None 
                            try:
                                numerology_pre = max(list(sub_dfs[sub_df_idx - 1]['5G KPI PCell RF Subcarrier Spacing'].dropna()), key=list(sub_dfs[sub_df_idx - 1]['5G KPI PCell RF Subcarrier Spacing'].dropna()).count)
                            except:
                                pass
                            try:
                                numerology_post = max(list(sub_dfs[sub_df_idx + 1]['5G KPI PCell RF Subcarrier Spacing'].dropna()), key=list(sub_dfs[sub_df_idx + 1]['5G KPI PCell RF Subcarrier Spacing'].dropna()).count)
                            except:
                                pass 
                            if numerology_pre == None and numerology_post == None:
                                continue
                            elif numerology_pre == None and numerology_post != None:
                                numerology = numerology_post
                            elif numerology_pre != None and numerology_post == None:
                                numerology = numerology_pre
                            elif numerology_pre == numerology_post:
                                numerology = numerology_pre
                            else:
                                if sub_dfs[sub_df_idx - 1]['Qualcomm 5G-NR MAC RACH Info RACH Attempt PCell RACH MSG2 N_TA Value [Tc]'].dropna().iloc[0] == sub_dfs[sub_df_idx]['Qualcomm 5G-NR MAC RACH Info RACH Attempt PCell RACH MSG2 N_TA Value [Tc]'].dropna().iloc[0]:
                                    numerology = numerology_pre
                                elif sub_dfs[sub_df_idx + 1]['Qualcomm 5G-NR MAC RACH Info RACH Attempt PCell RACH MSG2 N_TA Value [Tc]'].dropna().iloc[0] == sub_dfs[sub_df_idx]['Qualcomm 5G-NR MAC RACH Info RACH Attempt PCell RACH MSG2 N_TA Value [Tc]'].dropna().iloc[0]:
                                    numerology = numerology_post
                                else:
                                    a = 1
                        rach_nta_ts = sub_df[['Timestamp', 'Qualcomm 5G-NR MAC RACH Info RACH Attempt PCell RACH MSG2 N_TA Value [Tc]', 'Qualcomm 5G-NR MAC RACH Info RACH Attempt PCell RACH MSG2 TA Distance [meter]']].dropna()
                        rach_nta = list(rach_nta_ts['Qualcomm 5G-NR MAC RACH Info RACH Attempt PCell RACH MSG2 N_TA Value [Tc]'].dropna())[0]
                        rach_ts = list(rach_nta_ts['Timestamp'].dropna())[0]

                        mac_ce_ta_ts_sub_df = sub_df[['Timestamp', 'Qualcomm 5G-NR MAC CE Timing Advance Info PCell Timing Advance Command[Avg]']].dropna()

                        mac_ce_ta = list(mac_ce_ta_ts_sub_df['Qualcomm 5G-NR MAC CE Timing Advance Info PCell Timing Advance Command[Avg]'])
                        mac_ce_ta_ts = list(mac_ce_ta_ts_sub_df['Timestamp'])
                        
                        nta_old = rach_nta
                        for ts, ta in zip(mac_ce_ta_ts, mac_ce_ta):
                            nta_new = nta_old + ((ta - 31) * 16 * 64 ) / 2 ** (numerology)
                            ts_distance_dict[ts] = abs((nta_new * tc * 3 * (10 ** 8)) / 2)
                            nta_old = nta_new 
                        df_template['distance_fiveg'] = df_template['Timestamp'].map(ts_distance_dict)

                # Get RSRP-distance pairs for the entire file
                distance_rsrp_pair = match_distances_with_rsrp_forward_fill(df_template)

                # Process data by area type
                for area in area_types:
                    if area not in file_area_types:
                        continue  # Skip if this area type is not present in the file
                    
                    print(f"Processing area: {area}")
                    
                    # Filter dataframe for current area
                    area_df = df_template[df_template['area'] == area].copy()
                    
                    if len(area_df) == 0:
                        continue
                    
                    # Get area-specific RSRP-distance pairs
                    area_distance_rsrp_pair = match_distances_with_rsrp_forward_fill(area_df)
                    # Get references to the data structures for this area
                    area_ta_dict = hawaii_alaska_ta_dist_dict_by_area[area][path]
                    area_rsrp_dict = hawaii_alaska_rsrp_distance_dict_by_area[area][path]
                    
                    # LTE data
                    lte_rach_area = area_df['Random Access Procedure PCell Random Access Response(MSG2) Timing Advance [16Ts]'].dropna()
                    lte_non_rach_area = area_df['Qualcomm Lte/LteAdv Timing Advance Info Timing Advance [us]'].dropna()
                    
                    area_ta_dict[0][op].extend(lte_rach_area)  # LTE RACH distances
                    area_ta_dict[1][op].extend(lte_non_rach_area)  # LTE non-RACH distances
                    
                    # LTE RSRP data
                    area_rsrp_dict[0][op].extend([pair for pair in area_distance_rsrp_pair['lte_rach_distance_rsrp']])  # LTE RACH (distance, RSRP) pairs
                    area_rsrp_dict[1][op].extend([pair for pair in area_distance_rsrp_pair['lte_distance_rsrp']])  # LTE non-RACH (distance, RSRP) pairs
                    
                    # 5G data (only if columns are not missing)
                    if not missing_columns:
                        fiveg_rach_area = area_df['Qualcomm 5G-NR MAC RACH Info RACH Attempt PCell RACH MSG2 TA Distance [meter]'].dropna()
                        fiveg_non_rach_area = area_df['distance_fiveg'].dropna()
                        
                        area_ta_dict[2][op].extend(fiveg_rach_area)  # 5G RACH distances
                        area_ta_dict[3][op].extend(fiveg_non_rach_area)  # 5G non-RACH distances
                        
                        # 5G RSRP data
                        area_rsrp_dict[2][op].extend([pair for pair in area_distance_rsrp_pair['fiveg_rach_distance_rsrp']])  # 5G RACH (distance, RSRP) pairs
                        area_rsrp_dict[3][op].extend([pair for pair in area_distance_rsrp_pair['fiveg_distance_rsrp']])  # 5G non-RACH (distance, RSRP) pairs

                print(f"Completed processing {df_template}")

    # Save separate pickle files for each area type
    for area in area_types:
        print(f"Saving data for {area} areas...")
        
        # Save distance data
        dist_filename = f'/mnt/nuwinsshared/moinak/sigmetrics26-exploring-the-5g-digital-divide-in-the-non-contiguous-us-leo-satellites-to-the-rescue/processed_files/hawaii_alaska_ta_dist_dict_{area}.pkl'
        with open(dist_filename, 'wb') as fh:
            pkl.dump(hawaii_alaska_ta_dist_dict_by_area[area], fh)
        print(f"Saved: {dist_filename}")
        
        # Save RSRP-distance data
        rsrp_filename = f'/mnt/nuwinsshared/moinak/sigmetrics26-exploring-the-5g-digital-divide-in-the-non-contiguous-us-leo-satellites-to-the-rescue/processed_files/hawaii_alaska_rsrp_distance_dict_{area}.pkl'
        with open(rsrp_filename, 'wb') as fh:
            pkl.dump(hawaii_alaska_rsrp_distance_dict_by_area[area], fh)
        print(f"Saved: {rsrp_filename}")

    # Also save a combined dictionary for all areas (optional)
    combined_ta_dict = {
        'urban': hawaii_alaska_ta_dist_dict_by_area['urban'],
        'suburban': hawaii_alaska_ta_dist_dict_by_area['suburban'],
        'rural': hawaii_alaska_ta_dist_dict_by_area['rural']
    }
    
    combined_rsrp_dict = {
        'urban': hawaii_alaska_rsrp_distance_dict_by_area['urban'],
        'suburban': hawaii_alaska_rsrp_distance_dict_by_area['suburban'],
        'rural': hawaii_alaska_rsrp_distance_dict_by_area['rural']
    }
    
    # Save combined files
    with open('/mnt/nuwinsshared/moinak/sigmetrics26-exploring-the-5g-digital-divide-in-the-non-contiguous-us-leo-satellites-to-the-rescue/processed_files/hawaii_alaska_ta_dist_dict_all_areas.pkl', 'wb') as fh:
        pkl.dump(combined_ta_dict, fh)
    
    with open('/mnt/nuwinsshared/moinak/sigmetrics26-exploring-the-5g-digital-divide-in-the-non-contiguous-us-leo-satellites-to-the-rescue/processed_files/hawaii_alaska_rsrp_distance_dict_all_areas.pkl', 'wb') as fh:
        pkl.dump(combined_rsrp_dict, fh)
