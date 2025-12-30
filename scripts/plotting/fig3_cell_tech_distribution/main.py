import math
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from scripts.plotting.common import LaToBosDataLoader, calculate_tech_coverage_in_miles, cellular_operator_conf, cellular_location_conf, tech_order, tech_conf
from scripts.constants import XcalField, CommonField
from scripts.logging_utils import create_logger

current_dir = os.path.dirname(os.path.abspath(__file__))
logger = create_logger('cell_tech_distribution', filename=os.path.join(current_dir, 'outputs', 'cell_tech_distribution.log'))


def plot_tech_dist_stack_with_area_sidebyside(
        dfs: dict, 
        output_dir: str, 
        location_conf: Dict[str, Dict],
        operator_conf: Dict[str, Dict],
        tech_conf: Dict[str, Dict],
        area_types: list = ['urban', 'rural'],
        figsize: tuple = (5, 5),
        title: str = None,
        fig_name: str = 'fig_sidebyside',
        area_field: str = XcalField.AREA,
        segment_id_field: str = XcalField.SEGMENT_ID,
        lat_field: str = XcalField.LAT,
        lon_field: str = XcalField.LON,
        timestamp_field: str = CommonField.LOCAL_DT,
        legend_fontsize: int = 9,
    ):
    """Plot side-by-side stacked bars comparing tech distribution between area types for each operator
    """    
    # Sort operators by the order defined in operator_conf
    operators = sorted(list(dfs.keys()), key=lambda x: operator_conf[x]['order'])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate fractions for each operator and area type
    all_stats = {}
    all_tech_fractions = {}
    
    for operator in operators:
        print(f'Calculate fractions for {operator}')
        df = dfs[operator]
        operator_stats = {}
        operator_fractions = {}
        
        for area_type in area_types:
            print(f'  Processing area type: {area_type}')
            
            # Define area mask based on area type
            if area_type == 'urban':
                area_mask = (df[area_field] == 'urban') | (df[area_field] == 'suburban')
            elif area_type == 'rural':
                area_mask = df[area_field] == 'rural'
            else:
                raise ValueError(f'Invalid area type: {area_type}')
                
            data = df[area_mask]
            
            # Calculate total distance for each tech
            tech_distance_mile_map = {tech: 0 for tech in tech_order}
            
            if len(data) > 0:
                grouped_df = data.groupby([segment_id_field])
                tech_distance_mile_map, total_distance_miles = calculate_tech_coverage_in_miles(
                    grouped_df, 
                    lat_field=lat_field, 
                    lon_field=lon_field,
                    timestamp_field=timestamp_field,
                    max_speed_mph=130.0,
                    time_unit='ms',
                )
            else:
                total_distance_miles = 0
            
            area_stats = {
                'tech_distance_miles': tech_distance_mile_map,
                'total_distance_miles': total_distance_miles,
                'tech_fractions': {tech: tech_distance_mile_map[tech] / total_distance_miles if total_distance_miles > 0 else 0
                                  for tech in tech_order}
            }
            tech_fractions_digest = ', '.join([f'{tech} ({frac * 100:.2f}%)' for tech, frac in area_stats['tech_fractions'].items() if frac > 0])
            area_stats['tech_fractions_digest'] = f"{operator_conf[operator]['abbr']}-{area_type}: {tech_fractions_digest}"
            
            operator_stats[area_type] = area_stats
            operator_fractions[area_type] = area_stats['tech_fractions']
            
        # add a digest for tech_fractions and filter only the techs that are present
        
        all_stats[operator] = operator_stats
        all_tech_fractions[operator] = operator_fractions
    
    # Find which technologies are actually present in the data
    present_techs = set()
    for operator in operators:
        for area_type in area_types:
            for tech, fraction in all_tech_fractions[operator][area_type].items():
                if fraction > 0:
                    present_techs.add(tech)
    
    # Sort present_techs according to tech_order
    present_techs = sorted(list(present_techs), key=lambda x: tech_order.index(x))
    
    # Setup bar positioning
    bar_width = 0.3  # Width of each bar
    group_spacing = 1.0  # Space between operator groups
    x_positions = np.arange(len(operators)) * group_spacing
    
    # Create positions for the two bars within each group
    x_urban = x_positions - bar_width/2
    x_rural = x_positions + bar_width/2
    
    # Plot stacked bars for each area type
    for area_idx, area_type in enumerate(area_types):
        x_pos = x_urban if area_idx == 0 else x_rural
        bottom = np.zeros(len(operators))
        
        # Use hatching for rural areas to distinguish from urban
        hatch_pattern = None if area_idx == 0 else '//'  # No hatch for urban, stripes for rural
        
        # Plot each technology
        for tech in present_techs:
            values = [all_tech_fractions[operator][area_type][tech] for operator in operators]
            
            # Create the bars
            bars = ax.bar(x_pos, values, bottom=bottom, 
                         label=tech_conf[tech]["label"],
                         color=tech_conf[tech]['color'], 
                         width=bar_width,
                         hatch=hatch_pattern,
                         edgecolor='gray' if hatch_pattern else None,
                         linewidth=0.5 if hatch_pattern else 0)
            
            bottom += values
    
    # Customize the plot
    ax.set_ylabel('Fraction of Miles')
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([operator_conf[op]['abbr'] for op in operators])
    
    # Set x-axis limits
    ax.set_xlim(-0.5, len(operators) - 0.5)
    
    # Create technology legend handles manually (only solid colors, no hatching)
    tech_handles = []
    tech_labels = []
    for tech in present_techs:
        if tech_conf[tech]['label'] in tech_labels:
            continue
        tech_handles.append(plt.Rectangle((0,0),1,1, color=tech_conf[tech]['color']))
        tech_labels.append(tech_conf[tech]['label'])
        
    # Create technology legend on the left side (2x2 layout)
    if legend_fontsize < 9:
        bbox_to_anchor = (-0.23, 1.03, 0.85, 0.2)  # lowered y from 1.05 to 1.01
    else:
        bbox_to_anchor = (-0.23, 1.05, 0.85, 0.2)
    tech_legend = ax.legend(tech_handles[::-1], tech_labels[::-1],
                           bbox_to_anchor=bbox_to_anchor, 
                           loc='lower left',
                           mode='expand',
                           borderaxespad=0,
                           ncol=2,  # 2 columns for 2x2 layout
                           fontsize=legend_fontsize
                        )
    
    # Add the tech legend as an artist so it doesn't get overwritten
    ax.add_artist(tech_legend)
    
    # Create area type legend on the right side (1x2 layout)
    area_handles = []
    area_labels = []
    for i, area_type in enumerate(area_types):
        hatch_pattern = None if i == 0 else '//'  # No hatch for urban, stripes for rural
        # Make sure hatching is visible with proper edge color and linewidth
        area_handles.append(plt.Rectangle((0,0),1,1, 
                                        facecolor='lightgray', 
                                        hatch=hatch_pattern,
                                        edgecolor='gray', 
                                        linewidth=1.2))
        area_labels.append(area_type.title())
    
    # Add area type legend on the right side
    area_legend = ax.legend(
        area_handles, 
        area_labels,
        bbox_to_anchor=(0.65, 1.05, 0.35, 0.2), 
        loc='lower left',
        mode='expand',
        borderaxespad=0,
        fontsize=legend_fontsize,
        ncol=1,
    )
    
    ax.grid(True, axis='y', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    figure_path = os.path.join(output_dir, f'{fig_name}.pdf')
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()
    
    # Save stats to json
    stats_json_path = os.path.join(output_dir, f'{fig_name}.stats.json')
    with open(stats_json_path, 'w') as f:
        json.dump(all_stats, f, indent=4)
    
    logger.info(f"Saved side-by-side technology distribution plot to {figure_path}")
    logger.info(f"Saved side-by-side technology distribution stats to {stats_json_path}")


def plot_tech_dist_stack(
        dfs: dict, 
        output_dir: str, 
        location_conf: Dict[str, Dict],
        operator_conf: Dict[str, Dict],
        tech_conf: Dict[str, Dict],
        figsize: tuple = (3.5, 4),
        title: str = None,
        fig_name: str = 'fig',
        df_mask: Callable | None = None, 
        segment_id_field: str = XcalField.SEGMENT_ID,
        lat_field: str = XcalField.LAT,
        lon_field: str = XcalField.LON,
        timestamp_field: str = CommonField.LOCAL_DT,
        legend_fontsize = 9,
    ):
    """Plot a stack plot of the tech distribution for each operator
    
    Args:
        dfs (dict): Dictionary mapping operator names to their dataframes
        output_dir (str): Directory to save the plot
        location (str): Location name (e.g., 'alaska' or 'hawaii')
    """
    # Sort operators by the order of Verizon, T-Mobile, ATT
    operators = sorted(list(dfs.keys()), key=lambda x: operator_conf[x]['order'])
    
    # Dynamically adjust figure width based on number of operators while maintaining bar width consistency
    # bar_width = 0.3  # Reduced from 0.35
    # spacing_factor = 2.8  # Reduced from 2.8
    # max_operator_length = 3
    # total_width_needed = max_operator_length * bar_width * spacing_factor
    # fig_width = total_width_needed + 1.9
    # fig_height = fig_width * 0.6  # Make height 60% of width for rectangular shape
    
    fig, ax = plt.subplots(figsize=figsize)
    
    tech_fractions = []
    stats = {}

    # Calculate fractions for each operator
    for operator in operators:
        print(f'Calculate fractions for {operator}')
        df = dfs[operator]
        sub_stats = {}

        if df_mask is not None:
            data = df[df_mask(df)]
        else:
            data = df
        
        # Calculate total distance for each tech
        tech_distance_mile_map = {tech: 0 for tech in tech_order}
        
        grouped_df = data.groupby([segment_id_field])
        tech_distance_mile_map, total_distance_miles = calculate_tech_coverage_in_miles(
            grouped_df, 
            lat_field=lat_field, 
            lon_field=lon_field,
            timestamp_field=timestamp_field,
            max_speed_mph=130.0,
            time_unit='ms',
        )
        
        sub_stats['tech_distance_miles'] = tech_distance_mile_map
        sub_stats['total_distance_miles'] = total_distance_miles
        sub_stats['tech_fractions'] = {tech: tech_distance_mile_map[tech] / total_distance_miles 
                                      for tech in tech_order}

        tech_fractions_digest = ', '.join([f'{tech} ({frac * 100:.2f}%)' for tech, frac in sub_stats['tech_fractions'].items() if frac > 0])
        sub_stats['tech_fractions_digest'] = operator_conf[operator]['abbr'] + ': ' + tech_fractions_digest
        
        if total_distance_miles > 0:
            tech_fractions.append({tech: tech_distance_mile_map[tech] / total_distance_miles 
                                 for tech in tech_order})
            
        stats[operator] = sub_stats
    
    # Find which technologies are actually present in the data
    present_techs = set()
    for fractions in tech_fractions:
        for tech, fraction in fractions.items():
            if fraction > 0:
                present_techs.add(tech)
    
    # Sort present_techs according to tech_order
    present_techs = sorted(list(present_techs), key=lambda x: tech_order.index(x))
    
    # Plot stacked bars with consistent width
    bar_width = 0.2
    spacing_factor = 2.8
    x = np.arange(len(operators)) * spacing_factor * bar_width  # Evenly space the bars
    bottom = np.zeros(len(operators))

    # Only plot technologies that are present
    for i, tech in enumerate(present_techs):
        values = [f[tech] if tech in f else 0 for f in tech_fractions]
        ax.bar(x, values, bottom=bottom, label=tech, color=tech_conf[tech]['color'], 
               width=bar_width)
        bottom += values
    
    # ax.set_title(title)
    # ax.set_xlabel('Operator')
    ax.set_ylabel('Fraction of Miles')
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_xticks(x)
    ax.set_xticklabels(map(lambda x: operator_conf[x]['abbr'], operators))
    
    # Set x-axis limits to maintain consistent spacing
    ax.set_xlim(x[0] - bar_width * 1.5, x[-1] + bar_width * 1.5)
    
    # Move legend above the plot
    handles, labels = ax.get_legend_handles_labels()
    tech_handles = []
    tech_labels = []
    for tech in present_techs:
        if tech_conf[tech]['label'] in tech_labels:
            continue
        tech_handles.append(plt.Rectangle((0,0),1,1, color=tech_conf[tech]['color']))
        tech_labels.append(tech_conf[tech]['label'])

    # Calculate number of columns based on available width
    # Use max 3 columns to ensure readability
    ncol = min(2, len(present_techs))
    nrow = math.ceil(len(present_techs) / ncol)
    legend_height = 2 * nrow  # Adjust height based on number of rows
    
    ax.legend(tech_handles[::-1], tech_labels[::-1],
             bbox_to_anchor=(-0.4, 1.05, 1.4, legend_height), 
             loc='lower left',
             mode='expand',
             borderaxespad=0,
             ncol=ncol,
             fontsize=legend_fontsize)

    
    ax.grid(True, axis='y', alpha=0.3)  # Reduced grid line opacity
    
    # Adjust layout to be more compact
    plt.tight_layout()
    figure_path = os.path.join(output_dir, f'{fig_name}.pdf')
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()
    
    # Save stats to json
    stats_json_path = os.path.join(output_dir, f'{fig_name}.stats.json')
    with open(stats_json_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    logger.info(f"Saved technology distribution plot to {figure_path}")
    logger.info(f"Saved technology distribution stats to {stats_json_path}")


def main():
    location_map = {
        'alaska': {
            'xcal_dir': 'xcal/',
            'ping_dir': 'ping',
            'output_dir': 'outputs/',
            'xcal_filename': lambda operator: f'{operator}_xcal_smart_tput.csv',
            'ping_filename': lambda operator: f'{operator}_ping.csv',
        },
        'hawaii': {
            'xcal_dir': 'xcal/',
            'ping_dir': 'ping',
            'output_dir': 'outputs/',
            'xcal_filename': lambda operator: f'{operator}_xcal_smart_tput.csv',
            'ping_filename': lambda operator: f'{operator}_ping.csv',
        },
        'la_to_omaha': {
            'xcal_dir': 'throughput',
            'ping_dir': 'latency',
            'output_dir': 'outputs',
            'xcal_filename': lambda operator: f'xcal_smart_tput.tcp_downlink.{operator}.normal.2024-11-01-2024-11-05.csv',
            'ping_filename': lambda operator: f'icmp_ping.{operator}.normal.2024-11-01-2024-11-05.csv',
        },
    }



    for location in ['alaska', 'hawaii', 'la_to_omaha']:
        logger.info(f'-- Processing dataset: {location}')
        loc_map = location_map[location]
        base_dir = os.path.join(cellular_location_conf[location]['root_dir'], 'processed')
        xcal_dir = os.path.join(base_dir, loc_map['xcal_dir'])
        output_dir = os.path.join(current_dir, loc_map['output_dir'], location)
        legend_fontsize = 9 if location in ['alaska', 'hawaii'] else 7

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Store dataframes for all operators

        def get_tput_rtt_df(location: str):
            if location in ['alaska', 'hawaii']:
                return get_tput_rtt_df_for_ak_or_hi(location)
            elif location in ['la_to_omaha']:
                return get_tput_rtt_df_for_la_to_omaha(location)
            else:
                raise ValueError(f'Invalid location: {location}')

        def get_tput_rtt_df_for_ak_or_hi(location: str):
            operator_dfs = {}
            for operator in sorted(cellular_location_conf[location]['operators'], key=lambda x: cellular_operator_conf[x]['order']):
                logger.info(f'---- Processing operator: {operator}')
                smart_tput_csv_path = os.path.join(xcal_dir, loc_map['xcal_filename'](operator))
                rtt_csv_path = os.path.join(base_dir, loc_map['ping_dir'], loc_map['ping_filename'](operator))

                smart_tput_df = pd.read_csv(smart_tput_csv_path)
                smart_tput_df['type'] = smart_tput_df[XcalField.APP_TPUT_PROTOCOL] + '_' + smart_tput_df[XcalField.APP_TPUT_DIRECTION]
                rtt_df = pd.read_csv(rtt_csv_path)
                rtt_df['type'] = 'rtt'

                # Merge the dataframes to create a common structure
                tput_sub_df = smart_tput_df[[CommonField.LOCAL_DT, CommonField.AREA_TYPE, XcalField.SEGMENT_ID, XcalField.ACTUAL_TECH, XcalField.LON, XcalField.LAT, 'type']]
                rtt_sub_df = rtt_df[[CommonField.LOCAL_DT, CommonField.AREA_TYPE, XcalField.SEGMENT_ID, XcalField.ACTUAL_TECH, XcalField.LON, XcalField.LAT, 'type']]
                df = pd.concat(
                    [
                        tput_sub_df, 
                        rtt_sub_df
                    ],
                    ignore_index=True
                )
                df.sort_values(by=[CommonField.LOCAL_DT], inplace=True)
                # df.to_csv(os.path.join(xcal_dir, f'{operator}_coverage_with_tput_and_rtt.csv'), index=False)
                operator_dfs[operator] = df
            return operator_dfs

        def get_tput_rtt_df_for_la_to_omaha(location: str):
            operator_dfs = {}
            for operator in sorted(cellular_location_conf[location]['operators'], key=lambda x: cellular_operator_conf[x]['order']):
                logger.info(f'---- Processing operator: {operator}')                
                suffix = '2024-11-01-2024-11-05'
                tcp_dl_csv_path = os.path.join(base_dir, 'throughput', LaToBosDataLoader.get_xcal_filename(operator, 'tcp_downlink', suffix))
                tcp_ul_csv_path = os.path.join(base_dir, 'throughput', LaToBosDataLoader.get_xcal_filename(operator, 'tcp_uplink', suffix))
                rtt_csv_path = os.path.join(base_dir, 'latency', LaToBosDataLoader.get_rtt_filename(operator, suffix))

                tcp_dl_df = pd.read_csv(tcp_dl_csv_path)
                tcp_dl_df['type'] = 'tcp_downlink'
                tcp_ul_df = pd.read_csv(tcp_ul_csv_path)
                tcp_ul_df['type'] = 'tcp_uplink'
                rtt_df = pd.read_csv(rtt_csv_path)
                rtt_df['type'] = 'rtt'

                # Merge the dataframes to create a common structure
                tput_dl_sub_df = tcp_dl_df[[CommonField.LOCAL_DT, CommonField.UTC_TS, CommonField.AREA_TYPE, CommonField.SEGMENT_ID, CommonField.ACTUAL_TECH, CommonField.LON, CommonField.LAT, 'type']]
                tput_ul_sub_df = tcp_ul_df[[CommonField.LOCAL_DT, CommonField.UTC_TS, CommonField.AREA_TYPE, CommonField.SEGMENT_ID, CommonField.ACTUAL_TECH, CommonField.LON, CommonField.LAT, 'type']]
                rtt_sub_df = rtt_df[[CommonField.LOCAL_DT, CommonField.UTC_TS, CommonField.AREA_TYPE, CommonField.SEGMENT_ID, CommonField.ACTUAL_TECH, CommonField.LON, CommonField.LAT, 'type']]
                df = pd.concat(
                    [
                        tput_dl_sub_df, 
                        tput_ul_sub_df,
                        rtt_sub_df
                    ],
                    ignore_index=True
                )
                df.sort_values(by=[CommonField.LOCAL_DT], inplace=True)
                # df.to_csv(os.path.join(xcal_dir, f'{operator}_coverage_with_tput_and_rtt.csv'), index=False)
                operator_dfs[operator] = df
            return operator_dfs

        operator_dfs = get_tput_rtt_df(location)
        loc_label = cellular_location_conf[location]['label']

        if location in ['alaska', 'hawaii']:
            area_field = XcalField.AREA
            segment_id_field = XcalField.SEGMENT_ID
            lat_field = XcalField.LAT
            lon_field = XcalField.LON
        elif location in ['la_to_omaha']:
            area_field = CommonField.AREA_TYPE
            segment_id_field = CommonField.SEGMENT_ID
            lat_field = CommonField.LAT
            lon_field = CommonField.LON
        else:
            raise ValueError(f'Invalid location: {location}')

        # All Areas
        plot_tech_dist_stack(
            dfs=operator_dfs,
            output_dir=output_dir, 
            location_conf=cellular_location_conf,
            operator_conf=cellular_operator_conf,
            tech_conf=tech_conf,
            figsize=(3, 2.5),
            title=f'Technology Distribution ({loc_label}-All Areas)',
            fig_name=f'tech_dist_stack_all_areas.{location}',
            segment_id_field=segment_id_field,
            lat_field=lat_field,
            lon_field=lon_field,
            timestamp_field=CommonField.LOCAL_DT if location in ['alaska', 'hawaii'] else CommonField.UTC_TS,
            legend_fontsize=legend_fontsize,
        )

        # # Side-by-side comparison for Urban vs Rural
        plot_tech_dist_stack_with_area_sidebyside(
            dfs=operator_dfs,
            output_dir=output_dir,
            location_conf=cellular_location_conf,
            operator_conf=cellular_operator_conf,
            tech_conf=tech_conf,
            area_types=['urban', 'rural'],
            figsize=(3.7, 3.2),
            title=f'Technology Distribution ({loc_label}-Urban vs Rural)',
            fig_name=f'tech_dist_stack_urban_vs_rural.{location}',
            area_field=area_field,
            segment_id_field=segment_id_field,
            lat_field=lat_field,
            lon_field=lon_field,
            timestamp_field=CommonField.LOCAL_DT if location in ['alaska', 'hawaii'] else CommonField.UTC_TS,
            legend_fontsize=legend_fontsize,
        )


if __name__ == '__main__':
    main()
