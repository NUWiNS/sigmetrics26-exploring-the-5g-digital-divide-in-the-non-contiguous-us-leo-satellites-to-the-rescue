import json
import os
from typing import Any, Dict, List
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))


from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from scripts.utilities.data_collection import CdfStatCollector, TputCdfStatCollector
from scripts.utilities.io import JsonDataManager
from scripts.plotting.common import LaToBosDataLoader, operator_conf, location_conf
from scripts.constants import CommonField, XcalField
from scripts.logging_utils import create_logger
from scripts.utilities.DatasetHelper import DatasetHelper
from scripts.plotting.configs.mainland import cellular_location_conf as bos_la_cell_location_conf

current_dir = os.path.dirname(os.path.abspath(__file__))
logger = create_logger('dl_with_areas', filename=os.path.join(current_dir, 'outputs', 'dl_with_areas.log'))

# Define area configuration
area_conf = {
    'urban': {'line_style': '-'},
    'rural': {'line_style': 'dotted'}
}

def get_tput_data_for_alaska_and_hawaii(protocol: str, direction: str):
    al_dataset_helper = DatasetHelper(os.path.join(location_conf['alaska']['root_dir'], 'processed', 'throughput'))
    hawaii_dataset_helper = DatasetHelper(os.path.join(location_conf['hawaii']['root_dir'], 'processed', 'throughput'))

    alaska_tput_data = al_dataset_helper.get_tput_data(operator='starlink', protocol=protocol, direction=direction)

    hawaii_tput_data = hawaii_dataset_helper.get_tput_data(operator='starlink', protocol=protocol, direction=direction)

    alaska_tput_data['location'] = 'alaska'
    hawaii_tput_data['location'] = 'hawaii'

    combined_data = pd.concat([alaska_tput_data, hawaii_tput_data])
    combined_data['operator'] = 'starlink'
    return combined_data

def get_latency_data_for_alaska_and_hawaii():
    al_dataset_helper = DatasetHelper(os.path.join(location_conf['alaska']['root_dir'], 'processed', 'ping'))
    alaska_ping_data = al_dataset_helper.get_ping_data(operator='starlink')

    hawaii_dataset_helper = DatasetHelper(os.path.join(location_conf['hawaii']['root_dir'], 'processed', 'ping'))
    hawaii_ping_data = hawaii_dataset_helper.get_ping_data(operator='starlink')

    alaska_ping_data['location'] = 'alaska'
    hawaii_ping_data['location'] = 'hawaii'
    combined_data = pd.concat([alaska_ping_data, hawaii_ping_data])
    combined_data['operator'] = 'starlink'
    return combined_data

def plot_starlink_tput_comparison_by_location_and_area(
    plot_data: Dict[str, pd.DataFrame],
    locations: List[str],
    location_conf: Dict[str, Any],
    area_conf: Dict[str, Any],
    output_filepath: str,
):
    
    # Define metrics for each subplot
    metrics = ['tcp_downlink', 'tcp_uplink', 'icmp_ping']
    # Create figure with GridSpec to reserve space for legend
    fig = plt.figure(figsize=(len(metrics) * 3, 2.8))
    stats = {}
    # Create GridSpec with 2 rows - top row for legend, bottom row for plots
    gs = GridSpec(2, 1, height_ratios=[0.06, 0.96], figure=fig)
    
    # Create a subplot for the legend in the top row
    legend_ax = fig.add_subplot(gs[0, :])
    legend_ax.axis('off')  # Hide axes for legend subplot
    
    # Create a subgridspec for the plots in the bottom row
    plot_gs = gs[1].subgridspec(1, len(metrics), wspace=0.2)
    
    # Create the axes for the plots
    axes = [fig.add_subplot(plot_gs[0, i]) for i in range(len(metrics))]

    metrics_conf = {
        'tcp_downlink': {
            'xlabel': 'DL (Mbps)',
            'max_xlim': 200,
            'interval_x': 50,
            'data_field': CommonField.TPUT_MBPS,
        },
        'tcp_uplink': {
            'xlabel': 'UL (Mbps)',
            'max_xlim': 20,
            'interval_x': 5,
            'data_field': CommonField.TPUT_MBPS,
        },
        'icmp_ping': {
            'xlabel': 'RTT (ms)',
            'max_xlim': 200,
            'interval_x': 50,
            'data_field': CommonField.RTT_MS,
        }
    }
    
    legend_lines = []
    legend_labels = []
    
    # Process each subplot
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        df = plot_data[metric]
        metric_conf = metrics_conf[metric]
        stats[metric] = {}

        # Plot CDF for each location-area combination
        for location in locations:
            location_color = location_conf[location]['color']
            location_label = location_conf[location]['abbr']
            location_df = df[df['location'] == location]
            
            for area_type in ['urban', 'rural']:
                if area_type == 'urban':
                    mask = (location_df[CommonField.AREA_TYPE] == 'urban') | (location_df[CommonField.AREA_TYPE] == 'suburban')
                else:
                    # rural
                    mask = (location_df[CommonField.AREA_TYPE] == 'rural')
                location_area_df = location_df[mask]
                
                data = location_area_df[metric_conf['data_field']]

                if len(data) == 0:
                    logger.warning(f'No data for {location} {area_type}')
                    continue
                
                # Sort data for CDF
                sorted_data = np.sort(data)
                # Calculate cumulative probabilities
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                
                # Plot the CDF line
                line = ax.plot(
                    sorted_data,
                    cdf,
                    color=location_color,
                    linestyle=area_conf[area_type]['line_style'],
                    linewidth=3,
                    alpha=0.7,
                )[0]

                if metric in ['tcp_downlink', 'tcp_uplink']:
                    collector = TputCdfStatCollector(location_area_df, metric_conf['data_field'], name=f'{location_label}-{area_type}')
                else:
                    collector = CdfStatCollector(location_area_df, metric_conf['data_field'], name=f'{location_label}-{area_type}')
                stats[metric][f'{metric}-{location}-{area_type}'] = collector.get_statistics()
                stats[metric][f'{metric}-{location}-{area_type}']['digest'] = collector.get_digest(stats[metric][f'{metric}-{location}-{area_type}'])
            
                
                # Only add to legend for the first subplot
                if idx == 0:
                    legend_lines.append(line)
                    legend_labels.append(f"{location_label} {area_type.capitalize()}")
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel(metrics_conf[metric]['xlabel'])
        ax.set_ylabel('CDF' if idx == 0 else '')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, metric_conf['max_xlim'])
        ax.xaxis.set_ticks(np.arange(0, metric_conf['max_xlim'] + 1, metric_conf['interval_x']))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.4, linewidth=1)
        # Hide y-ticks for second subplot
        if idx != 0:
            ax.set_yticklabels([])
        
    # Add legend in the dedicated area at the top
    legend_ax.legend(
        legend_lines,
        legend_labels,
        loc='center',
        ncol=6,                      # Display 3 items per row to create 2 rows
        frameon=True,                # Add a frame around the legend
        borderaxespad=0,
        mode='expand',               # Expand to fill the width
    )
    
    plt.tight_layout()
    plt.savefig(output_filepath, bbox_inches='tight', dpi=300)
    stats_filename = output_filepath.replace('.pdf', '.stats.json')
    JsonDataManager.save(stats, stats_filename)
    logger.info(f'Saved plot to {output_filepath}')
    logger.info(f'Saved stats to {stats_filename}')
    for metric, metric_stats in stats.items():
        print(f'{metric} digest:')
        for _, value in metric_stats.items():
            print(f'{value["digest"]}\n')
        print('---')

def prepare_la_to_omaha_data(trace_type: str):
    suffix = '2024-11-01-2024-11-05'
    location = 'la_to_omaha'

    logger.info(f'-- Processing dataset: {location}')
    base_dir = os.path.join(bos_la_cell_location_conf[location]['root_dir'], 'processed')

    operator = 'starlink'
    logger.info(f'---- Processing operator: {operator}')
    if trace_type == 'icmp_ping':
        tput_csv_path = os.path.join(base_dir, 'latency', LaToBosDataLoader.get_rtt_filename(operator, suffix))
    elif trace_type == 'tcp_downlink' or trace_type == 'tcp_uplink':
        tput_csv_path = os.path.join(base_dir, 'throughput', LaToBosDataLoader.get_app_tput_filename(operator, trace_type, suffix))
    else:
        raise ValueError(f'Invalid test type: {trace_type}')
    
    df = pd.read_csv(tput_csv_path)
    df['operator'] = operator
    df['location'] = location
    logger.info(f'---- df ({df.shape[0]} rows): {tput_csv_path}')

    return df

def main():
    output_dir = os.path.join(current_dir, 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get data for both protocols
    plot_data = {}
    for protocol in ['tcp']:
        for direction in ['downlink', 'uplink']:
            key = f'{protocol}_{direction}'
            non_contiguous_df = get_tput_data_for_alaska_and_hawaii(protocol, direction)
            mainland_df = prepare_la_to_omaha_data(trace_type=f'{protocol}_{direction}')
            data_field = CommonField.DL_TPUT_MBPS if direction == 'downlink' else CommonField.UL_TPUT_MBPS
            mainland_df.rename(columns={data_field: CommonField.TPUT_MBPS}, inplace=True)

            # Keep necessary columns and add area information
            columns_to_keep = [CommonField.LOCATION, CommonField.AREA_TYPE, CommonField.TPUT_MBPS]
            non_contiguous_df = non_contiguous_df[columns_to_keep]
            mainland_df = mainland_df[columns_to_keep]
            
            # Store in the plot_data dictionary
            plot_data[key] = pd.concat([non_contiguous_df, mainland_df], ignore_index=True)

    starlink_ping_data = get_latency_data_for_alaska_and_hawaii()
    mainland_df = prepare_la_to_omaha_data(trace_type='icmp_ping')
    
    columns_to_keep = [CommonField.LOCATION, CommonField.AREA_TYPE, CommonField.RTT_MS]
    starlink_ping_data = starlink_ping_data[columns_to_keep]
    mainland_df = mainland_df[columns_to_keep]
    plot_data['icmp_ping'] = pd.concat([starlink_ping_data, mainland_df], ignore_index=True)

    # Create the plot
    plot_starlink_tput_comparison_by_location_and_area(
        plot_data=plot_data,
        locations=['alaska', 'hawaii', 'la_to_omaha'],
        location_conf=location_conf,
        area_conf=area_conf,
        output_filepath=os.path.join(output_dir, 'starlink.all_metrics.ak_hi.pdf')
    )

if __name__ == '__main__':
    main()