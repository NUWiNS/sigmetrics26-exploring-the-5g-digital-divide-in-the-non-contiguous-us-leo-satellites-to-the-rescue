import sys
import os
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from scripts.utilities.data_collection.CdfStatCollector import CdfStatCollector
from scripts.utilities.io import JsonDataManager
from scripts.logging_utils import create_logger
from scripts.plotting.common import location_conf, operator_conf, cellular_location_conf, style_confs, threshold_confs

current_dir = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(current_dir, 'outputs')

logger = create_logger('tput_diff', filename=os.path.join(output_dir, 'tput_diff.log'))

def get_mptcp_trace_path(base_dir: str, location: str, operator_a: str, operator_b: str, trace_type: str):
        return os.path.join(base_dir, f'mptcp_trace.{location}.{trace_type}.{operator_a}_{operator_b}.csv')

class DeltaTputPlotter:
    def __init__(
            self, 
            width: float = 4, 
            height: float = 3,
            xlim: Tuple[float, float] = (-50, 50),
            ylim: Tuple[float, float] = (0, 1),
            x_ticks: List[float] = [],
            font_size_1: float = 16,
            font_size_2: float = 15,    
            font_size_3: float = 12,
        ):
        fig, ax = plt.subplots(figsize=(width, height))
        self.fig = fig
        self.ax = ax
        
        # Add dashed coordinate system lines
        self.add_hline(0.5)
        self.add_vline(0)
        
        # Set axis limits and grid
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        self.ax.grid(True, linestyle='--', alpha=0.7)

        self.font_size_1 = font_size_1
        self.font_size_2 = font_size_2
        self.font_size_3 = font_size_3
        
        # Apply font size to tick labels
        self.ax.tick_params(axis='both', which='major', labelsize=self.font_size_2)
        if x_ticks:
            self.ax.set_xticks(x_ticks)


    def add_operator_pair_data(self, df: pd.DataFrame, operator_a: str, operator_b: str, style_conf: Dict):
        """
        Add CDF plot for a single operator pair's throughput difference data
        Args:
            df: DataFrame containing the diff_tput_mbps column
            operator_a: First operator name
            operator_b: Second operator name
            style_conf: Dictionary containing color and label information
        """
        diff_data = df['diff_throughput_mbps'].values
        sorted_data = np.sort(diff_data)
        cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        self.ax.plot(
            sorted_data, 
            cumulative_prob,
            color=style_conf['color'],
            linestyle=style_conf['linestyle'],
            label=f'{style_conf["label"]}',
            linewidth=3
        )

    def set_labels(self, title: str = None, show_y_label: bool = True, show_y_ticklabels: bool = True, show_legend: bool = True):
        """Set the title and axis labels"""
        # Apply font size 1 to labels and title
        self.ax.set_xlabel('Throughput Diff (Mbps)', fontsize=self.font_size_1)
        if title:
            self.ax.set_title(title, fontsize=self.font_size_1)

        if show_y_label:
            self.ax.set_ylabel('CDF', fontsize=self.font_size_1)

        # Always set yticks for grid lines but hide labels if show_y_ticks is False
        self.ax.set_yticks(np.arange(0, 1.1, 0.2))
        if not show_y_ticklabels:
            self.ax.set_yticklabels([])

        if show_legend:
            legend = self.ax.legend(loc='upper left', prop={'size': self.font_size_3})

    def add_hline(self, y: float, color: str = 'black', linestyle: str = '--', alpha: float = 0.3):
        self.ax.axhline(y=y, color=color, linestyle=linestyle, alpha=alpha)

    def add_vline(self, x: float, color: str = 'black', linestyle: str = '--', alpha: float = 0.3):
        self.ax.axvline(x=x, color=color, linestyle=linestyle, alpha=alpha)

    def save(self, output_filename: str):
        """Save the figure to a file"""
        plt.tight_layout()
        self.fig.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close(self.fig)

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_conf = {
        'alaska': {
            'tcp_downlink': {
                'xlim': (-200, 200),
                'show_y_label': True,
                'title': 'TCP Downlink',
                'show_legend': True,
            },
            'tcp_uplink': {
                'xlim': (-50, 50),
                'show_y_label': False,
                'show_y_ticklabels': False,
                'title': 'TCP Uplink',
                'show_legend': False,
            }
        },
        'hawaii': {
            'tcp_downlink': {
                'xlim': (-200, 200),
                'show_y_label': True,
                'title': 'TCP Downlink',
                'show_legend': False,
            },
            'tcp_uplink': {
                'xlim': (-50, 50),
                'show_y_label': False,
                'show_y_ticklabels': False,
                'title': 'TCP Uplink',
                'show_legend': True,
            }
        }
    }

    for location in ['alaska', 'hawaii']:
    # for location in ['hawaii']:
        loc_conf = location_conf[location]
        mptcp_dir = os.path.join(loc_conf['root_dir'], 'processed', 'mptcp')
        
        for trace_type in ['tcp_downlink', 'tcp_uplink']:
        # for trace_type in ['tcp_downlink']:
            # Create a new plotter for each location-trace_type combination
            conf = plot_conf[location][trace_type]

            plotter = DeltaTputPlotter(
                xlim=conf.get('xlim', None),
                x_ticks=conf.get('x_ticks', []),
            )

            operators = loc_conf['operators']
            stats = {}
            for i in range(len(operators) - 1):
                for j in range(i+1, len(operators)):
                    operator_a = operators[i]
                    operator_b = operators[j]

                    # Read the data
                    mptcp_df = pd.read_csv(get_mptcp_trace_path(mptcp_dir, location, operator_a, operator_b, trace_type))

                    # Swap operators if needed
                    if operator_a == 'verizon' and operator_b == 'tmobile':
                        operator_a, operator_b = operator_b, operator_a
                        mptcp_df['diff_throughput_mbps'] = -mptcp_df['diff_throughput_mbps']

                    # used for table 1 calculation
                    mptcp_df.to_csv(os.path.join(output_dir, f'diff_tput_cdf.{location}.{trace_type}.{operator_a}_{operator_b}.csv'), index=False)

                    style_conf = style_confs[f'{operator_a}_{operator_b}']
                    plotter.add_operator_pair_data(mptcp_df, operator_a, operator_b, style_conf)
                    cdf_stat_collector = CdfStatCollector(mptcp_df, 'diff_throughput_mbps')
                    stats[f'{operator_a}_{operator_b}'] = cdf_stat_collector.get_statistics()

            

            # Set labels and save the combined plot
            plotter.set_labels(
                show_y_label=conf.get('show_y_label', True),
                show_legend=conf.get('show_legend', True),
                show_y_ticklabels=conf.get('show_y_ticklabels', True),
            )
            output_filename = os.path.join(output_dir, f'diff_tput_cdf.{location}.{trace_type}.combined.pdf')
            plotter.save(output_filename)
            JsonDataManager.save(stats, os.path.join(output_dir, f'diff_tput_cdf_stat.{location}.{trace_type}.combined.json'))
            print(f'Saved combined diff tput CDF plot to {output_filename}')

if __name__ == '__main__':
    main()
