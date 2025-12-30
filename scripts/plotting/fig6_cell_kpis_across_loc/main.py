import logging
import os
import sys
from typing import Dict, override

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../'))

from scripts.utilities.data_collection import CdfStatCollector, TputCdfStatCollector
from scripts.utilities.io import JsonDataManager
from scripts.time_utils import now
from scripts.constants import CommonField, XcalField
from scripts.logging_utils import SilentLogger, create_logger
from scripts.shared.plotting import IPlotDataGenerator, IPlotter
from scripts.utilities.math_utils import MathUtils
from scripts.plotting.common import LaToBosDataLoader, aggregate_latency_data_by_location, aggregate_xcal_tput_data_by_location, cellular_operator_conf, cellular_location_conf
from scripts.plotting.common import cellular_operator_conf, cellular_location_conf

class DataGenerator(IPlotDataGenerator):
    def __init__(
            self, 
            operator_conf: Dict[str, Dict],
            location_conf: Dict[str, Dict],
            logger: logging.Logger = None,
        ):
        self.operator_conf = operator_conf
        self.location_conf = location_conf
        self.logger = logger or SilentLogger()
        # Collect all data
        self.data = None

    @override
    def get_plot_data(self):
        if not self.data:
            self.generate()
        return self.data
    
    @override
    def get_plot_data_config(self):
        return {}
    
    @override
    def get_plot_order(self):
        return ['tcp_dl', 'tcp_ul', 'rtt']

    def generate(self):
        self.data = {}
        self.data['tcp_dl'] = self.get_tput_data_across_locations(protocol='tcp', direction='downlink')
        self.data['tcp_ul'] = self.get_tput_data_across_locations(protocol='tcp', direction='uplink')
        self.data['rtt'] = self.get_rtt_data_across_locations()
        return self.data

    def get_tput_data_across_locations(self, protocol, direction):
        # AK, HI
        locations = ['alaska', 'hawaii']
        df = aggregate_xcal_tput_data_by_location(
            locations=locations,
            location_conf=self.location_conf,
            protocol=protocol,
            direction=direction,
        )
        df.rename(columns={    
            CommonField.TIME: CommonField.LOCAL_DT,
        }, inplace=True)
        if direction == 'downlink':
            df.rename(columns={    
                XcalField.TPUT_DL: CommonField.TPUT_MBPS,
            }, inplace=True)
        elif direction == 'uplink':
            df.rename(columns={    
                XcalField.TPUT_UL: CommonField.TPUT_MBPS,
            }, inplace=True)
        else:
            raise ValueError(f'Invalid direction: {direction}')
        ak_hi_df = df[[
            CommonField.LOCAL_DT,
            CommonField.TPUT_MBPS,
            CommonField.OPERATOR,
            CommonField.LOCATION,
        ]]
        # LA-OMA
        df = self.prepare_tput_data_for_la_to_omaha(
            protocol=protocol, 
            direction=direction
        )
        if direction == 'downlink':
            df.rename(columns={    
                CommonField.DL_TPUT_MBPS: CommonField.TPUT_MBPS,
            }, inplace=True)
        elif direction == 'uplink':
            df.rename(columns={    
                CommonField.UL_TPUT_MBPS: CommonField.TPUT_MBPS,
            }, inplace=True)
        else:
            raise ValueError(f'Invalid direction: {direction}')
        la_omaha_df = df[[
            CommonField.LOCAL_DT,   
            CommonField.TPUT_MBPS,
            CommonField.OPERATOR,
            CommonField.LOCATION,
        ]]
        main_df = pd.concat([ak_hi_df, la_omaha_df], ignore_index=True)
        return main_df

    def get_rtt_data_across_locations(self):
        # AK, HI
        locations = ['alaska', 'hawaii']
        df = aggregate_latency_data_by_location(
            locations=locations,
            location_conf=self.location_conf,
        )
        # Check available columns and select only those that exist
        available_cols = df.columns.tolist()
        required_cols = [CommonField.LOCAL_DT, CommonField.RTT_MS, CommonField.OPERATOR, CommonField.LOCATION]
        missing_cols = [col for col in required_cols if col not in available_cols]
        if missing_cols:
            self.logger.warning(f"Missing columns in AK/HI RTT data: {missing_cols}")
            self.logger.info(f"Available columns: {available_cols}")
        
        ak_hi_df = df[[col for col in required_cols if col in available_cols]].copy()
        
        # LA-OMA
        df = self.prepare_rtt_data_for_la_to_omaha()
        
        # Check available columns and select only those that exist
        available_cols = df.columns.tolist()
        missing_cols = [col for col in required_cols if col not in available_cols]
        if missing_cols:
            self.logger.warning(f"Missing columns in LA-Omaha RTT data: {missing_cols}")
            self.logger.info(f"Available columns: {available_cols}")
        
        la_omaha_df = df[[col for col in required_cols if col in available_cols]].copy()
        
        # Ensure both dataframes have the same columns
        common_cols = list(set(ak_hi_df.columns) & set(la_omaha_df.columns))
        ak_hi_df = ak_hi_df[common_cols]
        la_omaha_df = la_omaha_df[common_cols]
        
        main_df = pd.concat([ak_hi_df, la_omaha_df], ignore_index=True)
        return main_df

    def prepare_tput_data_for_la_to_omaha(self, protocol: str, direction: str):
        suffix = '2024-11-01-2024-11-05'
        location = 'la_to_omaha'
        self.logger.info(f'-- Processing dataset: {location}')
        base_dir = os.path.join(self.location_conf[location]['root_dir'], 'processed')

        operator_df_list = []
        for operator in sorted(self.location_conf[location]['operators'], key=lambda x: self.operator_conf[x]['order']):
            self.logger.info(f'---- Processing operator: {operator}')
            tput_csv_path = os.path.join(base_dir, 'throughput', LaToBosDataLoader.get_xcal_filename(
                operator, f'{protocol}_{direction}', suffix))

            df = pd.read_csv(tput_csv_path)
            df[CommonField.OPERATOR] = operator
            df[CommonField.LOCATION] = location
            self.logger.info(f'---- {protocol} {direction} df ({df.shape[0]} rows): {tput_csv_path}')
            operator_df_list.append(df)

        return pd.concat(operator_df_list)

    def prepare_rtt_data_for_la_to_omaha(self):
        suffix = '2024-11-01-2024-11-05'
        location = 'la_to_omaha'
        self.logger.info(f'-- Processing dataset: {location}')
        base_dir = os.path.join(self.location_conf[location]['root_dir'], 'processed')

        operator_df_list = []
        for operator in sorted(self.location_conf[location]['operators'], key=lambda x: self.operator_conf[x]['order']):
            self.logger.info(f'---- Processing operator: {operator}')
            csv_path = os.path.join(base_dir, 'latency', LaToBosDataLoader.get_rtt_filename(operator, suffix))
            df = pd.read_csv(csv_path)
            df[CommonField.OPERATOR] = operator
            df[CommonField.LOCATION] = location
            self.logger.info(f'---- ping df ({df.shape[0]} rows): {csv_path}')
            operator_df_list.append(df)
        return pd.concat(operator_df_list)

class NetworkKpiPlotter(IPlotter):
    def __init__(
            self, 
            data_generator: IPlotDataGenerator,
            operator_conf: Dict[str, Dict],
            location_conf: Dict[str, Dict],
            operator_legend_presentation = 'color',
            location_legend_presentation = 'linestyle',
        ):
        self.data_generator = data_generator
        self.operator_conf = operator_conf
        self.location_conf = location_conf
        self.operator_legend_presentation = operator_legend_presentation
        self.location_legend_presentation = location_legend_presentation

    def get_metric_configs(self, fig_width: float, fig_height: float):
        # Configuration for each metric
        return {
            'tcp_dl': {
                'fig_width': fig_width + 0.4,
                'fig_height': fig_height,
                'xlabel': 'Throughput (Mbps)',
                'title': 'TCP Downlink Throughput',
                'value_col': CommonField.TPUT_MBPS,
                'x_limit': (0, 600),
                'x_step': 100,
                'filename_suffix': 'tcp_dl',
                'show_y_label': True,
                'show_y_ticks': True,
                'show_legend': 'operator',
            },
            'tcp_ul': {
                'fig_width': fig_width,
                'fig_height': fig_height,
                'xlabel': 'Throughput (Mbps)', 
                'title': 'TCP Uplink Throughput',
                'value_col': CommonField.TPUT_MBPS,
                'x_limit': (0, 75),
                'x_step': 15,
                'filename_suffix': 'tcp_ul',
                'show_y_label': False,
                'show_y_ticks': False,
                'show_legend': 'location',
            },
            'rtt': {
                'fig_width': fig_width,
                'fig_height': fig_height,
                'xlabel': 'RTT (ms)',
                'title': 'ICMP Round-Trip Time',
                'value_col': CommonField.RTT_MS,
                'x_limit': (0, 200),
                'x_step': 50,
                'filename_suffix': 'rtt',
                'show_y_label': False,
                'show_y_ticks': False,
                'show_legend': False,
            },
        }

    @override
    def plot(
        self, 
        output_filename: str = None,
        fig_width: float = 3,
        fig_height: float = 4,
        line_width: float = 2.5,
        line_alpha: float = 0.8,
        label_font_size: int = None,
        tick_label_font_size: int = None,
        x_tick_rotation: int = None,
        dpi: int = 300,
    ):
        data = self.data_generator.get_plot_data()
        data_order = self.data_generator.get_plot_order()
        metric_configs = self.get_metric_configs(fig_width, fig_height)

        # Create separate plots for each metric
        for metric in data_order:
            config = metric_configs[metric]
            self._plot_single_metric(
                metric=metric,
                data=data[metric],
                config=config,
                fig_width=config.get('fig_width', fig_width),
                fig_height=config.get('fig_height', fig_height),
                output_filename=output_filename,
                line_width=line_width,
                line_alpha=line_alpha,
                dpi=dpi,
                left_margin=0.02 if not config.get('show_y_label', True) else 0.15,
                right_margin=0.98,
                top_margin=0.95,
                bottom_margin=0.15,
                label_font_size=label_font_size,
                tick_label_font_size=tick_label_font_size,
                x_tick_rotation=x_tick_rotation,
            )

    def _plot_single_metric(
        self,
        metric: str,
        data: pd.DataFrame,
        config: dict,
        output_filename: str = None,
        fig_width: float = 6.0,
        fig_height: float = 3.5,
        line_width: float = 2.5,
        line_alpha: float = 0.8,
        label_font_size: int = None,
        tick_label_font_size: int = None,
        x_tick_rotation: int = None,
        dpi: int = 300,
        left_margin: float = None,
        right_margin: float = None,
        top_margin: float = None,
        bottom_margin: float = None,
    ):
        # Create single figure
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        stats = {}
        
        # Get unique combinations of operator and location
        combinations = data[[CommonField.OPERATOR, CommonField.LOCATION]].drop_duplicates()
        
        # Plot each operator-location combination
        for _, row in combinations.iterrows():
            operator = row[CommonField.OPERATOR]
            location = row[CommonField.LOCATION]
            
            # Filter data for this combination
            subset = data[(data[CommonField.OPERATOR] == operator) & (data[CommonField.LOCATION] == location)]
            values = subset[config['value_col']].dropna()
            
            if len(values) == 0:
                continue
            
            # Calculate CDF
            x_val, y_val = MathUtils.cdf(values)
            
            # Get styling
            if self.operator_legend_presentation == 'color':
                color = self.operator_conf[operator]['color']
            elif self.operator_legend_presentation == 'linestyle':
                linestyle = self.operator_conf[operator]['linestyle']
            else:
                raise ValueError(f'Invalid operator legend presentation: {self.operator_legend_presentation}')

            if self.location_legend_presentation == 'linestyle':
                linestyle = self.location_conf[location]['linestyle']
            elif self.location_legend_presentation == 'color':
                color = self.location_conf[location]['color']
            else:
                raise ValueError(f'Invalid location legend presentation: {self.location_legend_presentation}')
            
            operator_abbr = self.operator_conf[operator]['abbr']
            location_abbr = self.location_conf[location]['abbr']
            label = f"{operator_abbr}-{location_abbr}"
            
            # Plot the line
            ax.plot(
                x_val,
                y_val,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=line_width,
                alpha=line_alpha,
            )
            if metric in ['tcp_dl', 'tcp_ul']:
                collector = TputCdfStatCollector(subset, config['value_col'], name=f'{operator_abbr}-{location_abbr}')
            else:
                collector = CdfStatCollector(subset, config['value_col'], name=f'{operator_abbr}-{location_abbr}')
            stats[f'{metric}-{operator}-{location}'] = collector.get_statistics()
            stats[f'{metric}-{operator}-{location}']['digest'] = collector.get_digest(stats[f'{metric}-{operator}-{location}'])
        
        # Configure plot
        # ax.set_title(config['title'], fontsize=12, fontweight='bold')
        if label_font_size:
            ax.set_xlabel(config['xlabel'], fontsize=label_font_size)
        else:
            ax.set_xlabel(config['xlabel'])
        
        print('x_tick_rotation', x_tick_rotation)
        if x_tick_rotation:
            ax.tick_params(axis='x', rotation=x_tick_rotation)

        if tick_label_font_size:
            ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size)

        if config.get('show_y_label', True):
            if tick_label_font_size:
                ax.set_ylabel('CDF', fontsize=tick_label_font_size)
            else:
                ax.set_ylabel('CDF')
        else:
            if tick_label_font_size:
                ax.set_ylabel('', fontsize=tick_label_font_size)
            else:
                ax.set_ylabel('')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Set x-axis limits and ticks
        x_min, x_max = config['x_limit']
        ax.set_xlim(x_min, x_max)
        x_step = config['x_step']
        x_ticks = np.arange(x_min, x_max + 1, x_step)
        ax.set_xticks(x_ticks)
        
        # Set y-axis ticks
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        if not config.get('show_y_ticks', True):
            ax.set_yticklabels([])  # Hide the tick labels but keep the ticks

        
        # Add horizontal line at 50th percentile (median)
        ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.4, linewidth=1)
        
        # Store lines for legend generation
        all_lines = []
        for line in ax.get_lines():
            if hasattr(line, '_label') and line._label and not line._label.startswith('_'):
                all_lines.append(line)
        
        # Add legends based on metric type
        if config.get('show_legend', False) == 'operator':
            self.add_operator_legend(ax, combinations)
        elif config.get('show_legend', False) == 'location':
            self.add_location_legend(ax, combinations)
        else:
            pass
        
        # Adjust layout and save
        # Apply custom margins if specified, otherwise use tight_layout
        if any(margin is not None for margin in [left_margin, right_margin, top_margin, bottom_margin]):
            plt.subplots_adjust(
                left=left_margin if left_margin is not None else 0.05,
                right=right_margin if right_margin is not None else 0.95,
                top=top_margin if top_margin is not None else 0.95,
                bottom=bottom_margin if bottom_margin is not None else 0.15
            )
        else:
            plt.tight_layout()
        
        if output_filename:
            # Create filename for this specific metric
            base_name = output_filename.rsplit('.', 1)[0]  # Remove extension
            extension = output_filename.rsplit('.', 1)[1] if '.' in output_filename else 'pdf'
            metric_filename = f"{base_name}.{config['filename_suffix']}.{extension}"
            plt.savefig(metric_filename, dpi=dpi, bbox_inches='tight', pad_inches=0.01)
            stats_filename = f"{base_name}.{config['filename_suffix']}.stats.json"

            JsonDataManager.save(stats, stats_filename)
            print(f"Plot saved to {metric_filename}")
            print(f"Stats saved to {stats_filename}")
            print('Digest: ---')
            for key, value in stats.items():
                print(f'{value["digest"]}\n')
            print('---')
        else:
            plt.show()
        
        plt.close()

    def add_operator_legend(self, ax: plt.Axes, combinations: pd.DataFrame):
        """Create operator legend emphasized by colors with solid lines."""
        op_legend_handles = []
        op_legend_labels = []
        seen_operators = set()
        
        # Sort combinations by operator order
        unique_operators = combinations[CommonField.OPERATOR].unique()
        sorted_operators = sorted(unique_operators, key=lambda x: self.operator_conf[x]['order'])

        for operator in sorted_operators:
            if operator not in seen_operators:
                if self.operator_legend_presentation == 'color':
                    color = self.operator_conf[operator]['color']
                    linestyle = '-'
                elif self.operator_legend_presentation == 'linestyle':
                    color = 'black'
                    linestyle = self.operator_conf[operator]['linestyle']
                else:
                    raise ValueError(f'Invalid operator legend presentation: {self.operator_legend_presentation}')
                # Clone the line for legend with solid line to emphasize color
                op_line = Line2D([], [], 
                                color=color,
                                linestyle=linestyle,
                                linewidth=2.5,
                                label=self.operator_conf[operator]['abbr'])
                op_legend_handles.append(op_line)
                op_legend_labels.append(self.operator_conf[operator]['abbr'])
                seen_operators.add(operator)
        
        ax.legend(handles=op_legend_handles, 
                    labels=op_legend_labels,
                    loc='lower right', 
                    fontsize=9,
                    framealpha=0.7)

    def add_location_legend(self, ax: plt.Axes, combinations: pd.DataFrame):
        """Create location legend emphasized by line styles with black color."""
        loc_legend_handles = []
        loc_legend_labels = []
        seen_locations = set()

        # Sort combinations by location order
        unique_locations = combinations[CommonField.LOCATION].unique()
        sorted_locations = sorted(unique_locations, key=lambda x: self.location_conf[x]['order'])
        
        for location in sorted_locations:
            if location not in seen_locations:
                if self.location_legend_presentation == 'linestyle':
                    color = 'black'
                    linestyle = self.location_conf[location]['linestyle']
                elif self.location_legend_presentation == 'color':
                    color = self.location_conf[location]['color']
                    linestyle = '-'
                else:   
                    raise ValueError(f'Invalid location legend presentation: {self.location_legend_presentation}')

                # Get location label (prefer abbr, fallback to label)
                location_label = self.location_conf[location].get('abbr', 
                                                                    self.location_conf[location]['label'])
                
                # Clone the line for legend with black color to emphasize line style
                loc_line = Line2D([], [], 
                                 color=color,
                                 linestyle=linestyle,
                                 linewidth=2.5,
                                 label=location_label)
                loc_legend_handles.append(loc_line)
                loc_legend_labels.append(location_label)
                seen_locations.add(location)

        ax.legend(handles=loc_legend_handles, 
                    labels=loc_legend_labels,
                    loc='lower right', 
                    fontsize=9,
                    framealpha=0.7)

def main():
    # Create output directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger = create_logger(
        'network_kpis', 
        filename=os.path.join(output_dir, f'plot_network_kpis.log'),
        level=logging.DEBUG,
    )
    
    # Generate data
    data_generator = DataGenerator(
        operator_conf=cellular_operator_conf,
        location_conf=cellular_location_conf,
        logger=logger,
    )
    data_generator.generate()
    
    # Create plotter and generate plot
    plotter = NetworkKpiPlotter(
        data_generator=data_generator,
        operator_conf=cellular_operator_conf,
        location_conf=cellular_location_conf,
        operator_legend_presentation='color',
        location_legend_presentation='linestyle',
    )
    output_filename = os.path.join(output_dir, 'cell_kpi_across_locations.pdf')
    plotter.plot(
        fig_width=2.8,
        fig_height=3.4,
        output_filename=output_filename,
        x_tick_rotation=30
    )
    
    print(f"Network performance comparison plot saved to: {output_filename}")

if __name__ == "__main__":
    main()