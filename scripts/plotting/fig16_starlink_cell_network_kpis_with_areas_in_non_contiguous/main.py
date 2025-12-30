import logging
import os
import sys
from typing import Dict, override

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../'))

from scripts.utilities.data_collection import CdfStatCollector, TputCdfStatCollector
from scripts.utilities.io import JsonDataManager
from scripts.utilities.math_utils import MathUtils
from scripts.plotting.fig15_starlink_cell_network_kpis_in_non_contiguous.main import NonContiguousDataGenerator
from scripts.constants import CommonField
from scripts.logging_utils import SilentLogger, create_logger
from scripts.shared.plotting import IPlotDataGenerator, IPlotter
from scripts.plotting.common import operator_conf, area_conf

class NonContiguousByLocationDataGenerator(NonContiguousDataGenerator):
    def __init__(self,
            logger: logging.Logger = None
        ):
        super().__init__(logger=logger)

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
        self.data = {
            'alaska': {},
            'hawaii': {},
        }
        tcp_dl_df = self.get_tput_data_of_non_contiguous_us(protocol='tcp', direction='downlink')
        tcp_ul_df = self.get_tput_data_of_non_contiguous_us(protocol='tcp', direction='uplink')
        rtt_df = self.get_rtt_data_of_non_contiguous_us()

        # Merge suburban into urban
        tcp_dl_df[CommonField.AREA_TYPE] = tcp_dl_df[CommonField.AREA_TYPE].apply(lambda x: 'urban' if x == 'suburban' else x)
        tcp_ul_df[CommonField.AREA_TYPE] = tcp_ul_df[CommonField.AREA_TYPE].apply(lambda x: 'urban' if x == 'suburban' else x)
        rtt_df[CommonField.AREA_TYPE] = rtt_df[CommonField.AREA_TYPE].apply(lambda x: 'urban' if x == 'suburban' else x)

        self.data['alaska']['tcp_dl'] = tcp_dl_df[tcp_dl_df[CommonField.LOCATION] == 'alaska']
        self.data['alaska']['tcp_ul'] = tcp_ul_df[tcp_ul_df[CommonField.LOCATION] == 'alaska']
        self.data['alaska']['rtt'] = rtt_df[rtt_df[CommonField.LOCATION] == 'alaska']

        self.data['hawaii']['tcp_dl'] = tcp_dl_df[tcp_dl_df[CommonField.LOCATION] == 'hawaii']
        self.data['hawaii']['tcp_ul'] = tcp_ul_df[tcp_ul_df[CommonField.LOCATION] == 'hawaii']
        self.data['hawaii']['rtt'] = rtt_df[rtt_df[CommonField.LOCATION] == 'hawaii']
        return self.data

class AlaskaDataGenerator(IPlotDataGenerator):
    def __init__(self, base_data_generator: NonContiguousByLocationDataGenerator, logger: logging.Logger = None):
        self.base_data_generator = base_data_generator
        self.logger = logger or SilentLogger()

    @override
    def get_plot_data(self):
        return self.base_data_generator.get_plot_data()['alaska']
    
    @override
    def get_plot_data_config(self):
        return {}
    
    @override
    def get_plot_order(self):
        return ['tcp_dl', 'tcp_ul', 'rtt']

class HawaiiDataGenerator(IPlotDataGenerator):
    def __init__(self, base_data_generator: NonContiguousByLocationDataGenerator, logger: logging.Logger = None):
        self.base_data_generator = base_data_generator
        self.logger = logger or SilentLogger()

    @override
    def get_plot_data(self):
        return self.base_data_generator.get_plot_data()['hawaii']
    
    @override
    def get_plot_data_config(self):
        return {}
    
    @override
    def get_plot_order(self):
        return ['tcp_dl', 'tcp_ul', 'rtt']

class NonContiguousWithAreasNetworkKpiPlotter(IPlotter):
    def __init__(self, 
        data_generator: IPlotDataGenerator,
        operator_conf: Dict[str, Dict],
        location: str,
        area_conf: Dict[str, Dict],
        operator_legend_presentation = 'color',
        area_legend_presentation = 'linestyle',
    ):
        self.data_generator = data_generator
        self.location = location
        self.operator_conf = operator_conf
        self.area_conf = area_conf
        self.operator_legend_presentation = operator_legend_presentation
        self.area_legend_presentation = area_legend_presentation

    def get_metric_configs(self, fig_width: float, fig_height: float):
        # Configuration for each metric
        return {
            'tcp_dl': {
                'fig_width': fig_width + 0.4,
                'fig_height': fig_height,
                'xlabel': 'Throughput (Mbps)',
                'title': 'TCP Downlink Throughput',
                'value_col': CommonField.TPUT_MBPS,
                'x_limit': (0, 300),
                'x_step': 50,
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
                'x_limit': (0, 60),
                'x_step': 10,
                'filename_suffix': 'tcp_ul',
                'show_y_label': False,
                'show_y_ticks': False,
                'show_legend': 'area',
            },
            'rtt': {
                'fig_width': fig_width,
                'fig_height': fig_height,
                'xlabel': 'RTT (ms)',
                'title': 'ICMP Round-Trip Time',
                'value_col': CommonField.RTT_MS,
                'x_limit': (0, 200),
                'x_step': 25,
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
        dpi: int = 300,
        left_margin: float = None,
        right_margin: float = None,
        top_margin: float = None,
        bottom_margin: float = None,
    ):
        # Create single figure
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        stats = {}
        
        # Get unique combinations of operator and area
        combinations = data[[CommonField.OPERATOR, CommonField.AREA_TYPE]].drop_duplicates()
        
        # Plot each operator-area combination
        for _, row in combinations.iterrows():
            operator = row[CommonField.OPERATOR]
            area = row[CommonField.AREA_TYPE]
            
            # Filter data for this combination
            subset = data[(data[CommonField.OPERATOR] == operator) & (data[CommonField.AREA_TYPE] == area)]
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

            if self.area_legend_presentation == 'linestyle':
                linestyle = self.area_conf[area]['linestyle']
            elif self.area_legend_presentation == 'color':
                color = self.area_conf[area]['color']
            else:
                raise ValueError(f'Invalid location legend presentation: {self.area_legend_presentation}')
            
            operator_abbr = self.operator_conf[operator]['abbr']
            area_label = self.area_conf[area]['label']
            label = f"{operator_abbr}-{area_label}"
            
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
                collector = TputCdfStatCollector(subset, config['value_col'], name=f'{operator_abbr}-{area_label}')
            else:
                collector = CdfStatCollector(subset, config['value_col'], name=f'{operator_abbr}-{area_label}')
            stats[f'{metric}-{operator}-{area}'] = collector.get_statistics()
            stats[f'{metric}-{operator}-{area}']['digest'] = collector.get_digest(stats[f'{metric}-{operator}-{area}'])
        
        # Configure plot
        # ax.set_title(config['title'], fontsize=12, fontweight='bold')
        ax.set_xlabel(config['xlabel'], fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        if config.get('show_y_label', True):
            ax.set_ylabel('CDF', fontsize=10)
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
        ax.tick_params(axis='both', labelsize=10)

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
        elif config.get('show_legend', False) == 'area':
            self.add_area_legend(ax, combinations)
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
                    fontsize=10,
                    framealpha=0.7)

    def add_area_legend(self, ax: plt.Axes, combinations: pd.DataFrame):
        """Create location legend emphasized by line styles with black color."""
        loc_area_handles = []
        loc_area_labels = []
        seen_areas = set()

        # Sort combinations by location order
        unique_areas = combinations[CommonField.AREA_TYPE].unique()
        sorted_areas = sorted(unique_areas, key=lambda x: self.area_conf[x]['order'])
        
        for area in sorted_areas:
            if area not in seen_areas:
                if self.area_legend_presentation == 'linestyle':
                    color = 'black'
                    linestyle = self.area_conf[area]['linestyle']
                elif self.area_legend_presentation == 'color':
                    color = self.area_conf[area]['color']
                    linestyle = '-'
                else:   
                    raise ValueError(f'Invalid location legend presentation: {self.area_legend_presentation}')

                # Get location label (prefer abbr, fallback to label)
                area_label = self.area_conf[area].get('abbr', 
                                                                    self.area_conf[area]['label'])
                
                # Clone the line for legend with black color to emphasize line style
                loc_line = Line2D([], [], 
                                 color=color,
                                 linestyle=linestyle,
                                 linewidth=2.5,
                                 label=area_label)
                loc_area_handles.append(loc_line)
                loc_area_labels.append(area_label)
                seen_areas.add(area)

        ax.legend(handles=loc_area_handles, 
                    labels=loc_area_labels,
                    loc='lower right', 
                    fontsize=10,
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
    non_contiguous_data_generator = NonContiguousByLocationDataGenerator(logger=logger)
    non_contiguous_data_generator.generate()

    ak_data_generator = AlaskaDataGenerator(non_contiguous_data_generator, logger=logger)
    hi_data_generator = HawaiiDataGenerator(non_contiguous_data_generator, logger=logger)

    NonContiguousWithAreasNetworkKpiPlotter(
        data_generator=ak_data_generator,
        operator_conf=operator_conf,
        location='alaska',
        area_conf=area_conf,
        operator_legend_presentation='color',
        area_legend_presentation='linestyle',
    ).plot(
        fig_width=3.4,
        fig_height=2.4,
        output_filename=os.path.join(output_dir, 'starlink_cell_kpi.ak.with_areas.pdf'),
    )

    NonContiguousWithAreasNetworkKpiPlotter(
        data_generator=hi_data_generator,
        operator_conf=operator_conf,
        location='hawaii',
        area_conf=area_conf,
        operator_legend_presentation='color',
        area_legend_presentation='linestyle',
    ).plot(
        fig_width=3.4,
        fig_height=2.4,
        output_filename=os.path.join(output_dir, 'starlink_cell_kpi.hi.with_areas.pdf'),
    )

if __name__ == "__main__":
    main()