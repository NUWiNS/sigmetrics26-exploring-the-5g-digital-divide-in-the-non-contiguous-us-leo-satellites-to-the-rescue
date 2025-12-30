import json
import os
from typing import Any, Dict, List
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from scripts.utilities.math_utils import MathUtils
from scripts.utilities.data_collection import TputCdfStatCollector
from scripts.plotting.common import LaToBosDataLoader, aggregate_xcal_tput_data_by_location, cellular_location_conf, cellular_operator_conf, tech_conf
from scripts.plotting.configs.mainland import cellular_location_conf as bos_la_cell_location_conf
from scripts.constants import CommonField, XcalField
from scripts.logging_utils import create_logger
from scripts.utilities.io.JsonDataManager import JsonDataManager

current_dir = os.path.dirname(os.path.abspath(__file__))
logger = create_logger('tcp_dl_with_areas', filename=os.path.join(current_dir, 'outputs', 'tcp_dl_with_areas.log'))

class TputWithTechDataGenerator:
    def __init__(self, df: pd.DataFrame, data_field: str = None):
        self.df = df
        self.data_field = data_field

    def generate(self):
        plot_data = {}
        grouped_df = self.df.groupby([CommonField.OPERATOR, XcalField.ACTUAL_TECH])
        stats = {}
        for (operator, tech), df in grouped_df:
            if operator not in plot_data:
                plot_data[operator] = {}
            plot_data[operator][tech] = df[self.data_field]
            if operator not in stats:
                stats[operator] = {}
            collector = TputCdfStatCollector(df, self.data_field)
            stats[operator][tech] = collector.get_statistics()
            stats[operator][tech]['digest'] = collector.get_digest(stats[operator][tech])

        for operator in stats:
            total_samples = 0
            valid_samples = 0
            for tech in stats[operator]:
                total_samples += stats[operator][tech]['total_samples']
                valid_samples += stats[operator][tech]['valid_samples']
            stats[operator]['total_samples'] = total_samples
            stats[operator]['valid_samples'] = valid_samples
        return {
            'plot_data': plot_data,
            'stats': stats,
        }

class TputWithTechPlotter:
    def __init__(self, 
                 plot_data: dict, 
                 data_field: str = None, 
                 operator_conf: Dict[str, Dict] = None, 
                 tech_conf: Dict[str, Dict] = None,
                 data_sample_threshold: int = 480,
        ):
        """Initialize the plotter with plot data organized by categories.
        
        plot_data structure:
        {
            'operator': {
                'tech': DataFrame
            }
        }
        """
        self.plot_data = plot_data
        self.operator_conf = operator_conf
        self.tech_conf = tech_conf
        self.data_field = data_field
        self.data_sample_threshold = data_sample_threshold

        self.tech_legend_handles = []
        self.tech_legend_labels = []
        self.seen_techs = set()

    def add_category_plot(self, ax: plt.Axes, operator: str, tech: str, data_field: str = None):
        """Add a CDF plot for a specific category to the given axis.
        
        Args:
            ax: The matplotlib axis to plot on
            operator: The operator name
            tech: The technology type (LTE, NR, etc.)
            data_field: Column name for the throughput data (if None, assumes data is already a Series)
        
        Returns:
            The plotted line object if successful, None otherwise
        """
        data = self.plot_data[operator][tech]
        if data is None or len(data) == 0:
            return None
            
        # Get configurations
        op_conf = self.operator_conf[operator]
        tech_config = self.tech_conf[tech] if tech in self.tech_conf else {}

        # Ignore NO SERVICE
        if tech.lower() == 'no service':
            return None
        
        # Extract throughput values and sort for CDF
        if isinstance(data, pd.DataFrame) and data_field is not None and data_field in data.columns:
            tput_values = data[data_field].dropna().values
        elif isinstance(data, pd.Series):
            tput_values = data.dropna().values
        else:
            raise ValueError(f'Failed to extract tput values')
        
        if len(tput_values) == 0:
            print(f'[Warning] No tput values found for {operator} {tech}')
            return None
            
        # Calculate CDF
        data_sorted, cdf = MathUtils.cdf(tput_values)
        
        # Plot the line
        line = ax.plot(
            data_sorted,
            cdf,
            linestyle=op_conf.get('linestyle', '-'),
            color=tech_config.get('color', 'black'),
            alpha=0.7,
            linewidth=5,
            # Don't set label here - we'll handle this separately for the two legend types
        )[0]
        
        # Store metadata for legend generation
        line.operator = operator
        line.tech = tech
        line.op_label = op_conf.get('abbr', operator)
        line.tech_label = tech_config.get('label', tech)
        
        return line
        
    def set_x_lim(self, ax: plt.Axes, x_lim_min: float = None, x_lim_max: float = None, x_step: float = None):
        _x_lim_min = x_lim_min if x_lim_min is not None else 0
        if x_lim_max is not None:
            ax.set_xlim(_x_lim_min, x_lim_max)

            if x_step is not None:
                ax.set_xticks(np.arange(_x_lim_min, x_lim_max + 1, x_step))

    def add_operator_legend(self, ax: plt.Axes, all_lines: List):
         # Create only operator legend
        op_legend_handles = []
        op_legend_labels = []
        seen_operators = set()
        
        # Sort all_lines by operator_conf order
        all_lines = sorted(all_lines, key=lambda x: self.operator_conf[x.operator].get('order', 999) if x.operator in self.operator_conf else 999)

        for line in all_lines:
            if line.operator not in seen_operators:
                # Clone the line for legend but with black color to emphasize line style
                op_line = plt.Line2D([], [], 
                                    color='black',
                                    linestyle=self.operator_conf[line.operator].get('linestyle', '-'),
                                    label=line.op_label)
                op_legend_handles.append(op_line)
                op_legend_labels.append(line.op_label)
                seen_operators.add(line.operator)
        
        ax.legend(handles=op_legend_handles, 
                    loc='lower right', 
                    framealpha=0.7,
                    # prop={'size': 13, 'weight': 'bold'},
                    )
        
    def track_tech_for_legend(self, line: Any):
        if not line:
            return
        if line.tech not in self.seen_techs:
            if any('mmwave' in tech.lower() for tech in self.seen_techs) and 'mmwave' in line.tech.lower():
                return
            # Clone the line for legend but with solid line to emphasize color
            tech_line = plt.Line2D([], [], 
                                color=self.tech_conf[line.tech].get('color', 'black') if line.tech in self.tech_conf else 'black',
                                linestyle='-',
                                label=line.tech_label)
            self.tech_legend_handles.append(tech_line)
            self.tech_legend_labels.append(line.tech_label)
            self.seen_techs.add(line.tech)
        
    def add_tech_legend(self, ax: plt.Axes):
        # Create only technology legend
        ax.legend(handles=self.tech_legend_handles, 
                    loc='lower right', 
                    framealpha=0.7,
                    # prop={'weight': 'bold'},
                    )

    def plot(self, 
             output_filepath: str, 
             figsize: tuple = (3.5, 4),
             title: str = None, 
             x_lim_min: float = None,
             x_lim_max: float = None,
             x_step: float = None,
             show_operator_legend: bool = False, 
             show_tech_legend: bool = False,
             show_y_axis_label: bool = True,
             show_y_tick_labels: bool = True,
             left_margin: float = None,
             right_margin: float = None,
             top_margin: float = None,
             bottom_margin: float = None,
        ):
        """Create a plot comparing throughput distributions across technologies.
        
        Args:
            output_filepath: Path to save the output plot
            title: Title for the plot
            show_operator_legend: Whether to show the operator legend (distinguished by line styles)
            show_tech_legend: Whether to show the technology legend (distinguished by colors)
            left_margin: Left margin (0-1, None for auto)
            right_margin: Right margin (0-1, None for auto)
            top_margin: Top margin (0-1, None for auto)
            bottom_margin: Bottom margin (0-1, None for auto)
        """
        # Create a single figure with one axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Store lines for legend generation
        all_lines = []
        
        # Sort operators by their order if available
        operators = list(sorted(self.plot_data.keys()))
        
        # Plot data for each operator and technology
        for operator in operators:
            tech_data = self.plot_data[operator]
            
            # Plot data for each technology
            for tech in tech_data:
                if len(tech_data[tech]) < self.data_sample_threshold:
                    logger.warning(f'{operator}-{tech} data sample is less than required threshold, skip plotting: {len(tech_data[tech])} < {self.data_sample_threshold}')
                    continue
                line = self.add_category_plot(ax, operator, tech, self.data_field)
                self.track_tech_for_legend(line)
                if line:
                    all_lines.append(line)
        
        # Configure plot appearance
        if title:
            ax.set_title(title)

        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.4, linewidth=1)

        if show_y_axis_label:
            ax.set_ylabel('CDF')
        if not show_y_tick_labels:
            ax.set_yticklabels([])
        
        ax.set_xlabel('Throughput (Mbps)')
        self.set_x_lim(ax, x_lim_min=x_lim_min, x_lim_max=x_lim_max, x_step=x_step)
        ax.tick_params(axis='x', rotation=30)
        
        # Add legends if requested
        if show_operator_legend:
            self.add_operator_legend(ax, all_lines)
        elif show_tech_legend:
            self.add_tech_legend(ax)
        
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
        
        # Save figure with tight bbox for minimal external margins
        plt.savefig(output_filepath, dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.close()

def plot_tech_breakdown_cdfs_in_a_row(
        df: pd.DataFrame,
        data_field: str,
        output_filepath: str,
        operators: List[str],
        operator_conf: Dict[str, Dict],
        tech_conf: Dict[str, Dict],
        title: str = '',
        max_xlim: float | None = None,
        interval_x: float | None = None,
        data_sample_threshold: int = 240, # 1 rounds data (~2min)
    ):
    # Create figure with horizontal subplots (one per operator)
    n_operators = len(operators)
    
    left_margin = 0.12 if n_operators > 2 else 0.12
    y_label_pos = 0
    fig = plt.figure(figsize=(2.5 * 3, 3.5))  # Fixed size for 3 operators (2.5 * 3)
    gs = fig.add_gridspec(1, n_operators, left=left_margin, bottom=0.2, top=0.85)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_operators)]
    
    # Ensure axes is always an array even with single operator
    if n_operators == 1:
        axes = np.array([axes])
    
    # Adjust spacing between subplots
    if n_operators > 2:
        plt.subplots_adjust(wspace=0.2)
    else:
        plt.subplots_adjust(wspace=0.15)
    
    # Add title at the top middle of the figure if requested
    # fig.suptitle(title, y=0.98, size=16, weight='bold')
    
    # Set up common y-axis label for all subplots
    fig.text(y_label_pos, 0.5, 'CDF', 
             rotation=90,
             verticalalignment='center',
             size=14,
             weight='bold')
    
    # Set up common x-axis label for all subplots
    fig.text(0.5, 0.08, 'Throughput (Mbps)', 
             horizontalalignment='center',
             size=14,
             weight='bold')

    # Plot content for each operator
    for idx, operator in enumerate(operators):
        ax = axes[idx]
        # Filter data for this operator
        operator_df = df[df['operator'] == operator]

        # Plot each technology's CDF
        for tech, tech_cfg in tech_conf.items():
            if tech == 'NO SERVICE':
                continue
            # Filter data for this technology
            operator_tech_df = operator_df[operator_df[XcalField.ACTUAL_TECH] == tech]
            
            if len(operator_tech_df) < data_sample_threshold:
                logger.warning(f'{operator}-{tech} data sample is less than required threshold, skip plotting: {len(operator_tech_df)} < {data_sample_threshold}')
                continue
            
            # Sort data and compute CDF
            data_sorted = np.sort(operator_tech_df[data_field])
            cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
            
            # Plot CDF line with tech color
            ax.plot(data_sorted, cdf, 
                    color=tech_cfg['color'],
                    label=tech_cfg['label'])

            print('=' * 20)
            print(f'[{title} {operator} {tech}] Tput CDF stats:')
            print(TputCdfStatCollector(operator_tech_df, data_field).get_statistics())
        
        # Set operator name as subplot title
        ax.set_title(operator_conf[operator]['label'])
        
        # Basic axis setup
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both')
        
        # Set y-axis ticks from 0 to 1 with steps of 0.2
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        
        # Only show y ticks for the first subplot
        if idx > 0:
            ax.set_yticklabels([])
        
        if max_xlim is not None:
            ax.set_xlim(0, max_xlim)
            if interval_x:
                ax.set_xticks(np.arange(0, max_xlim + 1, interval_x))
        else:
            # Find global max x value
            actual_max_x_value = 0
            for operator in operators:
                operator_df = df[df['operator'] == operator]
                for tech, tech_cfg in tech_conf.items():
                    operator_tech_data = operator_df[operator_df[XcalField.ACTUAL_TECH] == tech][data_field]
                    if len(operator_tech_data) < data_sample_threshold:
                        continue
                    actual_max_x_value = max(actual_max_x_value, np.max(operator_tech_data))
            ax.set_xlim(0, actual_max_x_value)
            if interval_x:
                ax.set_xticks(np.arange(0, actual_max_x_value + 1, interval_x))
        
        # Add legend to the first subplot only
        ax.legend(loc='lower right')
    
    # Save the figure
    plt.savefig(output_filepath, dpi=600)
    logger.info(f'Saved plot to {output_filepath}')
    plt.close()


def plot_tput_tech_breakdown_with_areas_for_one_location(
        df: pd.DataFrame,
        protocol: str, 
        direction: str,
        location: str,
        location_conf: Dict[str, Dict],
        operator_conf: Dict[str, Dict],
        tech_conf: Dict[str, Dict],
        data_sample_threshold: int = 480,
        output_dir: str = '.',
        dl_data_field: str = XcalField.TPUT_DL,
        ul_data_field: str = XcalField.TPUT_UL,
        fig_index: int = 0,
    ):

    data_field_map = {
        'downlink': dl_data_field,
        'uplink': ul_data_field,
    }

    data_field = data_field_map[direction]

    location_tput_conf = location_conf[location].get(f'{protocol}_{direction}', {})
    location_df = df[df[CommonField.LOCATION] == location]
    # merge suburban and urban into one
    location_urban_df = location_df[(location_df[CommonField.AREA_TYPE] == 'urban') | (location_df[CommonField.AREA_TYPE] == 'suburban')]

    stats = {
        'urban': {},
        'rural': {},
    }


    print(f'Location: {location} Urban')
    res = TputWithTechDataGenerator(
        df=location_urban_df,
        data_field=data_field,
    ).generate()
    plot_data = res['plot_data']
    stats['urban'] = res['stats']

    show_y_axis_label = True
    show_y_tick_labels = True
    figsize = (3.4, 4)
    if fig_index > 0:
        show_y_axis_label = False
        show_y_tick_labels = False
        figsize = (3, 4)

    TputWithTechPlotter(
        plot_data=plot_data,
        operator_conf=operator_conf,
        tech_conf=tech_conf,
        data_sample_threshold=data_sample_threshold,
    ).plot(
        figsize=figsize,
        output_filepath=os.path.join(output_dir, f'cell_{protocol}_{direction}.{location}.urban.pdf'),
        show_tech_legend=True,
        x_lim_max=location_tput_conf.get('max_xlim', None),
        x_step=location_tput_conf.get('interval_x', None),
        show_y_axis_label=show_y_axis_label,
        show_y_tick_labels=show_y_tick_labels,
        left_margin=0.02 if not show_y_axis_label else 0.15,
        right_margin=0.98,
        top_margin=0.95,
        bottom_margin=0.15,
    )

    print(f'Location: {location} Rural')
    location_rural_df = location_df[location_df[CommonField.AREA_TYPE] == 'rural']
    res = TputWithTechDataGenerator(
        df=location_rural_df,
        data_field=data_field,
    ).generate()
    plot_data = res['plot_data']
    stats['rural'] = res['stats']

    TputWithTechPlotter(
        plot_data=plot_data,
        operator_conf=operator_conf,
        tech_conf=tech_conf,
        data_sample_threshold=data_sample_threshold,
    ).plot(
        figsize=(3, 4),
        output_filepath=os.path.join(output_dir, f'cell_{protocol}_{direction}.{location}.rural.pdf'),
        show_operator_legend=True,
        x_lim_max=location_tput_conf.get('max_xlim', None),
        x_step=location_tput_conf.get('interval_x', None),
        show_y_axis_label=False,
        show_y_tick_labels=False,
        left_margin=0.02,
        right_margin=0.98,
        top_margin=0.95,
        bottom_margin=0.15,
    )

    stat_output_path = os.path.join(output_dir, f'cell_{protocol}_{direction}.{location}.stats.json')   
    JsonDataManager.save(
        data=stats,
        filename=stat_output_path,
        indent=4,
    )
    print(f'Saved stats to {stat_output_path}')

def prepare_data_for_la_to_omaha(protocol: str, direction: str):
    suffix = '2024-11-01-2024-11-05'
    location = 'la_to_omaha'

    logger.info(f'-- Processing dataset: {location}')
    base_dir = os.path.join(bos_la_cell_location_conf[location]['root_dir'], 'processed')

    tput_data = {
        protocol: {
            direction: {},
        }
    }
    operator_df_list = []
    for operator in sorted(bos_la_cell_location_conf[location]['operators'], key=lambda x: cellular_operator_conf[x]['order']):
        logger.info(f'---- Processing operator: {operator}')
        tput_csv_path = os.path.join(base_dir, 'throughput', LaToBosDataLoader.get_xcal_filename(operator, f'{protocol}_{direction}', suffix))
        df = pd.read_csv(tput_csv_path)
        df['operator'] = operator
        df['location'] = location
        logger.info(f'---- {protocol} {direction} df ({df.shape[0]} rows): {tput_csv_path}')
        operator_df_list.append(df)

    tput_data[protocol][direction] = pd.concat(operator_df_list)
    return tput_data

def main():
    output_dir = os.path.join(current_dir, 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # process the rest
    locations=['alaska', 'hawaii']
    protocol='tcp'
    direction='downlink'
    tput_df = aggregate_xcal_tput_data_by_location(
        locations=locations,
        location_conf=cellular_location_conf,
        protocol=protocol,
        direction=direction
    )
    # load la_to_omaha data
    la_to_omaha_tput_data = prepare_data_for_la_to_omaha(protocol, direction)
    tput_df = pd.concat([tput_df, la_to_omaha_tput_data[protocol][direction]])

    for location in locations:
        if location == 'alaska':
            fig_index = 0
        elif location == 'hawaii':
            fig_index = 1
        print(f'Processing {location}')
        plot_tput_tech_breakdown_with_areas_for_one_location(
            df=tput_df,
            protocol=protocol,
            direction=direction,
            location=location,
            location_conf=cellular_location_conf,
            operator_conf=cellular_operator_conf,
            tech_conf=tech_conf,
            data_sample_threshold=480,
            output_dir=output_dir,
            fig_index=fig_index,
        )
        print(f'Finished {location}, saved to {output_dir}')

    # plot la_to_omaha data
    print('Processing la_to_omaha')
    plot_tput_tech_breakdown_with_areas_for_one_location(
        location='la_to_omaha',
        df=tput_df,
        protocol=protocol,
        direction=direction,
        location_conf=bos_la_cell_location_conf,
        operator_conf=cellular_operator_conf,
        tech_conf=tech_conf,
        data_sample_threshold=480,
        output_dir=output_dir,
        dl_data_field=CommonField.DL_TPUT_MBPS,
        ul_data_field=CommonField.UL_TPUT_MBPS,
        fig_index=2,
    )

if __name__ == '__main__':
    main()