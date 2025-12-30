import logging
import os
import sys
from typing import Dict, override

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../'))

from scripts.plotting.fig6_cell_kpis_across_loc.main import NetworkKpiPlotter
from scripts.plotting.fig12_starlink_network_kpis_across_locations.main import get_tput_data_for_alaska_and_hawaii
from scripts.constants import CommonField, XcalField
from scripts.logging_utils import SilentLogger, create_logger
from scripts.shared.plotting import IPlotDataGenerator
from scripts.plotting.common import aggregate_xcal_tput_data_by_location, cellular_location_conf
from scripts.plotting.common import aggregate_latency_data_by_location, operator_conf, location_conf

class NonContiguousDataGenerator(IPlotDataGenerator):
    def __init__(self,
            logger: logging.Logger = None
        ):
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
        self.data = {
            'tcp_dl': {},
            'tcp_ul': {},
            'rtt': {},
        }
        self.data['tcp_dl'] = self.get_tput_data_of_non_contiguous_us(protocol='tcp', direction='downlink')
        self.data['tcp_ul'] = self.get_tput_data_of_non_contiguous_us(protocol='tcp', direction='uplink')
        self.data['rtt'] = self.get_rtt_data_of_non_contiguous_us()
        return self.data

    def get_tput_data_of_non_contiguous_us(self, protocol, direction):
        # Cellular Xcal AK, HI
        ak_hi_cell_df = aggregate_xcal_tput_data_by_location(
            locations=['alaska', 'hawaii'],
            location_conf=cellular_location_conf,
            protocol=protocol,
            direction=direction,
        )

        ak_hi_cell_df.rename(columns={    
            CommonField.TIME: CommonField.LOCAL_DT,
        }, inplace=True)
        if direction == 'downlink':
            ak_hi_cell_df.rename(columns={    
                XcalField.TPUT_DL: CommonField.TPUT_MBPS,
            }, inplace=True)
        elif direction == 'uplink':
            ak_hi_cell_df.rename(columns={    
                XcalField.TPUT_UL: CommonField.TPUT_MBPS,
            }, inplace=True)
        else:
            raise ValueError(f'Invalid direction: {direction}')
        ak_hi_cell_df = ak_hi_cell_df[[
            CommonField.LOCAL_DT,
            CommonField.TPUT_MBPS,
            CommonField.OPERATOR,
            CommonField.LOCATION,
            CommonField.AREA_TYPE,
        ]]

        # Starlink App Tput AK, HI
        ak_hi_starlink_df = get_tput_data_for_alaska_and_hawaii(protocol, direction)
        ak_hi_starlink_df.rename(columns={    
            CommonField.TIME: CommonField.LOCAL_DT,
        }, inplace=True)
        ak_hi_starlink_df = ak_hi_starlink_df[[
            CommonField.LOCAL_DT,
            CommonField.TPUT_MBPS,
            CommonField.OPERATOR,
            CommonField.LOCATION,
            CommonField.AREA_TYPE,
        ]]

        main_df = pd.concat([ak_hi_cell_df, ak_hi_starlink_df], ignore_index=True)
        return main_df

    def get_rtt_data_of_non_contiguous_us(self):
        # Cellular & Starlink RTT in AK, HI
        locations = ['alaska', 'hawaii']
        ak_hi_cell_starlink_df = aggregate_latency_data_by_location(
            locations=locations,
            location_conf=location_conf,
        )
        # Check available columns and select only those that exist
        available_cols = ak_hi_cell_starlink_df.columns.tolist()
        required_cols = [
            CommonField.LOCAL_DT, 
            CommonField.RTT_MS, 
            CommonField.OPERATOR, 
            CommonField.LOCATION,
            CommonField.AREA_TYPE,
        ]
        missing_cols = [col for col in required_cols if col not in available_cols]
        if missing_cols:
            self.logger.warning(f"Missing columns in AK/HI RTT data: {missing_cols}")
            self.logger.info(f"Available columns: {available_cols}")
        
        ak_hi_cell_starlink_df = ak_hi_cell_starlink_df[[col for col in required_cols if col in available_cols]].copy()
        
        return ak_hi_cell_starlink_df

class AlaskaDataGenerator(IPlotDataGenerator):
    def __init__(self, base_data_generator: NonContiguousDataGenerator, logger: logging.Logger = None):
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
    def __init__(self, base_data_generator: NonContiguousDataGenerator, logger: logging.Logger = None):
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

class NonContiguousNetworkKpiPlotter(NetworkKpiPlotter):
    def __init__(self, 
            data_generator: IPlotDataGenerator,
            operator_conf: Dict[str, Dict],
            location_conf: Dict[str, Dict],
        ):
        super().__init__(
            data_generator=data_generator,
            operator_conf=operator_conf,
            location_conf=location_conf,
        )
    
    @override
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
                'show_legend': 'location',
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
    non_contiguous_data_generator = NonContiguousDataGenerator(logger=logger)
    non_contiguous_data_generator.generate()

    NonContiguousNetworkKpiPlotter(
        data_generator=non_contiguous_data_generator,
        operator_conf=operator_conf,
        location_conf=location_conf,
    ).plot(
        fig_width=2.8,
        fig_height=2.4,
        label_font_size=10,
        tick_label_font_size=10,
        x_tick_rotation=30,
        output_filename=os.path.join(output_dir, 'starlink_cell_kpi.non_contiguous.pdf'),
    )

if __name__ == "__main__":
    main()