import logging
import os
import sys
from typing import Dict, override

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../'))

from scripts.plotting.fig6_cell_kpis_across_loc.main import NetworkKpiPlotter
from scripts.constants import CommonField
from scripts.logging_utils import SilentLogger, create_logger
from scripts.shared.plotting import IPlotDataGenerator
from scripts.utilities.DatasetHelper import DatasetHelper
from scripts.plotting.common import location_conf, operator_conf, LaToBosDataLoader, aggregate_operator_latency_data

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

class StarlinkDataGenerator(IPlotDataGenerator):
    def __init__(self,
            operator_conf: Dict[str, Dict],
            location_conf: Dict[str, Dict],
            logger: logging.Logger = None
        ):
        self.logger = logger or SilentLogger()
        # Collect all data
        self.operator_conf = operator_conf
        self.location_conf = location_conf
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
        self.data['tcp_dl'] = self.get_tput_data_across_locations(protocol='tcp', direction='downlink')
        self.data['tcp_ul'] = self.get_tput_data_across_locations(protocol='tcp', direction='uplink')
        self.data['rtt'] = self.get_rtt_data_of_non_contiguous_us()
        return self.data

    def get_tput_data_across_locations(self, protocol, direction):
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
        ]]
        # Starlink App Tput in mainland US
        mainland_df = self.prepare_tput_data_for_la_to_omaha(protocol, direction)
        mainland_df = mainland_df[[
            CommonField.LOCAL_DT,
            CommonField.TPUT_MBPS,
            CommonField.OPERATOR,
            CommonField.LOCATION,
        ]]
        main_df = pd.concat([ak_hi_starlink_df, mainland_df], ignore_index=True)
        return main_df

    def get_rtt_data_of_non_contiguous_us(self):
        required_cols = [CommonField.LOCAL_DT, CommonField.RTT_MS, CommonField.OPERATOR, CommonField.LOCATION]

        # Starlink RTT in AK, HI
        starlink_df_list = []
        for location in ['alaska', 'hawaii']:
            df = aggregate_operator_latency_data(
                root_dir=os.path.join(self.location_conf[location]['root_dir'], 'processed'),
                operators=['starlink'],
            )
            df[CommonField.LOCATION] = location
            df = df[required_cols].copy()
            starlink_df_list.append(df)

        # Starlink RTT in mainland US
        mainland_df = self.prepare_rtt_data_for_la_to_omaha()
        mainland_df = mainland_df[required_cols].copy()
        starlink_df_list.append(mainland_df)

        return pd.concat(starlink_df_list, ignore_index=True)
    

    def prepare_tput_data_for_la_to_omaha(self, protocol: str, direction: str):
        suffix = '2024-11-01-2024-11-05'
        location = 'la_to_omaha'
        self.logger.info(f'-- Processing dataset: {location}')
        base_dir = os.path.join(self.location_conf[location]['root_dir'], 'processed')

        operator_df_list = []
        for operator in ['starlink']:
            self.logger.info(f'---- Processing operator: {operator}')
            tput_csv_path = os.path.join(base_dir, 'throughput', LaToBosDataLoader.get_app_tput_filename(
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
        for operator in ['starlink']:
            self.logger.info(f'---- Processing operator: {operator}')
            csv_path = os.path.join(base_dir, 'latency', LaToBosDataLoader.get_rtt_filename(operator, suffix))
            df = pd.read_csv(csv_path)
            df[CommonField.OPERATOR] = operator
            df[CommonField.LOCATION] = location
            self.logger.info(f'---- ping df ({df.shape[0]} rows): {csv_path}')
            operator_df_list.append(df)
        return pd.concat(operator_df_list)

class StarlinkNetworkKpiPlotter(NetworkKpiPlotter):
    def __init__(self, 
            data_generator: IPlotDataGenerator,
            operator_conf: Dict[str, Dict],
            location_conf: Dict[str, Dict],
        ):
        super().__init__(
            data_generator=data_generator,
            operator_conf=operator_conf,
            location_conf=location_conf,
            operator_legend_presentation='linestyle',
            location_legend_presentation='color',
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
                'x_limit': (0, 400),
                'x_step': 100,
                'filename_suffix': 'tcp_dl',
                'show_y_label': True,
                'show_y_ticks': True,
                'show_legend': 'location',
            },
            'tcp_ul': {
                'fig_width': fig_width,
                'fig_height': fig_height,
                'xlabel': 'Throughput (Mbps)', 
                'title': 'TCP Uplink Throughput',
                'value_col': CommonField.TPUT_MBPS,
                'x_limit': (0, 30),
                'x_step': 5,
                'filename_suffix': 'tcp_ul',
                'show_y_label': False,
                'show_y_ticks': False,
                'show_legend': False,
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
    starlink_data_generator = StarlinkDataGenerator(
        operator_conf=operator_conf,
        location_conf=location_conf,
        logger=logger,
    )
    starlink_data_generator.generate()

    StarlinkNetworkKpiPlotter(
        data_generator=starlink_data_generator,
        operator_conf=operator_conf,
        location_conf=location_conf,
    ).plot(
        fig_width=2.6,
        fig_height=2,
        label_font_size=10,
        tick_label_font_size=10,
        output_filename=os.path.join(output_dir, 'starlink_kpi.across_locations.pdf'),
    )

if __name__ == "__main__":
    main()