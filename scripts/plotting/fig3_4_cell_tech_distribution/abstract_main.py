import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from scripts.plotting.fig3_4_cell_tech_distribution.main import (
    plot_tech_dist_stack,
    plot_tech_dist_stack_with_area_sidebyside,
)
from scripts.plotting.common import LaToBosDataLoader, cellular_operator_conf, cellular_location_conf, tech_conf
from scripts.constants import XcalField, CommonField
from scripts.logging_utils import create_logger

current_dir = os.path.dirname(os.path.abspath(__file__))
logger = create_logger('abs_cell_tech_distribution', filename=os.path.join(current_dir, 'outputs', 'abs_cell_tech_distribution.log'))


def get_tput_rtt_df_for_ak_or_hi(location: str, base_dir: str, loc_map: dict):
    xcal_dir = os.path.join(base_dir, loc_map['xcal_dir'])
    operator_dfs = {}
    for operator in sorted(cellular_location_conf[location]['operators'], key=lambda x: cellular_operator_conf[x]['order']):
        logger.info(f'---- Processing operator: {operator}')
        smart_tput_csv_path = os.path.join(xcal_dir, loc_map['xcal_filename'](operator))
        rtt_csv_path = os.path.join(base_dir, loc_map['ping_dir'], loc_map['ping_filename'](operator))

        smart_tput_df = pd.read_csv(smart_tput_csv_path)
        smart_tput_df['type'] = smart_tput_df[XcalField.APP_TPUT_PROTOCOL] + '_' + smart_tput_df[XcalField.APP_TPUT_DIRECTION]
        rtt_df = pd.read_csv(rtt_csv_path)
        rtt_df['type'] = 'rtt'

        tput_sub_df = smart_tput_df[[CommonField.LOCAL_DT, CommonField.AREA_TYPE, XcalField.SEGMENT_ID, XcalField.ACTUAL_TECH, XcalField.LON, XcalField.LAT, 'type']]
        rtt_sub_df = rtt_df[[CommonField.LOCAL_DT, CommonField.AREA_TYPE, XcalField.SEGMENT_ID, XcalField.ACTUAL_TECH, XcalField.LON, XcalField.LAT, 'type']]
        df = pd.concat([tput_sub_df, rtt_sub_df], ignore_index=True)
        df.sort_values(by=[CommonField.LOCAL_DT], inplace=True)
        operator_dfs[operator] = df
    return operator_dfs


def get_tput_rtt_df_for_la_to_omaha(location: str, base_dir: str):
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

        tput_dl_sub_df = tcp_dl_df[[CommonField.LOCAL_DT, CommonField.UTC_TS, CommonField.AREA_TYPE, CommonField.SEGMENT_ID, CommonField.ACTUAL_TECH, CommonField.LON, CommonField.LAT, 'type']]
        tput_ul_sub_df = tcp_ul_df[[CommonField.LOCAL_DT, CommonField.UTC_TS, CommonField.AREA_TYPE, CommonField.SEGMENT_ID, CommonField.ACTUAL_TECH, CommonField.LON, CommonField.LAT, 'type']]
        rtt_sub_df = rtt_df[[CommonField.LOCAL_DT, CommonField.UTC_TS, CommonField.AREA_TYPE, CommonField.SEGMENT_ID, CommonField.ACTUAL_TECH, CommonField.LON, CommonField.LAT, 'type']]
        df = pd.concat([tput_dl_sub_df, tput_ul_sub_df, rtt_sub_df], ignore_index=True)
        df.sort_values(by=[CommonField.LOCAL_DT], inplace=True)
        operator_dfs[operator] = df
    return operator_dfs


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
        output_dir = os.path.join(current_dir, loc_map['output_dir'], location)
        legend_fontsize = 13 if location in ['alaska', 'hawaii'] else 11  

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if location in ['alaska', 'hawaii']:
            operator_dfs = get_tput_rtt_df_for_ak_or_hi(location, base_dir, loc_map)
            area_field = XcalField.AREA
            segment_id_field = XcalField.SEGMENT_ID
            lat_field = XcalField.LAT
            lon_field = XcalField.LON
        elif location in ['la_to_omaha']:
            operator_dfs = get_tput_rtt_df_for_la_to_omaha(location, base_dir)
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
            figsize=(3.6, 4.2),
            title=f'Technology Distribution ({cellular_location_conf[location]["label"]}-All Areas)',
            fig_name=f'abs_tech_dist_stack_all_areas.{location}',
            segment_id_field=segment_id_field,
            lat_field=lat_field,
            lon_field=lon_field,
            timestamp_field=CommonField.LOCAL_DT if location in ['alaska', 'hawaii'] else CommonField.UTC_TS,
            legend_fontsize=legend_fontsize,
            label_fontsize=18,
            tick_label_fontsize=18,

        )


if __name__ == '__main__':
    main()
