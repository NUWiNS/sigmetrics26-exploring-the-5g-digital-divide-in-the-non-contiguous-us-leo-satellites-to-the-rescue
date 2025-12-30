import json
import os
from typing import Dict, List
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from scripts.constants import CommonField
from scripts.logging_utils import create_logger
from scripts.plotting.fig8_cell_tcp_dl_with_areas.main import plot_tput_tech_breakdown_with_areas_for_one_location, prepare_data_for_la_to_omaha
from scripts.plotting.common import aggregate_xcal_tput_data_by_location, cellular_location_conf, cellular_operator_conf, tech_conf
from scripts.plotting.configs.mainland import cellular_location_conf as bos_la_cell_location_conf

current_dir = os.path.dirname(os.path.abspath(__file__))
logger = create_logger('tcp_ul_with_areas', filename=os.path.join(current_dir, 'outputs', 'tcp_ul_with_areas.log'))


def main():
    output_dir = os.path.join(current_dir, 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # process the rest
    locations=['alaska', 'hawaii']
    protocol='tcp'
    direction='uplink'
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