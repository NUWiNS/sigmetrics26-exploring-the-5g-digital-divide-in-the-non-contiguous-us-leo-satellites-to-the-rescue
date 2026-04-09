import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../'))

from scripts.plotting.fig12_starlink_network_kpis_across_locations.main import (
    StarlinkDataGenerator,
    StarlinkNetworkKpiPlotter,
)
from scripts.plotting.common import location_conf, operator_conf
from scripts.logging_utils import create_logger


def main():
    # Create output directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Create logger
    logger = create_logger(
        'abs_network_kpis',
        filename=os.path.join(output_dir, 'abs_plot_network_kpis.log'),
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
        fig_width=3.2,
        fig_height=3.6,
        x_tick_rotation=30,
        override_tcp_dl_conf={
            'label_font_size': 16,
            'tick_label_font_size': 15,
            'legend_font_size': 14,
        },
        override_tcp_ul_conf={
            'label_font_size': 16,
            'tick_label_font_size': 16,
            'legend_font_size': 14,
        },
        override_rtt_conf={
            'label_font_size': 16,
            'tick_label_font_size': 16,
        },
        output_filename=os.path.join(output_dir, 'abs_starlink_kpi.across_locations.pdf'),
    )


if __name__ == "__main__":
    main()
