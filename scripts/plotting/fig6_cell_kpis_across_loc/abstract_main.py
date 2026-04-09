import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../'))

from scripts.plotting.fig6_cell_kpis_across_loc.main import (
    DataGenerator,
    NetworkKpiPlotter,
)
from scripts.plotting.common import cellular_operator_conf, cellular_location_conf
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
    plotter.plot(
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
            'legend_font_size': 15,
        },
        override_rtt_conf={
            'label_font_size': 16,
            'tick_label_font_size': 16,
            'legend_font_size': 15,
        },
        output_filename=os.path.join(output_dir, 'abs_cell_kpi_across_locations.pdf'),
    )


if __name__ == "__main__":
    main()
