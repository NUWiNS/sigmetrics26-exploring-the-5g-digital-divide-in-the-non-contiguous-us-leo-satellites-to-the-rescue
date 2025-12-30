import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from scripts.configs.alaska import ROOT_DIR
from scripts.logging_utils import create_logger
from scripts.cdf_tput_plotting_utils import get_data_frame_from_all_csv

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(ROOT_DIR, 'processed', 'throughput')
tput_cubic_dir = os.path.join(ROOT_DIR, 'processed', 'throughput_cubic')
tput_bbr_dir = os.path.join(ROOT_DIR, 'processed', 'throughput_bbr')
output_dir = os.path.join(current_dir, 'outputs')
xcal_dir = os.path.join(ROOT_DIR, 'processed', 'xcal')

logger = create_logger('logger', filename=os.path.join(output_dir, 'fig2_tput_with_cc_and_buffer.log'))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def get_data_frame_from_all_csv(operator: str, protocol: str, direction: str, base_dir: str = base_dir):
    csv_filename = f'{operator}_{protocol}_{direction}.csv'
    file_path = os.path.join(base_dir, csv_filename)
    df = pd.read_csv(file_path)
    logger.info(f'{csv_filename} count: {df.count()}')
    return df


def read_and_plot_cdf_tcp_tput_with_cubic_vs_bbr(protocol: str, direction: str, output_dir: str):
    cubic_df = pd.DataFrame()
    bbr_df = pd.DataFrame()
    for operator in ['att', 'verizon', 'starlink']:
        sub_cubic_df = get_data_frame_from_all_csv(operator, protocol, direction, base_dir=tput_cubic_dir)
        sub_cubic_df['operator'] = operator
        cubic_df = pd.concat([cubic_df, sub_cubic_df], ignore_index=True)

        sub_bbr_df = get_data_frame_from_all_csv(operator, protocol, direction, base_dir=tput_bbr_dir)
        sub_bbr_df['operator'] = operator
        bbr_df = pd.concat([bbr_df, sub_bbr_df], ignore_index=True)

    plot_cdf_tcp_tput_with_cubic_vs_bbr(
        cubic_df,
        bbr_df,
        protocol=protocol,
        direction=direction,
        output_dir=output_dir
    )



def plot_cdf_tcp_tput_with_cubic_vs_bbr(
        cubic_df: pd.DataFrame,
        bbr_df: pd.DataFrame,
        protocol: str,
        direction: str = 'downlink',
        output_dir='.'
):
    config = {
        'legends': ['Starlink (CUBIC)', 'Cellular (CUBIC)', 'Starlink (BBR)', 'Cellular (BBR)'],
        'filename': f'cubic_vs_bbr_{protocol}_{direction}.pdf'
    }
    all_throughputs = []

    cmap20 = plt.cm.tab20

    # Compare Urban performance only
    starlink_cubic = cubic_df[
        (cubic_df['operator'] == 'starlink') & (cubic_df['area'] == 'urban')
        ]['throughput_mbps']
    cellular_cubic = cubic_df[
        (cubic_df['operator'] != 'starlink') & (cubic_df['area'] == 'urban')
        ]['throughput_mbps']

    starlink_bbr = bbr_df[
        (bbr_df['operator'] == 'starlink') & (bbr_df['area'] == 'urban')
        ]['throughput_mbps']
    cellular_bbr = bbr_df[
        (bbr_df['operator'] != 'starlink') & (bbr_df['area'] == 'urban')
        ]['throughput_mbps']

    logger.info('Starlink CUBIC: %s', starlink_cubic.describe())
    logger.info('Cellular CUBIC: %s', cellular_cubic.describe())
    logger.info('Starlink BBR: %s', starlink_bbr.describe())
    logger.info('Cellular BBR: %s', cellular_bbr.describe())

    all_throughputs.extend([starlink_cubic, cellular_cubic, starlink_bbr, cellular_bbr])
    colors = ['black', cmap20(4), 'black', cmap20(4)]
    linestyles = ['--', '--', '-', '-']

    fig, ax = plt.subplots(figsize=(6, 4.6))

    for idx, data in enumerate(all_throughputs):
        sorted_data = np.sort(data)
        count, bins_count = np.histogram(sorted_data, bins=np.unique(sorted_data).shape[0])
        cdf = np.cumsum(count) / len(sorted_data)
        plt.plot(bins_count[1:], cdf, label=config['legends'][idx], color=colors[idx],
                 linestyle=linestyles[idx], linewidth=4)

    fzsize = 22
    ax.tick_params(axis='y', labelsize=fzsize)
    ax.tick_params(axis='x', labelsize=fzsize)
    ax.set_xlabel('Throughput (Mbps)', fontsize=fzsize)
    ax.set_ylabel('CDF', fontsize=fzsize)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(prop={'size': 20}, loc='lower right')
    if direction == 'uplink':
        max_tput = 100
        plt.xlim(0, max_tput)
        ax.set_xticks(range(0, max_tput + 1, 25))
    else:
        max_tput = 250
        plt.xlim(0, max_tput)
        ax.set_xticks(range(0, max_tput + 1, 50))
    plt.ylim(0, 1.02)
    plt.grid(True)
    plt.tight_layout()
    output_file_path = os.path.join(output_dir, config['filename'])
    plt.savefig(output_file_path)
    logger.info(f'Saved {output_file_path}')



def main():
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Dataset folder does not exist: {base_dir} ")

    # Plot the CDF of throughput
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Cubic vs BBR
    read_and_plot_cdf_tcp_tput_with_cubic_vs_bbr('tcp', 'downlink', output_dir)


if __name__ == '__main__':
    main()
