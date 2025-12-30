import os
import sys
from typing import Callable

import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../..'))

from scripts.utilities.data_collection.CdfStatCollector import CdfStatCollector
from scripts.utilities.io import JsonDataManager
from scripts.utilities.math_utils import MathUtils


from scripts.plotting.common import location_conf

def get_count_and_percentage_of_samples(df: pd.DataFrame, mask: Callable):
    samples = df[mask]
    return {
        'samples': len(samples),
        'total_samples': len(df),
        'percentage': MathUtils.percentage(len(samples), len(df)),
    }

def main():
    base_dir = os.path.join(current_dir, '../fig17_delta_tput_between_operators/outputs')
    output_dir = os.path.join(current_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    for location in ['alaska', 'hawaii']:
        print('=' * 20)
        print(f'{location.title()} stats:')

        loc_conf = location_conf[location]
        location_wise_stats = {}
        for trace_type in ['tcp_downlink', 'tcp_uplink']:
            print(f'-- {trace_type.title()}:')

            location_wise_stats[trace_type] = {}

            for op_a, op_b in loc_conf['op_pairs']:
                location_wise_stats[trace_type][f'{op_a}_{op_b}'] = {}

                diff_tput_csv_name = f'diff_tput_cdf.{location}.{trace_type}.{op_a}_{op_b}.csv'
                df = pd.read_csv(os.path.join(base_dir, diff_tput_csv_name))
                
                stats = {
                    'op_a': {
                        'name': op_a,
                    },
                    'op_b': {
                        'name': op_b,
                    },
                }
                stats['cdf'] = CdfStatCollector(df, 'diff_throughput_mbps').get_statistics()

                stats['op_a']['zero_tput'] = get_count_and_percentage_of_samples(df, mask=(
                    (df['A_throughput_mbps'] == 0)
                ))
                stats['op_b']['zero_tput'] = get_count_and_percentage_of_samples(df, mask=(
                    (df['B_throughput_mbps'] == 0)
                ))

                stats['concurrent_zero_tput'] = get_count_and_percentage_of_samples(df, mask=(
                    (df['A_throughput_mbps'] == 0) & (df['B_throughput_mbps'] == 0)
                ))
                stats['concurrent_small_tput_1mbps'] = get_count_and_percentage_of_samples(df, mask=(
                    (df['A_throughput_mbps'] < 1) & (df['B_throughput_mbps'] < 1)
                ))
                stats['concurrent_small_tput_5mbps'] = get_count_and_percentage_of_samples(df, mask=(
                    (df['A_throughput_mbps'] < 5) & (df['B_throughput_mbps'] < 5)
                ))
                
                location_wise_stats[trace_type][f'{op_a}_{op_b}'] = stats
                print(f'---- Operator A ({op_a}): {stats["op_a"]["zero_tput"]}')
                print(f'---- Operator B ({op_b}): {stats["op_b"]["zero_tput"]}')
                print(f'---- Concurrent Zero Tput: {stats["concurrent_zero_tput"]}\n')
                
        JsonDataManager.save(location_wise_stats, os.path.join(output_dir, f'diff_tput_cdf_stats.{location}.json'))
        print('=' * 20)
if __name__ == '__main__':
    main()
