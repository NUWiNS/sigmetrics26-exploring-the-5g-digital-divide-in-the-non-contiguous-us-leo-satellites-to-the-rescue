from typing import Dict
import pandas as pd

from scripts.utilities.math_utils import MathUtils
from .CdfStatCollector import CdfStatCollector


class TputCdfStatCollector(CdfStatCollector):
    def __init__(self, df: pd.DataFrame, data_field: str, name: str = 'default'):
        super().__init__(df, data_field, name)

    def get_statistics(self) -> Dict[str, float]:
        res = super().get_statistics()
        res['tput_0mbps_samples'] = len(self.series[self.series == 0])
        res['tput_0mbps_samples_percentage'] = MathUtils.percentage(res['tput_0mbps_samples'], res['total_samples'])
        res['tput_less_1mbps_samples'] = len(self.series[self.series < 1])
        res['tput_less_1mbps_samples_percentage'] = MathUtils.percentage(res['tput_less_1mbps_samples'], res['total_samples'])
        return res

    def get_digest(self, stats: Dict) -> str:
        digest = super().get_digest(stats)
        digest += f' / 0mbps:{stats["tput_0mbps_samples_percentage"]:.2f}% / <1mbps:{stats["tput_less_1mbps_samples_percentage"]:.2f}%'
        return digest
    
    def format_statistics(self) -> str:
        stats = self.get_statistics()
        
        # Format the statistics into a clean, readable string
        formatted_output = [
            f"Summary Statistics ({stats['total_samples']:,} samples):",
            f"  • Throughput Range: {stats['min']:.2f} - {stats['max']:.2f} Mbps",
            f"  • Median: {stats['median']:.2f} Mbps",
            f"  • 25th-75th Percentile: {stats['25th']:.2f} - {stats['75th']:.2f} Mbps",
            f"  • Outage Statistics:",
            f"    - 0 Mbps Samples: {stats['tput_0mbps_samples']:,} ({stats['tput_0mbps_samples_percentage']:.2f}%)",
            f"    - <1 Mbps Samples: {stats['tput_less_1mbps_samples']:,} ({stats['tput_less_1mbps_samples_percentage']:.2f}%)"
        ]
        
        # Check if there are any NaN values
        if stats['nan_samples'] > 0:
            formatted_output.append(
                f"  • Invalid Samples: {stats['nan_samples']:,} ({stats['nan_samples_percentage']:.2f}%)"
            )
            
        return "\n".join(formatted_output)