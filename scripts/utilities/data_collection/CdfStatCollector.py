from typing import Dict
import numpy as np
import pandas as pd

from scripts.utilities.math_utils import MathUtils


class CdfSeriesStatCollector:
    def __init__(self, series: pd.Series, name: str = 'default'):
        self.series = series
        self.name = name
        self._output_df = None

    def get_digest(self, stats: Dict) -> str:
        return f'{self.name}: min:{stats["min"]:.2f} / p5:{stats["5th"]:.2f} / p25:{stats["25th"]:.2f} / p50:{stats["median"]:.2f} / p75:{stats["75th"]:.2f} / p95:{stats["95th"]:.2f} / max:{stats["max"]:.2f}'

    def get_statistics(self) -> Dict[str, float]:
        """Calculate basic statistics from the data.
        """
        if len(self.series) == 0:
            return {}
        
        values = self.series.values
        total_samples = len(values)
        nan_count = np.isnan(values).sum()
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) == 0:
            return {
                'total_samples': total_samples,
                'valid_samples': 0,
                'valid_samples_percentage': 0,
                'nan_samples': int(nan_count),
                'nan_samples_percentage': MathUtils.percentage(int(nan_count), int(total_samples)),
            }
            
        res = {
            'min': float(np.min(valid_values)),
            '5th': float(np.percentile(valid_values, 5)),
            '25th': float(np.percentile(valid_values, 25)),
            'median': float(np.percentile(valid_values, 50)),
            '75th': float(np.percentile(valid_values, 75)),
            '95th': float(np.percentile(valid_values, 95)),
            'max': float(np.max(valid_values)),
            'total_samples': total_samples,
            'valid_samples': len(valid_values),
            'valid_samples_percentage': MathUtils.percentage(len(valid_values), total_samples),
            'nan_samples': int(nan_count),
            'nan_samples_percentage': MathUtils.percentage(int(nan_count), int(total_samples)),
        }
        return res

    def format_statistics(self) -> str:
        stats = self.get_statistics()
        
        # Format the statistics into a clean, readable string
        formatted_output = [
            f"Summary of series{self.name} ({stats['total_samples']:,} samples):",
            f"  • Value Range: {stats['min']:.2f} - {stats['max']:.2f}",
            f"  • Median: {stats['median']:.2f}",
            f"  • 25th-75th Percentile: {stats['25th']:.2f} - {stats['75th']:.2f}"
        ]
        
        # Check if there are any NaN values
        if stats['nan_samples'] > 0:
            formatted_output.append(
                f"  • Invalid Samples: {stats['nan_samples']:,} ({stats['nan_samples_percentage']:.2f}%)"
            )
            
        return "\n".join(formatted_output)
    
    def get_df(self) -> pd.DataFrame:
        if self._output_df is None:
            stats = self.get_statistics()
            rows = []
            for stat_name, stat_value in stats.items():
                rows.append({"series": self.name, "statistic": stat_name, "value": stat_value})
            self._output_df = pd.DataFrame(rows)
        return self._output_df
    
    def save_as_csv(self, output_filename: str):
        df = self.get_df()
        df.to_csv(output_filename, index=False)
    

class CdfStatCollector(CdfSeriesStatCollector):
    def __init__(self, df: pd.DataFrame, data_field: str, name: str = 'default'):
        super().__init__(df[data_field].astype(float), name)
        self.df = df
        self.data_field = data_field
