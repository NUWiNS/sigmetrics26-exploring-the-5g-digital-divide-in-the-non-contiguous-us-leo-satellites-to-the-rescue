from typing import List

import numpy as np
import pandas as pd


class MathUtils:
    @staticmethod
    def percentage(value: float, total: float, precision: int = 2) -> float | None:
        if total == 0:
            return None
        return round((value / total) * 100, precision)
    
    @staticmethod
    def cdf(df: List | pd.DataFrame | pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sorted_data = np.sort(df)
        ranks = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        return sorted_data, ranks
