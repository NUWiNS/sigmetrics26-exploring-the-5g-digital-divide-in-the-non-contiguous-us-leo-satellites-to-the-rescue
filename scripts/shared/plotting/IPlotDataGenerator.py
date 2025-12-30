from abc import ABC, abstractmethod
from typing import Dict, List

class IPlotDataGenerator(ABC):
    @abstractmethod
    def get_plot_data(self, *args, **kwargs) -> dict:
        pass

    @abstractmethod
    def get_plot_data_config(self, *args, **kwargs) -> dict:
        pass

    @abstractmethod
    def get_plot_order(self, *args, **kwargs) -> list:
        pass

    
    
