from abc import ABC, abstractmethod
from typing import Dict, List

from .IPlotDataGenerator import IPlotDataGenerator

class IMultiPlotDataGenerator(ABC):
    @abstractmethod
    def get_data_generator_map(self, *args, **kwargs) -> Dict[str, IPlotDataGenerator]:
        pass

    @abstractmethod
    def get_figure_data_order(self, *args, **kwargs) -> List[str]:
        pass

    @abstractmethod
    def get_figure_data_config(self, *args, **kwargs) -> Dict[str, Dict]:
        pass


    
    
