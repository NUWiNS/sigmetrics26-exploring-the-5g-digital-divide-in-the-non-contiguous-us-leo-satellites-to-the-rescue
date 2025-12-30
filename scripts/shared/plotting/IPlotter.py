from abc import ABC, abstractmethod

class IPlotter(ABC):
    @abstractmethod
    def plot(self, *args, **kwargs):
        pass
