from abc import ABC, abstractmethod
import pandas as pd


class BaseCycle(ABC):
    @abstractmethod
    def get_intervals(self, loader, user_id, start_date, end_date) -> pd.DataFrame:
        """Returns a DataFrame with dates as index and 'start' and 'end' columns for each cycle."""
        pass


class CycleRegistry:
    def __init__(self):
        self.cycles = {}
        self.features = {}

    def register_cycle(self, name):
        def decorator(cls):
            self.cycles[name] = cls
            return cls

        return decorator

    def register_feature(self, name):
        def decorator(func):
            self.features[name] = func
            return func

        return decorator

    def compute_feature(self, name):
        return self.features[name]

    def get_feature_names_for_cycle(self, cycle):
        return [f"cycle_{cycle}_{name}" for name in self.features.keys()]


# Create the internal instance
registry = CycleRegistry()
