import json

import numpy as np

class JsonDataManager:
    @staticmethod
    def load(filename: str) -> dict:
        with open(filename, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save(data: dict, filename: str, indent: int = 4):
        data = JsonDataManager.convert_to_native_types(data)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=indent)

    @staticmethod
    def convert_to_native_types(obj):
        """Convert numpy/pandas types to native Python types for JSON serialization."""
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: JsonDataManager.convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [JsonDataManager.convert_to_native_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
            