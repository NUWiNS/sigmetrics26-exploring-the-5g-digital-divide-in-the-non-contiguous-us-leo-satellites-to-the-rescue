import os
from typing import Dict, List

import pandas as pd

from scripts.constants import CommonField


class XcalDataAggregator:
    def __init__(
            self, 
            root_dir: str,
            operator_field: str = CommonField.OPERATOR,
            protocol_field: str = CommonField.PROTOCOL,
            direction_field: str = CommonField.DIRECTION,
            location_field: str = CommonField.LOCATION,
        ):
        self.root_dir = root_dir
        self.operator_field = operator_field
        self.protocol_field = protocol_field
        self.direction_field = direction_field
        self.location_field = location_field


    def get_xcal_tput_data_path(self, operator: str, label: str, protocol: str, direction: str):
        return os.path.join(self.root_dir, f'xcal_smart_tput.{protocol}_{direction}.{operator}.{label}.csv'.lower())

    def read_xcal_tput_data(self, operator: str, label: str, protocol: str = None, direction: str = None):
        input_csv_path = self.get_xcal_tput_data_path(operator, label, protocol, direction)
        df = pd.read_csv(input_csv_path)
        if protocol:
            df = df[df[self.protocol_field] == protocol]
        if direction:
            df = df[df[self.direction_field] == direction]
        return df

    def aggregate_xcal_tput_data(
            self,
            operators: List[str], 
            label: str,
            protocol: str = None, 
            direction: str = None, 
        ):
        data = pd.DataFrame()
        for operator in operators:
            df = self.read_xcal_tput_data(
                operator=operator, 
                label=label,
                protocol=protocol, 
                direction=direction, 
            )
            df[self.operator_field] = operator
            data = pd.concat([data, df])
        return data

    def aggregate_xcal_tput_data_by_location(
            self,
            dataset_label: str,
            locations: List[str], 
            location_conf: Dict[str, Dict],
            protocol: str = None, 
            direction: str = None, 
        ):
        data = pd.DataFrame()
        for location in locations:
            conf = location_conf[location]
            df = self.aggregate_xcal_tput_data(
                operators=conf['operators'], 
                label=dataset_label,
                protocol=protocol, 
                direction=direction, 
            )
            df[self.location_field] = location
            data = pd.concat([data, df])
        return data
