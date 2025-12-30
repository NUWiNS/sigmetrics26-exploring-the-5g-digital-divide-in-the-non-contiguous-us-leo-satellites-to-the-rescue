import sys
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../'))

from scripts.utilities.math_utils import MathUtils
from scripts.constants import CommonField, XcalField
from scripts.plotting.common import cellular_location_conf, cellular_operator_conf

class CellRbWithAreaDataGenerator:
    def __init__(self, df: pd.DataFrame, actual_tech: str, area_type: str):
        self.df = df
        self.actual_tech = actual_tech
        self.area_type = area_type

    def filter_area_type(self, df: pd.DataFrame, area_type: str):
        return df[df[CommonField.AREA_TYPE] == area_type]

    def generate(self):
        area_df = self.filter_area_type(self.df, self.area_type)

        res_df = pd.DataFrame()
        res_df[XcalField.SRC_IDX] = area_df[XcalField.SRC_IDX]
        res_df[XcalField.RUN_ID] = area_df[XcalField.RUN_ID]
        res_df[XcalField.LOCAL_TIME] = area_df[XcalField.LOCAL_TIME]
        res_df[CommonField.AREA_TYPE] = area_df[XcalField.AREA]
        
        res_df[XcalField.LTE_PRB_NUM_PDSCH] = area_df[XcalField.LTE_PRB_NUM_PDSCH]
        res_df[XcalField.LTE_PRB_NUM_PUSCH] = area_df[XcalField.LTE_PRB_NUM_PUSCH]

        if self.actual_tech.lower().startswith('5g'):
            for field in [
                XcalField._5G_RB_DL,
                XcalField._5G_RB_UL,
            ]:
                if field in area_df.columns:
                    res_df[field] = area_df[field]

        res_df[XcalField.ACTUAL_TECH] = area_df[XcalField.ACTUAL_TECH]
        res_df[XcalField.APP_TPUT_PROTOCOL] = area_df[XcalField.APP_TPUT_PROTOCOL]
        res_df[XcalField.APP_TPUT_DIRECTION] = area_df[XcalField.APP_TPUT_DIRECTION]

        return res_df
    
class CellTputRbDataGenerator:
    def __init__(self, 
                 df: pd.DataFrame, 
                 operator: str, 
                 protocol: str, 
                 direction: str,
                 actual_tech: str,
                 area_types: list[str],
    ):
        self.operator = operator
        self.protocol = protocol
        self.direction = direction
        self.actual_tech = actual_tech
        self.area_types = area_types

        tmp_df = self.filter_protocol_direction(df, protocol, direction)
        self.df = self.filter_actual_tech(tmp_df, actual_tech)

    def filter_protocol_direction(self, df: pd.DataFrame, protocol: str, direction: str):
        mask = (df[XcalField.APP_TPUT_PROTOCOL] == protocol) & \
              (df[XcalField.APP_TPUT_DIRECTION] == direction)
        return df[mask]
    
    def filter_actual_tech(self, df: pd.DataFrame, actual_tech: str):
        mask = (df[XcalField.ACTUAL_TECH].str.lower().str.startswith(actual_tech.lower()))
        return df[mask]
    
    def drop_cols(self, df: pd.DataFrame, cols: list[str]):
        for col in cols:
            if col not in df.columns:
                continue
            df = df.drop([col], axis=1)
        return df
    
    def drop_unnecessary_columns(self, df: pd.DataFrame):
        if self.direction == 'downlink':
            df = self.drop_cols(df, [XcalField._5G_RB_UL, XcalField.LTE_PRB_NUM_PUSCH])
        elif self.direction == 'uplink':
            df = self.drop_cols(df, [XcalField._5G_RB_DL, XcalField.LTE_PRB_NUM_PDSCH])
        return df

    def generate(self):
        res = {}
        for area_type in self.area_types:
            res_df = CellRbWithAreaDataGenerator(self.df, actual_tech=self.actual_tech, area_type=area_type).generate()
            res_df[CommonField.OPERATOR] = self.operator
            res_df = self.drop_unnecessary_columns(res_df)
            res[area_type] = res_df
        return res

    
class CellRbWithAreasPlotter:
    def __init__(
            self, 
            plot_data: dict,
            lte_rb_col: str,
            _5g_rb_col: str,
            location: str,
            operator: str,
        ):
        """Initialize the plotter with plot data organized by location and operator.
        
        plot_data structure:
        {
            'location': {
                'operator': {
                    'area_type': DataFrame
                }
            }
        }
        """
        self.plot_data = plot_data
        self.lte_rb_col = lte_rb_col
        self._5g_rb_col = _5g_rb_col
        self.location = location
        self.operator = operator

    def add_category_plot(self, ax, data: pd.DataFrame, area_type: str):
        """Add a CDF plot for a specific category to the given axis.
        
        Args:
            ax: The matplotlib axis to plot on
            data: DataFrame containing the RB data
            area_type: The area type (urban/rural)
        
        Returns:
            The plotted line object if successful, None otherwise
        """
        if len(data) == 0:
            return None

        # Get operator configuration
        op_conf = cellular_operator_conf[self.operator]
        
        # Define line styles for different areas
        area_styles = {
            'urban': '-',
            'rural': '--'
        }

        # Extract RB values and sort for CDF
        values = []
        if self._5g_rb_col in data.columns:
            values.extend(data[self._5g_rb_col].dropna().values)
        elif self.lte_rb_col in data.columns:
            values.extend(data[self.lte_rb_col].dropna().values)
        
        if not values:
            raise ValueError(f'Neither {self._5g_rb_col} nor {self.lte_rb_col} is in the data')

        # Calculate CDF
        data_sorted, cdf = MathUtils.cdf(values)
        
        # Plot the line
        line = ax.plot(
            data_sorted,
            cdf,
            linestyle=area_styles[area_type],
            color=op_conf['color'],
            label=area_type.title(),
        )[0]
        
        return line

    def plot(
            self, 
            output_path: str,
            x_lim: tuple[int, int] = (0, 100),
        ):
        """Create a CDF plot comparing urban and rural RB distributions."""
        # Create figure
        fig, ax = plt.subplots(figsize=(4, 2.8))
        
        # Get data for the specific location and operator
        area_data = self.plot_data[self.location][self.operator]
        
        # Plot each area type
        for area_type in ['urban', 'rural']:
            if area_type in area_data:
                self.add_category_plot(ax, area_data[area_type], area_type)
        
        # Configure plot
        ax.set_xlabel('Resource Blocks', fontsize=14)
        ax.set_ylabel('CDF', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_xlim(*x_lim)
        
        # Add reference lines
        ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.4, linewidth=1)
        # ax.axvline(x=4, color='black', linestyle=':', alpha=0.8, linewidth=1)
        ax.set_xticks(list(ax.get_xticks()))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Add legend in upper left
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # Create output directory if it doesn't exist
    output_dir = os.path.join(current_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    area_types = ['urban', 'rural']
    protocol = 'tcp'
    direction = 'downlink'
    actual_tech = '5G-low'

    plot_data = {}
    # NOTE: we only care about ALASKA for now
    for location in ['alaska']:
        root_dir = cellular_location_conf[location]['root_dir']
        data_dir = os.path.join(root_dir, 'processed', 'xcal')
        plot_data[location] = {}

        operators = ['att']
        for operator in operators:
            df = pd.read_csv(os.path.join(data_dir, f'{operator}_xcal_smart_tput.csv'))

            rb_data_with_areas = CellTputRbDataGenerator(
                df, 
                operator=operator,
                protocol=protocol,
                direction=direction,
                actual_tech=actual_tech,
                area_types=area_types,
            ).generate()

            # for debug
            for area_type, df in rb_data_with_areas.items():
                if len(df) == 0:
                    print(f'{location}.{operator}.{protocol}.{direction}.{actual_tech}.{area_type} is empty, skipped.')
                    continue

                output_csv_path = os.path.join(output_dir, f'cell_rb.{location}.{operator}.{protocol}.{direction}.{actual_tech}.{area_type}.csv')
                df.to_csv(output_csv_path, index=False)

            plot_data[location][operator] = rb_data_with_areas

            # Create a plot for this location and operator
            plotter = CellRbWithAreasPlotter(
                plot_data=plot_data,
                lte_rb_col=XcalField.LTE_PRB_NUM_PDSCH,
                _5g_rb_col=XcalField._5G_RB_DL,
                location=location,
                operator=operator,
            )
            plotter.plot(
                output_path=os.path.join(output_dir, f'cdf_cell_rb.{location}.{operator}.{protocol}.{direction}.{actual_tech}.pdf'),
                x_lim=(0, 55),
            )

if __name__ == '__main__':
    main()
