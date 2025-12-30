import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from scripts.constants import operator_color_map
from scripts.configs.mainland import ROOT_DIR as ML_ROOT_DIR

cellular_location_conf = {
    'la_to_omaha': {
        'label': 'LA to Omaha',
        'root_dir': ML_ROOT_DIR,
        'operators': ['verizon', 'att', 'tmobile'],
        'order': 1,
        'tcp_downlink': {
            'interval_x': 400,
            'max_xlim': 2000,
        },
        'tcp_uplink': {
            'interval_x': 50,
            'max_xlim': 250,
        },
    },
}

cellular_operator_conf = {
    'att': {
        'label': 'AT&T',
        'abbr': 'AT',
        'order': 1,
        'color': operator_color_map['att'],
        'linestyle': '--'
    },
    'verizon': {
        'label': 'Verizon',
        'abbr': 'VZ',
        'order': 2,
        'color': operator_color_map['verizon'],
        'linestyle': ':'
    },
    'tmobile': {
        'label': 'T-Mobile',
        'abbr': 'TM',
        'order': 3,
        'color': operator_color_map['tmobile'],
        'linestyle': '-.'
    },
}

operator_conf = {
    **cellular_operator_conf,
    'starlink': {
        'label': 'Starlink',
        'abbr': 'SL',
        'order': 4,
        'color': operator_color_map['starlink'],
        'linestyle': '-.',
    },
}


mptcp_operator_conf = {
    'starlink': {
        'label': 'SL',
        'color': operator_conf['starlink']['color'],
        'hatch': None,
    },
    'att': {
        'label': 'AT',
        'color': operator_conf['att']['color'],
        'hatch': None,
    },
    'verizon': {
        'label': 'VZ',
        'color': operator_conf['verizon']['color'],
        'hatch': None,
    },
    'tmobile': {
        'label': 'TM',
        'color': operator_conf['tmobile']['color'],
        'hatch': None,
    },
    'starlink_att': {
        'label': 'SL+AT',
        'color': operator_conf['att']['color'],
        'hatch': None,
        'order': 0, 
    },
    'starlink_verizon': {
        'label': 'SL+VZ',
        'color': operator_conf['verizon']['color'],
        'hatch': None,
        'order': 1,
    },
    'starlink_tmobile': {
        'label': 'SL+TM',
        'color': operator_conf['tmobile']['color'],
        'hatch': None,
        'order': 2,
    },
    'verizon_att': {
        'label': 'VZ+AT',
        'color': operator_conf['verizon']['color'],
        'hatch': '///',
        'order': 3,
    },
    'verizon_tmobile': {
        'label': 'TM+VZ',
        'color': operator_conf['tmobile']['color'],
        'hatch': '///',
        'order': 4,
    },
    'att_tmobile': {
        'label': 'AT+TM',
        'color': operator_conf['att']['color'],
        'hatch': '///',
        'order': 5,
    },
}

tech_conf = {
    'LTE': {
        'label': 'LTE',
        'color': '#326f21',
        'order': 1
    },
    'LTE-A': {
        'label': 'LTE-A',
        'color': '#86c84d',
        'order': 2
    },
    '5G-low': {
        'label': '5G Low',
        'color': '#ffd700',
        'order': 3
    },
    '5G-mid': {
        'label': '5G Mid',
        'color': '#ff9900',
        'order': 4
    },
    '5G-mmWave (28GHz)': {
        'label': '5G-mmWave (28GHz)',
        'color': '#ff4500',
        'order': 5
    },
    '5G-mmWave (39GHz)': {
        'label': '5G-mmWave (39GHz)',
        'color': '#ba281c',
        'order': 6
    },
    'NO SERVICE': {
        'label': 'No Service',
        'color': '#808080',
        'order': 7
    },
    'Unknown': {
        'label': 'Unknown',
        'color': '#000000',
        'order': 8
    },
}

style_confs = {
    'starlink_att': {
        'color': operator_color_map['att'],
        'linestyle': '-',
        'label': 'SL-AT',
    },
    'starlink_tmobile': {
        'color': operator_color_map['tmobile'],
        'linestyle': '-',
        'label': 'SL-TM',
    },
    'starlink_verizon': {
        'color': operator_color_map['verizon'],
        'linestyle': '-',
        'label': 'SL-VZ',
    },
    'att_tmobile': {
        'color': operator_color_map['att'],
        'linestyle': ':',
        'label': 'AT-TM',
    },
    'tmobile_verizon': {
        'color': operator_color_map['tmobile'],
        'linestyle': ':',
        'label': 'TM-VZ',
    },
    'verizon_tmobile': {
        'color': operator_color_map['tmobile'],
        'linestyle': ':',
        'label': 'TM-VZ',
    },
    'verizon_att': {
        'color': operator_color_map['verizon'],
        'linestyle': ':',
        'label': 'VZ-AT',
    },
}

threshold_confs = {
    'tcp_downlink': {
        0: {
            'color': 'blue',
            'label': '0 Mbps',
            'linestyle': '-',   
        },
        10: {
            'color': 'green',
            'label': '10 Mbps',
            'linestyle': 'dashed',
        },
        50: {
            'color': 'red',
            'label': '50 Mbps',
            'linestyle': 'dotted',
        }
    },
    'tcp_uplink': {
        0: {
            'color': 'blue',
            'label': '0 Mbps',
            'linestyle': '-',
        },
        5: {
            'color': 'green',
            'label': '5 Mbps',
            'linestyle': 'dashed',
        },
        10: {
            'color': 'red',
            'label': '10 Mbps',
            'linestyle': 'dotted',
        }
    }
}
# Colors for different technologies - from grey (NO SERVICE) to rainbow gradient (green->yellow->orange->red) for increasing tech
colors = ['#808080', '#326f21', '#86c84d', '#ffd700', '#ff9900', '#ff4500', '#ba281c']  # Grey, green, light green, yellow, amber, orange, red
tech_order = ['Unknown', 'NO SERVICE', 'LTE', 'LTE-A', '5G-low', '5G-mid', '5G-mmWave (28GHz)', '5G-mmWave (39GHz)']
