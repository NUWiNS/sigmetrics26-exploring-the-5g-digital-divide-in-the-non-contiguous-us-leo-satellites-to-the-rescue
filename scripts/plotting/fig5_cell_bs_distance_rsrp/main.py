import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def empty_measurement_groups():
    return [{}, {}, {}, {}]


def append_csv_pair(group, row):
    values = group.setdefault(row.operator, [])
    if pd.isna(row.distance_m):
        return
    values.append((float(row.distance_m), float(row.rsrp_dbm)))


def load_non_mainland_all_areas_csv(filename):
    data = {}
    df = pd.read_csv(filename, float_precision='round_trip')
    for row in df.itertuples(index=False):
        groups = data.setdefault(row.area, {}).setdefault(row.region, empty_measurement_groups())
        append_csv_pair(groups[int(row.measurement_index)], row)
    return data


def load_mainland_all_areas_csv(filename):
    data = {}
    df = pd.read_csv(filename, float_precision='round_trip')
    for row in df.itertuples(index=False):
        groups = data.setdefault(row.area, empty_measurement_groups())
        append_csv_pair(groups[int(row.measurement_index)], row)
    return data


# Load the combined area data
filename = REPO_ROOT / 'datasets/hawaii_road_test_202408/processed/bs_dist_rsrp/hawaii_alaska_rsrp_distance_dict_all_areas.csv'
non_mainland_all_areas_data = load_non_mainland_all_areas_csv(filename)

# Load the combined area data
filename = REPO_ROOT / 'datasets/la_to_omaha_road_test_202411/processed/bs_dist_rsrp/mainland_rsrp_distance_dict_all_areas.csv'
mainland_all_areas_data = load_mainland_all_areas_csv(filename)

op_label_dict = {'verizon': 'Verizon',
                'tmobile': 'T-Mobile',
                'atnt': 'AT&T',
                'VERIZON': 'Verizon',
                'TMOBILE': 'T-Mobile',
                'ATT': 'AT&T'}
main_region_distance_dict_lte = {'hawaii': {}, 'alaska': {}, 'mainland': {}}
main_region_rsrp_dict_lte = {'hawaii': {}, 'alaska': {}, 'mainland': {}}

main_region_distance_dict_fiveg = {'hawaii': {}, 'alaska': {}, 'mainland': {}}
main_region_rsrp_dict_fiveg = {'hawaii': {}, 'alaska': {}, 'mainland': {}}

for elem in non_mainland_all_areas_data['suburban'].keys():
    if 'hawaii' in elem:
        region_mod = 'hawaii'
    elif 'alaska' in elem:
        region_mod = 'alaska'

    for count in range(0, 4):
        if count == 1 or count == 3:
            continue
        for op in non_mainland_all_areas_data['suburban'][elem][count].keys():
            non_mainland_all_areas_data['rural'][elem][count][op].extend(non_mainland_all_areas_data['suburban'][elem][count][op])
            
            if count == 0 or count == 2:
                non_mainland_all_areas_data['rural'][elem][count][op].extend(non_mainland_all_areas_data['rural'][elem][count+1][op])
                non_mainland_all_areas_data['urban'][elem][count][op].extend(non_mainland_all_areas_data['urban'][elem][count+1][op])

            if count == 0:
                if op_label_dict[op] not in main_region_distance_dict_lte[region_mod].keys():
                    main_region_distance_dict_lte[region_mod][op_label_dict[op]] = {}
                    main_region_rsrp_dict_lte[region_mod][op_label_dict[op]] = {}

                main_region_distance_dict_lte[region_mod][op_label_dict[op]]['urban'] = [i[0] / 1000 for i in non_mainland_all_areas_data['urban'][elem][count][op]]
                main_region_rsrp_dict_lte[region_mod][op_label_dict[op]]['urban']     = [i[1] for i in non_mainland_all_areas_data['urban'][elem][count][op]]
                main_region_distance_dict_lte[region_mod][op_label_dict[op]]['rural'] = [i[0] / 1000 for i in non_mainland_all_areas_data['rural'][elem][count][op]]
                main_region_rsrp_dict_lte[region_mod][op_label_dict[op]]['rural']     = [i[1] for i in non_mainland_all_areas_data['rural'][elem][count][op]]
            elif count == 2:
                if op_label_dict[op] not in main_region_distance_dict_fiveg[region_mod].keys():
                    main_region_distance_dict_fiveg[region_mod][op_label_dict[op]] = {}
                    main_region_rsrp_dict_fiveg[region_mod][op_label_dict[op]] = {}

                main_region_distance_dict_fiveg[region_mod][op_label_dict[op]]['urban'] = [i[0] / 1000 for i in non_mainland_all_areas_data['urban'][elem][count][op]]
                main_region_rsrp_dict_fiveg[region_mod][op_label_dict[op]]['urban'] = [i[1] for i in non_mainland_all_areas_data['urban'][elem][count][op]]
                main_region_distance_dict_fiveg[region_mod][op_label_dict[op]]['rural'] = [i[0] / 1000 for i in non_mainland_all_areas_data['rural'][elem][count][op]]
                main_region_rsrp_dict_fiveg[region_mod][op_label_dict[op]]['rural'] = [i[1] for i in non_mainland_all_areas_data['rural'][elem][count][op]]


del non_mainland_all_areas_data['suburban']

region_mod = 'mainland'
for count in range(0, 4):
    if count == 1 or count == 3:
        continue
    for op in mainland_all_areas_data['suburban'][count].keys():
        mainland_all_areas_data['rural'][count][op].extend(mainland_all_areas_data['suburban'][count][op])

        if count == 0 or count == 2:
            mainland_all_areas_data['rural'][count][op].extend( mainland_all_areas_data['rural'][count+1][op])
            mainland_all_areas_data['urban'][count][op].extend( mainland_all_areas_data['urban'][count+1][op])
        if count == 0:
            if op_label_dict[op] not in main_region_distance_dict_lte[region_mod].keys():
                main_region_distance_dict_lte[region_mod][op_label_dict[op]] = {}
                main_region_rsrp_dict_lte[region_mod][op_label_dict[op]] = {}
        
            main_region_distance_dict_lte[region_mod][op_label_dict[op]]['urban'] = [i[0] / 1000 for i in mainland_all_areas_data['urban'][count][op]]
            main_region_rsrp_dict_lte[region_mod][op_label_dict[op]]['urban'] = [i[1] for i in mainland_all_areas_data['urban'][count][op]]
            main_region_distance_dict_lte[region_mod][op_label_dict[op]]['rural'] = [i[0] / 1000 for i in mainland_all_areas_data['rural'][count][op]]
            main_region_rsrp_dict_lte[region_mod][op_label_dict[op]]['rural'] = [i[1] for i in mainland_all_areas_data['rural'][count][op]]
        elif count == 2:
            if op_label_dict[op] not in main_region_distance_dict_fiveg[region_mod].keys():
                main_region_distance_dict_fiveg[region_mod][op_label_dict[op]] = {}
                main_region_rsrp_dict_fiveg[region_mod][op_label_dict[op]] = {}

            main_region_distance_dict_fiveg[region_mod][op_label_dict[op]]['urban'] = [i[0] / 1000 for i in mainland_all_areas_data['urban'][count][op]]
            main_region_rsrp_dict_fiveg[region_mod][op_label_dict[op]]['urban'] = [i[1] for i in mainland_all_areas_data['urban'][count][op]]
            main_region_distance_dict_fiveg[region_mod][op_label_dict[op]]['rural'] = [i[0] / 1000 for i in mainland_all_areas_data['rural'][count][op]]
            main_region_rsrp_dict_fiveg[region_mod][op_label_dict[op]]['rural'] = [i[1] for i in mainland_all_areas_data['rural'][count][op]]


del mainland_all_areas_data['suburban']

# --- Assumes these four dicts are already populated ---
#   main_region_distance_dict_lte
#   main_region_rsrp_dict_lte
#   main_region_distance_dict_fiveg
#   main_region_rsrp_dict_fiveg
regions       = ['alaska', 'hawaii', 'mainland']
techs         = ['lte', 'fiveg']
hatches       = {'urban': '\\\\', 'rural': '/'}
border_colors = {'AT&T': 'deepskyblue', 'Verizon': 'red', 'T-Mobile': 'magenta'}
short_codes   = {'AT&T': 'AT', 'Verizon': 'VZ', 'T-Mobile': 'TM'}
short_code_color = {'AT': 'deepskyblue', 'VZ': 'red', 'TM': 'magenta'}

metrics = {
    'distance': {
        'yrange': (0, 30),               # in km
        'ylabel': 'Distance (km)',
        'dict_lte': main_region_distance_dict_lte,
        'dict_5g':  main_region_distance_dict_fiveg
    },
    'rsrp': {
        'yrange': (-140, -60),
        'ylabel': 'RSRP (dBm)',
        'dict_lte': main_region_rsrp_dict_lte,
        'dict_5g':  main_region_rsrp_dict_fiveg
    }
}
a = []
for region in regions:
    for metric, props in metrics.items():
        fig, ax = plt.subplots(figsize=(4, 3))
        d_lte = props['dict_lte'][region]
        d_5g  = props['dict_5g'][region]
        # ops   = sorted(d_lte.keys())
        if len(d_lte.keys()) == 3:
            ops = ['AT&T', 'Verizon', 'T-Mobile']
        else:
            ops   = d_lte.keys()
        n_ops = len(ops)

        block = n_ops + 1
        positions = {
            'lte':   np.arange(n_ops),
            'fiveg': np.arange(n_ops) + block
        }
        width = 0.35

        # draw boxplots with white fill and colored borders
        for tech in techs:
            data_dict = d_lte if tech == 'lte' else d_5g
            for i, op_full in enumerate(ops):

                border = border_colors[op_full]
                for j, area in enumerate(('urban', 'rural')):
                    if area == 'urban':
                        if len(data_dict[op_full][area]) > 0:
                            a.append(np.quantile(data_dict[op_full][area], 0.5))
                            a.append(np.quantile(data_dict[op_full][area], 0.75))
                    data = data_dict[op_full][area]
                    if metric == 'distance':
                        data = [v for v in data]  # already in km
                    x = positions[tech][i] + (j - 0.5) * width
                    bp = ax.boxplot(
                        data,
                        positions=[x],
                        widths=width,
                        patch_artist=True,
                        manage_ticks=False,
                        whis=[5, 95]
                    )
                    for box in bp['boxes']:
                        box.set_facecolor('white')
                        box.set_edgecolor(border)
                        box.set_linewidth(1.5)
                        box.set_hatch(hatches[area])
                        box.set_joinstyle('round')
                        box.set_capstyle('round')

        # draw separator line
        ax.axvline(x=n_ops, color='black', linestyle='--', linewidth=1)

        # annotate LTE and 5G above their sections
        mid_lte = positions['lte'].mean()
        mid_5g  = positions['fiveg'].mean()
        y_min, y_max = props['yrange']
        y_text = y_max + 0.05 * (y_max - y_min)
        ax.text(mid_lte, y_text, 'LTE', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(mid_5g, y_text, '5G', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # x-axis: short codes with colors
        xticks, xlabels, xl_colors = [], [], []
        for tech in techs:
            for idx, op_full in enumerate(ops):
                xticks.append(positions[tech][idx])
                sc = short_codes[op_full]
                xlabels.append(sc)
                xl_colors.append(short_code_color[sc])
        ax.set_xticks(xticks)
        labels = ax.set_xticklabels(xlabels, rotation=0)
        for lbl, col in zip(labels, xl_colors):
            lbl.set_color('black')

        # y-axis, grid
        ax.set_ylim(*props['yrange'])
        ax.set_ylabel(props['ylabel'])
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        if region == regions[0] and metric == 'distance':
            # Area legend (hatches)
            area_handles = [
                Patch(facecolor='white', edgecolor='black', hatch=hatches['urban']),
                Patch(facecolor='white', edgecolor='black', hatch=hatches['rural'])
            ]
            area_labels = ['Urban', 'Rural']
            area_leg = ax.legend(area_handles, area_labels, loc='upper right')
            ax.add_artist(area_leg)

            # Operator legend (borders)
            op_handles = [
                Patch(facecolor='white', edgecolor=border_colors[op_full], linewidth=1.5)
                for op_full in ['AT&T', 'Verizon', 'T-Mobile']
            ]
            op_labels = [short_codes[op_full] for op_full in ops]
            ax.legend(op_handles, op_labels, loc='upper left')

        # save figure
        output_dir = SCRIPT_DIR / 'outputs'
        output_dir.mkdir(exist_ok=True)
        filename = output_dir / f"boxplots_{region}_{metric}_individual_v2.png"
        fig.tight_layout()
        fig.savefig(filename, dpi=600)
        plt.close(fig)
        print(f"Saved: {filename}")

a = 1
