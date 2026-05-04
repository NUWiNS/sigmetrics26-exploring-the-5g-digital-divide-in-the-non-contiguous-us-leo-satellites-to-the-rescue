from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CAMPAIGN_DIRS = {
    "alaska": REPO_ROOT / "datasets" / "alaska_road_test_202406",
    "hawaii": REPO_ROOT / "datasets" / "hawaii_road_test_202408",
}


class CommonField:
    TPUT_MBPS = "avg_tput"
    DELTA_MBPS = "avg_mptcp_minus_max"


location_conf = {
    "alaska": {
        "label": "AK",
        "operators": ["starlink", "verizon", "att"],
        "op_pairs": [
            ("starlink", "verizon"),
            ("starlink", "att"),
            ("verizon", "att"),
        ],
    },
    "hawaii": {
        "label": "HI",
        "operators": ["starlink", "verizon", "att", "tmobile"],
        "op_pairs": [
            ("starlink", "verizon"),
            ("starlink", "att"),
            ("starlink", "tmobile"),
            ("verizon", "att"),
            ("tmobile", "verizon"),
            ("att", "tmobile"),
        ],
    },
}

mptcp_operator_conf = {
    "starlink": {"label": "SL", "color": "black"},
    "att": {"label": "AT", "color": "deepskyblue"},
    "verizon": {"label": "VZ", "color": "red"},
    "tmobile": {"label": "TM", "color": "magenta"},
    "starlink_att": {"label": "SL+AT", "color": "deepskyblue", "hatch": None},
    "starlink_verizon": {"label": "SL+VZ", "color": "red", "hatch": None},
    "starlink_tmobile": {"label": "SL+TM", "color": "magenta", "hatch": None},
    "att_tmobile": {"label": "AT+TM", "color": "deepskyblue", "hatch": "///"},
    "verizon_att": {"label": "VZ+AT", "color": "red", "hatch": "///"},
    "tmobile_verizon": {"label": "TM+VZ", "color": "magenta", "hatch": "///"},
}

output_dir = SCRIPT_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

def result_dir(loc, direction, category):
    return (
        CAMPAIGN_DIRS[loc]
        / "processed"
        / "mpshell"
        / "emulation_results"
        / direction
        / category
    )


def plot_direction(direction, suffix):
    for loc, conf in location_conf.items():
        fig, axes = plt.subplots(1, 4, figsize=(9.6, 4))
        fig.subplots_adjust(wspace=0.3)

        box_kwargs = {
            "patch_artist": True,
            "whis": [0, 100],
            "medianprops": {"color": "white", "linewidth": 1},
            "boxprops": {"color": "black", "linewidth": 1},
            "whiskerprops": {"color": "black", "linewidth": 1.5},
            "capprops": {"color": "black", "linewidth": 1.5},
        }

        # --- SINGLE ---
        data1, labels1, colors1 = [], [], []
        for op in conf["operators"]:
            file = result_dir(loc, direction, "single") / f"single_{loc}_{op}.csv"
            if file.exists():
                df = pd.read_csv(file)
                if CommonField.TPUT_MBPS in df.columns and not df.empty:
                    data1.append(df[CommonField.TPUT_MBPS])
                    labels1.append(mptcp_operator_conf[op]["label"])
                    colors1.append(mptcp_operator_conf[op]["color"])
        if data1:
            bp1 = axes[0].boxplot(data1, tick_labels=labels1, **box_kwargs)
            for patch, color in zip(bp1["boxes"], colors1):
                patch.set_facecolor(color)
                patch.set_edgecolor("black")
        axes[0].set_title("Single Flow", fontweight="bold")
        axes[0].set_ylabel("Throughput (Mbps)", fontweight="bold")

        # --- MPTCP ---
        data2, labels2, colors2, hatches2 = [], [], [], []
        for op1, op2 in conf["op_pairs"]:
            sorted_pair = tuple(sorted([op1, op2]))
            key = f"{op1}_{op2}"
            file = (
                result_dir(loc, direction, "mptcp")
                / f"mptcp_{loc}_{sorted_pair[0]}_{sorted_pair[1]}.csv"
            )

            if file.exists():
                df = pd.read_csv(file)
                if CommonField.TPUT_MBPS in df.columns and not df.empty:
                    conf_entry = mptcp_operator_conf.get(key, {})
                    data2.append(df[CommonField.TPUT_MBPS])
                    labels2.append(conf_entry.get("label", key))
                    colors2.append(conf_entry.get("color", "gray"))
                    hatches2.append(conf_entry.get("hatch", None))
        if data2:
            bp2 = axes[1].boxplot(data2, tick_labels=labels2, **box_kwargs)
            for patch, color, hatch in zip(bp2["boxes"], colors2, hatches2):
                patch.set_facecolor(color)
                patch.set_edgecolor("black")
                if hatch:
                    patch.set_hatch(hatch)
        axes[1].set_title("MPTCP", fontweight="bold")

        if data1 and data2:
            all_vals = pd.concat(data1 + data2)
            y_min, y_max = all_vals.min() - 10, all_vals.max() + 10
            axes[0].set_ylim(y_min, y_max)
            axes[1].set_ylim(y_min, y_max)

        # --- MPTCP - MAX ---
        data3, labels3, colors3, hatches3 = [], [], [], []
        for op1, op2 in conf["op_pairs"]:
            sorted_pair = tuple(sorted([op1, op2]))
            key = f"{op1}_{op2}"
            file = (
                result_dir(loc, direction, "mptcp-max")
                / f"mptcp_minus_max_{loc}_{sorted_pair[0]}_{sorted_pair[1]}.csv"
            )
            if file.exists():
                df = pd.read_csv(file)
                if CommonField.DELTA_MBPS in df.columns and not df.empty:
                    conf_entry = mptcp_operator_conf.get(key, {})
                    data3.append(df[CommonField.DELTA_MBPS])
                    labels3.append(conf_entry.get("label", key))
                    colors3.append(conf_entry.get("color", "gray"))
                    hatches3.append(conf_entry.get("hatch", None))
        if data3:
            bp3 = axes[2].boxplot(data3, tick_labels=labels3, **box_kwargs)
            for patch, color, hatch in zip(bp3["boxes"], colors3, hatches3):
                patch.set_facecolor(color)
                patch.set_edgecolor("black")
                if hatch:
                    patch.set_hatch(hatch)
        axes[2].set_title("MPTCP − Max", fontweight="bold")
        axes[2].set_ylabel("Throughput Difference (Mbps)", fontweight="bold")

        # --- SUM - MPTCP ---
        data4, labels4, colors4, hatches4 = [], [], [], []
        for op1, op2 in conf["op_pairs"]:
            sorted_pair = tuple(sorted([op1, op2]))
            key = f"{op1}_{op2}"
            file = (
                result_dir(loc, direction, "sum-mptcp")
                / f"sum_minus_mptcp_{loc}_{sorted_pair[0]}_{sorted_pair[1]}.csv"
            )

            if file.exists():
                df = pd.read_csv(file)
                if CommonField.TPUT_MBPS in df.columns and not df.empty:
                    conf_entry = mptcp_operator_conf.get(key, {})
                    data4.append(df[CommonField.TPUT_MBPS])
                    labels4.append(conf_entry.get("label", key))
                    colors4.append(conf_entry.get("color", "gray"))
                    hatches4.append(conf_entry.get("hatch", None))

        if data4:
            bp4 = axes[3].boxplot(data4, tick_labels=labels4, **box_kwargs)
            for patch, color, hatch in zip(bp4["boxes"], colors4, hatches4):
                patch.set_facecolor(color)
                patch.set_edgecolor("black")
                if hatch:
                    patch.set_hatch(hatch)
        axes[3].set_title("Sum − MPTCP", fontweight="bold")
        axes[3].set_ylabel("Throughput Difference (Mbps)", fontweight="bold")

        for ax in axes:
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=30, fontsize=9, fontweight="bold"
            )
            for label in ax.get_yticklabels():
                label.set_fontweight("bold")
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_dir / f"mpshell_{loc}_{suffix}.pdf", format="pdf")
        plt.close()


for trace_direction, output_suffix in (("downlink", "dl"), ("uplink", "ul")):
    plot_direction(trace_direction, output_suffix)
