import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("results_summary.csv")

# Add class and dimensionality
# df["class"] = df["function_name"].map(lambda fn: function_metadata[fn][0])
# df["dimension"] = df["function_name"].map(lambda fn: function_metadata[fn][1])

# --- Step 1: Add function class to the DataFrame ---

function_metadata = {
    "Booth": ("plate", 2),
    "Zakharov (n=2)": ("plate", 2),
    "Zakharov (n=5)": ("plate", 5),
    "Zakharov (n=10)": ("plate", 10),
    "Rosenbrock (n=2)": ("rugged", 2),
    "Rosenbrock (n=5)": ("rugged", 5),
    "Rosenbrock (n=10)": ("rugged", 10),
    "Dixon-Price (n=2)": ("rugged", 2),
    "Dixon-Price (n=5)": ("rugged", 5),
    "Dixon-Price (n=10)": ("rugged", 10),
    "Six-Hump Camel": ("rugged", 2),
    "Easom": ("steep", 2),
    "Michalewicz (n=2)": ("steep", 2),
    "Michalewicz (n=5)": ("steep", 5),
    "Michalewicz (n=10)": ("steep", 10),
    "Powell (n=4)": ("other", 4),
    "Powell (n=8)": ("other", 8),
    "Powell (n=12)": ("other", 12),
    "Perm (n=2)": ("other", 2),
    "Perm (n=5)": ("other", 5),
    "Perm (n=10)": ("other", 10),
    "Goldstein-Price": ("other", 2),
    "Styblinski-Tang (n=2)": ("other", 2),
    "Styblinski-Tang (n=5)": ("other", 5),
    "Styblinski-Tang (n=10)": ("other", 10),
}

df["class"] = df["function_name"].map(lambda name: function_metadata.get(name, ("unknown", None))[0])

# --- Step 2: Define visuals ---
method_color_map = {
    "МСГ": ("#1f77b4", "#aec7e8"),
    "Метод Бройдена": ("#2ca02c", "#98df8a"),
    "Метод ДФП": ("#ff7f0e", "#ffbb78")
}

method_ua_label_map = {
    "МСГ (restart)": "МСГ (з відн.)",
    "МСГ (no restart)": "МСГ (без відн.)",
    "Метод Бройдена (restart)": "Метод Бройдена (з відн.)",
    "Метод Бройдена (no restart)": "Метод Бройдена (без відн.)",
    "Метод ДФП (restart)": "Метод ДФП (з відн.)",
    "Метод ДФП (no restart)": "Метод ДФП (без відн.)",
}

# Ukrainian labels for function classes
class_ua_map = {
    "plate": "пласких функцій",
    "rugged": "яристих функцій",
    "steep": "функцій з крутими спусками/поворотами",
    "other": "інших функцій"
}

def base_method(method_name):
    return method_name.replace(" (restart)", "").replace(" (no restart)", "").strip()

def is_restart(method_name):
    return "restart" in method_name

all_methods = sorted(
    df["method_name"].unique(),
    key=lambda s: (base_method(s), not is_restart(s))
)

xtick_labels = [method_ua_label_map.get(name, name) for name in all_methods]

error_metrics = ["f_error", "x_error"]
metric_titles = {
    "f_error": "Середнє значення $|\\hat{f} - f^*|$",
    "x_error": "Середнє значення $||\\hat{x} - x^*||$"
}

function_classes = ["plate", "rugged", "steep", "other"]

# --- Step 3: Plot ---
for metric in error_metrics:
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12.5, 13.5), constrained_layout=True)

    # # Inside plotting loop
    # for i, f_class in enumerate(function_classes):
    #     ax = axs[i]
    #     df_subset = df[df["class"] == f_class]
    #
    #     grouped = (
    #         df_subset.groupby("method_name")[metric]
    #         .mean()
    #         .reindex(all_methods)
    #     )
    #
    #     x = np.arange(len(all_methods))
    #     bar_colors = [
    #         method_color_map[base_method(m)][1 if is_restart(m) else 0]
    #         for m in all_methods
    #     ]
    #
    #     bars = ax.bar(x, grouped.values, color=bar_colors)
    #     ax.set_title(f"{metric_titles[metric]} для {class_ua_map[f_class]}", fontsize=14)
    #     ax.set_ylabel("$\hat{f} - f^*$", fontsize=12)
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(xtick_labels, rotation=30, fontsize=10)
    #     ax.grid(axis="y", linestyle="--", alpha=0.4)
    #
    #     y_max = grouped.max()
    #     ax.set_ylim(top=y_max * 1.15 if y_max > 0 else 1)
    #
    #     for bar, val in zip(bars, grouped.values):
    #         ax.text(
    #             bar.get_x() + bar.get_width() / 2,
    #             val + y_max * 0.02 if y_max > 0 else 0.02,
    #             f"{val:.0e}",  # scientific notation
    #             ha="center",
    #             va="bottom",
    #             fontsize=9
    #         )
    #
    # # fig.suptitle(f"{metric_titles[metric]} за класами функцій", fontsize=16)
    # plt.savefig(f"grouped_by_class_{metric}.png", dpi=80)
    # plt.close()

    # Plot average number of iterations per class
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12.5, 13.5), constrained_layout=True)

    for i, f_class in enumerate(function_classes):
        ax = axs[i]
        df_subset = df[df["class"] == f_class]

        grouped = (
            df_subset.groupby("method_name")["iterations"]
            .mean()
            .reindex(all_methods)
        )

        x = np.arange(len(all_methods))
        bar_colors = [
            method_color_map[base_method(m)][1 if is_restart(m) else 0]
            for m in all_methods
        ]

        bars = ax.bar(x, grouped.values, color=bar_colors)
        ax.set_title(f"Середня кількість ітерацій для {class_ua_map[f_class]}", fontsize=14)
        ax.set_ylabel("Ітерації", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=30, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        y_max = grouped.max()
        ax.set_ylim(top=y_max * 1.15 if y_max > 0 else 1)

        for bar, val in zip(bars, grouped.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + y_max * 0.02 if y_max > 0 else 0.02,
                f"{val:.0f}",  # integer format
                ha="center",
                va="bottom",
                fontsize=9
            )

    # Save iteration plot
    plt.savefig("grouped_by_class_iterations.png", dpi=80)
    plt.close()
