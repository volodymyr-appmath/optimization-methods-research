import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    "Beale": ("other", 2),
}



df = pd.read_csv("results_summary.csv")

# Add class and dimensionality
df["class"] = df["function_name"].map(lambda fn: function_metadata[fn][0])
df["dimension"] = df["function_name"].map(lambda fn: function_metadata[fn][1])


df_restart = df[df["method_name"].str.contains("restart")]


best_by_class = (
    df_restart.groupby(["method_name", "class"])["f_error"]
    .mean()
    .reset_index()
    .sort_values(["method_name", "f_error"])
    .groupby("method_name")
    .first()
)

print("--- Best class per method ---\n", best_by_class)


worst_by_class = (
    df_restart.groupby(["method_name", "class"])["f_error"]
    .mean()
    .reset_index()
    .sort_values(["method_name", "f_error"], ascending=[True, False])
    .groupby("method_name")
    .first()
)

print(f"\n{'-'*100}\n")

print("--- Worst class per method ---\n", worst_by_class)


dim_effect = (
    df_restart.groupby(["method_name", "dimension"])[["x_error", "f_error"]]
    .mean()
    .reset_index()
)

# print("--- Effect of dimension ---\n", dim_effect)
#
#
# for method in df_restart["method_name"].unique():
#     sub = dim_effect[dim_effect["method_name"] == method]
#     plt.plot(sub["dimension"], sub["f_error"], label=method)
#
# plt.xlabel("Number of dimensions")
# plt.ylabel("Mean $|\\hat{f} - f^*|$")
# plt.title("Accuracy vs. Dimensionality")
# plt.legend()
# plt.grid(True)
# plt.show()


# # Filter 2D, 5D, 10D
# df_subset = df[df["dimension"].isin([2, 5, 10])]
#
# # Only keep rows with both restarts and no restarts
# df_grouped = (
#     df_subset.groupby("method_name")[["f_error", "x_error"]]
#     .mean()
#     .reset_index()
# )
#
# # Sort methods in logical order (you can tweak if needed)
# method_order = sorted(df_grouped["method_name"].unique(), key=lambda s: (s.split()[0], "restart" not in s))
# df_grouped["method_name"] = pd.Categorical(df_grouped["method_name"], categories=method_order, ordered=True)
# df_grouped = df_grouped.sort_values("method_name")
#
# # Extract for plotting
# x_labels = df_grouped["method_name"]
# x = np.arange(len(x_labels))  # bar locations
#
# # Plot f_error
# plt.figure(figsize=(12, 6))
# plt.bar(x, df_grouped["f_error"], width=0.6)
# plt.xticks(x, x_labels, rotation=30)
# plt.ylabel("Mean $|\\hat{f} - f^*|$")
# plt.title("Function Value Error on 2D, 5D, 10D Functions")
# plt.grid(axis="y", linestyle="--", alpha=0.5)
# plt.tight_layout()
# plt.show()
#
# # Plot x_error
# plt.figure(figsize=(12, 6))
# plt.bar(x, df_grouped["x_error"], width=0.6, color="orange")
# plt.xticks(x, x_labels, rotation=30)
# plt.ylabel("Mean $||\\hat{x} - x^*||$")
# plt.title("Argument Error on 2D, 5D, 10D Functions")
# plt.grid(axis="y", linestyle="--", alpha=0.5)
# plt.tight_layout()
# plt.show()


# Colors for Ukrainian method names
method_color_map = {
    "МСГ": ("#1f77b4", "#aec7e8"),
    "Метод Бройдена": ("#2ca02c", "#98df8a"),
    "Метод ДФП": ("#ff7f0e", "#ffbb78")
}

# X-tick label override
method_ua_label_map = {
    "МСГ (restart)": "МСГ (з відн.)",
    "МСГ (no restart)": "МСГ (без відн.)",
    "Метод Бройдена (restart)": "Метод Бройдена (з відн.)",
    "Метод Бройдена (no restart)": "Метод Бройдена (без відн.)",
    "Метод ДФП (restart)": "Метод ДФП (з відн.)",
    "Метод ДФП (no restart)": "Метод ДФП (без відн.)",
}

# Helpers
def base_method(method_name):
    return method_name.replace(" (restart)", "").replace(" (no restart)", "").strip()

def is_restart(method_name):
    return "restart" in method_name

# Dimensions and metrics
dimensions = [2, 5, 10]
error_metrics = ["f_error", "x_error"]
metric_titles = {
    "f_error": "Середнє значення $|\\hat{f} - f^*|$",
    "x_error": "Середнє значення $||\\hat{x} - x^*||$"
}

y_axis_titles = {
    "f_error": "$|\\hat{f} - f^*|$",
    "x_error": "$||\\hat{x} - x^*||$"
}

# Ukrainian labels for function classes
class_ua_map = {
    "plate": "пласких функцій",
    "rugged": "яристих функцій",
    "steep": "функцій з крутими спусками",
    "other": "інших функцій"
}

# Ensure method order is stable
all_methods = sorted(
    df["method_name"].unique(),
    key=lambda s: (base_method(s), not is_restart(s))
)

# Plot
for metric in error_metrics:
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12.5, 13.5), constrained_layout=True)

    for i, dim in enumerate(dimensions):
        ax = axs[i]
        df_dim = df[df["dimension"] == dim]
        grouped = (
            df_dim.groupby("method_name")[metric]
            .mean()
            .reindex(all_methods)
        )

        x = np.arange(len(all_methods))
        bar_colors = [
            method_color_map[base_method(m)][1 if is_restart(m) else 0]
            for m in all_methods
        ]

        bars = ax.bar(x, grouped.values, color=bar_colors)
        ax.set_title(f"{metric_titles[metric]} для {dim}-вимірних функцій", fontsize=14)
        ax.set_ylabel(f"{y_axis_titles[metric]}", fontsize=14)
        ax.set_xticks(x)

        # ax.set_xticklabels(all_methods, rotation=30, fontsize=12)

        # Replace method names with Ukrainian variants
        xtick_labels = [method_ua_label_map.get(name, name) for name in all_methods]
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=30, fontsize=10)

        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Dynamic top margin and value annotation
        y_max = grouped.max()
        ax.set_ylim(top=y_max * 1.15)  # add top space

        for bar, val in zip(bars, grouped.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + y_max * 0.02,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    # fig.suptitle(metric_titles[metric] + " залежно від розмірності", fontsize=16)
    plt.savefig(f"2-5-10_statistics_{metric}", dpi=80)
    # plt.show()
    plt.close()
