import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(csv_filename):
    df = pd.read_csv(csv_filename)
    df = df[df["Project State"].isin([1, 2])]  # Filter for graduated (1) and retired (2) projects
    df["Merge to Reject Ratio"] = df["Merged PRs"] / df["Closed PRs"].replace(0, float('nan'))  # Avoid division by zero
    return df

def plot_and_save(fig, filename):
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_boxplots(df):
    metrics = ["PR Merge %", "PR Reject %", "Unresolved PRs", "Merge to Reject Ratio"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x="Project State", y=metric, data=df, ax=ax)
        ax.set_title(f"{metric} Distribution by Project State")
        ax.set_xlabel("Project State (1=Graduated, 2=Retired)")
        ax.set_ylabel(metric)
        plot_and_save(fig, f"{metric}_boxplot.png")

def plot_violin_plots(df):
    metrics = ["PR Merge %", "PR Reject %", "Unresolved PRs", "Merge to Reject Ratio"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(x="Project State", y=metric, data=df, ax=ax)
        ax.set_title(f"{metric} Density by Project State")
        ax.set_xlabel("Project State (1=Graduated, 2=Retired)")
        ax.set_ylabel(metric)
        plot_and_save(fig, f"{metric}_violinplot.png")

def plot_bar_charts(df):
    avg_metrics = df.groupby("Project State")[["PR Merge %", "PR Reject %"]].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    avg_metrics.plot(kind="bar", ax=ax)
    ax.set_title("Average PR Metrics by Project State")
    ax.set_xlabel("Project State (1=Graduated, 2=Retired)")
    ax.set_ylabel("Percentage / Ratio")
    ax.legend(["PR Merge %", "PR Reject %"])
    plot_and_save(fig, "average_pr_bar_chart.png")

def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x="PR Merge %", y="PR Reject %", hue="Project State", data=df, palette="coolwarm", ax=ax)
    ax.set_title("PR Merge % vs PR Reject % by Project State")
    ax.set_xlabel("PR Merge %")
    ax.set_ylabel("PR Reject %")
    plot_and_save(fig, "pr_merge_vs_reject_scatter.png")

def plot_pairplot(df):
    pairplot = sns.pairplot(df, hue="Project State", vars=["Merged PRs", "Closed PRs", "Unresolved PRs", "Merge to Reject Ratio"], palette="coolwarm")
    pairplot.savefig("pairplot.png", bbox_inches='tight')
    plt.close()

def print_summary_stats(df):
    for state, label in zip([1, 2], ["Graduated", "Retired"]):
        subset = df[df["Project State"] == state]
        num_projects = len(subset)
        total_prs = subset["Total PRs"].sum()
        merged_prs = subset["Merged PRs"].sum()
        closed_prs = subset["Closed PRs"].sum()
        unresolved_prs = subset["Unresolved PRs"].sum()
        merge_to_reject_ratio = merged_prs / closed_prs if closed_prs > 0 else float('nan')

        merge_percentage = (merged_prs / total_prs) * 100 if total_prs > 0 else 0
        reject_percentage = (closed_prs / total_prs) * 100 if total_prs > 0 else 0

        avg_total_prs = subset["Total PRs"].mean() if not subset.empty else 0
        avg_merged_prs = subset["Merged PRs"].mean() if not subset.empty else 0
        avg_closed_prs = subset["Closed PRs"].mean() if not subset.empty else 0
        avg_unresolved_prs = subset["Unresolved PRs"].mean() if not subset.empty else 0
        avg_merge_to_reject = subset["Merge to Reject Ratio"].mean() if not subset.empty else float('nan')

        print(f"{label} Projects: ({num_projects} projects)")
        print(f"  Total PRs: {total_prs}")
        print(f"  Merged PRs: {merged_prs}")
        print(f"  Rejected PRs: {closed_prs}")
        print(f"  Unresolved PRs: {unresolved_prs}")
        print(f"  Merge Percentage: {merge_percentage:.2f}%")
        print(f"  Reject Percentage: {reject_percentage:.2f}%")
        print(f"  Average Total PRs per Project: {avg_total_prs:.2f}")
        print(f"  Average Merged PRs per Project: {avg_merged_prs:.2f}")
        print(f"  Average Rejected PRs per Project: {avg_closed_prs:.2f}")
        print(f"  Average Unresolved PRs per Project: {avg_unresolved_prs:.2f}")
        print(f"  Average Merge to Reject Ratio: {avg_merge_to_reject:.2f}")
        print("-" * 40)

def main():
    csv_filename = "pr_summary.csv"  # Change to your actual CSV filename
    df = load_data(csv_filename)
    plot_boxplots(df)
    plot_violin_plots(df)
    plot_bar_charts(df)
    plot_scatter(df)
    plot_pairplot(df)
    print_summary_stats(df)

if __name__ == "__main__":
    main()
