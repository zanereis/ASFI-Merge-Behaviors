import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(csv_filename):
    df = pd.read_csv(csv_filename)
    df = df[df["Project State"].isin([1, 2])]  # Filter for graduated (1) and retired (2) projects
    return df

def plot_and_save(fig, filename):
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_boxplots(df):
    metrics = ["PR Merge %", "PR Reject %", "Unresolved PRs"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x="Project State", y=metric, data=df, ax=ax)
        ax.set_title(f"{metric} Distribution by Project State")
        ax.set_xlabel("Project State (1=Graduated, 2=Retired)")
        ax.set_ylabel(metric)
        plot_and_save(fig, f"{metric}_boxplot.png")

def plot_violin_plots(df):
    metrics = ["PR Merge %", "PR Reject %", "Unresolved PRs"]
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
    ax.set_title("Average PR Merge % and Reject % by Project State")
    ax.set_xlabel("Project State (1=Graduated, 2=Retired)")
    ax.set_ylabel("Percentage")
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
    pairplot = sns.pairplot(df, hue="Project State", vars=["Merged PRs", "Closed PRs", "Unresolved PRs"], palette="coolwarm")
    pairplot.savefig("pairplot.png", bbox_inches='tight')
    plt.close()

def main():
    csv_filename = "pr_summary.csv"  # Change to your actual CSV filename
    df = load_data(csv_filename)
    plot_boxplots(df)
    plot_violin_plots(df)
    plot_bar_charts(df)
    plot_scatter(df)
    plot_pairplot(df)

if __name__ == "__main__":
    main()
