from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from prefect import flow, task
from ydata_profiling import ProfileReport

from src import logger


class EDAReport:
    """
    Generate an exploratory data analysis (EDA) report for a given dataset.

    Args:
        data (pd.DataFrame): The input dataset to analyze.
        target_col (str, optional): The target column to analyze. Defaults to None.
        output_prefix (str, optional): The prefix for the output files. Defaults to 'eda_report'.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = None,
        output_prefix: str = "eda_report",
    ):
        self.data = data
        self.target_col = target_col
        self.output_prefix = output_prefix
        self.output_dir = Path("reports/eda_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats_path = self.output_dir / f"{self.output_prefix}_statistics.csv"
        self.pdf_path = self.output_dir / f"{self.output_prefix}_plots.pdf"
        self.html_path = self.output_dir / f"{self.output_prefix}_report.html"

    def profile_report(self):
        profile = ProfileReport(
            self.data, title="Weather Related Disease Report", explorative=True
        )
        profile.to_file(self.output_dir / f"{self.output_prefix}_report.html")
        logger.info(
            f"EDA complete. Files saved in '{self.output_dir}':\n- {self.html_path.name}"
        )

    def save_statistics(self):
        desc_stats = self.data.describe(include="all").transpose()
        desc_stats.to_csv(self.stats_path)
        logger.info(f"Statistics saved to {self.stats_path}")

    @task(
        name="plot_missing_values", retries=3, retry_delay_seconds=10, log_prints=True
    )
    def plot_missing_values(self, pdf):
        missing = self.data.isnull().sum()
        missing_nonzero = missing[missing > 0]
        if not missing_nonzero.empty:
            miss_df = pd.DataFrame(
                (missing_nonzero * 100 / self.data.shape[0])
            ).reset_index()
            miss_df.columns = ["Column", "Percentage"]
            miss_df["type"] = self.output_prefix

            fig = plt.figure(figsize=(18, 6))
            ax = sns.pointplot(data=miss_df, x="Column", y="Percentage", hue="type")
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.title(f"Percentage of Missing values in {self.output_prefix}")
            plt.ylabel("PERCENTAGE")
            plt.xlabel("COLUMNS")
            ax.set_facecolor("white")
            fig.set_facecolor("white")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            logger.info("Enhanced missing values plot saved")
        else:
            logger.info("No missing values found.")

    @task(
        name="plot_custom_missing_style",
        retries=3,
        retry_delay_seconds=10,
        log_prints=True,
    )
    def plot_custom_missing_style(self, pdf):
        fig = plt.figure(figsize=(18, 6))
        percent_zeros = pd.DataFrame(
            (self.data == 0).sum() * 100 / self.data.shape[0]
        ).reset_index()
        percent_zeros.columns = ["index", 0]
        percent_zeros["type"] = self.output_prefix

        ax = sns.pointplot(x="index", y=0, data=percent_zeros, hue="type")
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.title(f"Percentage of Zero Values in {self.output_prefix}")
        plt.ylabel("PERCENTAGE")
        plt.xlabel("COLUMNS")
        ax.set_facecolor("white")
        fig.set_facecolor("white")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        logger.info("Custom styled zero-values column plot saved")

    @task(name="plot_histograms", retries=3, retry_delay_seconds=10, log_prints=True)
    def plot_histograms(self, pdf):
        numeric_cols = [
            col
            for col in self.data.select_dtypes(include="number").columns
            if self.data[col].nunique() > 2
        ]
        for col in numeric_cols:
            plt.figure()
            sns.histplot(self.data[col].dropna(), kde=True, bins=30)
            plt.title(f"Histogram of {col}")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            logger.info(f"Histogram of {col} saved")

    @task(name="plot_value_counts", retries=3, retry_delay_seconds=10, log_prints=True)
    def plot_value_counts(self, pdf):
        numeric_cols = self.data.select_dtypes(include="number").columns
        categorical_cols = [
            col
            for col in self.data.columns
            if (self.data[col].nunique() <= 10 and col not in numeric_cols)
            or self.data[col].nunique() <= 2
        ]
        for col in categorical_cols:
            plt.figure()
            self.data[col].value_counts().plot(kind="bar")
            plt.title(f"Value Counts of {col}")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            logger.info(f"Value Counts of {col} saved")

    @task(
        name="plot_correlation_heatmap",
        retries=3,
        retry_delay_seconds=10,
        log_prints=True,
    )
    def plot_correlation_heatmap(self, pdf):
        numeric_cols = [
            col
            for col in self.data.select_dtypes(include="number").columns
            if self.data[col].nunique() > 2
        ]
        if (
            self.target_col
            and self.target_col in self.data.columns
            and self.data[self.target_col].nunique() > 2
        ):
            numeric_cols.append(self.target_col)
        subset = self.data[numeric_cols].copy()
        if len(subset.columns) > 1:
            plt.figure(figsize=(12, 8))
            corr = subset.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            logger.info("Correlation Heatmap saved")

    @task(
        name="plot_boxplots_by_target",
        retries=3,
        retry_delay_seconds=10,
        log_prints=True,
    )
    def plot_boxplots_by_target(self, pdf):
        if self.target_col and self.target_col in self.data.columns:
            numeric_cols = [
                col
                for col in self.data.select_dtypes(include="number").columns
                if self.data[col].nunique() > 2
            ]
            for col in numeric_cols:
                plt.figure(figsize=(8, 4))
                sns.boxplot(x=self.target_col, y=col, data=self.data)
                plt.title(f"{col} by {self.target_col}")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                pdf.savefig()
                plt.close()
                logger.info(f"Boxplot of {col} by {self.target_col} saved")

    @flow(name="generate_report", retries=3, retry_delay_seconds=10, log_prints=True)
    def generate_report(self):
        self.profile_report()
        self.save_statistics()

        with PdfPages(self.pdf_path) as pdf:
            self.plot_missing_values(pdf)
            self.plot_custom_missing_style(pdf)
            self.plot_histograms(pdf)
            self.plot_value_counts(pdf)
            self.plot_correlation_heatmap(pdf)
            self.plot_boxplots_by_target(pdf)

        logger.info(
            f"EDA complete. Files saved in '{self.output_dir}':\n- {self.stats_path.name}\n- {self.pdf_path.name}"
        )


def main():
    params_path = Path("params.yaml")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    data_path = params["data"]["raw_data_path"]
    target_col = params["data"]["target_col"]
    output_prefix = params["eda"]["output_prefix"]

    data = pd.read_csv(data_path)
    eda = EDAReport(data, target_col=target_col, output_prefix=output_prefix)
    eda.generate_report()


if __name__ == "__main__":
    main()
