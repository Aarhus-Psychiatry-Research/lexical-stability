"""Calculate aggregated summary statistics from all csv files in all subdirectories"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = "\\\\TSCLIENT\\P\\LASHA601\\documentLibrary\\lexical-dynamics-data\\descriptive_stats"
BIN_SIZE = "quarterly"


def load_csvs(directory: str):
    files = [p for p in Path(directory).glob("*.csv")]
    dfs = [pd.read_csv(file) for file in files]
    return pd.concat(dfs)


def se(x):
    return np.std(x, ddof=1) / np.sqrt(np.size(x))


def summarize_scores(
    df: pd.DataFrame,
    date_col: str = "datotid_senest_aendret_i_sfien",
    bin_size: str = "quarterly",
):
    """Summarize all numeric variables in dataframe with mean, median, sd per time interval.
    Also adds a column for counts
    Args:
        df (pd.DataFrame): A dataframe
        date_col (str, optional): the column containing dates. Defaults to "datotid_senest_aendret_i_sfien".
        bin_size (str, optional): time interval to bin to. One of ["daily", "weekly", "monthly"]. Defaults to "monthly".
    """
    df[date_col] = pd.to_datetime(df[date_col])

    drop_cols = ["aktivitetstypenavn", "elementledetekst", "konpattypetekst"]
    df = df.drop(drop_cols, axis=1)

    if bin_size == "daily":
        df_total = df.drop("overskrift", axis=1).groupby(df[date_col].dt.date)
        df = df.groupby(by=[df[date_col].dt.date, "overskrift"])
    elif bin_size == "weekly":
        df_total = df.drop("overskrift", axis=1).groupby(df[date_col].dt.week)
        df = df.groupby(by=[df[date_col].dt.week, "overskrift"])
    elif bin_size == "monthly":
        df_total = df.drop("overskrift", axis=1).groupby(df[date_col].dt.month)
        df = df.groupby(by=[df[date_col].dt.month, "overskrift"])
    elif bin_size == "quarterly":
        df_total = df.drop("overskrift", axis=1).groupby(df[date_col].dt.quarter)
        df = df.groupby(by=[df[date_col].dt.quarter, "overskrift"])

    aggregations = ["mean", "std", "count", se]

    df_total = df_total.agg(aggregations, axis="columns").reset_index()
    df = df.agg(aggregations, axis="columns").reset_index()
    # flatten multindex columns
    df.columns = ["_".join(c) for c in df.columns.to_flat_index()]
    df_total.columns = ["_".join(c) for c in df_total.columns.to_flat_index()]
    df_total["overskrift_"] = "Aggregate"
    return pd.concat([df, df_total])


if __name__ == "__main__":

    folders = list(p for p in Path(BASE_DIR).iterdir())

    for year in folders:
        print(f"Summarising csv files in directory: {year}")
        df = load_csvs(year)
        df = summarize_scores(df, bin_size=BIN_SIZE)
        save_folder = year / "summary"
        if not save_folder.exists():
            save_folder.mkdir()
        df.to_csv(
            save_folder / (year.name + "_" + BIN_SIZE + "_summary.csv"), index=False
        )
