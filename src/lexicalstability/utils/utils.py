from typing import List

import docx
import pandas as pd


def bin_continuous_data(series: pd.Series, bins: List[int]) -> pd.Series:
    """For prettier formatting of continuous binned data such as age.
    Args:
        series (pd.Series): Series with continuous data such as age
        bins (List[int]): Desired bins
    Returns:
        pd.Series: Binned data
    Example:
    >>> ages = pd.Series([15, 18, 20, 30, 32, 40, 50, 60, 61])
    >>> age_bins = [0, 18, 30, 50, 110]
    >>> bin_Age(ages, age_bins)
    0     0-18
    1     0-18
    2    19-30
    3    19-30
    4    31-50
    5    31-50
    6    31-50
    7      51+
    8      51+
    """
    labels = []
    for i, bin in enumerate(bins):
        if i == 0:
            labels.append(f"{bin}-{bins[i+1]}")
        elif i < len(bins) - 2:
            labels.append(f"{bin+1}-{bins[i+1]}")
        elif i == len(bins) - 2:
            labels.append(f"{bin+1}+")
        else:
            continue

    return pd.cut(series, bins=bins, labels=labels)


def table_to_docx(df, path):
    doc = docx.Document()
    t = doc.add_table(df.shape[0] + 1, df.shape[1])

    # add the header rows.
    for j in range(df.shape[-1]):
        t.cell(0, j).text = df.columns[j]

    # add the rest of the data frame
    for i in range(df.shape[0]):
        for j in range(df.shape[-1]):
            t.cell(i + 1, j).text = str(df.values[i, j])

    # save the doc
    doc.save(path)
