"""Extract novelty and resonance from keyword proportions"""
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from constants import ENTROPY_SAVE_DIR, GROUPED_KW_SAVE_DIR
from utils.infodynamics import InfoDynamics, jsd


def load_dataframe(path: str) -> pd.DataFrame:
    """Load dataframe from file"""
    return pd.read_csv(path)


def extract_novelty_resonance(
    df: pd.DataFrame,
    theta: Union[List[List], np.array],
    dates: Union[List, np.array],
    window: int,
):
    idmdl = InfoDynamics(data=theta, time=dates, window=window)
    idmdl.resonance(meas=jsd)

    df["novelty"] = idmdl.nsignal
    df["transience"] = idmdl.tsignal
    df["resonance"] = idmdl.rsignal
    df["nsigma"] = idmdl.nsigma
    df["tsigma"] = idmdl.tsigma
    df["rsigma"] = idmdl.rsigma
    return df


if __name__ == "__main__":

    ENTROPY_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    AGG_LEVEL = ["quarterly", "yearly"]

    WINDOW_SIZE = [2]

    for agg_level in AGG_LEVEL:
        files = list(p for p in Path(GROUPED_KW_SAVE_DIR).iterdir())
        # only run extraction on files maching the aggregation level
        files = filter(lambda f: agg_level in f.name, files)

        for file in files:
            df = load_dataframe(file)
            if agg_level == "yearly":
                # to make further processing easier, we add a date column that
                # matches the quarter column
                df["date"] = 1
            sort_by = ["year", "date"]
            df = df.sort_values(sort_by)
            # exclude anything before 2013
            df = df[df["year"] >= 2013]
            df = df.set_index(sort_by)
            theta = df.to_numpy()
            for window in WINDOW_SIZE:
                df = extract_novelty_resonance(
                    df, theta=theta, dates=range(len(df)), window=window
                )
                filename = f"entropy_{agg_level}_window_{window}_" + file.name
                filename = ENTROPY_SAVE_DIR / filename
                df.to_csv(filename)
