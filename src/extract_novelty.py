"""Extract novelty and resonance from keyword proportions"""
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

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

    BASE_SAVE_DIR = Path(
        "\\\\TSCLIENT\\P\\LASHA601\\documentLibrary\\lexical-dynamics-data"
    )
    GROUPED_KW_SAVE_DIR = BASE_SAVE_DIR / "kw_counts_per_group"
    ENTROPY_SAVE_DIR = BASE_SAVE_DIR / "entropy"
    if not ENTROPY_SAVE_DIR.exists():
        ENTROPY_SAVE_DIR.mkdir()

    AGG_LEVEL = "quarterly"

    WINDOW_SIZE = [2]

    files = list(p for p in Path(GROUPED_KW_SAVE_DIR).iterdir())
    # only run extraction on files maching the aggregation level
    files = filter(lambda f: AGG_LEVEL in f.name, files)

    for file in files:
        df = load_dataframe(file)
        df = df.sort_values(["year", "date"])
        # exclude anything before 2013
        df = df[df["year"] >= 2013]
        df = df.set_index(["year", "date"])
        theta = df.to_numpy()
        for window in WINDOW_SIZE:
            df = extract_novelty_resonance(
                df, theta=theta, dates=range(len(df)), window=window
            )
            filename = f"entropy_{AGG_LEVEL}_window_{window}_" + file.name
            filename = ENTROPY_SAVE_DIR / filename
            df.to_csv(filename)
