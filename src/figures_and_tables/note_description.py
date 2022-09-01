"""Calculate number of tokens per note type"""
import pandas as pd
from typing import Tuple

from wasabi import Printer

from pathlib import Path
from psycopmlutils.loaders.raw.sql_load import sql_load
from collections import Counter
import time
import pickle

import numpy as np


def yield_rows(df_gen: pd.DataFrame) -> Tuple[str, Tuple[str]]:
    """Yields rows from a dataframe generator as a tuple. 
    The first element is the text from the clinical note, second is the SFI heading

    Args:
        df (Generator[pd.DataFrame, None, None]): A generator of dataframes

    Yields:
        Tuple[str, Tuple[str]]: 
    """
    for df in df_gen:
        for row in df.itertuples():
            yield (row.fritekst, row.overskrift)


def n_tokens(paragraph: str):
    return len(paragraph.split(" "))


if __name__ == "__main__":
    msg = Printer(timestamp=True)

    years = [str(year) for year in np.arange(2011, 2022)]
    VIEW = "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret_"
    SCHEMA = "fct"
    SQL = f"SELECT * FROM [{SCHEMA}]."
    # save path
    BASE_SAVE_DIR = Path(
        "\\\\TSCLIENT\\P\\LASHA601\\documentLibrary\\lexical-dynamics-data"
    )
    CHUNKSIZE = 10000

    SERVER = "BI-DPA-PROD"
    DATABASE = "USR_PS_FORSK"

    results = {}
    for year in years:
        print(f"[INFO] Starting year {year}...")
        view = VIEW + year + "_inkl_2021]"

        sql = SQL + view
        # Get generator of data
        df_gen = sql_load(sql, SERVER, DATABASE, chunksize=CHUNKSIZE)
        print("[INFO] Fetched data...")

        yearly_results = Counter()
        t0 = time.time()

        for row in yield_rows(df_gen):
            yearly_results.update({row[1]: n_tokens(row[0])})

        results[year] = yearly_results
        t_diff = (time.time() - t0) / 60
        print(f"[INFO] Finished year {year} in {t_diff}...")

    with open(BASE_SAVE_DIR / "note_n_tokens.pkl", "wb") as f:
        pickle.dump(results, f)

    df = pd.DataFrame.from_dict(results).T.reset_index().to_csv("n_tokens_per_year.csv")

