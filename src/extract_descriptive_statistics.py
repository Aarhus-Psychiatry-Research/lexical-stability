"""Pipeline for extracting metrics from clinical notes using TextDescriptives with option to only sample a proportion"""

import os
import random
import time
from pathlib import Path
from typing import Generator, Tuple

import dacy
import numpy as np
import pandas as pd
import spacy
import textdescriptives as td

# from psycopmlutils.sql.loader import sql_load
from spacy.tokens import Doc

from constants import DESCRIPTIVE_STATS_DIR, sql_load


def yield_text(df_gen: Generator[pd.DataFrame, None, None]) -> Tuple[str, Tuple[str]]:
    """Yields rows from a dataframe generator as a tuple.
    The first element is the text, second is metadata

    Args:
        df_gen (Generator[pd.DataFrame, None, None]): A dataframe generator (e.g. loaded via sql_load)

    Yields:
        Tuple[str, Tuple[str]]: Tuple of text and relevant metadata
    """
    for df in df_gen:
        for row in df.itertuples():
            yield row.Fritekst, (
                row.krypteretCPR,
                row.LPR2_unikt_kontakt_nummer,
                row.afdeling_resultat_udfoert,
                row.datotid_senest_aendret_i_SFIen,
                row.Overskrift,
                row.AktivitetstypeNavn,
                row.elementledetekst,
            )


def yield_rows(df: pd.DataFrame) -> Tuple[str, Tuple[str]]:
    """Yields rows from a dataframe generator as a tuple.
    The first element is the text, second is metadata

    Args:
        df_gen (Generator[pd.DataFrame, None, None]): A dataframe generator (e.g. loaded via sql_load)

    Yields:
        Tuple[str, Tuple[str]]: Tuple of text and relevant metadata
    """
    for row in df.itertuples():
        yield row.fritekst, (
            row.dw_ek_borger,
            row.dw_sk_kontakt,
            row.dw_sk_lpr3kontakt,
            row.afdeling_resultat_udfoert,
            row.datotid_senest_aendret_i_sfien,
            row.overskrift,
            row.aktivitetstypenavn,
            row.elementledetekst,
            row.konpattypetekst,
        )


def extract_metrics_and_meta(
    docs: Generator[Tuple[Doc, Tuple[str]], None, None]
) -> pd.DataFrame:
    """Extract metrics and combine with metadata in dataframe

    Args:
        docs (Generator[Tuple[Doc, Tuple[str]], None, None]): A Doc object created with as_tuple=True

    Returns:
        pd.DataFrame: Merged dataframe
    """
    metrics = []
    meta = []
    for doc, context in docs:
        metrics.append(td.extract_df(doc, include_text=False))
        meta.append(context)
    metrics = pd.concat(metrics).reset_index(drop=True)
    meta = pd.DataFrame(
        meta,
        columns=[
            "dw_ek_borger",
            "dw_sk_kontakt",
            "dw_sk_lpr3kontakt",
            "afdeling_resultat_udfoert",
            "datotid_senest_aendret_i_sfien",
            "overskrift",
            "aktivitetstypenavn",
            "elementledetekst",
            "konpattypetekst",
        ],
    )
    return pd.concat([metrics, meta], axis=1)


if __name__ == "__main__":

    years = [str(year) for year in np.arange(2019, 2022)]
    VIEW = "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret_"
    SCHEMA = "fct"
    SQL = f"SELECT * FROM [{SCHEMA}]."

    SAMPLE_PROP = 0.1
    CHUNKSIZE = 10000

    SERVER = "BI-DPA-PROD"
    DATABASE = "USR_PS_FORSK"

    spacy.require_gpu()
    nlp = dacy.load("large")
    # nlp = spacy.load("da_core_news_lg")
    nlp.add_pipe("textdescriptives/readability")
    nlp.add_pipe("textdescriptives/dependency_distance")

    for year in years:
        print(f"[INFO] Starting year {year}...")
        view = VIEW + year + "_inkl_2021]"
        sql = SQL + view

        save_dir = Path(DESCRIPTIVE_STATS_DIR, year)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get generator of data
        df_gen = sql_load(
            sql,
            SERVER,
            DATABASE,
            chunksize=CHUNKSIZE,
            format_timestamp_cols_to_datetime=False,
        )
        print("[INFO] Fetched data...")
        # Establish connection database to store results
        # dbcon = sqlBI.Connection(SERVER, DATABASE)
        t0 = time.time()
        i = 0
        for n_chunk, chunk in enumerate(df_gen):
            if n_chunk % 20 == 0:
                print(f"Chunk n {n_chunk} of size {CHUNKSIZE} in year {year}")
            if random.random() < SAMPLE_PROP:
                filename = f"extraction_{year}_{i}.csv"
                save_name = os.path.join(save_dir, filename)
                if os.path.exists(save_name):
                    print(f"Already extracted {filename}. Skipping...")
                    i += 1
                    continue
                docs = nlp.pipe(yield_rows(chunk), as_tuples=True)
                extract_metrics_and_meta(docs).to_csv(save_name, index=False)
            i += 1

        t_diff = (time.time() - t0) / 60
        print(f"Time taken for {year}: {t_diff} mins")
