"""Extract number of occurences of keywords in clinical notes and calculate proportions. Preprocessing step before novelty/resonance calculation"""
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Set, Tuple

import numpy as np
import pandas as pd
import yaml
from psycopmlutils.loaders.raw.sql_load import sql_load


def load_keywords(path: str) -> Dict[str, str]:
    """Loads keywords from a yaml file"""
    with open(path, "r", encoding="utf8") as f:
        return yaml.full_load(f)


def combine_similar_columns(df: pd.DataFrame):
    """Combine counts from similar columns (e.g. 2nd person/second person)"""

    df["skizoaffektiv"] = df["skizoaffektiv"] + df["skizo-affektiv"]
    df["andenpersons"] = df["andenpersons"] + df["2.person"]
    df["tredjepersons"] = df["tredjepersons"] + df["3.person"]
    df["kataton*"] = df["kataton"] + df["katatoni"]
    df["mani*"] = df["mani"] + df["manisk"]
    df["hypomani*"] = df["hypomani"] + df["hypomanisk"]
    df["forvirret"] = df["forvirret"] + df["forvirring"]
    df["ambivalent"] = df["ambivalent"] + df["ambivalens"]
    to_drop = [
        "skizo-affektiv",
        "2.person",
        "3.person",
        "kataton",
        "katatoni",
        "mani",
        "manisk",
        "hypomani",
        "hypomanisk",
        "forvirring",
    ]
    return df.drop(to_drop, axis=1)


def extract_keyword_counts(
    df_gen: Generator[Tuple[str, str], None, None], keywords
) -> Dict[str, Counter]:
    """ "Extract keyword counts for all texts from a generator of datafranes containing the columns "datotid_senest_aendret_i_sfien"
    and "fritekst"
    """
    res = defaultdict(list)
    for row in yield_rows(df_gen):
        if res[row[0]]:
            res[row[0]].update(count_keywords(row[1], keywords))
        else:
            res[row[0]] = count_keywords(row[1], keywords)
    return res


def prepare_sql_and_load(year) -> Generator[pd.DataFrame, None, None]:
    """
    prepare string input to sql_load
    """
    sql = SQL + VIEW + year + "_inkl_2021]"
    return sql_load(sql, SERVER, DATABASE, chunksize=CHUNKSIZE)


def yield_rows(df_gen: pd.DataFrame) -> Generator[Tuple[str, str], None, None]:
    """Yields rows from a dataframe generator as a tuple.
    The first element is the date, second is the free text

    Args:
        df (Generator[pd.DataFrame, None, None]): A generator of dataframes

    Yields:
        Tuple[str, str]: A tuple of the date as a string and free text
    """
    for df in df_gen:
        for row in df.itertuples():
            yield (
                row.datotid_senest_aendret_i_sfien.date().strftime("%Y-%m-%d"),
                row.fritekst,
            )


def count_keywords(text: str, keywords: Set[str]) -> Counter:
    """Counts keywords in a text.
    Checks if the keyword is in the text and if so, adds one to the counter.
    Each keyword will only be counted once per text, even if it appears multiple times.
    This avoids possibly inflating counts for very long texts, perhaps at the expense of some granularity
    Args:
        text: str
        keywords (Set[str]): set of lowercased keywords
    """
    return Counter([keyword for keyword in keywords if keyword in text.lower()])


def unpack_list_of_dicts(list_of_dicts: List[Dict[str, Counter]]) -> Dict[str, Counter]:
    """Unpack list of dicts into one dict."""
    return {k: v for d in list_of_dicts for k, v in d.items()}


def save_keywords(keywords: Dict[str, str], path: str) -> None:
    """Saves a dictionary of keywords to a file."""
    with open(path, "wb") as f:
        pickle.dump(keywords, f)


def flatten_list_of_lists(list_of_lists: List[List[str]]) -> List[str]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]


def load_pickle(path: str):
    """loads a pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)


def tally_counts(df: pd.DataFrame, date_col: str, bin_size: str) -> pd.DataFrame:
    df["year"] = df[date_col].dt.year
    if bin_size == "weekly":
        df = df.groupby([df["year"], df[date_col].dt.isocalendar().week])
    if bin_size == "monthly":
        df = df.groupby([df["year"], df[date_col].dt.month])
    if bin_size == "quarterly":
        df = df.groupby([df["year"], df[date_col].dt.quarter])

    df = df.agg(["sum"], axis="columns")
    df.columns = df.columns.droplevel(1)
    return df


def rowwise_proportions(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda x: x / x.sum(), axis=1)


def get_sparse_columns(
    df: pd.DataFrame, min_counts: int, min_n_with_min_counts: int
) -> List[str]:
    """Check sparsity of a dataframe of counts.

    Args:
        df (pd.DataFrame): dataframe with count columns
        min_counts (int): minimum number of counts per timestep to be relevant
        min_timebins_with_min_counts (int): the minimum number of timebins (rows) with min_counts needed to be kept

    Returns:
        List of columns to remove based on the criteria
    """
    n_counts_per_keyword = (df > min_counts).astype(int).sum(axis=0)
    excluded = n_counts_per_keyword[n_counts_per_keyword < min_n_with_min_counts]

    print(
        f"Input df has {len(df.columns)} columns.\nBy requiring at least {min_counts} counts in at least {min_n_with_min_counts} timebins (rows), {len(excluded)} columns will be removed."
    )
    print(f"Excluded keywords: {excluded.index.tolist()}")
    excluded.to_csv(
        BASE_SAVE_DIR
        / f"excluded_keywords_{agg_level}_min_counts_{min_counts}_min_n_{min_n_with_min_counts}.csv",
        index=False,
    )
    return excluded.index.tolist()


if __name__ == "__main__":
    # set variables
    RUN_KEYWORD_EXTRACTION = False
    BASE_SAVE_DIR = Path(
        "\\\\TSCLIENT\\P\\LASHA601\\documentLibrary\\lexical-dynamics-data"
    )
    KW_PATH = Path.cwd() / "data" / "keywords.yaml"

    EXTRACTED_KW_SAVE_PATH = BASE_SAVE_DIR / "keyword_counts.pkl"
    GROUPED_KW_SAVE_DIR = BASE_SAVE_DIR / "kw_counts_per_group"
    if not GROUPED_KW_SAVE_DIR.exists():
        GROUPED_KW_SAVE_DIR.mkdir(parents=True)

    AGGREGATION_LEVELS = ["quarterly"]
    MIN_COUNTS = 10
    MIN_OCCURENCES = 5
    LOG_COUNTS = False

    if RUN_KEYWORD_EXTRACTION:

        VIEW = "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret_"
        SCHEMA = "fct"
        SQL = f"SELECT * FROM [{SCHEMA}]."

        CHUNKSIZE = 10000

        SERVER = "BI-DPA-PROD"
        DATABASE = "USR_PS_FORSK"

        years = [str(year) for year in np.arange(2011, 2022)]
        # load keywords and the f category they are from
        f_categories_and_keywords = load_keywords(KW_PATH)
        # extract the keywords only
        keywords = flatten_list_of_lists(f_categories_and_keywords.values())
        keywords = set([keyword.lower() for keyword in keywords])
        ## consider making keywords a set and change count_keywords accordingly - test performance to see if necessary

        kw_counts = []
        for year in years:
            print(f"Starting year {year}")
            t0 = time.time()
            df_gen = prepare_sql_and_load(year)
            kw_counts.append(extract_keyword_counts(df_gen, keywords))
            print(f"Finished year {year} in {time.time() - t0} seconds")

        kw_counts = unpack_list_of_dicts(kw_counts)
        save_keywords(kw_counts, EXTRACTED_KW_SAVE_PATH)

    kw_counts = load_pickle(EXTRACTED_KW_SAVE_PATH)
    f_categories_and_keywords = load_keywords(KW_PATH)

    # remove empty entries
    kw_counts = {k: v for k, v in kw_counts.items() if v}

    kw_counts = pd.DataFrame.from_dict(kw_counts).T.rename_axis("date").reset_index()
    kw_counts["date"] = pd.to_datetime(kw_counts["date"], format="%Y-%m-%d")
    # kw_counts = combine_similar_columns(kw_counts)

    for agg_level in AGGREGATION_LEVELS:
        agg_counts = tally_counts(kw_counts, "date", agg_level)
        agg_counts = combine_similar_columns(agg_counts)
        sparse_cols = get_sparse_columns(
            agg_counts,
            min_counts=MIN_COUNTS,
            # has to be minimum MIN_COUNTS in a third of the timesteps
            min_n_with_min_counts=MIN_OCCURENCES,
        )
        agg_counts = agg_counts.drop(sparse_cols, axis=1)
        # to avoid zeros and NaNs when taking the proportion we add 1 to each count
        agg_counts += 1
        if LOG_COUNTS:
            agg_counts = agg_counts.apply(lambda x: np.log(x))
            MIN_COUNTS = np.log(MIN_COUNTS)
        for f_category, keywords in f_categories_and_keywords.items():
            print(f"Processing {f_category}, with min_counts = {MIN_COUNTS}")
            keywords = [keyword.lower() for keyword in keywords]
            # some keywords were removed due to pruning. Only selecting the subset that
            # still remains
            sub_counts = agg_counts[
                agg_counts.columns[agg_counts.columns.isin(keywords)]
            ]

            sub_counts = rowwise_proportions(sub_counts)
            filename = (
                GROUPED_KW_SAVE_DIR
                / f"keyword_prop_{f_category}_{agg_level}_is_logged_{LOG_COUNTS}_min_counts_{MIN_COUNTS}_min_occ_{MIN_OCCURENCES}.csv"
            )
            sub_counts.to_csv(filename)

        # for all together
        print("Processing aggregate")
        keywords = flatten_list_of_lists(f_categories_and_keywords.values())
        agg_prop = rowwise_proportions(agg_counts)
        filename = (
            GROUPED_KW_SAVE_DIR
            / f"keyword_prop_aggregate_{agg_level}_is_logged_{LOG_COUNTS}_min_counts_{MIN_COUNTS}_min_occ_{MIN_OCCURENCES}.csv"
        )
        agg_prop.to_csv(filename)
