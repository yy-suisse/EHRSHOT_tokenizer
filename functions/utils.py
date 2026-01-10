import yaml
import polars as pl 
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_config() -> dict:
    config_path = "configs.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_ulms(config):

    MRCONSO_COLUMNS = [
        "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI",
        "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"
    ]

    path_mrconso = config["ontologies"]["ULMS_info"]  # e.g. "D:/UMLS/2025AA/META/MRCONSO.RRF"

    # Your target source vocabularies in UMLS (example)
    # EHR_ontologies = ["SNOMEDCT_US", "ICD10CM", "RXNORM", "LOINC", "ATC"]  # <-- set yours

    keep_ttys = ["PT", "SY", "FSN", "FN"]  # NOTE: "FN" is sometimes used; depends on SAB

    df_umls = (
        pl.read_csv(
            path_mrconso,
            separator="|",
            has_header=False,
            quote_char=None,                 # ignore quotes
            truncate_ragged_lines=True,      # tolerate occasional ragged rows
            ignore_errors=True,              # safer on giant RRFs
            encoding="utf8-lossy",           # avoid crashing on weird chars
        )
    )

    # Drop trailing empty column caused by the final "|" at end of each line
    # (only if it's actually there)
    if df_umls.width == len(MRCONSO_COLUMNS) + 1:
        df_umls = df_umls.select(pl.all().exclude(df_umls.columns[-1]))

    # Assign column names
    df_umls.columns = MRCONSO_COLUMNS

    df_umls_filtered = (
        df_umls
        .filter(
            (pl.col("LAT") == "ENG") &
            (pl.col("SUPPRESS") == "N") &
            (
                pl.col("TTY").is_in(keep_ttys) |
                (pl.col("ISPREF") == "Y")
            )
        )
        .select(
            "CUI",
            "SAB",
            "CODE",
            "TTY",
            "ISPREF",
            "STR",
        )
        .with_columns(
            (pl.col("ISPREF") == "Y").alias("is_preferred")
        )
    )
    return df_umls_filtered

def get_tensor_from_numpy(np_array, device):
    return torch.tensor(np_array, device=device, dtype=torch.float32)