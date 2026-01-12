from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib
import pandas as pd

ART_DIR = Path(__file__).resolve().parents[1] / "artifacts"

@dataclass
class ModelBundle:
    name: str
    model: object
    prep: dict
    library: pd.DataFrame

def _normalize_library_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize SMILES column
    if "SMILES" not in df.columns:
        for alt in ["smiles", "Smiles", "SMILE"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "SMILES"})
                break

    # Normalize ID column
    if "ID" not in df.columns:
        for alt in ["Name", "name", "Compound", "compound", "ID "]:
            if alt in df.columns:
                df = df.rename(columns={alt: "ID"})
                break

    if "SMILES" not in df.columns:
        raise ValueError("Library file must contain a SMILES column.")
    if "ID" not in df.columns:
        raise ValueError("Library file must contain an ID (or Name) column.")
    return df

def load_bundle(model_name: str) -> ModelBundle:
    """
    model_name: "q-RASAR" or "QSAR"
    Required files in artifacts/:
      <model_name>_model.joblib
      <model_name>_prep_data.joblib
      <model_name>_library.xlsx
    """
    model_fp = ART_DIR / f"{model_name}_model.joblib"
    prep_fp  = ART_DIR / f"{model_name}_prep_data.joblib"
    lib_fp   = ART_DIR / f"{model_name}_library.xlsx"

    if not model_fp.exists():
        raise FileNotFoundError(f"Missing model file: {model_fp}")
    if not prep_fp.exists():
        raise FileNotFoundError(f"Missing prep_data file: {prep_fp}")
    if not lib_fp.exists():
        raise FileNotFoundError(f"Missing library file: {lib_fp}")

    model = joblib.load(model_fp)
    prep = joblib.load(prep_fp)
    lib = pd.read_excel(lib_fp)
    lib = _normalize_library_columns(lib)

    return ModelBundle(name=model_name, model=model, prep=prep, library=lib)
