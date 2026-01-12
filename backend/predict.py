from __future__ import annotations
import numpy as np
import pandas as pd
from .ad import ADEvaluator

def _canon_smiles(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(s)
        if m is None:
            return s
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return s

def _get_required_features(prep: dict) -> list[str]:
    """
    Expect prep_data contains key 'Xtrain' as pandas DataFrame.
    The required feature list is Xtrain.columns.
    """
    if isinstance(prep, dict) and "Xtrain" in prep:
        Xtrain = prep["Xtrain"]
        if hasattr(Xtrain, "columns"):
            return list(Xtrain.columns)
    raise ValueError("prep_data.joblib must contain key 'Xtrain' as a pandas DataFrame.")

def _build_ad(bundle) -> tuple[list[str], ADEvaluator]:
    feat_cols = _get_required_features(bundle.prep)
    Xtrain = bundle.prep["Xtrain"]
    Xtrain_mat = np.asarray(Xtrain[feat_cols].values, dtype=float)
    ad = ADEvaluator(Xtrain_mat, k=1.0)  # change k if needed
    return feat_cols, ad

def predict_from_library(bundle, smiles_list: list[str]) -> pd.DataFrame:
    """
    Library-based prediction (SMILES-only): only compounds present in the library can be predicted.
    Returns FOUND flag and predictions/AD for hits.
    """
    lib = bundle.library.copy()
    lib["_canon"] = lib["SMILES"].astype(str).map(_canon_smiles)

    feat_cols, ad = _build_ad(bundle)

    idx = {cs: i for i, cs in enumerate(lib["_canon"].tolist()) if cs}

    rows = []
    for smi in smiles_list:
        cs = _canon_smiles(smi)
        if not cs or cs not in idx:
            rows.append({
                "ID": "",
                "SMILES": smi,
                "FOUND": False,
                "Log kSR": np.nan,
                "AD_Distance": np.nan,
                "AD_Threshold": np.nan,
                "In_AD": np.nan
            })
            continue

        rec = lib.iloc[idx[cs]]

        # Extract feature vector from library row
        Xrow = rec.reindex(feat_cols)
        Xrow = pd.to_numeric(Xrow, errors="coerce")
        X = np.asarray([Xrow.values], dtype=float)

        # Predict
        y = float(bundle.model.predict(X)[0])

        # AD
        d, in_dom, thr = ad.evaluate(X)

        rows.append({
            "ID": rec.get("ID", ""),
            "SMILES": rec.get("SMILES", smi),
            "FOUND": True,
            "Log kSR": y,
            "AD_Distance": float(d[0]),
            "AD_Threshold": float(thr),
            "In_AD": bool(in_dom[0]),
        })

    return pd.DataFrame(rows)

def predict_from_descriptors(bundle, df: pd.DataFrame) -> pd.DataFrame:
    """
    External prediction (with descriptors):
    User uploads a table containing SMILES + all required feature columns.
    Output includes Log kSR + AD metrics.
    """
    if "SMILES" not in df.columns:
        raise ValueError("Input table must contain a SMILES column.")

    feat_cols, ad = _build_ad(bundle)

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required descriptor columns (first 30): {missing[:30]}")

    X = df[feat_cols].copy()
    for c in feat_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    Xv = np.asarray(X.values, dtype=float)

    y = bundle.model.predict(Xv).astype(float)
    d, in_dom, thr = ad.evaluate(Xv)

    out = df.copy()
    out["Log kSR"] = y
    out["AD_Distance"] = d.astype(float)
    out["AD_Threshold"] = float(thr)
    out["In_AD"] = in_dom.astype(bool)
    return out
