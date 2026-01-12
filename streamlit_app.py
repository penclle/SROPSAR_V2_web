from __future__ import annotations

import io
import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.model_loader import load_bundle
from backend.predict import predict_from_library, predict_from_descriptors
from backend.smiles_draw import smiles_to_png


# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"


# ---------------- Helpers ----------------
def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _safe_read_table(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


# ---------------- Page config ----------------
st.set_page_config(page_title="SROPSAR V2", layout="wide")

# ---- CSS polish (simple + stable) ----
st.markdown(
    """
<style>
div.stButton > button {border-radius: 10px !important; font-weight: 700 !important;}
div[data-testid="stMetric"] {background:white; border:1px solid #C9D2EA; border-radius:12px; padding:12px;}
div[data-testid="stDataFrame"] {border:1px solid #C9D2EA; border-radius:12px; overflow:hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Header (icons + bigger title + single-line full name) ----------------
left_icon = ASSET_DIR / "sulfate.png"
right_icon = ASSET_DIR / "organic.png"

left_b64 = _img_b64(left_icon) if left_icon.exists() else ""
right_b64 = _img_b64(right_icon) if right_icon.exists() else ""

# ---------------- 修改后的 Header 代码 ----------------
# ---------------- 修改后的 Header 代码 ----------------
st.markdown(
    f"""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">
<div style="width:18%; text-align:left;">
{"<img src='data:image/png;base64," + left_b64 + "' style='height:78px;'/>" if left_b64 else ""}
</div>
<div style="width:64%; text-align:center;">
<div style="font-size:36px; font-weight:900; letter-spacing:0.6px; line-height:1.05;">
SROPSAR V2
</div>
<div style="font-size:16px; font-weight:700; margin-top:6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
Sulfate Radicals–Organic Pollutants Structure–Activity Relationships
</div>
</div>
<div style="width:18%; text-align:right;">
{"<img src='data:image/png;base64," + right_b64 + "' style='height:78px;'/>" if right_b64 else ""}
</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- Sidebar: model switch ----------------
model_name = st.sidebar.radio("Model", ["q-RASAR", "QSAR"], index=0)

@st.cache_resource
def get_bundle(name: str):
    return load_bundle(name)

bundle = get_bundle(model_name)
st.sidebar.success(f"Loaded: {bundle.name}")
st.sidebar.info(f"Library rows: {len(bundle.library)}")

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(
    [
        "Single prediction (library SMILES)",
        "Batch prediction",
        "Search Info (ID / SMILES)",
    ]
)

# ---------------- Tab 1: Single (library only, show quality+structure) ----------------
with tab1:
    st.write("**Single prediction (SMILES-only)** works only for compounds present in the built-in descriptor library.")
    smi = st.text_input("SMILES", value="", placeholder="Paste a SMILES here...")
    run = st.button("Run", type="primary")

    if run:
        res = predict_from_library(bundle, [smi])

        if not bool(res.loc[0, "FOUND"]):
            st.error("Not found in the built-in library. Use Batch → External (with descriptors) for new compounds.")
        else:
            in_ad = bool(res.loc[0, "In_AD"])
            logksr = float(res.loc[0, "Log kSR"])

            colA, colB = st.columns([1, 1])

            with colA:
                st.metric("Log kSR", f"{logksr:.6f}")
                st.metric("Applicability Domain", "In AD" if in_ad else "Out AD")
                st.caption(
                    f"AD distance = {float(res.loc[0,'AD_Distance']):.4f} "
                    f"(threshold {float(res.loc[0,'AD_Threshold']):.4f})"
                )

                good_fp = ASSET_DIR / "quality_good.png"
                bad_fp = ASSET_DIR / "quality_poor.png"
                qimg = good_fp if in_ad else bad_fp
                if qimg.exists():
                    st.image(str(qimg), caption="Predictive quality", use_container_width=True)

            with colB:
                png = smiles_to_png(str(res.loc[0, "SMILES"]))
                if png:
                    st.image(png, caption="Structure", use_container_width=True)
                else:
                    st.warning("RDKit drawing failed on server. (Structure image unavailable)")

            st.subheader("Result")
            st.dataframe(res, use_container_width=True)

# ---------------- Tab 2: Batch (two modes) ----------------
with tab2:
    mode = st.radio(
        "Batch mode",
        ["Library (SMILES-only, returns FOUND)", "External (with descriptors, predicts all rows)"],
        horizontal=True,
    )

    st.write("Upload CSV/Excel. Must contain a **SMILES** column.")
    upl = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])

    if upl:
        df = _safe_read_table(upl)

        if "SMILES" not in df.columns:
            st.error("Uploaded file must contain a SMILES column.")
        else:
            if mode.startswith("Library"):
                out = predict_from_library(bundle, df["SMILES"].astype(str).tolist())
                # Merge user-provided ID if any
                if "ID" in df.columns and "ID" in out.columns:
                    out["ID"] = out["ID"].where(out["ID"].astype(str).str.len() > 0, df["ID"].astype(str))
                st.success(f"Done. FOUND={int(out['FOUND'].sum())}/{len(out)}")
            else:
                # External with descriptors
                try:
                    out = predict_from_descriptors(bundle, df)
                    st.success(f"Done. Predicted rows: {len(out)}")
                except Exception as e:
                    st.error(str(e))
                    st.stop()

            st.dataframe(out.head(200), use_container_width=True)

            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as w:
                out.to_excel(w, index=False, sheet_name="predictions")
            st.download_button(
                "Download results (Excel)",
                data=bio.getvalue(),
                file_name=f"SROPSAR_V2_{bundle.name}_batch_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ---------------- Tab 3: Search Info ----------------
with tab3:
    st.write("This table shows **ID** and **SMILES** from the built-in library (model-specific).")
    show_n = st.slider("Rows to display", min_value=50, max_value=5000, value=500, step=50)
    df_show = bundle.library[["ID", "SMILES"]].copy().head(show_n)
    st.dataframe(df_show, use_container_width=True)
    st.caption("Tip: copy an ID/SMILES and paste into the Single prediction tab.")
