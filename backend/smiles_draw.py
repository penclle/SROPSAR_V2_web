from __future__ import annotations
from io import BytesIO

def smiles_to_png(smiles: str, size=(360, 240)) -> bytes | None:
    """Return PNG bytes of RDKit 2D depiction; None if failed."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        m = Chem.MolFromSmiles((smiles or "").strip())
        if m is None:
            return None
        img = Draw.MolToImage(m, size=size)
        bio = BytesIO()
        img.save(bio, format="PNG")
        return bio.getvalue()
    except Exception:
        return None
