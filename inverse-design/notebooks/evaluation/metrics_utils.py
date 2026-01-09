# metrics_utils.py
# Common utilities for PolyGen / minGPT metric evaluation + JSON SMILES extraction.
# Keep this file in the same folder as your notebook, then:
#   from metrics_utils import extract_smiles_list, normalize_polymer_ends, evaluate_smiles_list, patch_deepchem_scipy_gibrat

from __future__ import annotations

from typing import Any, List, Tuple
import numpy as np
import pandas as pd


def patch_deepchem_scipy_gibrat() -> None:
    """
    Fix for environments where deepchem expects scipy.stats.gibrat but SciPy doesn't provide it.
    Call this BEFORE importing minGPT.metrics (which imports deepchem).
    """
    import scipy.stats as st
    if not hasattr(st, "gibrat"):
        # gibrat distribution corresponds to lognormal with s=1 in SciPy
        st.gibrat = st.lognorm


def extract_smiles_list(obj: Any) -> List[str]:
    """
    Extract a SMILES list from common JSON structures:
      - list[str]
      - list[dict] with SMILES under common keys
      - dict containing a list under some key
    """
    if obj is None:
        return []

    # list[str]
    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], str)):
        return [s.strip() for s in obj if isinstance(s, str) and s.strip()]

    # list[dict]
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        keys = ["mol_smiles", "smiles", "p_smiles", "psmiles", "polymer_smiles", "polymer"]
        out: List[str] = []
        for d in obj:
            if not isinstance(d, dict):
                continue
            for k in keys:
                v = d.get(k, None)
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
                    break
        return out

    # dict -> find a list inside
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return extract_smiles_list(v)

    raise ValueError("Unrecognized JSON structure for SMILES list.")


def normalize_polymer_ends(smiles: str) -> str:
    """
    Normalize polymer repeat-unit ends:
      - If already contains [Cu]/[Au], keep.
      - If exactly two '*' exist, replace first '*' -> [Cu], second '*' -> [Au].
      - Else keep as-is.
    """
    if smiles is None:
        return ""
    s = str(smiles).strip()
    if not s:
        return ""

    if "[Cu]" in s or "[Au]" in s:
        return s

    if s.count("*") == 2:
        first = s.find("*")
        s = s[:first] + "[Cu]" + s[first + 1 :]
        second = s.find("*")
        s = s[:second] + "[Au]" + s[second + 1 :]

    return s


def evaluate_smiles_list(
    gen_smiles_list: List[str],
    train_smiles_list: List[str],
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute 6 metrics aligned with pipeline.evaluate() in your PolyGen/minGPT code:
      (uniqueness, novelty, validity, synthesizability, similarity, diversity)

    Notes:
      - This function imports minGPT.metrics lazily to avoid import-time failures.
      - Make sure to call patch_deepchem_scipy_gibrat() before using this if your environment hits the gibrat error.
    """
    # Patch first to avoid deepchem/scipy import crash
    patch_deepchem_scipy_gibrat()

    # Lazy import (after patch)
    from rdkit import Chem
    from rdkit.DataStructs import TanimotoSimilarity
    from minGPT.metrics import (
        check_novelty,
        validate_mol,
        has_two_ends,
        calculateScore,
        calculate_morgan_fingerprint,
        calculate_diversity,
    )

    # DataFrames
    df_train = pd.DataFrame({"mol_smiles": train_smiles_list})
    df_gen = pd.DataFrame({"mol_smiles": gen_smiles_list})
    num_samples = len(df_gen)
    if num_samples == 0:
        raise ValueError("No generated SMILES provided.")

    # -------------------------------
    # Novelty & Uniqueness (aligned)
    # -------------------------------
    df_train["duplicate"] = df_train["mol_smiles"].duplicated()
    df_train = df_train[df_train["duplicate"] == False]

    df_gen["duplicate"] = df_gen["mol_smiles"].duplicated()
    uniqueness = 1 - len(df_gen[df_gen["duplicate"] == True]) / num_samples

    # novelty: adds "diversity" column with values like "novel"
    df_gen_2 = check_novelty(df_gen.copy(), df_train, "mol_smiles")
    count_not_novel = df_gen_2["mol_smiles"][df_gen_2["diversity"] != "novel"].count()
    novelty = 1 - count_not_novel / num_samples

    # -------------------------------
    # Validity (based on df_gen_2)
    # -------------------------------
    df_gen_valid = validate_mol(df_gen_2.copy(), column_name="mol_smiles")
    df_gen_valid = has_two_ends(df_gen_valid)

    df_valid = df_gen_valid.loc[
        (df_gen_valid["validity"] == "ok") & (df_gen_valid["has_two_ends"] == True)
    ]
    validity = len(df_valid) / num_samples

    # -------------------------------
    # Clean set (for SA, similarity, diversity)
    # -------------------------------
    df_clean = df_gen_valid.loc[
        (df_gen_valid["duplicate"] == False)
        & (df_gen_valid["diversity"] == "novel")
        & (df_gen_valid["validity"] == "ok")
        & (df_gen_valid["has_two_ends"] == True)
    ]

    # -------------------------------
    # Synthesizability (SA < 5)
    # -------------------------------
    if len(df_clean) > 0:
        sa_scores_clean = [calculateScore(Chem.MolFromSmiles(s)) for s in df_clean["mol_smiles"]]
        synthesizability = len([x for x in sa_scores_clean if x < 5]) / len(df_clean)
    else:
        synthesizability = np.nan

    # -------------------------------
    # Similarity & Diversity
    # -------------------------------
    if len(df_clean) > 0:
        morgan_gen = calculate_morgan_fingerprint(df_clean["mol_smiles"])
        morgan_train = calculate_morgan_fingerprint(df_train["mol_smiles"])

        # similarity: for each gen, mean similarity to all train, then mean across gens
        tanimoto_similarity = []
        for i in range(len(df_clean["mol_smiles"])):
            f1 = morgan_gen[i]
            scores = [TanimotoSimilarity(f1, f2) for f2 in morgan_train]
            tanimoto_similarity.append(np.mean(scores))
        similarity = float(np.mean(tanimoto_similarity))

        # diversity
        _, diversity = calculate_diversity(df_clean["mol_smiles"].to_list())
    else:
        similarity = np.nan
        diversity = np.nan

    return float(uniqueness), float(novelty), float(validity), float(synthesizability), float(similarity), float(diversity)
