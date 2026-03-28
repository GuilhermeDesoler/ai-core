from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class TreeNodeRecord:
    node_id: str
    name: str
    parent_id: str | None
    level: int
    path_ids: list[str]
    path_names: list[str]
    is_leaf: bool
    icon: str = ""
    resume_url: str = ""

@dataclass(frozen=True)
class QMatrixArtifacts:
    q_matrix: sparse.csr_matrix
    questions_df: pd.DataFrame
    tree_df: pd.DataFrame
    question_id_to_index: dict[str, int]
    concept_id_to_index: dict[str, int]
    concept_index_to_id: dict[int, str]
    question_concepts_df: pd.DataFrame
    metadata: dict[str, Any]


# ============================================================
# I/O helpers
# ============================================================

def _read_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Tree loading and flattening
# ============================================================

def extract_subject_nodes(raw_tree: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Accepts multiple common wrappers and always returns the root list of subject nodes.
    Supported formats:
      - [ ...subject nodes... ]
      - {"subjectNodes": [...]}
      - {"data": {"subjectNodes": [...]}}
    """
    if isinstance(raw_tree, list):
        return raw_tree

    if not isinstance(raw_tree, dict):
        raise TypeError("raw_tree must be a dict or list")

    if "data" in raw_tree and isinstance(raw_tree["data"], dict):
        raw_tree = raw_tree["data"]

    if "subjectNodes" in raw_tree and isinstance(raw_tree["subjectNodes"], list):
        return raw_tree["subjectNodes"]

    raise ValueError("Could not find subjectNodes in tree payload")

def flatten_subject_tree(subject_nodes: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def walk(
        node: dict[str, Any],
        parent_id: str | None = None,
        level: int = 0,
        path_ids: list[str] | None = None,
        path_names: list[str] | None = None,
    ) -> None:
        path_ids = [] if path_ids is None else path_ids.copy()
        path_names = [] if path_names is None else path_names.copy()

        node_id = node["externalId"]
        node_name = node["name"]
        current_path_ids = path_ids + [node_id]
        current_path_names = path_names + [node_name]
        children = node.get("children", []) or []

        rows.append(
            asdict(
                TreeNodeRecord(
                    node_id=node_id,
                    name=node_name,
                    parent_id=parent_id,
                    level=level,
                    path_ids=current_path_ids,
                    path_names=current_path_names,
                    is_leaf=len(children) == 0,
                    icon=node.get("icon", ""),
                    resume_url=node.get("resumeUrl", ""),
                )
            )
        )

        for child in children:
            walk(
                child,
                parent_id=node_id,
                level=level + 1,
                path_ids=current_path_ids,
                path_names=current_path_names,
            )

    for root in subject_nodes:
        walk(root)

    df = pd.DataFrame(rows).sort_values(["level", "name", "node_id"]).reset_index(drop=True)
    df["path_str"] = df["path_names"].apply(lambda xs: " > ".join(xs))
    return df

def load_tree(tree_json_path: str | Path) -> pd.DataFrame:
    raw = _read_json(tree_json_path)
    roots = extract_subject_nodes(raw)
    return flatten_subject_tree(roots)


# ============================================================
# Questions loading and validation
# ============================================================

_REQUIRED_QUESTION_COLUMNS = {
    "questionId",
    "subjectId",
    "subjectAccumulatedNames",
}

def load_questions(questions_json_path: str | Path) -> pd.DataFrame:
    """
    Supports JSON array payload or newline-delimited JSON if the file was preprocessed.
    """
    path = Path(questions_json_path)
    text = path.read_text(encoding="utf-8").strip()

    # Try standard JSON first
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Fallback to JSON Lines
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        payload = rows

    if isinstance(payload, dict):
        # Accept wrappers like {"questions": [...]}
        if "questions" in payload and isinstance(payload["questions"], list):
            payload = payload["questions"]
        else:
            raise ValueError("Questions JSON must be a list or contain a 'questions' list")

    if not isinstance(payload, list):
        raise TypeError("Questions payload must be a list of objects")

    df = pd.DataFrame(payload).copy()
    missing = _REQUIRED_QUESTION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required question columns: {sorted(missing)}")

    rename_map = {
        "questionId": "question_id",
        "subjectId": "subject_id",
        "examYear": "exam_year",
        "institutionName": "institution_name",
        "subjectAccumulatedNames": "subject_path_names",
        "_id": "raw_id",
    }
    df = df.rename(columns=rename_map)

    if "raw_id" not in df.columns:
        df["raw_id"] = df["question_id"]

    if "exam_year" not in df.columns:
        df["exam_year"] = np.nan

    if "institution_name" not in df.columns:
        df["institution_name"] = None

    df = df[
        [
            "raw_id",
            "question_id",
            "subject_id",
            "exam_year",
            "institution_name",
            "subject_path_names",
        ]
    ].copy()

    # ============================================================
    # LIMPEZA DE DADOS CRÍTICOS (NOVO BLOCO)
    # ============================================================

    # Padroniza como string
    df["question_id"] = df["question_id"].astype("string").str.strip()
    df["subject_id"] = df["subject_id"].astype("string").str.strip()

    before = len(df)

    # Remove linhas inválidas
    df = df[
        df["question_id"].notna()
        & df["subject_id"].notna()
        & (df["question_id"] != "")
        & (df["subject_id"] != "")
    ].copy()

    after = len(df)

    removed = before - after

    if removed > 0:
        print(f"[QMatrix] Removidas {removed} questões com IDs inválidos")

    # ============================================================

    return df

def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def validate_questions(questions_df: pd.DataFrame, tree_df: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}

    report["n_rows"] = int(len(questions_df))
    report["n_unique_questions"] = int(questions_df["question_id"].nunique())
    report["duplicated_question_ids"] = int(questions_df["question_id"].duplicated().sum())
    report["null_question_id"] = int(questions_df["question_id"].isna().sum())
    report["null_subject_id"] = int(questions_df["subject_id"].isna().sum())
    report["null_path_names"] = int(questions_df["subject_path_names"].isna().sum())

    if report["duplicated_question_ids"] > 0:
        raise ValueError("Found duplicated question_id values")
    if report["null_question_id"] > 0:
        raise ValueError("Found null question_id values")
    if report["null_subject_id"] > 0:
        raise ValueError("Found null subject_id values")
    if report["null_path_names"] > 0:
        raise ValueError("Found null subject_path_names values")

    tree_ids = set(tree_df["node_id"].tolist())
    invalid_subject_ids = questions_df.loc[~questions_df["subject_id"].isin(tree_ids), "subject_id"].unique().tolist()
    report["invalid_subject_ids"] = invalid_subject_ids
    if invalid_subject_ids:
        raise ValueError(f"Found subject_id values missing from tree: {invalid_subject_ids[:10]}")

    leaf_ids = set(tree_df.loc[tree_df["is_leaf"], "node_id"].tolist())
    non_leaf_mask = ~questions_df["subject_id"].isin(leaf_ids)

    non_leaf_subject_ids = questions_df.loc[non_leaf_mask, "subject_id"].unique().tolist()
    report["non_leaf_subject_ids"] = non_leaf_subject_ids[:50]
    report["n_non_leaf_subject_ids"] = len(non_leaf_subject_ids)
    report["pct_non_leaf_questions"] = float(non_leaf_mask.mean())

    if non_leaf_subject_ids:
        tree_info = tree_df.set_index("node_id")[["name", "level", "path_str", "is_leaf"]].to_dict(orient="index")

        examples = []
        bad_rows = questions_df.loc[non_leaf_mask].head(20)

        for row in bad_rows.itertuples(index=False):
            info = tree_info.get(row.subject_id, {})
            examples.append({
                "question_id": row.question_id,
                "subject_id": row.subject_id,
                "subject_path_names_from_question": row.subject_path_names,
                "tree_name": info.get("name"),
                "tree_level": info.get("level"),
                "tree_path": info.get("path_str"),
                "tree_is_leaf": info.get("is_leaf"),
            })

        report["non_leaf_examples"] = examples

        print(f"[QMatrix] Aviso: {len(non_leaf_subject_ids)} subject_id estão em nós não-folha (isso é permitido).")
        print(pd.DataFrame(examples))

    tree_path_names_by_id = tree_df.set_index("node_id")["path_names"].to_dict()

    mismatched_paths: list[dict[str, Any]] = []
    for row in questions_df.itertuples(index=False):
        expected = tree_path_names_by_id[row.subject_id]
        got = row.subject_path_names
        if len(expected) != len(got):
            mismatched_paths.append(
                {
                    "question_id": row.question_id,
                    "subject_id": row.subject_id,
                    "expected": expected,
                    "got": got,
                }
            )
            continue

        expected_norm = [_normalize_text(x) for x in expected]
        got_norm = [_normalize_text(x) for x in got]
        if expected_norm != got_norm:
            mismatched_paths.append(
                {
                    "question_id": row.question_id,
                    "subject_id": row.subject_id,
                    "expected": expected,
                    "got": got,
                }
            )

    report["n_mismatched_paths"] = len(mismatched_paths)
    report["mismatched_path_examples"] = mismatched_paths[:10]
    report["n_mismatched_paths"] = len(mismatched_paths)
    report["mismatched_path_examples"] = mismatched_paths[:10]

    if mismatched_paths:
        print(f"[QMatrix] Aviso: {len(mismatched_paths)} questões com path divergente da árvore oficial.")

        # opcional: debug
        print(pd.DataFrame(mismatched_paths[:10]))

        return report


# ============================================================
# Q-matrix building
# ============================================================

def build_q_matrix(tree_df: pd.DataFrame, questions_df: pd.DataFrame) -> QMatrixArtifacts:
    """
    Builds a hierarchical multi-hot Q-matrix where each question activates:
      - its leaf node (subject_id)
      - all ancestors in the official path
    """
    tree_by_id = tree_df.set_index("node_id")

    # Stable concept ordering: tree order
    concept_ids = tree_df["node_id"].tolist()
    concept_id_to_index = {cid: idx for idx, cid in enumerate(concept_ids)}
    concept_index_to_id = {idx: cid for cid, idx in concept_id_to_index.items()}

    # Stable question ordering: incoming file order
    question_ids = questions_df["question_id"].tolist()
    question_id_to_index = {qid: idx for idx, qid in enumerate(question_ids)}

    row_idx: list[int] = []
    col_idx: list[int] = []
    question_concept_rows: list[dict[str, Any]] = []

    for q_row in questions_df.itertuples(index=False):
        q_index = question_id_to_index[q_row.question_id]
        path_ids = tree_by_id.at[q_row.subject_id, "path_ids"]
        path_names = tree_by_id.at[q_row.subject_id, "path_names"]

        for node_id, node_name in zip(path_ids, path_names):
            c_index = concept_id_to_index[node_id]
            row_idx.append(q_index)
            col_idx.append(c_index)
            question_concept_rows.append(
                {
                    "question_id": q_row.question_id,
                    "subject_id": q_row.subject_id,
                    "concept_id": node_id,
                    "concept_name": node_name,
                    "is_leaf_concept": node_id == q_row.subject_id,
                    "concept_level": int(tree_by_id.at[node_id, "level"]),
                }
            )

    data = np.ones(len(row_idx), dtype=np.float32)
    q_matrix = sparse.csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(len(question_ids), len(concept_ids)),
        dtype=np.float32,
    )

    question_concepts_df = pd.DataFrame(question_concept_rows)
    avg_concepts_per_question = float(q_matrix.nnz / max(q_matrix.shape[0], 1))
    density = float(q_matrix.nnz / max(q_matrix.shape[0] * q_matrix.shape[1], 1))

    metadata = {
        "n_questions": int(q_matrix.shape[0]),
        "n_concepts": int(q_matrix.shape[1]),
        "n_nonzero": int(q_matrix.nnz),
        "density": density,
        "avg_concepts_per_question": avg_concepts_per_question,
        "n_leaf_concepts": int(tree_df["is_leaf"].sum()),
        "n_root_concepts": int((tree_df["level"] == 0).sum()),
        "max_depth": int(tree_df["level"].max()),
    }

    return QMatrixArtifacts(
        q_matrix=q_matrix,
        questions_df=questions_df,
        tree_df=tree_df,
        question_id_to_index=question_id_to_index,
        concept_id_to_index=concept_id_to_index,
        concept_index_to_id=concept_index_to_id,
        question_concepts_df=question_concepts_df,
        metadata=metadata,
    )


# ============================================================
# Export helpers
# ============================================================

def save_qmatrix_artifacts(artifacts: QMatrixArtifacts, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse.save_npz(output_dir / "q_matrix.npz", artifacts.q_matrix)

    (output_dir / "question_id_to_index.json").write_text(
        json.dumps(artifacts.question_id_to_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "concept_id_to_index.json").write_text(
        json.dumps(artifacts.concept_id_to_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "concept_index_to_id.json").write_text(
        json.dumps(artifacts.concept_index_to_id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "qmatrix_metadata.json").write_text(
        json.dumps(artifacts.metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    artifacts.tree_df.to_parquet(output_dir / "tree_df.parquet", index=False)
    artifacts.questions_df.to_parquet(output_dir / "questions_df.parquet", index=False)
    artifacts.question_concepts_df.to_parquet(output_dir / "question_concepts.parquet", index=False)


# ============================================================
# High-level convenience API
# ============================================================

def build_qmatrix_from_files(
    tree_json_path: str | Path,
    questions_json_path: str | Path,
    output_dir: str | Path | None = None,
) -> QMatrixArtifacts:
    tree_df = load_tree(tree_json_path)
    questions_df = load_questions(questions_json_path)
    validate_questions(questions_df, tree_df)
    artifacts = build_q_matrix(tree_df, questions_df)

    if output_dir is not None:
        save_qmatrix_artifacts(artifacts, output_dir)

    return artifacts


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Example:
    # python qmatrix_builder.py
    # then edit the paths below.
    TREE_PATH = Path("tree.json")
    QUESTIONS_PATH = Path("questions.json")
    OUTPUT_DIR = Path("artifacts/qmatrix_v1")

    if TREE_PATH.exists() and QUESTIONS_PATH.exists():
        artifacts = build_qmatrix_from_files(
            tree_json_path=TREE_PATH,
            questions_json_path=QUESTIONS_PATH,
            output_dir=OUTPUT_DIR,
        )
        print("Q-matrix construída com sucesso")
        print(json.dumps(artifacts.metadata, ensure_ascii=False, indent=2))
    else:
        print("Defina TREE_PATH e QUESTIONS_PATH para arquivos válidos antes de executar.")
