from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_DATA_PATH = DATA_DIR / "processed_qa.csv"
DEFAULT_INDEX_DIR = DATA_DIR / "index"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_and_filter_data(data_path: Path, max_rows: int) -> pd.DataFrame:
    usecols = [
        "question_id",
        "answer_id",
        "question_score",
        "answer_score",
        "question_title",
        "question_text",
        "answer_text",
    ]
    df = pd.read_csv(data_path, usecols=usecols)

    df["question_text"] = df["question_text"].fillna("").astype(str).str.strip()
    df["question_title"] = df["question_title"].fillna("").astype(str).str.strip()
    df["answer_text"] = df["answer_text"].fillna("").astype(str).str.strip()

    df = df[
        (df["question_text"].str.len() >= 20)
        & (df["answer_text"].str.len() >= 30)
        & (df["question_score"] >= 0)
        & (df["answer_score"] >= 0)
    ].copy()

    # Берем более качественные пары "вопрос-ответ", чтобы индекс был компактным.
    df = df.sort_values(
        by=["question_score", "answer_score"],
        ascending=False,
    )
    if max_rows > 0:
        df = df.head(max_rows).copy()

    df = df.reset_index(drop=True)
    return df


def build_index(
    data_path: Path,
    output_dir: Path,
    model_name: str,
    max_rows: int,
    batch_size: int,
) -> None:
    print(f"Reading dataset: {data_path}")
    df = load_and_filter_data(data_path, max_rows=max_rows)
    print(f"Rows selected for index: {len(df)}")

    texts = df["question_text"].tolist()
    if not texts:
        raise ValueError("No rows selected. Check filters or source CSV.")

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding questions...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = normalize_rows(embeddings).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "question_embeddings.npy"
    metadata_path = output_dir / "metadata.json"

    metadata = df.to_dict(orient="records")
    np.save(embeddings_path, embeddings)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved embeddings: {embeddings_path}")
    print(f"Saved metadata:   {metadata_path}")
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build embedding index for Stack Overflow QA dataset."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to processed_qa.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory for index files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20000,
        help="How many rows to keep in index (0 = all rows)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding inference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
