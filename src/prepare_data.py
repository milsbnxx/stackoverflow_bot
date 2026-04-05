from pathlib import Path
import re
import html
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

QUESTIONS_PATH = DATA_DIR / "Questions.csv"
ANSWERS_PATH = DATA_DIR / "Answers.csv"
OUTPUT_PATH = DATA_DIR / "processed_qa.csv"


def clean_html_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    questions = pd.read_csv(QUESTIONS_PATH, encoding="latin-1")
    answers = pd.read_csv(ANSWERS_PATH, encoding="latin-1")

    print("Questions shape:", questions.shape)
    print("Answers shape:", answers.shape)

    questions = questions.rename(columns={"Id": "question_id"})
    answers = answers.rename(columns={"ParentId": "question_id", "Id": "answer_id"})

    questions["clean_title"] = questions["Title"].apply(clean_html_text)
    questions["clean_body"] = questions["Body"].apply(clean_html_text)
    answers["clean_answer"] = answers["Body"].apply(clean_html_text)

    questions["question_title"] = questions["clean_title"].fillna("").str.strip()
    questions["question_body"] = questions["clean_body"].fillna("").str.strip()
    questions["question_text"] = (
        questions["question_title"] + " " + questions["question_body"]
    ).str.strip()

    best_answers = (
        answers.sort_values("Score", ascending=False)
        .drop_duplicates(subset=["question_id"])
        .copy()
    )

    dataset = questions.merge(
        best_answers[["question_id", "answer_id", "Score", "clean_answer"]],
        on="question_id",
        how="inner",
        suffixes=("_question", "_answer")
    )

    dataset = dataset[
        (dataset["question_title"].str.len() > 10) &
        (dataset["answer_text"].str.len() > 30)
    ].copy() if "answer_text" in dataset.columns else dataset.copy()

    dataset["answer_text"] = dataset["clean_answer"].astype(str)

    dataset = dataset[
        (dataset["question_title"].str.len() > 10) &
        (dataset["answer_text"].str.len() > 30)
    ].copy()

    dataset = dataset[
        [
            "question_id",
            "answer_id",
            "Score_question",
            "Score_answer",
            "question_title",
            "question_body",
            "question_text",
            "answer_text",
        ]
    ].rename(
        columns={
            "Score_question": "question_score",
            "Score_answer": "answer_score",
        }
    )

    dataset = dataset[
        (dataset["question_score"] >= 0) &
        (dataset["answer_score"] >= 0)
    ].copy()

    dataset = dataset.reset_index(drop=True)

    dataset.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()