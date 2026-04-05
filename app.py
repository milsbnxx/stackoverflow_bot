from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.retriever import EmbeddingRetriever


BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "data" / "index"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "frontend"),
    static_folder=str(BASE_DIR / "frontend"),
    static_url_path="/assets",
)

retriever = None
retriever_error = ""


def get_retriever():
    global retriever
    global retriever_error

    if retriever is not None:
        return retriever

    try:
        retriever = EmbeddingRetriever(index_dir=INDEX_DIR)
    except Exception as e:  # noqa: BLE001
        retriever_error = str(e)
        return None

    return retriever


def shorten(text: str, max_len: int = 700) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/search")
def api_search():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    top_k = int(payload.get("top_k", 3))
    top_k = max(1, min(top_k, 5))

    if not query:
        return jsonify({"ok": False, "error": "Введите вопрос."}), 400

    local_retriever = get_retriever()
    if local_retriever is None:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Индекс не найден. Сначала выполните: python build_index.py",
                    "details": retriever_error,
                }
            ),
            500,
        )

    results = local_retriever.search(query, top_k=top_k)
    normalized = []
    for item in results:
        normalized.append(
            {
                "score": round(float(item["score"]), 4),
                "question_title": item["question_title"],
                "answer_text": shorten(item["answer_text"]),
            }
        )

    return jsonify({"ok": True, "results": normalized})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
