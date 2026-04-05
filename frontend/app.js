const queryInput = document.getElementById("query");
const topkInput = document.getElementById("topk");
const topkValue = document.getElementById("topkValue");
const searchBtn = document.getElementById("searchBtn");
const resultsBox = document.getElementById("results");
const statusBox = document.getElementById("status");

topkInput.addEventListener("input", () => {
  topkValue.textContent = topkInput.value;
});

queryInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    runSearch();
  }
});

searchBtn.addEventListener("click", runSearch);

function showStatus(message, isError = false) {
  statusBox.classList.remove("hidden");
  statusBox.textContent = message;
  statusBox.style.background = isError ? "#ffe9e9" : "#fff7e8";
  statusBox.style.color = isError ? "#8a2424" : "#734d00";
  statusBox.style.borderColor = isError ? "#f5b3b3" : "#ffd48f";
}

function hideStatus() {
  statusBox.classList.add("hidden");
  statusBox.textContent = "";
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderResults(items) {
  if (!items.length) {
    resultsBox.innerHTML = `
      <article class="result-card">
        <p class="answer">По вашему запросу ничего не найдено.</p>
      </article>
    `;
    return;
  }

  const html = items
    .map((item, index) => {
      return `
        <article class="result-card">
          <div class="result-head">
            <h3 class="result-title">${index + 1}. ${escapeHtml(item.question_title)}</h3>
            <span class="score-badge">Score: ${item.score.toFixed(4)}</span>
          </div>
          <p class="answer">${escapeHtml(item.answer_text)}</p>
        </article>
      `;
    })
    .join("");

  resultsBox.innerHTML = html;
}

async function runSearch() {
  const query = queryInput.value.trim();
  if (!query) {
    showStatus("Введите вопрос.", true);
    return;
  }

  hideStatus();
  searchBtn.disabled = true;
  searchBtn.textContent = "Ищу...";

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: Number(topkInput.value),
      }),
    });

    const data = await response.json();

    if (!response.ok || !data.ok) {
      showStatus(data.error || "Ошибка поиска.", true);
      resultsBox.innerHTML = "";
      return;
    }

    renderResults(data.results || []);
  } catch (error) {
    showStatus("Ошибка сети или сервера. Проверьте, что backend запущен.", true);
    resultsBox.innerHTML = "";
  } finally {
    searchBtn.disabled = false;
    searchBtn.textContent = "Найти ответ";
  }
}
