const $ = (id) => document.getElementById(id);

function currentMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

document.querySelectorAll('input[name="mode"]').forEach((el) =>
  el.addEventListener("change", () => {
    $("mode-url").hidden = currentMode() !== "url";
    $("mode-key").hidden = currentMode() !== "key";
  })
);

function signed(n) {
  return n > 0 ? `+${n}` : `${n}`;
}

function setStatus(text, { error = false, loading = false } = {}) {
  const el = $("status");
  el.hidden = !text;
  el.className = error ? "error" : "";
  el.replaceChildren();
  if (loading) {
    const s = document.createElement("div");
    s.className = "spinner";
    el.append(s);
  }
  el.append(document.createTextNode(text || ""));
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node[k] = v;
  node.append(...children.filter((c) => c != null));
  return node;
}

function formatDuration(sec) {
  if (!sec) return "";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function renderCard(target, role, video) {
  const card = $(target);
  card.replaceChildren();
  if (!video) return;
  if (video.thumbnail) card.append(el("img", { src: video.thumbnail, alt: "", loading: "lazy" }));
  card.append(el("div", { className: "role", textContent: role }));
  if (video.url) {
    card.append(
      el("div", { className: "title" },
        el("a", { href: video.url, target: "_blank", rel: "noopener", textContent: video.title }))
    );
    const meta = [video.uploader, formatDuration(video.duration)].filter(Boolean).join(" ・ ");
    if (meta) card.append(el("div", { className: "meta", textContent: meta }));
  }
  card.append(el("div", { className: "key", textContent: `推定キー: ${video.key}` }));
}

function renderChart(scores, best) {
  const chart = $("chart");
  chart.replaceChildren();
  const values = scores.map((s) => s.score);
  const min = Math.min(...values);
  const max = Math.max(...values);
  for (const { shift, score } of scores) {
    const ratio = max > min ? (score - min) / (max - min) : 1;
    const bar = el("div", { className: "bar" + (shift === best ? " best" : ""), title: `${signed(shift)}: ${score.toFixed(3)}` });
    const fill = el("span");
    fill.style.height = `${Math.max(2, ratio * 100)}%`;
    bar.append(fill, el("small", { textContent: signed(shift) }));
    chart.append(bar);
  }
}

function render(data) {
  const shift = $("shift");
  shift.textContent = data.semitones === 0 ? "±0" : signed(data.semitones);
  shift.className = "shift " + (data.semitones > 0 ? "up" : data.semitones < 0 ? "down" : "zero");
  $("summary").textContent = data.summary;

  const details = [];
  if (data.precise != null && Math.abs(data.precise - data.semitones) >= 0.15) {
    details.push(`細かく見ると ${signed(data.precise.toFixed(2))} 半音（再生速度の変更などの可能性）`);
  }
  details.push(`信頼度 ${Math.round(data.confidence * 100)}%`);
  details.push(`原曲 ${data.original.key} → この動画 ${data.cover.key}`);
  $("detail").textContent = details.join(" ・ ");

  $("warning").hidden = !data.warning;
  $("warning").textContent = data.warning || "";

  renderCard("cover-card", "判定した動画", data.cover);
  renderCard("original-card", "原曲", data.method === "compare" ? data.original : null);

  $("chart-wrap").hidden = !data.scores;
  if (data.scores) renderChart(data.scores, data.semitones);

  $("result").hidden = false;
}

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = { cover_url: $("cover").value.trim() };
  if (currentMode() === "url") {
    body.original_url = $("original").value.trim();
    if (!body.original_url) return setStatus("原曲のURLを入力してください", { error: true });
  } else {
    body.original_key = $("key").value.trim();
    if (!body.original_key) return setStatus("原曲キーを入力してください", { error: true });
  }

  $("submit").disabled = true;
  $("result").hidden = true;
  setStatus("音声をダウンロードして解析しています（1分ほどかかることがあります）…", { loading: true });
  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `エラーが発生しました (${res.status})`);
    setStatus("");
    render(data);
  } catch (err) {
    setStatus(err.message, { error: true });
  } finally {
    $("submit").disabled = false;
  }
});
