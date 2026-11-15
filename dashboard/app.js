const $ = (s) => document.querySelector(s);
const fmt$ = (v) => "$" + Math.round(v).toLocaleString();
const pct = (v, d = 1) => (100 * v).toFixed(d) + "%";
const api = (p, body) => fetch(p, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => r.json());
const C = { allow: "#2fbf71", hold: "#f2b134", decline: "#ef5b5b", fraud: "#c86bfa", grid: "#243049", text: "#8a96ad", accent: "#5b8cff" };
Chart.defaults.color = C.text; Chart.defaults.borderColor = C.grid; Chart.defaults.font.family = "-apple-system, Inter, sans-serif";

// ---------------------------------------------------------------- tabs
document.querySelectorAll(".tab").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tab, .view").forEach((e) => e.classList.remove("active"));
  b.classList.add("active"); $("#" + b.dataset.view).classList.add("active");
  if (b.dataset.view === "perf") loadReports();
  if (b.dataset.view === "whatif") whatif();
});

// ---------------------------------------------------------------- live
let running = false, timer = null, selected = null;
const hourly = new Chart($("#hourly"), {
  type: "bar", data: { labels: [], datasets: [
    { label: "allowed", data: [], backgroundColor: C.allow, stack: "a" },
    { label: "held → verify", data: [], backgroundColor: C.hold, stack: "a" },
    { label: "declined", data: [], backgroundColor: C.decline, stack: "a" },
    { label: "actual fraud", data: [], type: "line", borderColor: C.fraud, backgroundColor: C.fraud, yAxisID: "y2", pointRadius: 0, tension: .3 },
  ] },
  options: { animation: false, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } } },
    scales: { x: { stacked: true, ticks: { maxTicksLimit: 8 } }, y: { stacked: true, title: { display: true, text: "orders" } },
      y2: { position: "right", grid: { display: false }, title: { display: true, text: "fraud" }, beginAtZero: true } } },
});
const risk = new Chart($("#risk"), {
  type: "bar", data: { labels: Array.from({ length: 20 }, (_, i) => (i * 5) + "%"), datasets: [{ data: [], backgroundColor: Array.from({ length: 20 }, (_, i) => i < 2 ? C.allow : i < 10 ? C.hold : C.decline) }] },
  options: { animation: false, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { type: "logarithmic", title: { display: true, text: "orders" } } } },
});

function kpi(label, value, sub, cls = "") { return `<div class="kpi"><div class="l">${label}</div><div class="v ${cls}">${value}</div>${sub ? `<div class="s">${sub}</div>` : ""}</div>`; }

function render(s) {
  const k = s.kpis;
  $("#bar").style.width = (100 * s.cursor / s.total) + "%";
  $("#simday").textContent = `sim day ${s.sim_day.toFixed(1)} · ${s.cursor.toLocaleString()}/${s.total.toLocaleString()} test orders`;
  $("#kpis").innerHTML = [
    kpi("Orders processed", k.transactions.toLocaleString(), fmt$(k.volume_usd) + " volume"),
    kpi("Auto-resolved", k.auto_resolved_pct.toFixed(2) + "%", `${k.merchant_escalations} needed merchant`, "good"),
    kpi("Fraud $ stopped", k.fraud_usd_total ? pct(k.fraud_usd_stopped / k.fraud_usd_total) : "–", `${fmt$(k.fraud_usd_stopped)} of ${fmt$(k.fraud_usd_total)}`, "good"),
    kpi("Fraud orders caught", `${k.fraud_caught}/${k.fraud_total}`, "count-based"),
    kpi("False declines", k.false_decline_rate_pct.toFixed(2) + "%", "of genuine customers", k.false_decline_rate_pct > 0.5 ? "bad" : "good"),
    kpi("Asked to verify", k.legit_friction_pct.toFixed(2) + "%", `genuine customers · ${k.trusted_skips} trust-memory skips · ${k.novelty_holds} novelty holds`, "warn"),
    kpi("Net savings", fmt$(k.net_savings_usd), "vs. approving everything", k.net_savings_usd >= 0 ? "good" : "bad"),
  ].join("");
  hourly.data.labels = s.hourly.map((h) => "d" + Math.floor(h.hour / 24) + " " + String(h.hour % 24).padStart(2, "0") + "h");
  hourly.data.datasets[0].data = s.hourly.map((h) => h.allow);
  hourly.data.datasets[1].data = s.hourly.map((h) => h.hold);
  hourly.data.datasets[2].data = s.hourly.map((h) => h.decline);
  hourly.data.datasets[3].data = s.hourly.map((h) => h.fraud);
  hourly.update();
  risk.data.datasets[0].data = s.risk_hist.map((v) => v || null); risk.update();
  $("#feed tbody").innerHTML = s.recent.map((r) => `<tr data-id="${r.tx_id}" class="${r.tx_id === selected ? "sel" : ""}">
    <td>d${Math.floor(r.ts / 86400)} ${new Date(r.ts * 1000).toISOString().slice(11, 16)}</td>
    <td>$${r.amount.toFixed(2)}</td><td>${r.category}</td><td>${r.ip_country}</td><td>${pct(r.p, 1)}</td>
    <td><span class="pill ${r.decision}">${r.decision}</span>${r.trusted_skip ? ' <span class="muted" title="known device passed verification before">trusted</span>' : ""}${r.novelty_gate ? ' <span class="muted" title="novelty safety net">novel</span>' : ""}</td>
    <td class="muted">${r.verification || ""}</td>
    <td>${r.truth_fraud ? `<span class="pill fraud">${r.truth_type}</span>` : '<span class="pill ok">genuine</span>'}</td></tr>`).join("");
}
$("#feed").onclick = (e) => { const tr = e.target.closest("tr[data-id]"); if (tr) showDetail(tr.dataset.id); };

async function showDetail(id) {
  selected = id;
  const d = await api(`/api/tx/${id}`);
  const r = d.record, f = d.facts;
  const maxImp = Math.max(0.01, ...d.reasons.map((x) => x.impact));
  const ec = ["allow", "hold", "decline"].map((a) => `<div class="${a === r.decision ? "chosen" : ""}">${a}<b>${fmt$(r.expected_costs[a] * 100 / 100)}</b><span class="muted">${r.expected_costs[a].toFixed(2)} exp. cost</span></div>`).join("");
  $("#detail").innerHTML = `<h3>${r.tx_id} · $${r.amount.toFixed(2)} ${r.category} <span class="pill ${r.decision}">${r.decision}</span></h3>
    <div>Calibrated fraud probability: <b>${pct(r.p, 2)}</b> ${r.verification ? `· verification <b>${r.verification}</b>` : ""}</div>
    <div class="ec">${ec}</div>
    <div class="narr" id="narr">${d.narrative}</div>
    <h3 style="margin-top:12px">Top risk drivers <span class="muted">(drop in P(fraud) if this signal looked normal)</span></h3>
    ${d.reasons.length ? d.reasons.map((x) => `<div class="reason">${x.description} <span class="muted">−${pct(x.impact, 1)}</span><div class="bar" style="width:${100 * x.impact / maxImp}%"></div></div>`).join("") : '<p class="muted">No signal raises risk above normal.</p>'}
    <h3 style="margin-top:12px">Facts</h3>
    <div class="facts">${Object.entries(f).map(([k, v]) => `<div><span>${k.replaceAll("_", " ")}</span> ${v === null ? "–" : v}</div>`).join("")}</div>
    <p class="foot">Ground truth (simulation): ${r.truth_fraud ? r.truth_type : "genuine"}</p>`;
  document.querySelectorAll("#feed tr").forEach((t) => t.classList.toggle("sel", t.dataset.id === id));
}

async function tickOnce(n) { const s = await api("/api/tick", { n }); render(s); if (s.cursor >= s.total) stop(); }
function stop() { running = false; clearInterval(timer); $("#play").textContent = "▶ Run"; }
$("#play").onclick = () => {
  if (running) return stop();
  running = true; $("#play").textContent = "⏸ Pause";
  timer = setInterval(() => tickOnce(+$("#speed").value), 700);
};
$("#step").onclick = () => tickOnce(100);
$("#reset").onclick = async () => { stop(); $("#kpis").innerHTML = '<div class="kpi">Rebuilding behavioural state…</div>'; render(await api("/api/reset", {})); };
$("#brief").onclick = async () => { $("#briefing").textContent = "Writing…"; const b = await api("/api/briefing", {}); $("#briefing").textContent = b.text; $("#briefing").classList.remove("muted"); };
api("/api/state").then(render);

// ---------------------------------------------------------------- reports
let reportsLoaded = false;
async function loadReports() {
  if (reportsLoaded) return; reportsLoaded = true;
  const R = await api("/api/reports");
  if (!R.results) { $("#det").innerHTML = "<tr><td>Run python -m sentinel.evaluate first.</td></tr>"; return; }
  const detAll = R.results.mean.detection, gateKey = Object.keys(detAll).find((k) => k.includes("novelty gate"));
  const gate = detAll[gateKey];
  const det = Object.fromEntries(Object.entries(detAll).filter(([k]) => k !== gateKey)), names = Object.keys(det);
  $("#det").innerHTML = `<tr><th>model</th><th>AUPRC</th><th>mean interval endpoints*</th><th>recall @0.5% FPR</th><th>$ recall @0.5% FPR</th><th>AUROC</th></tr>` +
    names.map((n) => { const m = det[n]; return `<tr class="${n.startsWith("Sentinel") ? "best" : ""}"><td>${n}</td><td>${m.auprc.toFixed(3)}</td><td class="muted">${m.auprc_ci95.map((v) => v.toFixed(2)).join("–")}</td><td>${pct(m["recall@0.5%fpr"])}</td><td>${pct(m["dollar_recall@0.5%fpr"])}</td><td>${m.auroc.toFixed(3)}</td></tr>`; }).join("") +
    (gate ? `<tr class="best"><td>Sentinel + novelty gate (flag = verify)</td><td colspan=2 class="muted">${pct(gate.genuine_flagged, 2)} genuine flagged</td><td>${pct(gate.recall)}</td><td>${pct(gate.dollar_recall)}</td><td></td></tr>` : "") +
    `<tr><td colspan=6 class="foot">*per-world bootstrap interval endpoints averaged; not a confidence interval for the mean. Calibration on test: Brier ${R.results.mean.calibration.brier.toFixed(4)}, ECE ${R.results.mean.calibration.ece.toFixed(4)}</td></tr>`;
  const types = Object.keys(det[names[0]]["recall_by_type@0.5%fpr"]);
  new Chart($("#types"), { type: "bar", data: { labels: types.map((t) => t === "sleeper_ring" ? "sleeper ring (unseen)" : t),
    datasets: names.map((n, i) => ({ label: n, data: types.map((t) => det[n]["recall_by_type@0.5%fpr"][t]), backgroundColor: ["#3a4660", "#56657f", "#7d8aa3", C.accent][i] }))
      .concat(gate ? [{ label: "Sentinel + novelty gate", data: types.map((t) => gate.recall_by_type[t]), backgroundColor: "#9fb8ff" }] : []) },
    options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } } }, scales: { y: { max: 1, ticks: { callback: (v) => v * 100 + "%" } } } } });
  const pol = R.results.mean.policy;
  $("#pol").innerHTML = `<tr><th>policy</th><th>total cost</th><th>fraud $ stopped</th><th>false declines</th><th>genuine asked to verify</th><th>genuine customers lost</th><th>merchant reviews</th></tr>` +
    Object.entries(pol).map(([n, s]) => `<tr class="${n.startsWith("Sentinel full") ? "best" : ""}"><td>${n}</td><td>${fmt$(s.total_cost)}</td><td>${pct(s.fraud_dollar_stop_rate)}</td><td>${pct(s.false_decline_rate, 2)}</td><td>${pct(s.legit_friction_rate, 2)}</td><td>${pct(s.legit_lost_rate, 2)}</td><td>${Math.round(s.merchant_reviews)}</td></tr>`).join("");
  if (R.experiments) {
    const v = R.experiments.valid;
    $("#exp").innerHTML = `<tr><th>experiment</th><th>AUPRC</th><th>±sd</th><th>R@0.5%</th></tr>` + Object.entries(v).map(([n, m]) => `<tr><td>${n}</td><td>${m.auprc.toFixed(3)}</td><td class="muted">${m.auprc_sd.toFixed(3)}</td><td>${pct(m["recall@0.5%fpr"])}</td></tr>`).join("");
  }
  if (R.public_benchmark) {
    const p = R.public_benchmark.results;
    $("#pub").innerHTML = `<tr><th>model</th><th>valid AUPRC</th><th>TEST AUPRC</th><th>95% CI</th><th>recall @0.5% FPR</th></tr>` + Object.entries(p).map(([n, m]) => `<tr><td>${n}</td><td>${m.valid.auprc.toFixed(3)}</td><td>${m.test.auprc.toFixed(3)}</td><td class="muted">${m.test.auprc_ci95.map((x) => x.toFixed(2)).join("–")}</td><td>${pct(m.test["recall@0.5%fpr"])}</td></tr>`).join("");
  }
}

// ---------------------------------------------------------------- what-if
const SL = [["margin", "Gross margin", 0.05, 0.8, 0.01, 0.30, (v) => pct(v, 0)], ["chargeback_fee", "Chargeback fee ($)", 0, 100, 1, 25, (v) => "$" + v],
  ["churn_cost", "Cost of insulting a good customer ($)", 0, 100, 1, 20, (v) => "$" + v], ["friction", "Cost per verification ($)", 0, 5, 0.1, 0.4, (v) => "$" + v],
  ["catch_rate", "Fraudsters stopped by verification", 0.3, 1, 0.01, 0.85, (v) => pct(v, 0)], ["abandon", "Genuine buyers who abandon verification", 0, 0.4, 0.01, 0.07, (v) => pct(v, 0)]];
$("#sliders").innerHTML = "<h3>Business assumptions</h3>" + SL.map(([k, l, mn, mx, st, v, f]) => `<div class="slider"><label>${l}<b id="v_${k}">${f(v)}</b></label><input type="range" id="s_${k}" min="${mn}" max="${mx}" step="${st}" value="${v}"></div>`).join("");
let wiChart = null, wiT = null;
SL.forEach(([k, , , , , , f]) => $("#s_" + k).oninput = (e) => { $("#v_" + k).textContent = f(+e.target.value); clearTimeout(wiT); wiT = setTimeout(whatif, 150); });
async function whatif() {
  const body = Object.fromEntries(SL.map(([k]) => [k, +$("#s_" + k).value]));
  const s = await api("/api/whatif", body);
  $("#wiout").innerHTML = [kpi("Savings vs approve-all", fmt$(s.savings_vs_approve_all), "", "good"), kpi("Fraud $ stopped", pct(s.fraud_dollar_stop_rate), "", "good"),
    kpi("False declines", pct(s.false_decline_rate, 2), "genuine customers", "bad"), kpi("Asked to verify", pct(s.legit_friction_rate, 2), "genuine customers", "warn")].join("");
  const data = [s.allow_rate, s.hold_rate, s.decline_rate];
  if (!wiChart) wiChart = new Chart($("#wichart"), { type: "doughnut", data: { labels: ["allow", "hold → verify", "decline"], datasets: [{ data, backgroundColor: [C.allow, C.hold, C.decline], borderWidth: 0 }] }, options: { maintainAspectRatio: false, plugins: { legend: { position: "right" } } } });
  else { wiChart.data.datasets[0].data = data; wiChart.update(); }
}
