/* Consola de operador — logica de la UI (vanilla JS, sin build ni dependencias externas: todo
   offline). Consume la API local (/api) que re-sirve el pipeline agentico validado. Las REGLAS
   deciden los numeros; el LLM solo narra. La verdad del dataset se pide SOLO tras decidir. */
(function () {
  "use strict";
  const API = "/api";
  const MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"];
  const REDUCE = matchMedia("(prefers-reduced-motion:reduce)").matches;

  const S = {
    farm: "C", seed: 42, cases: [], sel: null,
    playback: null, diag: null, plan: null, truth: null,
    decision: null, animId: null, affCell: null, chatBusy: false,
  };

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
  const fmtNum = (n) => (n == null ? "-" : Math.round(n).toLocaleString("es-ES"));
  const fmtClock = (iso) => (iso ? iso.slice(0, 16).replace("T", " ") : "--");
  // Los timestamps del dataset son naive (reloj del evento): se parsean como UTC para NO aplicarles
  // el desfase horario local al mostrarlos (interpolamos y formateamos siempre en UTC).
  const parseTs = (iso) => new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z");

  async function getJSON(path) {
    const r = await fetch(API + path);
    if (!r.ok) throw new Error(path + " -> " + r.status);
    return r.json();
  }

  // ============================================================ arranque
  async function init() {
    try {
      const f = await getJSON("/farms");
      $("env-era5").textContent = f.era5 ? "OK" : "no";
      $("env-llm").textContent = f.ollama ? "OK" : "no";
      renderFarmButtons(f.farms);
    } catch (e) {
      $("cases").innerHTML = "";
      $("cases").appendChild(el("div", "notice", "No se pudo contactar con el backend: " + e.message));
      return;
    }
    await loadCases(S.farm);
    $("back").addEventListener("click", toSelect);
    $("replay").addEventListener("click", () => S.playback && startPlayback());
    setupChat();
  }

  function renderFarmButtons(farms) {
    const c = $("farm-buttons");
    c.innerHTML = "";
    farms.forEach((f) => {
      const b = el("button", "btn" + (f.id === S.farm ? " on" : ""), f.id + " (" + f.n_turbinas + ")");
      b.addEventListener("click", () => {
        S.farm = f.id;
        [...c.children].forEach((x) => x.classList.remove("on"));
        b.classList.add("on");
        loadCases(f.id);
      });
      c.appendChild(b);
    });
  }

  // ============================================================ seleccion de caso
  async function loadCases(farm) {
    const box = $("cases");
    box.innerHTML = "";
    box.appendChild(el("div", "loading", "Cargando eventos de Farm " + farm + " ..."));
    let cases;
    try {
      cases = (await getJSON("/farms/" + farm + "/cases")).cases;
    } catch (e) {
      box.innerHTML = "";
      box.appendChild(el("div", "notice", "Error cargando casos: " + e.message));
      return;
    }
    S.cases = cases;
    box.innerHTML = "";
    cases.forEach((c) => {
      const card = el("button", "case");
      const id = el("div", "id");
      id.appendChild(el("span", null, farm + "-" + c.asset_id + "  ev" + c.event_id));
      if (c.caso_estudio) id.appendChild(el("span", "study", "CASO ESTUDIO"));
      card.appendChild(id);
      const k1 = el("div", "kv"); k1.appendChild(el("span", null, "margen S/h"));
      const b1 = el("b", null, c.margen_sh); k1.appendChild(b1); card.appendChild(k1);
      const k2 = el("div", "kv"); k2.appendChild(el("span", null, "alarma"));
      k2.appendChild(el("b", null, c.mes ? c.mes + " 2023" : "sin alarma")); card.appendChild(k2);
      const sigCls = c.senal === "fuerte" ? "fuerte" : (c.senal === "debil" ? "debil" : "sin");
      const sig = el("span", "sig " + sigCls, "senal " + c.senal);
      card.appendChild(sig);
      card.addEventListener("click", () => selectCase(c));
      box.appendChild(card);
    });
  }

  function toSelect() {
    if (S.animId) cancelAnimationFrame(S.animId);
    $("screen-cockpit").classList.add("hidden");
    $("screen-select").classList.remove("hidden");
  }

  // ============================================================ cockpit
  async function selectCase(c) {
    S.sel = c; S.plan = null; S.diag = null; S.truth = null; S.decision = null;
    $("screen-select").classList.add("hidden");
    $("screen-cockpit").classList.remove("hidden");
    $("cap-case").textContent = "Farm " + S.farm + " / " + S.farm + "-" + c.asset_id + " / ev" + c.event_id;
    $("cap-turb").textContent = "Aerogenerador " + S.farm + "-" + c.asset_id;
    $("cap-farm").textContent = "Parque - Farm " + S.farm;
    $("diag").innerHTML = "<div class='loading'>Compilando expediente XAI<span class='blink'>_</span></div>";
    $("decision-area").innerHTML = "";
    $("reveal").classList.add("hidden");
    $("timeline-panel").classList.add("hidden");
    $("log").innerHTML = "";
    $("status").textContent = "Reproduciendo evento";
    setTurbine("nominal");
    buildMeter();

    let pb, diag;
    try {
      [pb, diag] = await Promise.all([
        getJSON("/cases/" + S.farm + "/" + c.event_id + "/playback"),
        getJSON("/cases/" + S.farm + "/" + c.event_id + "/diagnose?use_llm=" + llmOn()),
      ]);
    } catch (e) {
      $("diag").innerHTML = "";
      $("diag").appendChild(el("div", "notice", "Error: " + e.message));
      return;
    }
    S.playback = pb; S.diag = diag;
    $("cap-det").textContent = diag.detector;
    $("margin").textContent = pb.margen_sh + "x";
    $("channel").textContent = diag.canal_disparador || "-";
    buildFarmMap(pb);
    renderWeather(pb.meteo);
    renderDiagnosis(diag);
    greetChat(c);
    startPlayback();
    loadPlan();
  }

  const llmOn = () => ($("env-llm").textContent === "OK" ? "true" : "false");

  // ------------------------------------------------ barra de nivel + mapa
  function buildMeter() {
    const m = $("meter"); m.innerHTML = ""; const SEG = 14;
    for (let i = 0; i < SEG; i++) {
      const zone = i < 9 ? "g" : (i < 12 ? "a" : "r");
      m.appendChild(el("div", "seg " + zone));
    }
  }
  function setMeter(frac) {
    const segs = $("meter").querySelectorAll(".seg");
    const n = Math.round(Math.max(0, Math.min(1, frac)) * segs.length);
    segs.forEach((e, i) => e.classList.toggle("on", i < n));
  }
  function buildFarmMap(pb) {
    // Esquematico: los asset_id no son 1..N (IDs de turbina no contiguos), asi que las celdas van
    // sin numerar y solo la turbina AFECTADA se etiqueta con su asset_id.
    const f = $("farm"); f.innerHTML = "";
    const n = pb.n_turbinas || 22;
    const affIdx = Math.max(0, Math.min(n - 1, pb.asset_id - 1));
    S.affCell = null;
    for (let i = 0; i < n; i++) {
      const cell = el("div", "t ok", i === affIdx ? String(pb.asset_id) : "");
      if (i === affIdx) S.affCell = cell;
      f.appendChild(cell);
    }
  }

  // ------------------------------------------------ animacion CUSUM
  function startPlayback() {
    if (S.animId) cancelAnimationFrame(S.animId);
    const pb = S.playback;
    const canvas = $("cusum"), ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const s = pb.cusum, N = s.length, yMax = pb.y_max || Math.max(...s, pb.h) * 1.1;
    const h = pb.h, cross = pb.alarma_frac;   // null si no hay alarma
    const confirmada = !!pb.confirmada;
    const t0 = parseTs(pb.t_inicio), t1 = parseTs(pb.t_decision || pb.t_inicio);
    const vy = (v) => H - 6 - (v / yMax) * (H - 12);

    // La reproduccion PARA en el instante de decision: cruza umbral (ambar) y se MANTIENE ->
    // confirma la alarma (rojo) y se congela. No se dibuja lo posterior (el eje ya viene truncado).
    function draw(p, frozen) {
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = "rgba(120,150,130,.10)"; ctx.lineWidth = 1;
      for (let g = 1; g < 4; g++) { const y = (H / 4) * g; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
      const ty = vy(h);
      ctx.strokeStyle = "rgba(217,130,0,.8)"; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(W, ty); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "rgba(217,130,0,.7)"; ctx.font = "9px monospace"; ctx.fillText("umbral h", W - 58, ty - 4);
      const last = Math.max(1, Math.floor(p * (N - 1)));
      ctx.beginPath(); ctx.moveTo(0, H);
      for (let a = 0; a <= last; a++) ctx.lineTo((a / (N - 1)) * W, vy(s[a]));
      ctx.lineTo((last / (N - 1)) * W, H); ctx.closePath();
      const crossed = cross != null && p >= cross;
      const g1 = frozen ? "rgba(255,91,77,.30)" : (crossed ? "rgba(255,182,56,.28)" : "rgba(55,245,140,.30)");
      const gr = ctx.createLinearGradient(0, 0, 0, H);
      gr.addColorStop(0, g1); gr.addColorStop(1, "rgba(55,245,140,.02)"); ctx.fillStyle = gr; ctx.fill();
      ctx.beginPath();
      for (let j = 0; j <= last; j++) { const px = (j / (N - 1)) * W, py = vy(s[j]); j ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.strokeStyle = frozen ? "#ff5b4d" : (crossed ? "#ffb638" : "#37f58c");
      ctx.lineWidth = frozen ? 2.2 : 1.8; ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur = frozen ? 12 : 7;
      ctx.stroke(); ctx.shadowBlur = 0;
      // marca ambar del cruce del umbral (aviso)
      if (crossed && !frozen) {
        const cx = cross * W;
        ctx.strokeStyle = "rgba(255,182,56,.5)"; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke(); ctx.setLineDash([]);
      }
      // punto de DECISION (rojo, congelado) al final del tramo visible
      if (frozen) {
        const cx = last / (N - 1) * W, cy = vy(s[last]);
        ctx.strokeStyle = "rgba(255,91,77,.7)"; ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
        ctx.fillStyle = "#ff5b4d"; ctx.shadowColor = "#ff5b4d"; ctx.shadowBlur = 14;
        ctx.beginPath(); ctx.arc(cx, cy, 4.5, 0, 7); ctx.fill(); ctx.shadowBlur = 0;
      } else {
        const curx = p * W; ctx.strokeStyle = "rgba(142,255,192,.85)";
        ctx.beginPath(); ctx.moveTo(curx, 0); ctx.lineTo(curx, H); ctx.stroke();
      }
    }

    function phaseOf(value, frozen) {
      if (confirmada && frozen) return "fired";      // sostenida -> rojo y parada
      return value > h ? "rising" : "idle";          // ambar mientras cruza/sostiene; verde si no
    }
    function apply(p, frozen) {
      const cur = s[Math.max(0, Math.floor(p * (N - 1)))];
      setMeter(cur / yMax);
      $("prog").style.width = (p * 100).toFixed(0) + "%";
      $("pct").textContent = Math.round(p * 100) + "%";
      const dt = new Date(t0.getTime() + (t1 - t0) * p);
      $("clock").textContent = fmtClock(dt.toISOString());
      $("status-clock").textContent = "UTC " + fmtClock(dt.toISOString());
      const ph = phaseOf(cur, frozen);
      $("alarm").className = "alarm " + (ph === "idle" ? "" : ph);
      const A = $("astate"), sub = $("asub"), ts = $("tstate");
      if (ph === "idle") { A.textContent = "NORMAL"; sub.textContent = "vigilancia - sin precursor"; ts.textContent = "REPRODUCIENDO"; }
      else if (ph === "rising") { A.textContent = "AVISO"; sub.textContent = "CUSUM sobre umbral - sostenimiento"; ts.textContent = "CONFIRMANDO ALARMA"; }
      else { A.textContent = "ALARMA"; sub.textContent = "CUSUM sostenido > umbral - alarma confirmada"; ts.textContent = "ALARMA - DECISION"; }
      if (S.affCell && !S.decision) {
        S.affCell.className = ph === "fired" ? "t aff fired" : (ph === "rising" ? "t aff" : "t ok");
      }
      // sin decision aun: la turbina se auto-deratea de forma protectora al confirmarse la alarma
      if (!S.decision) setTurbine(ph === "fired" ? "derated" : "nominal");
    }

    if (REDUCE) { draw(1, confirmada); apply(1, confirmada); afterPlayback(); return; }
    const DUR = 8000; let start = null;
    function step(tms) {
      if (!start) start = tms;
      let p = Math.min(1, (tms - start) / DUR);
      const e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
      const frozen = p >= 1 && confirmada;
      draw(e, frozen); apply(e, frozen);
      if (p < 1) S.animId = requestAnimationFrame(step);
      else { S.animId = null; afterPlayback(); }
    }
    S.animId = requestAnimationFrame(step);
  }

  function afterPlayback() {
    $("status").textContent = S.playback && S.playback.confirmada
      ? "Alarma confirmada - decision del operador" : "Sin alarma sostenida - vigilancia";
  }

  // ------------------------------------------------ turbina
  function setTurbine(mode) {
    const t = $("turbine");
    t.className = "turbine" + (mode === "nominal" ? "" : " " + mode);
    const label = { nominal: "nominal", derated: "derateada -20%", stopped: "parada (0%)", broken: "hacia el fallo" }[mode];
    const power = { nominal: "100% nom", derated: "~80% nom", stopped: "0% (parada)", broken: "100% -> fallo" }[mode];
    const tm = { nominal: "NOMINAL", derated: "DERATE", stopped: "PARADA", broken: "RUN-TO-FAIL" }[mode];
    $("tmode2").textContent = label; $("power").textContent = power; $("tmode").textContent = tm;
  }

  // ------------------------------------------------ meteo
  function renderWeather(wx) {
    const box = $("wx");
    if (!wx) { box.innerHTML = "<div class='na'>sin datos meteo para este evento</div>"; $("wx-mes").textContent = "-"; return; }
    $("wx-mes").textContent = wx.mes + " (clim. 2014-2023)";
    box.innerHTML = "";
    box.appendChild(wxLine("VIENTO 10m", wx.viento + " m/s"));
    box.appendChild(wxLine("OLEAJE Hs", wx.hs + " m"));
    const acc = el("div", "acc" + (wx.ventana_ctv ? "" : " no"));
    acc.appendChild(el("span", "k", "VENTANA CTV (Hs<" + wx.hs_max + " & v<" + wx.wind_max + ")"));
    acc.appendChild(el("span", "v", (wx.ventana_ctv ? "SI" : "NO") + " - " + wx.pct_accesible + "% dias"));
    box.appendChild(acc);
  }
  function wxLine(k, v) {
    const l = el("div", "line"); l.appendChild(el("span", "k", k)); l.appendChild(el("span", "v tnum", v)); return l;
  }

  // ------------------------------------------------ diagnostico
  function twoLayer(txt) {
    let t = (txt || "").replace(/</g, "&lt;");
    ["Recomendacion:", "Que pasa:", "A que atenerse:", "Por que:"].forEach((e) => { t = t.split(e).join("<b>" + e + "</b>"); });
    return t.replace(/\n/g, "<br>");
  }
  function renderDiagnosis(d) {
    const box = $("diag"); box.innerHTML = "";
    const chips = el("div", "chips");
    const tri = { escalar: "esc", vigilar: "vig", descartar: "desc" }[d.triage] || "k";
    chips.appendChild(el("span", "chip " + tri, "triaje: " + d.triage));
    chips.appendChild(el("span", "chip k", "precursor: " + d.tipo_precursor));
    chips.appendChild(el("span", "chip k", "margen " + d.margen_sh + "x"));
    if (d.urgencia_dias != null) chips.appendChild(el("span", "chip k", "urgencia ~" + d.urgencia_dias + "d"));
    if (d.ig_propio_pct != null) chips.appendChild(el("span", "chip k", "IG propio " + d.ig_propio_pct + "%"));
    if (d.persistencia_frac != null) chips.appendChild(el("span", "chip k", "persist " + d.persistencia_frac));
    box.appendChild(chips);
    const p = el("div", "two"); p.innerHTML = twoLayer(d.hipotesis); box.appendChild(p);
    const src = el("div", "dhint", "Lectura " + (d.usa_llm ? "(qwen3)" : "(plantilla)") +
      " - canal indicador anonimizado; subsistema y costes son representativos y relativos.");
    box.appendChild(src);
  }

  // ------------------------------------------------ plan (asincrono)
  async function loadPlan() {
    const area = $("decision-area");
    area.innerHTML = "<div class='loading'>Optimizando plan (NSGA-II, 3 semillas + greedy + validacion exacta)<span class='blink'>_</span><br><small>puede tardar en el primer calculo de un caso</small></div>";
    const c = S.sel;
    const mes = c.mes_num ? "&mes=" + c.mes_num : "";
    let plan;
    try {
      plan = await getJSON("/cases/" + S.farm + "/" + c.event_id + "/plan?use_llm=" + llmOn() + mes);
    } catch (e) {
      area.innerHTML = ""; area.appendChild(el("div", "notice", "Error optimizando: " + e.message)); return;
    }
    S.plan = plan;
    if (!plan.escalado) {
      area.innerHTML = "";
      area.appendChild(el("div", "notice",
        "El Agente 1 marco este evento como " + (plan.triage || "").toUpperCase() +
        " (no 'escalar'): no se genera plan de intervencion. Solo se planifican las alarmas escaladas."));
      return;
    }
    renderDecision(plan);
    renderTimeline(plan.timeline);
  }

  function riskCls(cual) {
    if (["nulo", "muy bajo", "bajo"].includes(cual)) return "okv";
    if (["moderado", "alto"].includes(cual)) return "warnv";
    return "bad";
  }
  function renderDecision(plan) {
    const area = $("decision-area"); area.innerHTML = "";
    if (plan.aviso) {
      const av = el("div", "notice", plan.aviso); av.style.marginBottom = "8px"; area.appendChild(av);
    }
    const wrap = el("div"); wrap.style.overflowX = "auto";
    const table = el("table", "dtbl tnum");
    const opts = plan.opciones;
    // cabecera
    const thead = el("thead"); const htr = el("tr"); htr.appendChild(el("th"));
    opts.forEach((o) => {
      const th = el("th", o.recomendado ? "reco" : ""); th.dataset.opt = o.id;
      th.appendChild(el("span", null, o.titulo));
      if (o.recomendado) th.appendChild(el("span", "reco-tag", "RECOMENDADO"));
      th.addEventListener("click", () => decide(o.id));
      htr.appendChild(th);
    });
    thead.appendChild(htr); table.appendChild(thead);
    // filas
    const tb = el("tbody");
    tb.appendChild(rowCoste(opts));
    tb.appendChild(rowRiesgo(opts));
    tb.appendChild(rowImpacto(opts));
    tb.appendChild(rowCuando(opts));
    table.appendChild(tb); wrap.appendChild(table); area.appendChild(wrap);

    area.appendChild(el("div", "dhint",
      "Pulsa una columna para DECIDIR. 'Optimo' equilibra coste/riesgo/impacto (deratea esperando la " +
      "reparacion planificada). Subsistema representativo: " + plan.subsistema +
      (plan.requiere_jackup ? " (jack-up)" : " (CTV)") + "."));
    if (plan.narrativa) {
      const nar = el("div", "two"); nar.style.marginTop = "6px";
      nar.style.fontSize = ".7rem"; nar.innerHTML = twoLayer(plan.narrativa); area.appendChild(nar);
    }
    highlightDecision();
  }
  const optCell = (o, cls, txt) => { const td = el("td", cls); td.dataset.opt = o.id; td.textContent = txt; return td; };
  function rowCoste(opts) {
    const tr = el("tr"); tr.appendChild(el("th", null, "Coste"));
    opts.forEach((o) => {
      const cls = "hcost" + (o.id === "romper" ? " bad" : "") + (o.recomendado ? " reco" : "");
      tr.appendChild(optCell(o, cls, o.coste == null ? "-" : fmtNum(o.coste)));
    });
    return tr;
  }
  function rowRiesgo(opts) {
    const tr = el("tr"); tr.appendChild(el("th", null, "Riesgo"));
    opts.forEach((o) => {
      let cls = (o.recomendado ? "reco " : "");
      let txt;
      if (o.id === "romper") { cls += "bad"; txt = "seguro (1.00)"; }
      else { cls += riskCls(o.riesgo_cual); txt = o.riesgo.toFixed(2) + " " + o.riesgo_cual; }
      tr.appendChild(optCell(o, cls, txt));
    });
    return tr;
  }
  function rowImpacto(opts) {
    const tr = el("tr"); tr.appendChild(el("th", null, "Impacto"));
    opts.forEach((o) => {
      let cls = (o.recomendado ? "reco " : "");
      let txt;
      if (o.impacto == null) { cls += "warnv"; txt = o.impacto_cual; }
      else if (!o.alcanzable) { cls = "na"; txt = "no alcanz. (" + o.impacto.toFixed(1) + ")"; }
      else { txt = o.impacto.toFixed(1) + " " + o.impacto_cual; }
      tr.appendChild(optCell(o, cls, txt));
    });
    return tr;
  }
  function rowCuando(opts) {
    const tr = el("tr"); tr.appendChild(el("th", null, "Cuando"));
    opts.forEach((o) => tr.appendChild(optCell(o, o.recomendado ? "reco" : "", o.cuando || "-")));
    return tr;
  }

  function decide(optId) {
    if (!S.plan || !S.plan.escalado) return;
    S.decision = optId;
    highlightDecision();
    const mode = optId === "parar" ? "stopped" : (optId === "romper" ? "broken" : "derated");
    setTurbine(mode);
    if (S.affCell) S.affCell.className = optId === "parar" ? "t ok" : "t aff fired";
    const o = S.plan.opciones.find((x) => x.id === optId);
    $("status").textContent = "Decision: " + (o ? o.titulo : optId);
    fetchTruth();
  }
  function highlightDecision() {
    document.querySelectorAll(".dtbl th,.dtbl td").forEach((c) => c.classList.remove("chosen"));
    if (!S.decision) return;
    document.querySelectorAll('.dtbl [data-opt="' + S.decision + '"]').forEach((c) => c.classList.add("chosen"));
  }

  // ------------------------------------------------ revelacion (tras decidir)
  async function fetchTruth() {
    if (!S.truth) {
      try { S.truth = await getJSON("/cases/" + S.farm + "/" + S.sel.event_id + "/truth"); }
      catch (e) { return; }
    }
    renderReveal(S.truth);
  }
  function renderReveal(t) {
    const box = $("reveal");
    const miss = (t.veredicto === "falsa_alarma" || t.veredicto === "no_detectado");
    const tag = { acierto: "[ ANOMALIA REAL ]", falsa_alarma: "[ FALSA ALARMA ]",
      correcto_negativo: "[ NORMAL ]", no_detectado: "[ NO DETECTADO ]" }[t.veredicto] || "[ ? ]";
    const nota = { acierto: "el diagnostico ACERTO", falsa_alarma: "el diagnostico se equivoco (normal)",
      correcto_negativo: "correcto: no habia nada", no_detectado: "anomalia sin alarma" }[t.veredicto] || "";
    box.className = "reveal" + (miss ? " miss" : "");
    box.innerHTML = "";
    box.appendChild(el("span", "k", "Verdad del dataset"));
    box.appendChild(el("span", "tag", tag));
    box.appendChild(el("span", null, t.glosa + " - " + nota));
    box.appendChild(el("span", "note", "se revela tras decidir - el agente nunca la vio"));
    box.classList.remove("hidden");
  }

  // ------------------------------------------------ linea de tiempo
  function fechaCorta(t0iso, dias) {
    const d = new Date(parseTs(t0iso).getTime() + dias * 86400000);
    return d.getUTCDate() + " " + MESES[d.getUTCMonth()];   // UTC: sin desfase horario
  }
  function renderTimeline(tl) {
    if (!tl) return;
    $("timeline-panel").classList.remove("hidden");
    $("tl-sub").textContent = tl.veto && tl.veto.length ? ("veto: " + tl.veto.join("; ")) : "sin veto ambiental";
    const cv = $("timeline"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
    const sens = tl.sensibilidad, N = sens.length, smax = Math.max(...sens, 0.01);
    const AX = 12, TOP = 48;                 // eje de meses abajo; franja superior para etiquetas
    const base = H - AX;
    const x = (d) => (d / (N - 1)) * W, y = (v) => base - (v / smax) * (base - TOP);
    ctx.clearRect(0, 0, W, H); ctx.font = "9px monospace";
    // area de sensibilidad de fauna (banda inferior, deja libre la franja de etiquetas)
    ctx.beginPath(); ctx.moveTo(0, base);
    sens.forEach((v, d) => ctx.lineTo(x(d), y(v))); ctx.lineTo(W, base); ctx.closePath();
    ctx.fillStyle = "rgba(230,149,0,.4)"; ctx.fill();
    // derating (t0 -> intervencion) + intervencion, solo en la banda inferior
    const di = tl.dia_intervencion, dur = tl.dur_intervencion;
    ctx.fillStyle = "rgba(241,196,15,.22)"; ctx.fillRect(0, TOP, x(di), base - TOP);
    ctx.fillStyle = "rgba(44,127,184,.5)"; ctx.fillRect(x(di), TOP, Math.max(2, x(di + dur) - x(di)), base - TOP);
    // zona de VETO de jack-up (temporada sensible de fauna): explica por que se pospone la intervencion
    (tl.veto_dias || []).forEach(([a, c]) => {
      ctx.fillStyle = "rgba(194,35,35,.14)"; ctx.fillRect(x(a), TOP, x(c) - x(a), base - TOP);
      ctx.strokeStyle = "rgba(194,35,35,.45)"; ctx.setLineDash([3, 3]);
      ctx.strokeRect(x(a), TOP, x(c) - x(a), base - TOP); ctx.setLineDash([]);
      ctx.fillStyle = "rgba(140,20,20,.75)";
      ctx.fillText("veto jack-up", x(a) + 4, base - 4);
    });
    // marcadores con FECHAS REALES, en filas escalonadas (nunca se solapan)
    mark(ctx, x(0), base, "#0f6d4a", "alarma " + fechaCorta(tl.t0, 0), 9);
    if (tl.dia_rotura < N)
      mark(ctx, x(tl.dia_rotura), base, "#b81f1f", "fallo probable " + fechaCorta(tl.t0, tl.dia_rotura), 20, true);
    if (tl.dia_rotura_derating < N)
      mark(ctx, x(tl.dia_rotura_derating), base, "#8a5300", "con derating " + fechaCorta(tl.t0, tl.dia_rotura_derating), 31, true);
    mark(ctx, x(di), base, "#1c5f8f", "intervencion " + fechaCorta(tl.t0, di), 42);
    // eje de meses (UTC, sin desfase): UNA etiqueta por mes de CALENDARIO en su dia 1 -> nunca se
    // repite un mes (saltar de 30 en 30 dias caia dos veces en el mismo mes al desalinearse del calendario)
    ctx.fillStyle = "rgba(20,20,20,.6)"; ctx.textAlign = "left";
    const t0 = parseTs(tl.t0);
    let cur = new Date(Date.UTC(t0.getUTCFullYear(), t0.getUTCMonth(), 1));
    if (cur.getTime() < t0.getTime()) cur = new Date(Date.UTC(t0.getUTCFullYear(), t0.getUTCMonth() + 1, 1));
    while (true) {
      const d = Math.round((cur.getTime() - t0.getTime()) / 86400000);
      if (d >= N) break;
      if (d >= 0) ctx.fillText(MESES[cur.getUTCMonth()], x(d) + 2, H - 3);
      cur = new Date(Date.UTC(cur.getUTCFullYear(), cur.getUTCMonth() + 1, 1));
    }
  }
  function mark(ctx, px, ybase, color, label, ly, dash) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; if (dash) ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(px, ly + 2); ctx.lineTo(px, ybase); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = color; ctx.textAlign = "left";
    const w = ctx.measureText(label).width;
    ctx.fillText(label, Math.max(2, Math.min(px + 3, ctx.canvas.width - 4 - w)), ly);
  }

  // ============================================================ terminal LLM (streaming)
  function setupChat() {
    const inp = $("chat-input");
    inp.addEventListener("keydown", (ev) => { if (ev.key === "Enter") sendChat(); });
  }
  function greetChat(c) {
    const inp = $("chat-input");
    if ($("env-llm").textContent !== "OK") {
      inp.disabled = true; inp.placeholder = "asistente no disponible (Ollama apagado)";
      addLine("a", "Asistente offline: arranca 'ollama serve' con el modelo qwen3 para el terminal.");
    } else {
      inp.disabled = false; inp.placeholder = "consultar sobre la alarma..."; inp.focus();
    }
  }
  function addLine(cls, txt) {
    const d = el("div", cls, txt || ""); $("log").appendChild(d);
    $("log").scrollTop = $("log").scrollHeight; return d;
  }
  async function sendChat() {
    const inp = $("chat-input");
    const q = inp.value.trim();
    if (!q || S.chatBusy || !S.sel) return;
    S.chatBusy = true; inp.value = ""; inp.disabled = true;
    addLine("q", q);
    const ans = addLine("a", "");
    const c = S.sel;
    try {
      const r = await fetch(API + "/chat", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ farm: S.farm, event: c.event_id, seed: S.seed, question: q,
          mes: c.mes_num || null }),
      });
      if (r.status === 503) { ans.textContent = "asistente no disponible (Ollama)."; }
      else if (!r.ok) { ans.textContent = "error " + r.status; }
      else {
        if (r.headers.get("X-Scope") === "out") ans.className = "a oos";
        const reader = r.body.getReader(); const dec = new TextDecoder();
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          ans.textContent += dec.decode(value, { stream: true });
          $("log").scrollTop = $("log").scrollHeight;
        }
      }
    } catch (e) {
      ans.textContent = "error: " + e.message;
    } finally {
      S.chatBusy = false; inp.disabled = false; inp.focus();
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
