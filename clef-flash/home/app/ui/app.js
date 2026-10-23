"use strict";
// The clef-flash home page. The page owns the editor (the state text, the question cards, and the wish line, which
// writes the state's "request") and the picture of the house; it sends the editor's content to Python on Decide
// (pywebview.api.decide); Python builds the request (app/editor.py), runs it, and draws the outcome through HOME:
// running(), answers() or error(). Every number on screen arrives as a finished string (formatted in Python), and
// shown() reads the strings back for the run record. mark(name) returns the paint times of a step (epoch ms): raf1 =
// the animation frame that paints the change, raf2 = the frame after it. autoType / autoTap / autoPaste are the
// autoplay's hands: they type into the same fields and press the same buttons a user would.
window.HOME = (() => {
  const $ = (id) => document.getElementById(id);
  const marks = {};
  const now = () => performance.timeOrigin + performance.now();
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const TYPE_WORDS = { noul: "yes / no", choice: "choice", score: "score" };
  const OPTION_HINTS = {
    noul: "Optional: Yes: what yes means, and No: what no means",
    choice: "One option per line: name: description",
    score: "One level per line, from the lowest",
  };
  // app/house.py, written out again for the picture while you type: the devices and the words of their settings
  const DEVICES = ["light", "curtains", "tv", "aircon", "door"];
  const NAMES = { light: "Light", curtains: "Curtains", tv: "TV", aircon: "Air conditioner", door: "Front door" };
  const WORDS = {
    light: [["off", ["off", "dark"]], ["dim", ["dim", "dimmed", "low"]], ["bright", ["bright", "brightness", "full", "on"]]],
    curtains: [["closed", ["closed", "close", "shut", "drawn"]], ["open", ["open", "opened"]]],
    tv: [["on", ["on"]], ["off", ["off"]]],
    aircon: [["cool", ["cool", "cooling"]], ["heat", ["heat", "heating", "warm"]], ["off", ["off"]]],
    door: [["unlocked", ["unlocked", "unlock"]], ["locked", ["locked", "lock"]]],
  };
  const SKY = [["night", ["dark", "night", "midnight"]],
    ["dusk", ["dusk", "dawn", "sunset", "sunrise", "evening", "twilight"]]];
  const AC_WORD = { off: "OFF", cool: "COOL", heat: "HEAT" };
  const LOCK_WORD = { locked: "LOCKED", unlocked: "UNLOCKED" };
  const UNKNOWN_AT = { light: [252, 168], curtains: [214, 190], tv: [412, 262], aircon: [414, 152], door: [548, 250] };
  let phase = "loading";        // loading | ready | running | done
  let wishOn = null;
  let cardKey = 0;
  let loads = 0;                // how many times load() filled the editor (scripts wait on it)

  function markPaint(name, extra) {
    requestAnimationFrame(() => {
      const raf1 = now();
      requestAnimationFrame(() => { marks[name] = Object.assign({ raf1, raf2: now() }, extra || {}); });
    });
  }

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function rectOf(e) {
    const r = e.getBoundingClientRect();
    return { x: r.x, y: r.y, w: r.width, h: r.height };
  }

  function api() {
    return window.pywebview && window.pywebview.api;
  }

  // ---------------------------------------------------------------- the house

  function words(text) {
    return new Set(String(text).toLowerCase().match(/[a-z]+/g) || []);
  }

  function setting(device, text) {
    const w = words(text);
    for (const [value, names] of WORDS[device]) if (names.some((n) => w.has(n))) return value;
    return null;
  }

  function skyOf(outside) {
    const w = words(outside === undefined || outside === null ? "" : outside);
    for (const [value, names] of SKY) if (names.some((n) => w.has(n))) return value;
    return "day";
  }

  function plain(v) {
    return typeof v === "string" || typeof v === "number" || typeof v === "boolean";
  }

  function houseNow(state) {
    const s = state && typeof state === "object" && !Array.isArray(state) ? state : {};
    const out = {};
    for (const d of DEVICES) out[d] = plain(s[d]) ? setting(d, s[d]) : null;
    out.sky = skyOf(s.outside);
    out.inside = typeof s.inside === "string" || typeof s.inside === "number" ? String(s.inside) : "";
    out.time = typeof s.time === "string" || typeof s.time === "number" ? String(s.time) : "";
    return out;
  }

  function renderHouse(h, changed) {
    const svg = $("house");
    for (const d of DEVICES) svg.dataset[d] = h[d] || "unknown";
    svg.dataset.sky = h.sky;
    $("time").textContent = h.time;
    $("inside").textContent = h.inside;
    $("ac-word").textContent = AC_WORD[h.aircon] || "?";
    $("lock-word").textContent = LOCK_WORD[h.door] || "?";
    for (const t of svg.querySelectorAll(".tag")) t.classList.toggle("changed", changed.includes(t.dataset.device));
    const unknown = $("unknown");
    unknown.replaceChildren();
    for (const d of DEVICES) {
      if (h[d]) continue;
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", UNKNOWN_AT[d][0]);
      t.setAttribute("y", UNKNOWN_AT[d][1]);
      t.dataset.device = d;
      t.textContent = "?";
      unknown.appendChild(t);
    }
  }

  function houseShown() {
    const svg = $("house");
    const out = {};
    for (const d of DEVICES) out[d] = svg.dataset[d] === "unknown" ? null : svg.dataset[d];
    out.sky = svg.dataset.sky;
    out.inside = $("inside").textContent;
    out.time = $("time").textContent;
    out.ac_word = $("ac-word").textContent;
    out.lock_word = $("lock-word").textContent;
    out.changed = [...svg.querySelectorAll(".tag.changed")].map((t) => t.dataset.device);
    out.unknown = [...$("unknown").children].map((t) => t.dataset.device);
    return out;
  }

  // ---------------------------------------------------------------- phase, status, footer

  function setPhase(p, word) {
    phase = p;
    const pill = p === "running" ? "running" : p === "done" ? "done" : "idle";
    $("pill").className = "pill " + pill;
    const s = $("status");
    s.className = "status " + pill;
    if (word !== undefined) s.textContent = word;
    const locked = p === "running" || p === "loading";
    for (const f of document.querySelectorAll(".side textarea, .side input")) f.readOnly = p === "running";
    for (const b of document.querySelectorAll("#pane-edit button, #wishes button")) b.disabled = p === "running";
    $("decide").disabled = locked || cards().length === 0;
    if (p !== "running") $("decide").className = "decide";
  }

  function cards() {
    return [...document.querySelectorAll("#questions .q-card")];
  }

  function clearFoot() {
    $("time-big").textContent = "";
    $("foot-note").textContent = "";
  }

  function showError(text) {
    const e = $("error");
    e.textContent = text;
    e.className = "error on";
  }

  function hideError() {
    $("error").textContent = "";
    $("error").className = "error";
  }

  // ---------------------------------------------------------------- the editor

  function grow(t) {
    t.style.height = "auto";
    t.style.height = t.scrollHeight + 2 + "px";
  }

  function stateObject() {
    const text = $("state").value;
    const head = text.trim()[0];
    if (head !== "{") return null;
    try {
      const v = JSON.parse(text);
      return v !== null && typeof v === "object" && !Array.isArray(v) ? v : null;
    } catch (e) {
      return null;
    }
  }

  function stateKind() {
    const text = $("state").value;
    let kind = "";
    if (text.trim()) {
      kind = "text";
      const head = text.trim()[0];
      if (head === "{" || head === "[") {
        try {
          const v = JSON.parse(text);
          kind = v !== null && typeof v === "object" ? "JSON" : "text";
        } catch (e) {
          kind = "text (not valid JSON)";
        }
      }
    }
    $("state-kind").textContent = kind;
  }

  // The STATE text of a JSON value, as app/editor.py json_text writes it (one member per line when it is long).
  function pyDumps(v) {
    if (Array.isArray(v)) return "[" + v.map(pyDumps).join(", ") + "]";
    if (v !== null && typeof v === "object") {
      return "{" + Object.entries(v).map(([k, x]) => JSON.stringify(k) + ": " + pyDumps(x)).join(", ") + "}";
    }
    return JSON.stringify(v);
  }

  function jsonText(value) {
    const one = pyDumps(value);
    if (one.length <= 60) return one;
    if (!Array.isArray(value)) {
      return "{\n" + Object.entries(value).map(([k, v]) => "  " + JSON.stringify(k) + ": " + pyDumps(v)).join(",\n") + "\n}";
    }
    return "[\n" + value.map((v) => "  " + pyDumps(v)).join(",\n") + "\n]";
  }

  // The wish line shows the state's "request"; it is off while the state is not a JSON object.
  function syncWish() {
    const obj = stateObject();
    const wish = $("wish");
    if (obj === null) {
      wish.value = "";
      wish.disabled = true;
      $("wish-note").textContent = "The state is not a JSON object: write the wish into it as \"request\".";
      return;
    }
    wish.disabled = false;
    if (obj.request === undefined || typeof obj.request === "string") {
      if (wish.value !== (obj.request || "")) wish.value = obj.request || "";
      $("wish-note").textContent = "";
    } else {
      wish.value = "";
      wish.disabled = true;
      $("wish-note").textContent = "The state's \"request\" is not a text.";
    }
  }

  function writeWish() {
    const obj = stateObject();
    if (obj === null) return;
    obj.request = $("wish").value;
    $("state").value = jsonText(obj);
  }

  function setType(card, type) {
    card.dataset.type = type;
    for (const b of card.querySelectorAll(".seg button")) b.className = b.dataset.type === type ? "on" : "";
    card.querySelector(".q-opts").placeholder = OPTION_HINTS[type];
  }

  function newName() {
    const used = new Set(cards().map((c) => c.querySelector(".q-name").value.trim()));
    for (let i = 1; ; i++) if (!used.has("q" + i)) return "q" + i;
  }

  function questionCard(q) {
    const card = el("div", "card q-card");
    card.dataset.key = String(++cardKey);
    const edit = el("div", "q-edit");
    const row = el("div", "q-row");
    const name = el("input", "q-name");
    name.type = "text";
    name.spellcheck = false;
    name.placeholder = "name";
    name.value = q.name;
    row.appendChild(name);
    const seg = el("div", "seg");
    for (const t of ["noul", "choice", "score"]) {
      const b = el("button", "", TYPE_WORDS[t]);
      b.type = "button";
      b.dataset.type = t;
      b.addEventListener("click", () => { setType(card, t); edited(); });
      seg.appendChild(b);
    }
    row.appendChild(seg);
    const rm = el("button", "q-remove", "−");
    rm.type = "button";
    rm.title = "Remove this question";
    rm.addEventListener("click", () => { card.remove(); edited(); });
    row.appendChild(rm);
    edit.appendChild(row);
    const text = el("input", "q-text");
    text.type = "text";
    text.spellcheck = false;
    text.placeholder = "The question";
    text.value = q.text;
    edit.appendChild(text);
    const opts = el("textarea", "q-opts");
    opts.rows = 1;
    opts.spellcheck = false;
    opts.value = q.options;
    edit.appendChild(opts);
    for (const f of [name, text, opts]) f.addEventListener("input", () => { if (f === opts) grow(opts); edited(); });
    card.appendChild(edit);
    card.appendChild(el("div", "q-answer"));
    // an answered card goes back to its fields on a click (the answers go, as with any edit)
    card.addEventListener("click", () => { if (phase === "done" && card.classList.contains("answered")) edited(); });
    setType(card, q.type);
    return card;
  }

  function readEditor() {
    return {
      state: $("state").value,
      questions: cards().map((c) => ({
        name: c.querySelector(".q-name").value,
        type: c.dataset.type,
        text: c.querySelector(".q-text").value,
        options: c.querySelector(".q-opts").value,
      })),
      wish: $("wish").value,
    };
  }

  // One row per question card: before Decide the device's name and what the state says now (dim), after Decide the
  // setting picked and its probability.
  function devicesNow() {
    const obj = stateObject() || {};
    const box = $("devices");
    box.className = "devices";
    box.replaceChildren(...cards().map((c) => {
      const name = c.querySelector(".q-name").value.trim();
      const row = el("div", "dev");
      row.dataset.name = name;
      row.appendChild(el("span", "d-name", NAMES[name] || name));
      row.appendChild(el("span", "d-value", plain(obj[name]) ? String(obj[name]) : ""));
      row.appendChild(el("span", "d-p", ""));
      return row;
    }));
  }

  function clearAnswers() {
    for (const c of cards()) {
      c.classList.remove("answered", "stale");
      c.querySelector(".q-answer").replaceChildren();
      delete c.dataset.answerName;
    }
  }

  // Any change to the editor: the answers go, the house is the state's again, the next Decide runs the edited request.
  function edited(fromWish) {
    hideError();
    hideCaretIfIdle();
    if (wishOn !== null) {
      wishOn = null;
      highlightWish();
    }
    if (phase === "done") {
      clearAnswers();
      setPhase("ready", "READY");
    } else {
      setPhase(phase);
    }
    stateKind();
    if (!fromWish) syncWish();
    renderHouse(houseNow(stateObject()), []);
    devicesNow();
    clearFoot();
    for (const t of document.querySelectorAll(".side textarea")) grow(t);
  }

  function highlightWish() {
    for (const b of document.querySelectorAll(".ex")) b.className = b.dataset.id === wishOn ? "ex on" : "ex";
  }

  function showTab(name) {
    for (const b of document.querySelectorAll(".tab")) b.className = b.dataset.tab === name ? "tab on" : "tab";
    $("pane-wish").className = name === "wish" ? "pane on" : "pane";
    $("pane-edit").className = name === "edit" ? "pane on" : "pane";
    if (name === "edit") for (const t of document.querySelectorAll(".side textarea")) grow(t);
  }

  function wire() {
    $("state").addEventListener("input", () => { grow($("state")); edited(); });
    $("wish").addEventListener("input", () => { writeWish(); edited(true); });
    $("wish").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !$("decide").disabled) $("decide").click();
    });
    $("add-q").addEventListener("click", () => {
      const card = questionCard({ name: newName(), type: "choice", text: "", options: "" });
      $("questions").appendChild(card);
      grow(card.querySelector(".q-opts"));
      edited();
    });
    for (const b of document.querySelectorAll(".tab")) b.addEventListener("click", () => showTab(b.dataset.tab));
    $("decide").addEventListener("click", async () => {
      if (phase !== "ready" && phase !== "done") return;
      hideError();
      const r = await api().decide(readEditor());
      if (r && r.error) showError(r.error);
    });
  }

  // ---------------------------------------------------------------- the autoplay's hands

  const caret = () => $("caret");
  let caretOn = null;
  let typing = false;

  function caretPoint(field) {
    const cs = getComputedStyle(field);
    const m = document.createElement("div");
    for (const p of ["boxSizing", "width", "borderTopWidth", "borderRightWidth", "borderBottomWidth",
      "borderLeftWidth", "paddingTop", "paddingRight", "paddingBottom", "paddingLeft", "fontFamily", "fontSize",
      "fontWeight", "fontStyle", "letterSpacing", "lineHeight", "textTransform", "wordSpacing", "textIndent",
      "tabSize"]) m.style[p] = cs[p];
    const single = field.tagName === "INPUT";
    m.style.position = "absolute";
    m.style.visibility = "hidden";
    m.style.left = "-10000px";
    m.style.top = "0";
    m.style.whiteSpace = single ? "pre" : "pre-wrap";
    m.style.overflowWrap = single ? "normal" : "break-word";
    m.textContent = field.value;
    const mark = document.createElement("span");
    mark.textContent = "​";
    m.appendChild(mark);
    document.body.appendChild(m);
    const r = field.getBoundingClientRect();
    const lh = parseFloat(cs.lineHeight) || mark.offsetHeight;
    const out = single
      ? { x: r.left + mark.offsetLeft - field.scrollLeft, y: r.top + (r.height - lh) / 2, h: lh }
      : { x: r.left + mark.offsetLeft - field.scrollLeft, y: r.top + mark.offsetTop - field.scrollTop, h: lh };
    m.remove();
    return out;
  }

  function placeCaret(field) {
    const p = caretPoint(field);
    const c = caret();
    c.style.left = p.x + "px";
    c.style.top = p.y + 3 + "px";
    c.style.height = p.h - 6 + "px";
    c.style.display = "block";
    caretOn = field;
  }

  function hideCaret() {
    caret().style.display = "none";
    caretOn = null;
  }

  function hideCaretIfIdle() {
    if (caretOn && !typing) hideCaret();
  }

  async function typeInto(selector, text, cps, replace, name) {
    const field = document.querySelector(selector);
    typing = true;
    const start = now();
    if (replace && field.value) {
      placeCaret(field);
      await sleep(150);
      field.value = "";
      field.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "deleteContentBackward" }));
      placeCaret(field);
      await sleep(150);
    } else {
      placeCaret(field);
    }
    const dt = cps > 0 ? 1000 / cps : 0;
    const t0 = performance.now();
    for (let i = 0; i < text.length; i++) {
      if (dt) {
        const wait = t0 + (i + 1) * dt - performance.now();
        if (wait > 0) await sleep(wait);
      }
      field.value += text[i];
      field.dispatchEvent(new InputEvent("input", {
        bubbles: true, data: text[i], inputType: text[i] === "\n" ? "insertLineBreak" : "insertText" }));
      placeCaret(field);
    }
    typing = false;
    markPaint(name, { start, end: now(), chars: text.length, value: field.value, rect: rectOf(field) });
  }

  async function tap(selector, name, holdMs) {
    const b = document.querySelector(selector);
    const rect = rectOf(b);
    hideCaret();
    if (holdMs > 0) {
      b.classList.add("tap");
      await sleep(holdMs);
      b.classList.remove("tap");
    }
    b.click();
    markPaint(name, { rect });
  }

  // ---------------------------------------------------------------- what Python calls

  return {
    hud(h) {
      $("model").textContent = h.model;
      $("device").textContent = h.device;
    },

    // The wish buttons (each fills the editor with its house and the questions).
    wishes(list) {
      const row = $("wishes");
      for (const x of list) {
        const b = el("button", "ex", x.title);
        b.type = "button";
        b.dataset.id = x.id;
        b.addEventListener("click", async () => {
          if (phase === "running") return;
          const r = await api().wish(x.id);
          HOME.load(r.editor, r.id);
        });
        row.appendChild(b);
      }
      wire();
    },

    // Fill the editor: {state, questions: [{name, type, text, options}]} and the wish's id (or null).
    load(form, wishId) {
      hideError();
      hideCaret();
      $("state").value = form.state;
      $("questions").replaceChildren(...form.questions.map(questionCard));
      if (phase === "done") setPhase("ready", "READY");
      else setPhase(phase);
      stateKind();
      syncWish();
      renderHouse(houseNow(stateObject()), []);
      devicesNow();
      clearFoot();
      wishOn = wishId || null;
      highlightWish();
      for (const t of document.querySelectorAll(".side textarea")) grow(t);
      loads += 1;
    },

    loads() {
      return loads;
    },

    tab(name) {
      showTab(name);
      return name;
    },

    status(p, word, mark) {
      setPhase(p, word);
      if (mark) markPaint(mark);
    },

    ready(mark) {
      clearAnswers();
      hideCaret();
      setPhase("ready", "READY");
      renderHouse(houseNow(stateObject()), []);
      devicesNow();
      clearFoot();
      markPaint(mark || "ready");
    },

    running(mark) {
      hideError();
      hideCaret();
      const answered = cards().some((c) => c.classList.contains("answered"));
      if (answered) {
        for (const c of cards()) c.classList.add("stale");
        $("devices").classList.add("stale");
      }
      clearFoot();
      setPhase("running", "RUNNING");
      $("decide").className = "decide pressed";
      markPaint(mark || "press");
    },

    // The answers, every card and row at once: {questions: [{name, device, text, answer, answer_p, top, options}],
    // time, note, house_before, house_after, changed}. Card i and row i show question i of the request.
    answers(a, mark) {
      const list = cards();
      const box = $("devices");
      box.className = "devices";
      const rows = [];
      a.questions.forEach((q, i) => {
        const card = list[i];
        const face = card.querySelector(".q-answer");
        face.replaceChildren();
        face.appendChild(el("div", "q-atext", q.text));
        const ans = el("div", "answer");
        ans.appendChild(el("span", "a-name", q.answer));
        ans.appendChild(el("span", "a-p", q.answer_p));
        face.appendChild(ans);
        const opts = el("div", "options");
        for (const o of q.options) {
          const row = el("div", o.key === q.top ? "opt top" : "opt");
          row.dataset.key = o.key;
          row.appendChild(el("span", "o-name", o.label));
          const track = el("span", "o-track");
          const fill = el("span", "o-fill");
          fill.style.width = (100 * o.width).toFixed(2) + "%";
          track.appendChild(fill);
          row.appendChild(track);
          row.appendChild(el("span", "o-p", o.text));
          opts.appendChild(row);
        }
        face.appendChild(opts);
        card.dataset.answerName = q.name;
        card.classList.remove("stale");
        card.classList.add("answered");
        const r = el("div", a.changed.includes(q.name) ? "dev answered changed" : "dev answered");
        r.dataset.name = q.name;
        r.appendChild(el("span", "d-name", q.device));
        r.appendChild(el("span", "d-value", q.answer));
        r.appendChild(el("span", "d-p", q.answer_p));
        rows.push(r);
      });
      box.replaceChildren(...rows);
      renderHouse(a.house_after, a.changed);
      $("time-big").textContent = a.time;
      $("foot-note").textContent = a.note || "";
      setPhase("done", "DONE");
      markPaint(mark || "answers");
    },

    error(text) {
      showError(text);
      if (phase === "running") setPhase("ready", "READY");
    },

    autoType(selector, text, cps, replace, mark) {
      typeInto(selector, text, cps, replace, mark);
      return null;
    },

    // Paste: the whole text at once, one input event, as a paste from the clipboard arrives.
    autoPaste(selector, text, mark) {
      const field = document.querySelector(selector);
      hideCaret();
      field.value = text;
      field.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertFromPaste" }));
      markPaint(mark, { chars: text.length, rect: rectOf(field) });
      return null;
    },

    autoTap(selector, mark, holdMs) {
      tap(selector, mark, holdMs === undefined ? (selector === "#decide" ? 0 : 140) : holdMs);
      return null;
    },

    readEditor,
    houseShown,

    phase() {
      return phase;
    },

    mark(name) {
      return marks[name] || null;
    },

    shown() {
      const out = {
        status: $("status").textContent,
        pill: $("pill").textContent.replace(/\s+/g, " ").trim(),
        model: $("model").textContent,
        device: $("device").textContent,
        wish: $("wish").value,
        time: $("time-big").textContent,
        note: $("foot-note").textContent,
        error: $("error").textContent,
        devices: {},
        questions: {},
        house: houseShown(),
      };
      for (const row of document.querySelectorAll("#devices .dev.answered")) {
        out.devices[row.dataset.name] = {
          device: row.querySelector(".d-name").textContent,
          answer: row.querySelector(".d-value").textContent,
          answer_p: row.querySelector(".d-p").textContent,
          changed: row.classList.contains("changed"),
        };
      }
      for (const card of cards()) {
        if (!card.classList.contains("answered")) continue;
        const q = {
          text: card.querySelector(".q-atext").textContent,
          answer: card.querySelector(".a-name").textContent,
          answer_p: card.querySelector(".a-p").textContent,
          options: {},
          labels: {},
        };
        for (const row of card.querySelectorAll(".opt")) {
          q.options[row.dataset.key] = row.querySelector(".o-p").textContent;
          q.labels[row.dataset.key] = row.querySelector(".o-name").textContent;
        }
        out.questions[card.dataset.answerName] = q;
      }
      return out;
    },

    layout() {
      const out = { css_width: window.innerWidth, css_height: window.innerHeight,
        device_pixel_ratio: window.devicePixelRatio, rects: {}, bars: {} };
      for (const id of ["pill", "status", "model", "device", "house", "wish", "wishes", "devices", "decide", "error",
        "time-big", "foot-note", "pane-wish", "pane-edit"]) out.rects[id] = rectOf($(id));
      for (const id of ["pane-wish", "pane-edit"]) {
        const p = $(id);
        out[id.replace("-", "_") + "_scroll"] = { scroll_height: p.scrollHeight, client_height: p.clientHeight };
      }
      for (const card of cards()) {
        if (!card.classList.contains("answered")) continue;
        const bars = {};
        for (const row of card.querySelectorAll(".opt")) {
          const track = row.querySelector(".o-track").getBoundingClientRect();
          const fill = row.querySelector(".o-fill").getBoundingClientRect();
          bars[row.dataset.key] = track.width > 0 ? fill.width / track.width : null;
        }
        out.bars[card.dataset.answerName] = bars;
      }
      // anything shown that is drawn past the window or past its own box would be cut: report it (inside a pane, which
      // scrolls, only past the window's sides)
      const cut = [];
      for (const e of document.querySelectorAll("#app *")) {
        if (e.closest(".pane:not(.on)") || e.closest("svg")) continue;
        const r = e.getBoundingClientRect();
        if (r.width === 0 && r.height === 0) continue;
        const cs = getComputedStyle(e);
        if (cs.display === "none" || cs.visibility === "hidden") continue;
        const sides = r.right > window.innerWidth + 0.5 || r.left < -0.5;
        const outside = sides || (!e.closest(".pane") && (r.bottom > window.innerHeight + 0.5 || r.top < -0.5));
        const spills = cs.display !== "inline" && e.tagName !== "TEXTAREA" && !e.classList.contains("pane")
          && e.clientWidth > 0 && (e.scrollWidth > e.clientWidth + 1 || e.scrollHeight > e.clientHeight + 1);
        if (outside || spills) cut.push((e.id || e.className || e.tagName) + " " + JSON.stringify(rectOf(e)));
      }
      const pane = $("pane-wish");
      if (pane.classList.contains("on") && pane.scrollHeight > pane.clientHeight + 1) cut.push("pane-wish scrolls");
      out.cut = cut;
      return out;
    },
  };
})();
