"use strict";
// The clef-flash mac page. The page owns the editor (the state text, the question cards, and the command line, which
// writes the state's "request") and the rows of this Mac's devices; it sends the editor's content to Python on Decide
// (pywebview.api.decide), with whether the state is still the one read from the Mac ("live"), whether the questions
// still follow the state ("autoQuestions") and whether "Apply to this Mac" is on. Python reads the Mac again when the
// state is live (setState), builds the questions from the state when they follow it (setQuestions), builds the
// request (app/editor.py), runs it, and draws the outcome through MAC: running(), answers(), then applying() /
// applied() per device, or error(). Every number on screen arrives as a finished string (formatted in Python), and
// shown() reads the strings back for the run record. mark(name) returns the paint times of a step (epoch ms): raf1 =
// the animation frame that paints the change, raf2 = the frame after it. autoType / autoTap / autoPaste are the
// autoplay's hands: they type into the same fields and press the same buttons a user would.
window.MAC = (() => {
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
  // app/actions.py NAMES and app/mac_demo.py value_word / device_now, written out again for the rows while you type
  const NAMES = { tv: "TV app", calculator: "Calculator", draft_1: "Draft (1)", draft_2: "Draft (2)", music: "Music",
    sound: "Sound", screen: "Screen" };
  const SETTABLE = { app: ["running"], document: ["open"], player: ["playing"], speaker: ["volume"], lock: ["locked"] };
  const WORDS = { running: ["Running", "Not running"], open: ["Open", "Closed"], playing: ["Playing", "Off"],
    locked: ["Locked", "Unlocked"], muted: ["Muted", "Not muted"], available: ["Available", "Not available"] };
  const ICONS = {
    app: '<rect x="5" y="6" width="22" height="18" rx="3" fill="none" stroke="currentColor" stroke-width="2.4"/>' +
      '<path d="M5 11 h22" stroke="currentColor" stroke-width="2.4"/>',
    document: '<path d="M9 4 h10 l6 6 v18 h-16 z" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linejoin="round"/>' +
      '<path d="M12 16 h9 M12 21 h9" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>',
    player: '<path d="M12 23 V8 l13 -3 v15" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linejoin="round"/>' +
      '<ellipse cx="9" cy="23.5" rx="3.6" ry="2.8" fill="currentColor"/><ellipse cx="22" cy="20.5" rx="3.6" ry="2.8" fill="currentColor"/>',
    speaker: '<path d="M5 12 h5 l7 -6 v20 l-7 -6 h-5 z" fill="currentColor"/>' +
      '<path d="M21 11 q4 5 0 10 M24.5 8 q6.5 8 0 16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>',
    lock: '<rect x="7" y="14" width="18" height="13" rx="2.5" fill="currentColor"/>' +
      '<path d="M11 14 v-4 a5 5 0 0 1 10 0 v4" fill="none" stroke="currentColor" stroke-width="2.4"/>',
    other: '<circle cx="16" cy="16" r="9" fill="none" stroke="currentColor" stroke-width="2.4"/>',
  };
  let phase = "loading";        // loading | ready | running | done
  let commandOn = null;
  let cardKey = 0;
  let loads = 0;                // how many times load() filled the editor (scripts wait on it)
  let live = true;              // the state text is the one read from the Mac (not edited by hand)
  let autoQ = true;             // the question cards follow the state (not edited by hand)
  let auto = false;             // autoplay: the fields take no keys (the autoplay's hands still write them)
  let qTimer = null;

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

  // ---------------------------------------------------------------- this Mac's rows

  function valueWord(prop, v) {
    if (v === undefined || v === null) return "?";
    if (typeof v === "boolean") return WORDS[prop] ? WORDS[prop][v ? 0 : 1] : (v ? "Yes" : "No");
    return String(v);
  }

  function settable(d) {
    const props = SETTABLE[d.kind] || Object.keys(d).filter((k) => typeof d[k] === "boolean" && k !== "available");
    return props.filter((p) => p in d && (typeof d[p] === "boolean" || typeof d[p] === "number"));
  }

  function deviceNow(d) {
    if (d.kind === "speaker" && typeof d.volume === "number") return String(d.volume) + (d.muted === true ? " (muted)" : "");
    const props = settable(d);
    if (!props.length) return "—";
    return props.map((p) => valueWord(p, d[p])).join(" · ");
  }

  function devicesOf(state) {
    if (!state || !Array.isArray(state.devices)) return [];
    return state.devices.filter((d) => d && typeof d === "object" && typeof d.id === "string" && d.id);
  }

  function row(id) {
    return document.querySelector(`.row[data-device="${CSS.escape(id)}"]`);
  }

  function rowsNow() {
    const state = stateObject();
    const devs = devicesOf(state);
    const box = $("rows");
    box.className = "rows";
    box.replaceChildren(...devs.map((d) => {
      const r = el("div", "row");
      r.dataset.device = d.id;
      const ns = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(ns, "svg");
      svg.setAttribute("class", "icon");
      svg.setAttribute("viewBox", "0 0 32 32");
      svg.innerHTML = ICONS[d.kind] || ICONS.other;
      r.appendChild(svg);
      r.appendChild(el("span", "r-name", NAMES[d.id] || String(d.name || d.id)));
      r.appendChild(el("span", "r-before", deviceNow(d)));
      r.appendChild(el("span", "r-arrow", ""));
      r.appendChild(el("span", "r-after", ""));
      r.appendChild(el("span", "r-p", ""));
      r.appendChild(el("span", "r-mark", ""));
      r.appendChild(el("span", "r-error", ""));
      return r;
    }));
    clearWay();
    $("mac-note").textContent = devs.length ? "" : "The state lists no devices.";
    $("mac-time").textContent = state && typeof state.now === "string" ? state.now.replace("T", " ").slice(0, 16) : "";
    $("mac-source").textContent = live ? "read from this Mac" : "edited by you";
  }

  function clearWay() {
    $("way").className = "way";
    $("way-word").textContent = "";
    $("way-p").textContent = "";
    $("way-detail").textContent = "";
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
    for (const f of document.querySelectorAll(".side textarea, .side input[type=text]")) f.readOnly = auto || p === "running";
    for (const b of document.querySelectorAll("#pane-edit button, #commands button, #scene")) b.disabled = p === "running";
    $("apply").disabled = p === "running";
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
    $("q-kind").textContent = autoQ ? "built from the state" : "edited by you";
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

  // The command line shows the state's "request"; it is off while the state is not a JSON object.
  function syncCommand() {
    const obj = stateObject();
    const c = $("command");
    if (obj === null) {
      c.value = "";
      c.disabled = true;
      $("command-note").textContent = "The state is not a JSON object: write the command into it as \"request\".";
      return;
    }
    c.disabled = false;
    if (obj.request === undefined || obj.request === null || typeof obj.request === "string") {
      if (c.value !== (obj.request || "")) c.value = obj.request || "";
      $("command-note").textContent = "";
    } else {
      c.value = "";
      c.disabled = true;
      $("command-note").textContent = "The state's \"request\" is not a text.";
    }
  }

  function writeCommand() {
    const obj = stateObject();
    if (obj === null) return;
    obj.request = $("command").value;
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
    const r = el("div", "q-row");
    const name = el("input", "q-name");
    name.type = "text";
    name.spellcheck = false;
    name.placeholder = "name";
    name.value = q.name;
    r.appendChild(name);
    const seg = el("div", "seg");
    for (const t of ["noul", "choice", "score"]) {
      const b = el("button", "", TYPE_WORDS[t]);
      b.type = "button";
      b.dataset.type = t;
      b.addEventListener("click", () => { setType(card, t); cardEdited(); });
      seg.appendChild(b);
    }
    r.appendChild(seg);
    const rm = el("button", "q-remove", "−");
    rm.type = "button";
    rm.title = "Remove this question";
    rm.addEventListener("click", () => { card.remove(); cardEdited(); });
    r.appendChild(rm);
    edit.appendChild(r);
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
    for (const f of [name, text, opts]) f.addEventListener("input", () => { if (f === opts) grow(opts); cardEdited(); });
    card.appendChild(edit);
    card.appendChild(el("div", "q-answer"));
    // an answered card goes back to its fields on a click (the answers go, as with any edit)
    card.addEventListener("click", () => { if (phase === "done" && card.classList.contains("answered")) edited(); });
    setType(card, q.type);
    return card;
  }

  function fillCards(list) {
    $("questions").replaceChildren(...list.map(questionCard));
    for (const t of document.querySelectorAll(".q-opts")) grow(t);
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
      command: $("command").value,
      live,
      autoQuestions: autoQ,
      apply: $("apply").checked,
    };
  }

  function clearAnswers() {
    for (const c of cards()) {
      c.classList.remove("answered", "stale");
      c.querySelector(".q-answer").replaceChildren();
      delete c.dataset.answerName;
    }
  }

  // The cards follow the state: built again (app/form.py, through Python) a moment after the state changes.
  function followState() {
    if (!autoQ || !api()) return;
    clearTimeout(qTimer);
    qTimer = setTimeout(async () => {
      const r = await api().questions($("state").value);
      if (!autoQ || phase === "running" || !r || r.error) return;
      fillCards(r.questions);
      setPhase(phase);
    }, 200);
  }

  // Any change to the editor: the answers go, the rows show the state again, the next Decide runs the edited request.
  function edited(fromCommand) {
    hideError();
    hideCaretIfIdle();
    if (commandOn !== null && !fromCommand) {
      commandOn = null;
      highlightCommand();
    }
    if (phase === "done") {
      clearAnswers();
      setPhase("ready", "READY");
    } else {
      setPhase(phase);
    }
    stateKind();
    if (!fromCommand) syncCommand();
    rowsNow();
    clearFoot();
    for (const t of document.querySelectorAll(".side textarea")) grow(t);
  }

  function cardEdited() {
    autoQ = false;
    edited();
  }

  function highlightCommand() {
    for (const b of document.querySelectorAll(".ex")) b.className = b.dataset.id === commandOn ? "ex on" : "ex";
  }

  function showTab(name) {
    for (const b of document.querySelectorAll(".tab")) b.className = b.dataset.tab === name ? "tab on" : "tab";
    $("pane-command").className = name === "command" ? "pane on" : "pane";
    $("pane-edit").className = name === "edit" ? "pane on" : "pane";
    if (name === "edit") for (const t of document.querySelectorAll(".side textarea")) grow(t);
  }

  async function readMac() {
    if (phase === "running") return;
    const r = await api().mac($("command").value);
    $("state").value = r.state;
    live = true;
    edited();
    followState();
  }

  async function rebuild() {
    if (phase === "running") return;
    const r = await api().questions($("state").value);
    if (r.error) {
      showError(r.error);
      return;
    }
    autoQ = true;
    fillCards(r.questions);
    edited();
  }

  function wire() {
    $("state").addEventListener("input", () => { live = false; grow($("state")); edited(); followState(); });
    $("command").addEventListener("input", () => {
      writeCommand();
      if (commandOn !== null) {
        commandOn = null;
        highlightCommand();
      }
      edited(true);
      followState();
    });
    $("command").addEventListener("keydown", (e) => {
      if (e.key !== "Enter" || e.shiftKey) return;
      e.preventDefault();   // the command is one line that wraps: Enter decides
      if (!auto && !$("decide").disabled) $("decide").click();
    });
    $("read-mac").addEventListener("click", readMac);
    $("rebuild-q").addEventListener("click", rebuild);
    $("scene").addEventListener("click", async () => {
      if (phase === "running") return;
      $("mac-note").textContent = "Opening the demo's devices…";
      const r = await api().scene();
      if (!r.ok) showError("Could not set: " + r.errors.join("; "));
      if (live) await readMac();
    });
    $("add-q").addEventListener("click", () => {
      const card = questionCard({ name: newName(), type: "choice", text: "", options: "" });
      $("questions").appendChild(card);
      grow(card.querySelector(".q-opts"));
      cardEdited();
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

    // The command buttons (each writes its command into the command line) and the "Apply to this Mac" switch.
    commands(list, applyOn) {
      const box = $("commands");
      for (const x of list) {
        const b = el("button", "ex", x.title);
        b.type = "button";
        b.dataset.id = x.id;
        b.addEventListener("click", async () => {
          if (phase === "running") return;
          const r = await api().command(x.id);
          const c = $("command");
          if (c.disabled) return;
          c.value = r.command;
          writeCommand();
          commandOn = r.id;
          highlightCommand();
          edited(true);
          followState();
          loads += 1;
        });
        box.appendChild(b);
      }
      $("apply").checked = applyOn;
      wire();
    },

    // Fill the editor: {state, questions: [{name, type, text, options}]}, the command's id (or null), and whether the
    // state is the one read from the Mac. The questions follow the state again.
    load(form, commandId, isLive) {
      hideError();
      hideCaret();
      $("state").value = form.state;
      fillCards(form.questions);
      live = isLive !== false;
      autoQ = true;
      if (phase === "done") setPhase("ready", "READY");
      else setPhase(phase);
      stateKind();
      syncCommand();
      rowsNow();
      clearFoot();
      commandOn = commandId || null;
      highlightCommand();
      for (const t of document.querySelectorAll(".side textarea")) grow(t);
      loads += 1;
    },

    // The state read from the Mac again right before a Decide (the editor stays live).
    setState(text) {
      $("state").value = text;
      live = true;
      stateKind();
      rowsNow();
      for (const t of document.querySelectorAll(".side textarea")) grow(t);
    },

    // The Mac read again outside a Decide (a recorded scene opened the drafts): the state and the rows.
    macRead(text, mark) {
      $("state").value = text;
      live = true;
      stateKind();
      syncCommand();
      rowsNow();
      for (const t of document.querySelectorAll(".side textarea")) grow(t);
      markPaint(mark || "mac-read");
    },

    // The questions built from the state right before a Decide (they follow the state).
    setQuestions(list) {
      fillCards(list);
      autoQ = true;
      stateKind();
    },

    loads() {
      return loads;
    },

    // Autoplay: the page takes no keys of its own (a key pressed during a recording changes nothing).
    autoplay(on) {
      auto = !!on;
      setPhase(phase);
      return auto;
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
      rowsNow();
      clearFoot();
      markPaint(mark || "ready");
    },

    running(mark) {
      hideError();
      hideCaret();
      const answered = cards().some((c) => c.classList.contains("answered"));
      if (answered) {
        for (const c of cards()) c.classList.add("stale");
        $("rows").classList.add("stale");
      }
      clearWay();
      clearFoot();
      setPhase("running", "RUNNING");
      $("decide").className = "decide pressed";
      markPaint(mark || "press");
    },

    // The answers, every card, the way of handling and every row at once: {questions: [{name, text, answer, answer_p,
    // top, options}], way: {key, word, p, text, detail}, rows: [{id, before, after, target, target_p, candidate}],
    // time, note}. Card i shows question i of the request.
    answers(a, mark) {
      const list = cards();
      rowsNow();
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
          const r = el("div", o.key === q.top ? "opt top" : "opt");
          r.dataset.key = o.key;
          r.appendChild(el("span", "o-name", o.label));
          const track = el("span", "o-track");
          const fill = el("span", "o-fill");
          fill.style.width = (100 * o.width).toFixed(2) + "%";
          track.appendChild(fill);
          r.appendChild(track);
          r.appendChild(el("span", "o-p", o.text));
          opts.appendChild(r);
        }
        face.appendChild(opts);
        card.dataset.answerName = q.name;
        card.classList.remove("stale");
        card.classList.add("answered");
      });
      if (a.way) {
        $("way").className = "way on " + a.way.key;
        $("way-word").textContent = a.way.word;
        $("way-p").textContent = a.way.p;
        $("way-detail").textContent = a.way.detail;
      }
      for (const r of a.rows) {
        const rw = row(r.id);
        if (!rw) continue;
        if (r.after) {
          rw.querySelector(".r-arrow").textContent = "→";
          rw.querySelector(".r-after").textContent = r.after;
        }
        rw.querySelector(".r-p").textContent = r.target_p === null ? "" : r.target_p;
        rw.classList.toggle("target", !!r.target);
        rw.classList.toggle("changed", !!r.after);
        rw.classList.toggle("candidate", !!r.candidate);
        if (r.candidate) rw.querySelector(".r-mark").textContent = "?";
      }
      $("time-big").textContent = a.time;
      $("foot-note").textContent = a.note || "";
      setPhase("done", "DONE");
      markPaint(mark || "answers");
    },

    // The devices being set: a waiting mark on their rows.
    applying(devices) {
      for (const d of devices) {
        const rw = row(d);
        if (!rw) continue;
        const m = rw.querySelector(".r-mark");
        m.textContent = "…";
        m.className = "r-mark wait";
      }
      $("mac-note").textContent = "Setting this Mac…";
    },

    // One device set and read back: {ok, shown_only, error}.
    applied(device, r, mark) {
      const rw = row(device);
      if (rw) {
        const m = rw.querySelector(".r-mark");
        m.textContent = r.shown_only ? "◌" : r.ok ? "✓" : "✗";
        m.className = "r-mark " + (r.shown_only ? "shown" : r.ok ? "ok" : "bad");
        rw.querySelector(".r-error").textContent = r.shown_only ? "Shown only: this app never locks the screen."
          : r.ok ? "" : "Could not set: " + r.error;
      }
      if (![...document.querySelectorAll(".r-mark.wait")].length) {
        const bad = [...document.querySelectorAll(".r-mark.bad")].length;
        $("mac-note").textContent = bad ? `This Mac took the other settings; ${bad} could not be set.`
          : "This Mac is set: each device read back after it was set.";
      }
      markPaint(mark);
    },

    applyOff(mark, hasChanges) {
      $("mac-note").textContent = hasChanges ? "Not applied: \"Apply to this Mac\" is off." : "";
      markPaint(mark);
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

    macShown() {
      const out = { time: $("mac-time").textContent, source: $("mac-source").textContent,
        note: $("mac-note").textContent, way: { word: $("way-word").textContent, p: $("way-p").textContent,
          detail: $("way-detail").textContent }, rows: {} };
      for (const r of document.querySelectorAll(".row")) {
        out.rows[r.dataset.device] = {
          name: r.querySelector(".r-name").textContent,
          before: r.querySelector(".r-before").textContent,
          after: r.querySelector(".r-after").textContent,
          p: r.querySelector(".r-p").textContent,
          mark: r.querySelector(".r-mark").textContent,
          error: r.querySelector(".r-error").textContent,
          target: r.classList.contains("target"),
          changed: r.classList.contains("changed"),
          candidate: r.classList.contains("candidate"),
        };
      }
      return out;
    },

    phase() {
      return phase;
    },

    live() {
      return live;
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
        command: $("command").value,
        apply: $("apply").checked,
        time: $("time-big").textContent,
        note: $("foot-note").textContent,
        error: $("error").textContent,
        mac: MAC.macShown(),
        questions: {},
        auto_questions: autoQ,
        scheme: matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light",
      };
      for (const card of cards()) {
        if (!card.classList.contains("answered")) continue;
        const q = {
          text: card.querySelector(".q-atext").textContent,
          answer: card.querySelector(".a-name").textContent,
          answer_p: card.querySelector(".a-p").textContent,
          options: {},
          labels: {},
        };
        for (const r of card.querySelectorAll(".opt")) {
          q.options[r.dataset.key] = r.querySelector(".o-p").textContent;
          q.labels[r.dataset.key] = r.querySelector(".o-name").textContent;
        }
        out.questions[card.dataset.answerName] = q;
      }
      return out;
    },

    layout() {
      const out = { css_width: window.innerWidth, css_height: window.innerHeight,
        device_pixel_ratio: window.devicePixelRatio, rects: {}, bars: {} };
      for (const id of ["pill", "status", "model", "device", "way", "way-word", "way-detail", "rows", "command",
        "commands", "decide", "error", "time-big", "foot-note", "pane-command", "pane-edit", "mac-time", "mac-source",
        "mac-note"]) out.rects[id] = rectOf($(id));
      for (const r of document.querySelectorAll(".row")) out.rects["row-" + r.dataset.device] = rectOf(r);
      for (const id of ["pane-command", "pane-edit"]) {
        const p = $(id);
        out[id.replace("-", "_") + "_scroll"] = { scroll_height: p.scrollHeight, client_height: p.clientHeight };
      }
      for (const card of cards()) {
        if (!card.classList.contains("answered")) continue;
        const bars = {};
        for (const r of card.querySelectorAll(".opt")) {
          const track = r.querySelector(".o-track").getBoundingClientRect();
          const fill = r.querySelector(".o-fill").getBoundingClientRect();
          bars[r.dataset.key] = track.width > 0 ? fill.width / track.width : null;
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
      const pane = $("pane-command");
      if (pane.classList.contains("on") && pane.scrollHeight > pane.clientHeight + 1) cut.push("pane-command scrolls");
      out.cut = cut;
      return out;
    },
  };
})();
