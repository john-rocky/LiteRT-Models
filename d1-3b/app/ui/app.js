"use strict";
// The d1-3B editor. The page owns the editor (state text, question cards) and sends its content to Python on Decide
// (pywebview.api.decide); Python builds the request (app/editor.py), runs it, and draws the outcome through D1:
// running(), answers() or error(). Every number on screen arrives as a finished string (Python rounds), and shown()
// reads the strings back for the run record. mark(name) returns the paint times of a step (epoch ms): raf1 = the
// animation frame that paints the change, raf2 = the frame after it. autoType / autoTap / autoDropStart are the
// autoplay's hands: they type into the same fields and press the same buttons a user would.
window.D1 = (() => {
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
  let phase = "loading";        // loading | ready | running | done | fatal
  let photo = null;             // {id, name, w, h, bytes, format, thumb}
  let exampleOn = null;
  let cardKey = 0;

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

  // ---------------------------------------------------------------- phase, status, footer

  function setPhase(p, word) {
    phase = p;
    const pill = p === "running" ? "running" : p === "done" ? "done" : "idle";
    $("pill").className = "pill " + pill;
    const s = $("status");
    s.className = "status " + pill;
    if (word !== undefined) s.textContent = word;
    const locked = p === "running" || p === "loading" || p === "fatal";
    for (const f of document.querySelectorAll("#editor textarea, #editor input")) f.readOnly = p === "running" || p === "fatal";
    for (const b of document.querySelectorAll("#editor button")) b.disabled = p === "running" || p === "fatal";
    $("state-clear").disabled = p === "running" || p === "fatal" || !$("state").value;
    $("decide").disabled = locked || cards().length === 0;
    if (p !== "running") $("decide").className = "decide";
  }

  function cards() {
    return [...document.querySelectorAll("#questions .q-card")];
  }

  function countWords() {
    const n = cards().length;
    const q = n + " question" + (n === 1 ? "" : "s");
    return photo ? "1 picture · " + q : q;
  }

  function footerCount() {
    $("foot-main").textContent = countWords();
    $("foot-note").textContent = "";
    $("foot-hint").textContent = "";
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
      photo: photo ? { id: photo.id, name: photo.name } : null,
    };
  }

  function clearAnswers() {
    for (const c of cards()) {
      c.classList.remove("answered", "stale");
      c.querySelector(".q-answer").replaceChildren();
      delete c.dataset.answerName;
    }
  }

  // Any change to the editor: the answers go, the count updates, the next Decide runs the edited request.
  function edited() {
    hideError();
    hideCaretIfIdle();
    if (exampleOn !== null) {
      exampleOn = null;
      highlightExample();
    }
    if (phase === "done") {
      clearAnswers();
      setPhase("ready", "READY");
    } else {
      setPhase(phase);
    }
    stateKind();
    footerCount();
    for (const t of document.querySelectorAll("#editor textarea")) grow(t);
  }

  function highlightExample() {
    for (const b of document.querySelectorAll(".ex")) b.className = b.dataset.id === exampleOn ? "ex on" : "ex";
  }

  function showPhoto(meta) {
    photo = meta;
    const zone = $("photo");
    zone.classList.remove("over");
    if (!meta) {
      zone.className = "card photo empty";
      $("photo-name").textContent = "";
      $("photo-img").removeAttribute("src");
      return null;
    }
    zone.className = "card photo";
    $("photo-name").textContent = meta.name + " · " + meta.w + " × " + meta.h;
    const img = $("photo-img");
    img.src = meta.thumb;
    return img;
  }

  async function photoResult(r) {
    if (!r) return;
    if (r.error) {
      showError(r.error);
      return;
    }
    if (r.cancelled) return;
    showPhoto(r);
    edited();
  }

  function wire() {
    $("state").addEventListener("input", () => { grow($("state")); edited(); });
    $("state-clear").addEventListener("click", () => {
      $("state").value = "";
      $("state").dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "deleteContent" }));
    });
    $("add-q").addEventListener("click", () => {
      const card = questionCard({ name: newName(), type: "choice", text: "", options: "" });
      $("questions").appendChild(card);
      grow(card.querySelector(".q-opts"));
      edited();
    });
    $("photo-choose").addEventListener("click", async () => {
      if (phase === "running") return;
      photoResult(await api().choose_photo());
    });
    $("photo-clear").addEventListener("click", async () => {
      await api().clear_photo();
      showPhoto(null);
      edited();
    });
    const zone = $("photo");
    // A file dragged over the window: the page takes it (the web view would otherwise open the file).
    for (const t of ["dragenter", "dragover"]) {
      document.addEventListener(t, (e) => {
        e.preventDefault();
        if (phase !== "running") zone.classList.add("over");
      });
    }
    document.addEventListener("dragleave", (e) => {
      if (e.relatedTarget === null) zone.classList.remove("over");
    });
    document.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("over");
      if (phase === "running") return;
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) dropFileIn(file);
    });
    $("decide").addEventListener("click", async () => {
      if (phase !== "ready" && phase !== "done") return;
      hideError();
      const r = await api().decide(readEditor());
      if (r && r.error) showError(r.error);
    });
  }

  function dropFileIn(file) {
    const reader = new FileReader();
    reader.onload = async () => {
      const b64 = String(reader.result).split(",", 2)[1] || "";
      photoResult(await api().drop_photo(file.name, b64));
    };
    reader.readAsDataURL(file);
  }

  // ---------------------------------------------------------------- the autoplay's hands

  const caret = () => $("caret");
  let caretOn = null;

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
    const out = { x: r.left + mark.offsetLeft - field.scrollLeft, y: r.top + mark.offsetTop - field.scrollTop, h: lh };
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

  let typing = false;

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

    examples(list) {
      const row = $("examples");
      for (const x of list) {
        const b = el("button", "ex", x.title);
        b.type = "button";
        b.dataset.id = x.id;
        b.addEventListener("click", async () => {
          if (phase === "running") return;
          const r = await api().example(x.id);
          D1.load(r.editor, r.photo, r.id);
        });
        row.appendChild(b);
      }
      wire();
    },

    // Fill the editor: {state, questions: [{name, type, text, options}]}, the photo (or null), the example's id.
    load(form, photoMeta, exampleId) {
      hideError();
      hideCaret();
      $("state").value = form.state;
      $("questions").replaceChildren(...form.questions.map(questionCard));
      showPhoto(photoMeta || null);
      exampleOn = exampleId || null;
      highlightExample();
      if (phase === "done") setPhase("ready", "READY");
      else setPhase(phase);
      stateKind();
      footerCount();
      for (const t of document.querySelectorAll("#editor textarea")) grow(t);
    },

    status(p, word, mark) {
      setPhase(p, word);
      if (mark) markPaint(mark);
    },

    ready(mark) {
      clearAnswers();
      hideCaret();
      setPhase("ready", "READY");
      footerCount();
      markPaint(mark || "ready");
    },

    hold(mark) {
      markPaint(mark);
    },

    fatal(text) {
      setPhase("fatal", "MISSING FILES");
      showError(text);
    },

    running(mark) {
      hideError();
      hideCaret();
      const answered = cards().some((c) => c.classList.contains("answered"));
      if (answered) for (const c of cards()) c.classList.add("stale");
      $("foot-main").textContent = countWords();
      $("foot-note").textContent = "";
      $("foot-hint").textContent = "";
      setPhase("running", "RUNNING");
      $("decide").className = "decide pressed";
      markPaint(mark || "press");
    },

    // The answers, every card at once: {questions: [{name, text, answer, answer_p, top, options: [...]}], footer,
    // note, hint}. Card i shows question i of the request.
    answers(a, mark) {
      const list = cards();
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
      });
      $("foot-main").textContent = a.footer;
      $("foot-note").textContent = a.note || "";
      $("foot-hint").textContent = a.hint || "";
      setPhase("done", "DONE");
      markPaint(mark || "answers");
    },

    error(text) {
      showError(text);
      if (phase === "running") setPhase("ready", "READY");
    },

    photoSet(meta, mark) {
      const img = showPhoto(meta);
      edited();
      if (!mark) return null;
      if (img && !img.complete) img.addEventListener("load", () => markPaint(mark, { rect: rectOf($("photo")) }), { once: true });
      else markPaint(mark, { rect: rectOf($("photo")) });
      return null;
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

    autoDropStart(mark) {
      hideCaret();
      $("photo").classList.add("over");
      markPaint(mark, { rect: rectOf($("photo")) });
      return null;
    },

    // A scripted drop (scripts/drive_editor.py): the file goes through the page's own drop handler.
    dropFile(name, b64, type) {
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const dt = new DataTransfer();
      dt.items.add(new File([bytes], name, { type }));
      const zone = $("photo");
      for (const t of ["dragenter", "dragover", "drop"]) {
        zone.dispatchEvent(new DragEvent(t, { bubbles: true, cancelable: true, dataTransfer: dt }));
      }
      return dt.files.length;
    },

    readEditor,

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
        footer: $("foot-main").textContent,
        footer_note: $("foot-note").textContent,
        hint: $("foot-hint").textContent,
        error: $("error").textContent,
        photo: $("photo-name").textContent,
        questions: {},
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
        for (const row of card.querySelectorAll(".opt")) {
          q.options[row.dataset.key] = row.querySelector(".o-p").textContent;
          q.labels[row.dataset.key] = row.querySelector(".o-name").textContent;
        }
        out.questions[card.dataset.answerName] = q;
      }
      return out;
    },

    layout() {
      const out = { css_width: window.innerWidth, css_height: window.innerHeight, device_pixel_ratio: window.devicePixelRatio,
        rects: {}, bars: {} };
      for (const id of ["pill", "status", "model", "device", "examples", "state", "photo", "decide", "foot-main",
        "foot-note", "foot-hint", "editor"]) out.rects[id] = rectOf($(id));
      for (const card of cards()) {
        const name = card.dataset.answerName || card.querySelector(".q-name").value;
        out.rects["q:" + name] = rectOf(card);
        if (!card.classList.contains("answered")) continue;
        const bars = {};
        for (const row of card.querySelectorAll(".opt")) {
          const track = row.querySelector(".o-track").getBoundingClientRect();
          const fill = row.querySelector(".o-fill").getBoundingClientRect();
          bars[row.dataset.key] = track.width > 0 ? fill.width / track.width : null;
        }
        out.bars[card.dataset.answerName] = bars;
      }
      // anything drawn past the window or past its own box would be cut: report it
      const cut = [];
      for (const e of document.querySelectorAll("#app *")) {
        const r = e.getBoundingClientRect();
        if (r.width === 0 && r.height === 0) continue;
        const cs = getComputedStyle(e);
        if (cs.display === "none" || cs.visibility === "hidden") continue;
        const outside = r.right > window.innerWidth + 0.5 || r.bottom > window.innerHeight + 0.5 || r.left < -0.5
          || r.top < -0.5;
        const spills = cs.display !== "inline" && e.tagName !== "TEXTAREA" && e.clientWidth > 0
          && (e.scrollWidth > e.clientWidth + 1 || e.scrollHeight > e.clientHeight + 1);
        if (outside || spills) cut.push((e.id || e.className || e.tagName) + " " + JSON.stringify(rectOf(e)));
      }
      for (const t of document.querySelectorAll("#editor textarea")) {
        if (t.scrollHeight > t.clientHeight + 1) cut.push((t.id || t.className) + " scrolls");
      }
      out.cut = cut;
      return out;
    },
  };
})();
