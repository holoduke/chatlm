// Custom combobox that progressively enhances a native <select>.
//
// The native element stays in the DOM (hidden) as the source of truth:
// picking an option sets `select.value` and dispatches a `change` event,
// so all existing wiring (postModelChange / refreshModels / restore) keeps
// working untouched. This module only owns the presentation: a styled
// button + a grouped, searchable popup panel with capability badges.
//
// Per-option metadata is read from data-attributes set by the filler:
//   data-group   section header the option belongs under
//   data-title   primary label (falls back to textContent)
//   data-sub     muted secondary line (backend · params · size)
//   data-badges  comma-separated chip tokens (e.g. "UNCENSORED,TOOLS")
// Disabled options render greyed and are not selectable.

const OPEN_PANELS = new Set();

function closeAllPanels(except) {
  for (const api of OPEN_PANELS) {
    if (api !== except) api.close();
  }
}

// One document-level listener dismisses any open panel on outside click.
let _outsideBound = false;
function bindOutside() {
  if (_outsideBound) return;
  _outsideBound = true;
  document.addEventListener("click", (e) => {
    for (const api of OPEN_PANELS) {
      // Panel is portaled to <body>, so check both it and the button root.
      // Also ignore the widget's own (hidden) native select — a wrapping
      // <label> forwards clicks to it, which would otherwise self-close.
      if (e.target === api.select) continue;
      if (!api.root.contains(e.target) && !api.panel.contains(e.target)) api.close();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeAllPanels(null);
  });
}

function badgeKind(token) {
  const t = token.toLowerCase();
  if (t.includes("uncensored")) return "uncensored";
  if (t.includes("tools")) return "tools";
  if (t.includes("vision")) return "vision";
  if (t.includes("audio")) return "audio";
  if (t.includes("think")) return "think";
  if (t.includes("heavy") || t.includes("⚠")) return "heavy";
  if (t.includes("mlx") || t.includes("ollama") || t.includes("hf") || t.includes("llama-srv")) return "backend";
  return "default";
}

function renderBadges(container, badgesAttr) {
  container.textContent = "";
  if (!badgesAttr) return;
  for (const raw of badgesAttr.split(",")) {
    const token = raw.trim();
    if (!token) continue;
    const chip = document.createElement("span");
    chip.className = `cs-badge cs-badge--${badgeKind(token)}`;
    chip.textContent = token;
    container.appendChild(chip);
  }
}

export function enhanceSelect(select) {
  if (select._cs) return select._cs;
  bindOutside();

  const root = document.createElement("div");
  root.className = "cs";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "cs-button";
  button.setAttribute("aria-haspopup", "listbox");
  button.setAttribute("aria-expanded", "false");

  const buttonLabel = document.createElement("span");
  buttonLabel.className = "cs-button-label";
  const caret = document.createElement("span");
  caret.className = "cs-caret";
  caret.textContent = "▾";
  button.append(buttonLabel, caret);

  const panel = document.createElement("div");
  panel.className = "cs-panel";
  panel.setAttribute("role", "listbox");
  panel.hidden = true;

  const searchWrap = document.createElement("div");
  searchWrap.className = "cs-search";
  const search = document.createElement("input");
  search.type = "text";
  search.placeholder = "filter…";
  search.spellcheck = false;
  search.autocomplete = "off";
  searchWrap.appendChild(search);

  const list = document.createElement("div");
  list.className = "cs-list";

  panel.append(searchWrap, list);
  root.append(button);
  // Portal the panel to <body> so a transformed/filtered ancestor (the
  // glass top-bar) can't become its containing block and clip it.
  document.body.appendChild(panel);

  // Insert the widget right after the native select and hide the native.
  select.classList.add("cs-native-hidden");
  select.after(root);

  let activeIndex = -1; // index into the currently-visible rows

  function visibleRows() {
    return Array.from(list.querySelectorAll(".cs-option:not([hidden])"));
  }

  function setActive(idx) {
    const rows = visibleRows();
    activeIndex = Math.max(-1, Math.min(idx, rows.length - 1));
    rows.forEach((r, i) => r.classList.toggle("cs-option--active", i === activeIndex));
    if (activeIndex >= 0) rows[activeIndex].scrollIntoView({ block: "nearest" });
  }

  function commit(value) {
    if (select.value === value) {
      close();
      return;
    }
    select.value = value;
    select.dispatchEvent(new Event("change", { bubbles: true }));
    syncButton();
    close();
  }

  function syncButton() {
    const opt = select.selectedOptions[0] || select.options[0];
    if (!opt) {
      buttonLabel.textContent = "--";
      return;
    }
    buttonLabel.textContent = opt.dataset.title || opt.textContent || "--";
    button.title = opt.title || opt.value || "";
    // Reflect a current uncensored pick with an accent dot on the button.
    button.classList.toggle("cs-button--uncensored", /uncensored/i.test(opt.dataset.badges || ""));
  }

  function build() {
    list.textContent = "";
    const groups = new Map();
    for (const opt of Array.from(select.options)) {
      const group = opt.dataset.group || "Models";
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push(opt);
    }
    for (const [groupName, opts] of groups) {
      const header = document.createElement("div");
      header.className = "cs-group";
      header.textContent = groupName;
      header.dataset.group = groupName;
      list.appendChild(header);
      for (const opt of opts) {
        const row = document.createElement("div");
        row.className = "cs-option";
        row.setAttribute("role", "option");
        row.dataset.value = opt.value;
        row.dataset.group = groupName;
        if (opt.disabled) row.classList.add("cs-option--disabled");
        if (opt.selected) row.classList.add("cs-option--selected");

        const main = document.createElement("div");
        main.className = "cs-option-main";
        const title = document.createElement("span");
        title.className = "cs-option-title";
        title.textContent = opt.dataset.title || opt.textContent || opt.value;
        const badges = document.createElement("span");
        badges.className = "cs-option-badges";
        renderBadges(badges, opt.dataset.badges);
        main.append(title, badges);

        row.appendChild(main);
        if (opt.dataset.sub) {
          const sub = document.createElement("div");
          sub.className = "cs-option-sub";
          sub.textContent = opt.dataset.sub;
          row.appendChild(sub);
        }

        row.title = opt.title || opt.value;
        if (!opt.disabled) {
          row.addEventListener("click", () => commit(opt.value));
        }
        list.appendChild(row);
      }
    }
  }

  function applyFilter() {
    const q = search.value.trim().toLowerCase();
    const groupHasVisible = new Map();
    for (const row of list.querySelectorAll(".cs-option")) {
      const hay = `${row.textContent} ${row.dataset.value}`.toLowerCase();
      const show = !q || hay.includes(q);
      row.hidden = !show;
      if (show) groupHasVisible.set(row.dataset.group, true);
    }
    for (const header of list.querySelectorAll(".cs-group")) {
      header.hidden = !groupHasVisible.get(header.dataset.group);
    }
    setActive(visibleRows().findIndex((r) => r.classList.contains("cs-option--selected")));
  }

  function position() {
    const r = button.getBoundingClientRect();
    const margin = 8;
    const maxW = 400;
    // Clamp horizontally so the panel never runs off the right edge.
    const left = Math.max(margin, Math.min(r.left, window.innerWidth - maxW - margin));
    panel.style.left = `${left}px`;
    panel.style.minWidth = `${Math.max(r.width, 300)}px`;
    // Drop below the button; flip above if there isn't room.
    const belowRoom = window.innerHeight - r.bottom - margin;
    if (belowRoom < 240 && r.top > belowRoom) {
      panel.style.top = "auto";
      panel.style.bottom = `${window.innerHeight - r.top + 4}px`;
      panel.style.maxHeight = `${r.top - margin}px`;
    } else {
      panel.style.bottom = "auto";
      panel.style.top = `${r.bottom + 4}px`;
      panel.style.maxHeight = `${belowRoom}px`;
    }
  }

  function open() {
    if (select.disabled) return;
    closeAllPanels(api);
    build();
    panel.hidden = false;
    position();
    button.setAttribute("aria-expanded", "true");
    root.classList.add("cs--open");
    OPEN_PANELS.add(api);
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    search.value = "";
    applyFilter();
    setTimeout(() => search.focus(), 0);
  }

  const reposition = () => { if (!panel.hidden) position(); };

  function close() {
    panel.hidden = true;
    button.setAttribute("aria-expanded", "false");
    root.classList.remove("cs--open");
    OPEN_PANELS.delete(api);
    window.removeEventListener("resize", reposition);
    window.removeEventListener("scroll", reposition, true);
  }

  button.addEventListener("click", (e) => {
    e.stopPropagation();
    // Stop the wrapping <label> from forwarding this click to the hidden
    // native <select> (which would steal focus / re-close the panel).
    e.preventDefault();
    if (panel.hidden) open(); else close();
  });

  search.addEventListener("input", applyFilter);
  search.addEventListener("keydown", (e) => {
    const rows = visibleRows();
    if (e.key === "ArrowDown") { e.preventDefault(); setActive(activeIndex + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive(activeIndex - 1); }
    else if (e.key === "Enter") {
      e.preventDefault();
      const row = rows[activeIndex];
      if (row && !row.classList.contains("cs-option--disabled")) commit(row.dataset.value);
    }
  });

  // Keep the button label in sync if the value changes elsewhere
  // (e.g. the restore-from-localStorage path sets select.value directly).
  select.addEventListener("change", syncButton);

  const api = { root, panel, open, close, refresh: syncButton, select };
  select._cs = api;
  syncButton();
  return api;
}
