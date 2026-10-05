/* Control Center: local project library, notes, folders, JSON backup and list view. */
(() => {
  "use strict";

  const page = document.getElementById("page");
  const modalRoot = document.getElementById("modal-root");
  const toastStack = document.getElementById("toast-stack");
  const searchInput = document.getElementById("global-search");
  const importInput = document.getElementById("import-file");

  const ICONS = {
    rocket: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4c2.8-1.5 5.4-1.3 6-1-.2 2.2-.7 4.7-2.6 6.7l-5.6 5.6-4.2-4.2L13.2 5.5c.3-.5.5-1 .8-1.5Z"/><path d="m7.6 11.2-3.2.5-1.7 3.1 5.2.3M12.8 16.4l-.5 3.2-3.1 1.7-.3-5.2M7.1 16.9l-2.4 2.4"/><circle cx="15.5" cy="8.5" r="1.3"/></svg>',
    arena: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 20 7.5v9L12 21l-8-4.5v-9L12 3Z"/><path d="m4.5 7.8 7.5 4.4 7.5-4.4M12 12.2V21M8 5.2l8 4.5"/><circle cx="12" cy="12" r="2"/></svg>',
    dungeon: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20V6l4-3 4 3 4-3 4 3v14H4Z"/><path d="M8 20v-5h4v5m4-10h.01M8 10h.01m8 5h.01"/></svg>',
    hub: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="2.5"/><circle cx="5" cy="5" r="1.5"/><circle cx="19" cy="5" r="1.5"/><circle cx="5" cy="19" r="1.5"/><circle cx="19" cy="19" r="1.5"/><path d="m6.2 6.2 4 4m7.6-4-4 4m-7.6 7.6 4-4m7.6 4-4-4"/></svg>',
    gamepad: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 8h10a4 4 0 0 1 3.8 5.2l-1 3.3a2 2 0 0 1-3.3.9l-2.1-2H9.6l-2.1 2a2 2 0 0 1-3.3-.9l-1-3.3A4 4 0 0 1 7 8Z"/><path d="M7 11v4m-2-2h4m7-1h.01m2 2h.01"/></svg>',
    spark: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 3 1.6 6.1L20 11l-6.4 1.9L12 19l-1.6-6.1L4 11l6.4-1.9L12 3Z"/><path d="m19 16 .7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7L19 16Z"/></svg>',
    plus: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>',
    clock: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.5 2"/></svg>',
    star: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 3 2.7 5.5 6.1.9-4.4 4.3 1 6.1-5.4-2.9-5.4 2.9 1-6.1-4.4-4.3 6.1-.9L12 3Z"/></svg>',
    arrow: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14m-6-6 6 6-6 6"/></svg>',
    grid: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/><rect x="4" y="14" width="6" height="6" rx="1"/><rect x="14" y="14" width="6" height="6" rx="1"/></svg>',
    list: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01"/></svg>',
    search: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 4.2 4.2"/></svg>',
    info: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 11v5m0-8h.01"/></svg>',
    close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg>',
    check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6"/></svg>',
    folder: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/><path d="M3 10h18"/></svg>',
    download: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v12m-5-5 5 5 5-5M4 17v3h16v-3"/></svg>',
    upload: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4m-5 5 5-5 5 5M4 17v3h16v-3"/></svg>',
    edit: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m15 5 4 4M4 20l4-.8L19.5 7.7a2.1 2.1 0 0 0-3-3L5 16.2 4 20Z"/></svg>',
    trash: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M9 7V5h6v2m-8 0 1 13h8l1-13M10 11v6m4-6v6"/></svg>',
    lock: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 5v2"/></svg>',
    history: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7M3 4v5h5"/><path d="M12 8v4l3 2"/></svg>',
  };

  const ICON_NAMES = ["rocket", "arena", "dungeon", "hub", "gamepad", "spark", "plus"];
  const COLOR_NAMES = ["lime", "blue", "violet", "orange"];
  const STATUS_OPTIONS = [
    ["idea", "Idee"],
    ["project", "Projekt"],
    ["prototype", "Prototyp"],
    ["active", "In Arbeit"],
    ["paused", "Pausiert"],
    ["archived", "Archiviert"],
    ["reference", "GUI-Vorlage"],
  ];
  const CATEGORIES = ["Spielidee", "KI & Experimente", "Tool", "Web", "Projektordner", "Allgemein"];
  const SORT_OPTIONS = [
    ["updated", "Zuletzt geändert"],
    ["title", "Name (A–Z)"],
    ["opened", "Zuletzt geöffnet"],
    ["created", "Neueste zuerst"],
    ["status", "Status"],
  ];

  const state = {
    projects: [],
    query: "",
    filter: "all",
    layout: "grid",
    sort: "updated",
    favoritesOnly: false,
    readOnly: false,
    workspace: "",
    workspaceConfigured: false,
    loading: true,
    error: "",
    lastFocused: null,
  };

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  }

  function icon(name) { return ICONS[name] || ICONS.hub; }

  function loadPreference(key, fallback) {
    try { return window.localStorage.getItem(key) || fallback; } catch (_) { return fallback; }
  }

  function savePreference(key, value) {
    try { window.localStorage.setItem(key, value); } catch (_) { /* storage may be blocked */ }
  }

  const dateFormat = new Intl.DateTimeFormat("de-DE", { day: "2-digit", month: "2-digit", year: "numeric" });

  function formatTimestamp(value) {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0) return "";
    return dateFormat.format(new Date(number));
  }

  function relativeTime(value) {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0) return "noch nicht geöffnet";
    const minutes = Math.round((Date.now() - number) / 60000);
    if (minutes < 1) return "gerade eben";
    if (minutes < 60) return `vor ${minutes} Min.`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `vor ${hours} Std.`;
    const days = Math.round(hours / 24);
    if (days < 30) return `vor ${days} Tag${days === 1 ? "" : "en"}`;
    return formatTimestamp(number);
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      cache: "no-store",
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Anfrage fehlgeschlagen (${response.status}).`);
    return data;
  }

  function visibleProjects(list) {
    return list.filter((project) => !project.isDemo || project.folderPath || project.hasFolder);
  }

  async function loadLibrary() {
    state.loading = true;
    state.error = "";
    render();
    try {
      const health = await api("/api/health");
      state.readOnly = Boolean(health.readOnly);
      state.workspace = health.workspace || "";
      state.workspaceConfigured = Boolean(health.workspaceConfigured);
      if (!state.readOnly && health.libraryExists === false) {
        await migrateLegacyLibrary();
      }
      const result = await api("/api/projects");
      state.projects = visibleProjects(Array.isArray(result.projects) ? result.projects : []);
    } catch (error) {
      state.error = error.message || "Control Center konnte die lokale Bibliothek nicht laden.";
    } finally {
      state.loading = false;
      render();
    }
  }

  async function migrateLegacyLibrary() {
    try {
      const legacy = window.localStorage.getItem("control-center-projects-v1");
      const oldProjects = legacy ? JSON.parse(legacy) : null;
      if (Array.isArray(oldProjects) && oldProjects.length) {
        await api("/api/projects/import", {
          method: "POST",
          body: JSON.stringify({ format: "control-center-library", version: 1, projects: oldProjects }),
        });
        window.localStorage.setItem("control-center-projects-v1-backup", legacy);
        window.localStorage.removeItem("control-center-projects-v1");
      }
    } catch (migrationError) {
      console.warn("Alte Browser-Bibliothek wurde nicht automatisch übernommen.", migrationError);
    }
  }

  function statusClass(project) {
    return STATUS_OPTIONS.some(([key]) => key === project.status) ? project.status : "prototype";
  }

  function folderState(project) {
    if (state.readOnly) return project.hasFolder ? "linked" : "missing";
    if (project.folderPath) return "linked";
    return project.folderAvailable ? "outside" : "missing";
  }

  const FOLDER_LABELS = {
    linked: "Projektordner verbunden",
    outside: "Ordner außerhalb des Hauptordners",
    missing: "Noch keinem Ordner zugeordnet",
  };

  function searchBlob(project) {
    const steps = Array.isArray(project.nextSteps) ? project.nextSteps.join(" ") : "";
    return [project.title, project.description, project.category, project.notes, steps, ...(project.tags || [])]
      .join(" ")
      .toLowerCase();
  }

  function filterProjects(items) {
    let filtered = [...items];
    if (state.filter === "ideas") filtered = filtered.filter((project) => project.status === "idea");
    if (state.filter === "projects") filtered = filtered.filter((project) => project.status !== "idea");
    if (state.filter === "missing") filtered = filtered.filter((project) => folderState(project) === "missing");
    if (state.favoritesOnly) filtered = filtered.filter((project) => project.favorite);
    if (state.query.trim()) {
      const query = state.query.trim().toLowerCase();
      filtered = filtered.filter((project) => searchBlob(project).includes(query));
    }
    return filtered;
  }

  function sortProjects(items) {
    const sorted = [...items];
    const byTitle = (a, b) => String(a.title).localeCompare(String(b.title), "de", { sensitivity: "base" });
    if (state.sort === "title") return sorted.sort(byTitle);
    if (state.sort === "opened") {
      return sorted.sort((a, b) => Number(b.lastOpened || 0) - Number(a.lastOpened || 0) || byTitle(a, b));
    }
    if (state.sort === "created") {
      return sorted.sort((a, b) => String(b.createdAt || "").localeCompare(String(a.createdAt || "")) || byTitle(a, b));
    }
    if (state.sort === "status") {
      return sorted.sort((a, b) => statusClass(a).localeCompare(statusClass(b)) || byTitle(a, b));
    }
    return sorted.sort((a, b) => String(b.updatedAt || "").localeCompare(String(a.updatedAt || "")) || byTitle(a, b));
  }

  function favoriteButton(project) {
    return `<button class="favorite-button ${project.favorite ? "is-favorite" : ""}" type="button" data-action="toggle-favorite" data-id="${escapeHtml(project.id)}" aria-pressed="${project.favorite ? "true" : "false"}" title="${project.favorite ? "Favorit entfernen" : "Als Favorit markieren"}">${icon("star")}</button>`;
  }

  function openButton(project, compact) {
    const hasFolder = folderState(project) === "linked";
    const label = state.readOnly ? "Nur am PC" : hasFolder ? "Projekt öffnen" : "Ordner fehlt";
    const disabled = state.readOnly || !hasFolder ? "disabled" : "";
    return `<button class="project-open ${compact ? "slim" : ""}" type="button" data-action="open-project" data-id="${escapeHtml(project.id)}" ${disabled}>${escapeHtml(label)} ${icon("arrow")}</button>`;
  }

  function projectCard(project) {
    const tags = Array.isArray(project.tags) ? project.tags : [];
    const folder = folderState(project);
    return `
      <article class="project-card" data-kind="${escapeHtml(project.status || "prototype")}">
        <div class="project-top">
          <div class="project-icon ${escapeHtml(project.color || "lime")}">${icon(project.icon)}</div>
          <span class="project-state ${statusClass(project)}">${escapeHtml(project.statusLabel || "Projekt")}</span>
          ${favoriteButton(project)}
        </div>
        <h3 class="project-title">${escapeHtml(project.title)}</h3>
        <p class="project-description">${escapeHtml(project.description || (folder === "linked" ? "Der Projektordner enthält die Dateien dieses Projekts." : "Projektordner verbinden, um den echten Inhalt zu öffnen."))}</p>
        <div class="project-tags">${tags.map((tag) => `<span class="project-tag">${escapeHtml(tag)}</span>`).join("")}</div>
        <div class="project-footer">
          <span class="project-updated">${icon("folder")} ${escapeHtml(FOLDER_LABELS[folder])}</span>
          <span class="card-actions">${openButton(project, true)}<button class="icon-button tiny" type="button" data-action="edit-project" data-id="${escapeHtml(project.id)}" title="Details und Notizen" aria-label="Details von ${escapeHtml(project.title)}">${icon("edit")}</button></span>
        </div>
      </article>`;
  }

  function projectRow(project) {
    const tags = Array.isArray(project.tags) ? project.tags : [];
    const steps = Array.isArray(project.nextSteps) ? project.nextSteps.length : 0;
    const folder = folderState(project);
    const notes = project.notes ? "Notizen" : "";
    return `
      <article class="project-row" data-kind="${escapeHtml(project.status || "prototype")}">
        <span class="project-icon ${escapeHtml(project.color || "lime")}">${icon(project.icon)}</span>
        <div class="row-main">
          <div class="row-headline">
            <b class="row-title">${escapeHtml(project.title)}</b>
            <span class="project-state ${statusClass(project)}">${escapeHtml(project.statusLabel || "Projekt")}</span>
          </div>
          <p class="row-description">${escapeHtml(project.description || "")}</p>
          <div class="row-meta">
            <span class="row-cell">${icon("folder")} ${escapeHtml(FOLDER_LABELS[folder])}</span>
            <span class="row-cell">${escapeHtml(project.category || "Allgemein")}</span>
            <span class="row-cell">${icon("clock")} ${escapeHtml(relativeTime(project.lastOpened))}</span>
            ${steps ? `<span class="row-cell">${steps} nächste Schritte</span>` : ""}
            ${notes ? `<span class="row-cell">${escapeHtml(notes)}</span>` : ""}
            ${tags.map((tag) => `<span class="project-tag">${escapeHtml(tag)}</span>`).join("")}
          </div>
        </div>
        <div class="row-actions">
          ${favoriteButton(project)}
          ${openButton(project)}
          <button class="icon-button tiny" type="button" data-action="edit-project" data-id="${escapeHtml(project.id)}" title="Details und Notizen" aria-label="Details von ${escapeHtml(project.title)}">${icon("edit")}</button>
          <button class="icon-button tiny danger" type="button" data-action="delete-project" data-id="${escapeHtml(project.id)}" title="Eintrag löschen" aria-label="${escapeHtml(project.title)} löschen">${icon("trash")}</button>
        </div>
      </article>`;
  }

  function pageHeading(eyebrow, title, description, action = "") {
    return `<div class="page-heading"><div><div class="eyebrow">${escapeHtml(eyebrow)}</div><h1>${escapeHtml(title)}</h1><p>${escapeHtml(description)}</p></div>${action}</div>`;
  }

  function writeActions() {
    if (state.readOnly) return `<span class="readonly-chip">${icon("lock")} Ordnerzugriff nur lokal am PC</span>`;
    return `<div class="heading-actions">
      <button class="button small" type="button" data-action="backups">${icon("history")} Sicherungen</button>
      <button class="button small" type="button" data-action="import">${icon("upload")} Import</button>
      <button class="button small" type="button" data-action="export">${icon("download")} Export</button>
      <button class="button primary" type="button" data-action="new-project" ${state.workspaceConfigured ? "" : "disabled title=\"Verbinde zuerst deinen Desktop-Projektordner\""}>${icon("plus")} Neues Projekt</button>
    </div>`;
  }

  function workspacePanel() {
    const workspace = state.workspace || "";
    return `
      <section class="workspace-panel">
        <div class="workspace-panel-heading">
          <div class="workspace-panel-icon">${icon("folder")}</div>
          <div><span class="workspace-kicker">DEIN PROJEKTORDNER</span><h2>Ordner mit deinen Projekten verbinden</h2>
          <p>Wähle den Ordner, in dem deine Projektordner liegen. Versteckte Ordner und Build-Ordner werden übersprungen; umbenannte Ordner werden wiedererkannt.</p></div>
        </div>
        ${state.readOnly ? `<div class="workspace-readonly-note">Die Arena-Vorschau kann nicht auf deinen Desktop zugreifen. Starte Control Center auf deinem PC, um den lokalen Projektordner zu verbinden.</div>` : `<form id="workspace-form" class="workspace-form"><label for="workspace-path">Hauptordner</label><div class="workspace-input-row"><input id="workspace-path" name="path" type="text" value="${escapeHtml(workspace)}" placeholder="z. B. C:\\\\Users\\\\Du\\\\Desktop\\\\Meine Projekte" required><button class="button" type="button" data-action="choose-workspace">${icon("folder")} Ordner wählen</button><button class="button primary" type="submit">Verbinden &amp; einlesen ${icon("arrow")}</button></div><small>Control Center legt keine Dateien in deinen Projektordnern ab. „Ordner wählen“ nutzt den Dateidialog des Systems, falls verfügbar, sonst den eingebauten Ordnerbrowser.</small></form>`}
      </section>`;
  }

  function statsPanel() {
    const ideaCount = state.projects.filter((project) => project.status === "idea").length;
    const folderCount = state.projects.filter((project) => folderState(project) === "linked").length;
    const favoriteCount = state.projects.filter((project) => project.favorite).length;
    return `
      <div class="overview-stats library-stats" aria-label="Bibliotheksübersicht">
        <div class="overview-stat"><div class="stat-icon">${icon("folder")}</div><div class="stat-copy"><strong>${state.projects.length}</strong><span>Projekte in der Bibliothek</span></div></div>
        <div class="overview-stat"><div class="stat-icon violet">${icon("spark")}</div><div class="stat-copy"><strong>${ideaCount}</strong><span>Ideen</span></div></div>
        <div class="overview-stat"><div class="stat-icon blue">${icon("check")}</div><div class="stat-copy"><strong>${folderCount}</strong><span>Verbunden mit Projektordner</span></div></div>
        <div class="overview-stat"><div class="stat-icon orange">${icon("star")}</div><div class="stat-copy"><strong>${favoriteCount}</strong><span>Favoriten</span></div></div>
      </div>`;
  }

  function filterPills() {
    const pills = [
      ["all", "Alle"],
      ["projects", "Projekte"],
      ["ideas", "Ideen"],
      ["favorites", "Favoriten"],
      ["missing", "Ordner fehlt"],
    ];
    return `<div class="filter-pills" role="group" aria-label="Bibliothek filtern">${pills.map(([key, label]) => `<button class="filter-pill ${state.filter === key ? "active" : ""}" type="button" data-filter="${key}">${label}</button>`).join("")}</div>`;
  }

  function sortSelect() {
    const options = SORT_OPTIONS.map(([key, label]) => `<option value="${key}" ${state.sort === key ? "selected" : ""}>${label}</option>`).join("");
    return `<label class="sort-select"><span>Sortierung</span><select id="sort-select" aria-label="Sortierung">${options}</select></label>`;
  }

  function renderEmpty(title, description, action = "") {
    return `<div class="empty-state"><div class="empty-icon">${icon("folder")}</div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(description)}</p>${action}</div>`;
  }

  function renderLibrary() {
    const items = sortProjects(filterProjects(state.projects));
    const emptyAction = state.readOnly ? "" : state.workspaceConfigured
      ? `<button class="button small" type="button" data-action="new-project">${icon("plus")} Eintrag hinzufügen</button>`
      : `<button class="button small" type="button" data-action="choose-workspace">${icon("folder")} Desktop-Ordner wählen</button>`;
    const emptyTitle = state.query
      ? "Keine Treffer"
      : state.filter === "ideas" ? "Noch keine Ideen"
      : state.filter === "favorites" ? "Keine Favoriten"
      : state.filter === "missing" ? "Keine nicht zugeordneten Projekte"
      : "Bibliothek ist leer";
    const emptyDescription = state.query
      ? "Passe die Suche an — es werden auch Notizen und nächste Schritte durchsucht."
      : state.filter === "missing" ? "Alle sichtbaren Einträge sind mit einem Ordner verbunden."
      : state.filter === "favorites" ? "Markiere Einträge mit dem Stern, um sie hier zu sammeln."
      : "Verbinde oben deinen Projektordner oder füge einen Bibliothekseintrag hinzu.";
    const listMarkup = state.layout === "list"
      ? `<div class="project-list" role="list">${items.map(projectRow).join("")}</div>`
      : `<div class="project-grid">${items.map(projectCard).join("")}</div>`;
    return `
      ${pageHeading("WORKSPACE", "Meine Bibliothek", "Hier sind deine Projekte und Ideen. Ein Projekt öffnen zeigt seinen echten Ordner in einem neuen Browser-Tab.", writeActions())}
      ${workspacePanel()}
      ${statsPanel()}
      <section class="section">
        <div class="section-head">
          <div><div class="section-title"><h2>Projekte und Ideen</h2><span>${items.length} ${items.length === 1 ? "Eintrag" : "Einträge"}</span></div>
          <p class="section-description">Rasteransicht für einen schnellen Überblick, Listenansicht mit Status, Kategorie, Ordnern und letzten Schritten.</p></div>
          <div class="section-tools">
            ${filterPills()}
            ${sortSelect()}
            <div class="view-toggle" role="group" aria-label="Ansicht wechseln">
              <button type="button" class="${state.layout === "grid" ? "active" : ""}" data-layout="grid" aria-label="Rasteransicht" title="Rasteransicht">${icon("grid")}</button>
              <button type="button" class="${state.layout === "list" ? "active" : ""}" data-layout="list" aria-label="Listenansicht" title="Listenansicht">${icon("list")}</button>
            </div>
          </div>
        </div>
        ${items.length ? listMarkup : renderEmpty(emptyTitle, emptyDescription, emptyAction)}
      </section>
      <div class="preview-note">${icon("info")}<span><b>Projekt öffnen:</b> Wenn ein Projektordner eine <code>index.html</code> enthält, wird die Projektseite geladen. Sonst öffnet sich eine Dateiansicht des Ordners.</span></div>`;
  }

  function renderPage() {
    if (state.loading) return renderEmpty("Bibliothek wird geladen", "Control Center verbindet sich mit deiner lokalen Projektbibliothek.");
    if (state.error) return renderEmpty("Server nicht erreichbar", state.error, `<button class="button small" type="button" data-action="retry">${icon("clock")} Erneut versuchen</button>`);
    return renderLibrary();
  }

  function render() {
    document.querySelectorAll(".nav-item[data-view]").forEach((button) => button.classList.toggle("active", button.dataset.view === "library"));
    document.getElementById("breadcrumb-current").textContent = "Bibliothek";
    document.getElementById("project-count").textContent = String(state.projects.length).padStart(2, "0");
    const localPill = document.querySelector(".local-card");
    if (localPill) {
      const pillTitle = localPill.querySelector("b");
      const pillText = localPill.querySelector("small");
      if (pillTitle) pillTitle.textContent = state.readOnly ? "Desktop nicht verfügbar" : "Lokale Projektbibliothek";
      if (pillText) pillText.textContent = state.readOnly ? "Starte die App auf deinem PC" : "Projektdateien bleiben bei dir";
    }
    page.innerHTML = renderPage();
  }

  function showToast(message, isError = false) {
    const toast = document.createElement("div");
    toast.className = `toast ${isError ? "error" : ""}`;
    toast.innerHTML = `${icon(isError ? "info" : "check")}<span></span>`;
    toast.querySelector("span").textContent = message;
    toastStack.appendChild(toast);
    window.setTimeout(() => toast.remove(), 4200);
  }

  function openModal(html) {
    state.lastFocused = document.activeElement;
    modalRoot.innerHTML = html;
    const focusTarget = modalRoot.querySelector("input, textarea, select, button");
    if (focusTarget) focusTarget.focus();
  }

  function closeModal() {
    modalRoot.innerHTML = "";
    if (state.lastFocused && document.contains(state.lastFocused)) {
      try { state.lastFocused.focus(); } catch (_) { /* element may be gone */ }
    }
    state.lastFocused = null;
  }

  function openNewProjectModal(defaults = {}) {
    if (state.readOnly) return showToast("Neue Projekte können nur direkt am Server-PC angelegt werden.", true);
    if (!state.workspaceConfigured) return showToast("Verbinde zuerst den Hauptordner, in dem deine Projekte liegen.", true);
    openModal(`
      <div class="modal-backdrop" data-action="backdrop-close">
        <section class="modal" role="dialog" aria-modal="true" aria-labelledby="new-project-title">
          <div class="modal-head"><div><h2 id="new-project-title">Neues Projekt</h2><p>Erfasse eine Idee und lege den zugehörigen Ordner gleich mit an.</p></div><button class="icon-button" type="button" data-action="close-modal" aria-label="Dialog schließen">${icon("close")}</button></div>
          <form class="modal-body" id="new-project-form">
            <div class="form-field"><label for="project-name">Projektname</label><input id="project-name" name="name" maxlength="80" required placeholder="z. B. Neon Arena" value="${escapeHtml(defaults.title || "")}"></div>
            <div class="form-field"><label for="project-description">Kurzbeschreibung</label><textarea id="project-description" name="description" maxlength="500" placeholder="Worum geht es bei der Idee?">${escapeHtml(defaults.description || "")}</textarea></div>
            <div class="form-field"><label for="project-category">Bereich</label><select id="project-category" name="category">${CATEGORIES.map((name) => `<option ${name === "Allgemein" ? "selected" : ""}>${escapeHtml(name)}</option>`).join("")}</select></div>
            <label class="checkbox-field"><input type="checkbox" name="createFolder" checked><span><b>Projektordner gleich mit anlegen</b><small>Legt ihn im verbundenen Hauptordner zusammen mit deinen anderen Projekten an.</small></span></label>
            <div class="modal-actions"><button class="button" type="button" data-action="close-modal">Abbrechen</button><button class="button primary" type="submit">${icon("plus")} Projekt hinzufügen</button></div>
          </form>
        </section>
      </div>`);
    document.getElementById("project-name")?.focus();
  }

  function optionList(values, selected) {
    return values.map((value) => `<option value="${escapeHtml(value)}" ${value === selected ? "selected" : ""}>${escapeHtml(value)}</option>`).join("");
  }

  function openEditModal(id) {
    const project = state.projects.find((item) => item.id === id);
    if (!project) return showToast("Eintrag nicht gefunden.", true);
    const steps = Array.isArray(project.nextSteps) ? project.nextSteps.join("\n") : "";
    const tags = Array.isArray(project.tags) ? project.tags.join(", ") : "";
    const folder = folderState(project);
    openModal(`
      <div class="modal-backdrop" data-action="backdrop-close">
        <section class="modal wide" role="dialog" aria-modal="true" aria-labelledby="edit-project-title">
          <div class="modal-head"><div><h2 id="edit-project-title">${escapeHtml(project.title)}</h2><p>Notizen, Status, Schlagworte und Projektordner.</p></div><button class="icon-button" type="button" data-action="close-modal" aria-label="Dialog schließen">${icon("close")}</button></div>
          <form class="modal-body" id="edit-project-form">
            <input type="hidden" name="id" value="${escapeHtml(project.id)}">
            <div class="form-row">
              <div class="form-field"><label for="edit-title">Projektname</label><input id="edit-title" name="title" maxlength="80" required value="${escapeHtml(project.title)}"></div>
              <div class="form-field status-field"><label for="edit-status">Status</label><select id="edit-status" name="status">${STATUS_OPTIONS.map(([key, label]) => `<option value="${key}" ${project.status === key ? "selected" : ""}>${label}</option>`).join("")}</select></div>
            </div>
            <div class="form-row">
              <div class="form-field"><label for="edit-category">Bereich</label><input id="edit-category" name="category" list="category-options" maxlength="50" value="${escapeHtml(project.category || "")}"><datalist id="category-options">${CATEGORIES.map((name) => `<option value="${escapeHtml(name)}"></option>`).join("")}</datalist></div>
              <div class="form-field"><label for="edit-tags">Schlagworte</label><input id="edit-tags" name="tags" maxlength="240" placeholder="Mit Komma getrennt" value="${escapeHtml(tags)}"></div>
            </div>
            <div class="form-field"><label for="edit-description">Kurzbeschreibung</label><textarea id="edit-description" name="description" maxlength="500">${escapeHtml(project.description || "")}</textarea></div>
            <div class="form-field"><label for="edit-notes">Notizen</label><textarea id="edit-notes" name="notes" class="notes-area" maxlength="20000" placeholder="Ideen, Links, Entscheidungen …">${escapeHtml(project.notes || "")}</textarea></div>
            <div class="form-field"><label for="edit-steps">Nächste Schritte <small>eine Zeile pro Schritt</small></label><textarea id="edit-steps" name="nextSteps" class="steps-area" maxlength="6000" placeholder="Ersten Prototyp bauen&#10;Steuerung testen">${escapeHtml(steps)}</textarea></div>
            <div class="form-row">
              <div class="form-field"><label for="edit-icon">Symbol</label><select id="edit-icon" name="icon">${optionList(ICON_NAMES, project.icon || "hub")}</select></div>
              <div class="form-field"><label for="edit-color">Farbe</label><select id="edit-color" name="color">${optionList(COLOR_NAMES, project.color || "lime")}</select></div>
            </div>
            <div class="folder-path">
              ${icon("folder")}
              <div><b>${escapeHtml(FOLDER_LABELS[folder])}</b><small>${escapeHtml(project.folderPath || project.folderAvailable || "Beim Öffnen wird sonst nur die Dateiansicht angezeigt.")}</small></div>
            </div>
            <div class="detail-form-footer">
              ${state.readOnly ? "" : `
              <button class="button small" type="button" data-action="pick-folder" data-id="${escapeHtml(project.id)}">${icon("folder")} Ordner wählen</button>
              <button class="button small" type="button" data-action="ensure-folder" data-id="${escapeHtml(project.id)}">${icon("plus")} Ordner anlegen</button>
              <button class="button small" type="button" data-action="system-open" data-id="${escapeHtml(project.id)}">${icon("arrow")} Dateimanager</button>
              ${project.folderPath || project.folderAvailable ? `<button class="button small" type="button" data-action="detach-folder" data-id="${escapeHtml(project.id)}">Verknüpfung lösen</button>` : ""}`}
              <button class="button small danger" type="button" data-action="delete-project" data-id="${escapeHtml(project.id)}">${icon("trash")} Eintrag löschen</button>
            </div>
            <label class="checkbox-field"><input type="checkbox" name="favorite" ${project.favorite ? "checked" : ""}><span><b>Favorit</b><small>Favoriten erscheinen oben in der Schnellaktionen-Palette.</small></span></label>
            <div class="modal-actions"><button class="button" type="button" data-action="close-modal">Abbrechen</button><button class="button primary" type="submit">${icon("check")} Speichern</button></div>
          </form>
        </section>
      </div>`);
  }

  async function openFolderPicker(projectId = "") {
    if (state.readOnly) return showToast("Ordner kannst du nur auf dem PC verknüpfen, auf dem sie liegen.", true);
    openModal(`
      <div class="modal-backdrop" data-action="backdrop-close">
        <section class="modal wide" role="dialog" aria-modal="true" aria-labelledby="picker-title">
          <div class="modal-head"><div><h2 id="picker-title">Ordner wählen</h2><p id="picker-path" class="picker-path">—</p></div><button class="icon-button" type="button" data-action="close-modal" aria-label="Dialog schließen">${icon("close")}</button></div>
          <div class="modal-body">
            <div class="picker-shortcuts" id="picker-shortcuts"></div>
            <div class="picker-list" id="picker-list" role="list"><div class="picker-empty">Ordner werden geladen …</div></div>
            <div class="modal-actions">
              <button class="button" type="button" data-action="close-modal">Abbrechen</button>
              <button class="button" type="button" id="picker-up" data-action="picker-up">Übergeordneter Ordner</button>
              <button class="button primary" type="button" id="picker-choose" data-action="picker-choose" data-id="${escapeHtml(projectId)}">Diesen Ordner verwenden</button>
            </div>
          </div>
        </section>
      </div>`);
    await browseTo("");
  }

  let pickerPath = "";
  let pickerParent = "";

  async function browseTo(path) {
    try {
      const result = await api(`/api/browse?path=${encodeURIComponent(path || pickerPath)}`);
      pickerPath = result.path;
      pickerParent = result.parent || "";
      const list = document.getElementById("picker-list");
      const pathLabel = document.getElementById("picker-path");
      const shortcuts = document.getElementById("picker-shortcuts");
      if (pathLabel) pathLabel.textContent = result.path;
      if (shortcuts) {
        shortcuts.innerHTML = (result.shortcuts || [])
          .map((item) => `<button class="filter-pill" type="button" data-action="picker-goto" data-path="${escapeHtml(item.path)}">${escapeHtml(item.label)}</button>`)
          .join("");
      }
      if (list) {
        list.innerHTML = (result.directories || []).length
          ? result.directories.map((item) => `<button class="picker-row" type="button" role="listitem" data-action="picker-goto" data-path="${escapeHtml(item.path)}">${icon("folder")}<b>${escapeHtml(item.name)}</b>${icon("arrow")}</button>`).join("")
          : `<div class="picker-empty">Keine Unterordner.</div>`;
      }
    } catch (error) {
      showToast(error.message || "Ordner konnten nicht gelesen werden.", true);
    }
  }

  async function commandData(query = "") {
    const commands = [];
    if (!state.readOnly) {
      commands.push({ kind: "action", id: "connect-workspace", title: "Projektordner verbinden", detail: "Vorhandene Ordner einlesen", icon: "folder", terms: "desktop ordner verbinden sync einlesen" });
      if (state.workspaceConfigured) commands.push({ kind: "action", id: "new-project", title: "Eintrag hinzufügen", detail: "Idee oder Projekt erfassen", icon: "plus", terms: "create new idee hinzufügen" });
      commands.push({ kind: "action", id: "export", title: "Bibliothek exportieren", detail: "JSON-Sicherung herunterladen", icon: "download", terms: "backup sichern export" });
      commands.push({ kind: "action", id: "import", title: "Sicherung importieren", detail: "JSON-Bibliothek wiederherstellen", icon: "upload", terms: "restore import" });
      commands.push({ kind: "action", id: "backups", title: "Sicherungen verwalten", detail: "Alte Stände wiederherstellen", icon: "history", terms: "backup sicherung restore" });
    }
    commands.push({ kind: "action", id: "toggle-layout", title: state.layout === "grid" ? "Listenansicht" : "Rasteransicht", detail: "Ansicht der Bibliothek wechseln", icon: state.layout === "grid" ? "list" : "grid", terms: "ansicht liste raster layout" });
    const normalized = query.trim().toLowerCase();
    const matchingCommands = commands.filter((item) => `${item.title} ${item.detail} ${item.terms}`.toLowerCase().includes(normalized));
    const favorites = state.projects.filter((project) => project.favorite);
    const pool = normalized ? state.projects : (favorites.length ? favorites : state.projects);
    const matchingProjects = pool
      .filter((project) => !normalized || `${project.title} ${project.description} ${project.category}`.toLowerCase().includes(normalized))
      .slice(0, normalized ? 8 : 5);
    return { commands: matchingCommands, projects: matchingProjects };
  }

  function renderCommandResults(query = "") {
    const results = document.getElementById("command-results");
    if (!results) return;
    const { commands, projects: matches } = commandData(query);
    const commandRows = commands.map((item) => `<button class="command-item" type="button" data-action="command-run" data-kind="${item.kind}" data-id="${escapeHtml(item.id)}"><span class="command-icon">${icon(item.icon)}</span><span class="command-copy"><b>${escapeHtml(item.title)}</b><small>${escapeHtml(item.detail)}</small></span><kbd>↵</kbd></button>`).join("");
    const projectRows = matches.map((project) => `<button class="command-item" type="button" data-action="command-run" data-kind="project" data-id="${escapeHtml(project.id)}"><span class="command-icon ${escapeHtml(project.color || "lime")}">${icon(project.icon)}</span><span class="command-copy"><b>${escapeHtml(project.title)}</b><small>${escapeHtml(FOLDER_LABELS[folderState(project)])}</small></span><kbd>↵</kbd></button>`).join("");
    results.innerHTML = `${commandRows ? `<div class="command-group-label">AKTIONEN</div>${commandRows}` : ""}${projectRows ? `<div class="command-group-label">PROJEKTE</div>${projectRows}` : ""}${!commandRows && !projectRows ? `<div class="command-empty">Keine passenden Aktionen oder Projekte.</div>` : ""}`;
    results.querySelector(".command-item")?.classList.add("is-selected");
  }

  function openCommandPalette() {
    if (state.loading || modalRoot.firstElementChild) return;
    openModal(`<div class="modal-backdrop command-backdrop" data-action="backdrop-close"><section class="command-modal" role="dialog" aria-modal="true" aria-labelledby="command-title"><h2 class="sr-only" id="command-title">Schnellaktionen</h2><label class="command-input-row">${icon("search")}<input id="command-search" type="search" autocomplete="off" placeholder="Projekt oder Aktion suchen…" aria-label="Schnellaktionen suchen"><kbd>ESC</kbd></label><div class="command-results" id="command-results"></div><div class="command-footer"><span><kbd>↑</kbd><kbd>↓</kbd> navigieren</span><span><kbd>↵</kbd> auswählen</span><span><kbd>Esc</kbd> schließen</span></div></section></div>`);
    const input = document.getElementById("command-search");
    input.addEventListener("input", () => renderCommandResults(input.value));
    renderCommandResults("");
    input.focus();
  }

  async function runCommand(kind, id) {
    closeModal();
    if (kind === "project") {
      openProject(id);
    } else if (id === "connect-workspace") {
      document.getElementById("workspace-path")?.focus();
      document.getElementById("workspace-path")?.scrollIntoView({ behavior: "smooth", block: "center" });
    } else if (id === "new-project") {
      openNewProjectModal();
    } else if (id === "export") {
      exportLibrary();
    } else if (id === "import") {
      importInput.click();
    } else if (id === "backups") {
      await openBackupsModal();
    } else if (id === "toggle-layout") {
      state.layout = state.layout === "grid" ? "list" : "grid";
      savePreference("control-center-layout", state.layout);
      render();
    }
  }

  async function openBackupsModal() {
    if (state.readOnly) return showToast("Sicherungen sind nur auf dem Server-PC verfügbar.", true);
    let backups = [];
    try {
      const result = await api("/api/backups");
      backups = result.backups || [];
    } catch (error) {
      return showToast(error.message, true);
    }
    const rows = backups.length
      ? backups.map((item) => `<div class="backup-row"><div><b>${escapeHtml(formatTimestamp(new Date(item.created).getTime()))}</b><small>${item.entries} Einträge · ${Math.max(1, Math.round(item.size / 1024))} KB</small></div><button class="button small" type="button" data-action="restore-backup" data-name="${escapeHtml(item.name)}">Wiederherstellen</button></div>`).join("")
      : `<div class="picker-empty">Noch keine Sicherung vorhanden. Vor jedem Import wird automatisch eine angelegt.</div>`;
    openModal(`
      <div class="modal-backdrop" data-action="backdrop-close">
        <section class="modal" role="dialog" aria-modal="true" aria-labelledby="backup-title">
          <div class="modal-head"><div><h2 id="backup-title">Sicherungen</h2><p>Die letzten zehn Stände deiner Bibliothek.</p></div><button class="icon-button" type="button" data-action="close-modal" aria-label="Dialog schließen">${icon("close")}</button></div>
          <div class="modal-body">
            <div class="backup-list">${rows}</div>
            <div class="modal-actions">
              <button class="button" type="button" data-action="close-modal">Schließen</button>
              <button class="button primary" type="button" data-action="create-backup">${icon("plus")} Jetzt sichern</button>
            </div>
          </div>
        </section>
      </div>`);
  }

  function showAbout() {
    openModal(`
      <div class="modal-backdrop" data-action="backdrop-close"><section class="modal" role="dialog" aria-modal="true" aria-labelledby="about-title">
        <div class="modal-head"><div><h2 id="about-title">Control Center</h2><p>Lokales Projekt-Hub — keine Cloud.</p></div><button class="icon-button" type="button" data-action="close-modal" aria-label="Dialog schließen">${icon("close")}</button></div>
        <div class="modal-body">
          <div class="preview-note">${icon("info")}<span>Control Center zeigt deine vorhandenen Projekte aus dem lokalen Hauptordner. Beim Öffnen lädt es den echten Projektinhalt in einem neuen Tab; Dateien und Ordner bleiben auf deinem PC.</span></div>
          <p class="form-hint">Kürzel: <b>Ctrl/⌘ + K</b> Schnellaktionen, <b>/</b> Suche, <b>Esc</b> schließen.</p>
          <div class="modal-actions"><button class="button primary" type="button" data-action="close-modal">Verstanden</button></div>
        </div>
      </section></div>`);
  }

  function startDownload(filename, text) {
    const blob = new Blob([text], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function exportLibrary() {
    const exportable = state.projects.map((project) => {
      const { folderPath, folderAvailable, hasFolder, lastOpened, folderSignature, ...portable } = project;
      return { ...portable, folderPath: null, folderSignature: null, hasFolder: false, lastOpened: null, isDemo: false };
    });
    const data = { format: "control-center-library", version: 1, exportedAt: new Date().toISOString(), projects: exportable };
    const date = new Date().toISOString().slice(0, 10);
    startDownload(`control_center_library_${date}.json`, `${JSON.stringify(data, null, 2)}\n`);
    showToast("Bibliothek exportiert. Arbeitsordner sind nicht enthalten.");
  }

  async function importLibrary(file) {
    try {
      const parsed = JSON.parse(await file.text());
      if (parsed?.format !== "control-center-library" || !Array.isArray(parsed.projects)) {
        throw new Error("Die Datei ist keine gültige Control-Center-Sicherung.");
      }
      if (!window.confirm(`Die Sicherung enthält ${parsed.projects.length} Einträge und ersetzt die aktuelle Bibliothek. Vorhandene Projektordner werden nicht gelöscht; die aktuelle Bibliothek wird vor dem Ersetzen automatisch gesichert. Fortfahren?`)) return;
      const result = await api("/api/projects/import", { method: "POST", body: JSON.stringify(parsed) });
      state.projects = visibleProjects(result.projects || []);
      let folderMessage = "";
      if (state.workspace && !state.readOnly) {
        try {
          const scanned = await api("/api/workspace/scan", { method: "POST", body: "{}" });
          state.projects = visibleProjects(scanned.projects || []);
          folderMessage = ` ${scanned.linked} Ordner erneut verbunden.`;
          (scanned.warnings || []).forEach((warning) => showToast(warning, true));
        } catch (_) {
          folderMessage = " Verbinde den Projektordner oben erneut.";
        }
      }
      state.filter = "all";
      state.query = "";
      searchInput.value = "";
      render();
      showToast(`${result.imported} Einträge importiert. ${result.backupCreated ? "Vorherige Bibliothek gesichert." : "Keine vorherige Bibliothek vorhanden."}${folderMessage}`);
    } catch (error) {
      showToast(error.message || "Import fehlgeschlagen.", true);
    } finally {
      importInput.value = "";
    }
  }

  function projectUrlFor(id) {
    const port = window.location.port;
    const path = `/project/${encodeURIComponent(id)}/`;
    return port ? `${window.location.protocol}//${window.location.hostname}:${port}${path}` : `${window.location.origin}${path}`;
  }

  function openProject(id) {
    const project = state.projects.find((item) => item.id === id);
    if (!project) return;
    if (state.readOnly) return showToast("Projektdateien kannst du nur lokal auf dem PC öffnen, auf dem sie liegen.", true);
    if (folderState(project) !== "linked") return showToast("Verbinde zuerst den Hauptordner mit deinen Projektordnern.", true);
    const projectTab = window.open(projectUrlFor(id), "_blank");
    if (!projectTab) {
      showToast("Der Browser hat den neuen Tab blockiert. Erlaube Pop-ups für Control Center und versuche es erneut.", true);
      return;
    }
    try { projectTab.opener = null; } catch (_) { /* tab isolation is best-effort */ }
  }

  async function refreshWorkspaceProjects(options = {}) {
    const result = await api("/api/workspace/scan", { method: "POST", body: "{}" });
    state.projects = visibleProjects(result.projects || []);
    render();
    (result.warnings || []).forEach((warning) => showToast(warning, true));
    if (!options.silent) {
      showToast(`${result.added} neue Ordner aufgenommen; ${result.linked} Verknüpfungen aktualisiert.`);
    }
  }

  async function saveWorkspacePath(form) {
    if (state.readOnly) return;
    const formData = new FormData(form);
    const path = String(formData.get("path") || "").trim();
    if (!path) return;
    try {
      const result = await api("/api/workspace", { method: "POST", body: JSON.stringify({ path }) });
      state.workspace = result.workspace;
      state.workspaceConfigured = true;
      (result.warnings || []).forEach((warning) => showToast(warning, true));
      await refreshWorkspaceProjects();
    } catch (error) { showToast(error.message, true); }
  }

  async function chooseWorkspace() {
    if (state.readOnly) return;
    try {
      const result = await api("/api/workspace/select", { method: "POST", body: "{}" });
      if (result.cancelled) return;
      state.workspace = result.workspace;
      state.workspaceConfigured = true;
      (result.warnings || []).forEach((warning) => showToast(warning, true));
      await refreshWorkspaceProjects();
    } catch (error) {
      showToast(`${error.message} Öffne stattdessen den eingebauten Ordnerbrowser.`, true);
      await openFolderPicker("");
    }
  }

  async function createProject(form) {
    if (!state.workspaceConfigured) return showToast("Verbinde zuerst deinen Desktop-Hauptordner.", true);
    const formData = new FormData(form);
    const payload = {
      title: String(formData.get("name") || "").trim(),
      description: String(formData.get("description") || "").trim(),
      category: String(formData.get("category") || "Allgemein"),
      createFolder: formData.get("createFolder") === "on",
    };
    if (!payload.title) return;
    try {
      const response = await api("/api/projects", { method: "POST", body: JSON.stringify(payload) });
      state.projects.unshift(response.project);
      state.filter = "all";
      state.query = "";
      searchInput.value = "";
      closeModal();
      render();
      showToast(payload.createFolder ? "Projekt und Arbeitsordner angelegt." : "Projekt zur Bibliothek hinzugefügt.");
    } catch (error) { showToast(error.message, true); }
  }

  async function saveProject(form, id) {
    const formData = new FormData(form);
    const payload = {
      title: String(formData.get("title") || "").trim(),
      description: String(formData.get("description") || "").trim(),
      category: String(formData.get("category") || "").trim(),
      status: String(formData.get("status") || "idea"),
      notes: String(formData.get("notes") || ""),
      icon: String(formData.get("icon") || "hub"),
      color: String(formData.get("color") || "lime"),
      favorite: formData.get("favorite") === "on",
      tags: String(formData.get("tags") || "").split(",").map((tag) => tag.trim()).filter(Boolean).slice(0, 10),
      nextSteps: String(formData.get("nextSteps") || "").split("\n").map((step) => step.trim()).filter(Boolean).slice(0, 30),
    };
    if (!payload.title) return showToast("Der Projektname darf nicht leer sein.", true);
    try {
      const response = await api(`/api/projects/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(payload) });
      state.projects = state.projects.map((item) => (item.id === id ? response.project : item));
      closeModal();
      render();
      showToast("Eintrag gespeichert.");
    } catch (error) { showToast(error.message, true); }
  }

  function replaceProject(project) {
    state.projects = state.projects.map((item) => (item.id === project.id ? project : item));
    render();
  }

  async function toggleFavorite(id) {
    const project = state.projects.find((item) => item.id === id);
    if (!project) return;
    try {
      const response = await api(`/api/projects/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify({ favorite: !project.favorite }) });
      replaceProject(response.project);
    } catch (error) { showToast(error.message, true); }
  }

  async function deleteProject(id) {
    const project = state.projects.find((item) => item.id === id);
    if (!project) return;
    if (!window.confirm(`„${project.title}“ aus der Bibliothek löschen? Der Projektordner auf der Festplatte bleibt unberührt.`)) return;
    try {
      await api(`/api/projects/${encodeURIComponent(id)}`, { method: "DELETE" });
      state.projects = state.projects.filter((item) => item.id !== id);
      closeModal();
      render();
      showToast(`„${project.title}“ gelöscht. Der Ordner bleibt erhalten.`);
    } catch (error) { showToast(error.message, true); }
  }

  async function ensureFolder(id) {
    try {
      const response = await api(`/api/projects/${encodeURIComponent(id)}/folder`, { method: "POST", body: "{}" });
      replaceProject(response.project);
      showToast("Projektordner ist bereit.");
    } catch (error) { showToast(error.message, true); }
  }

  async function openInFileManager(id) {
    try {
      const response = await api(`/api/projects/${encodeURIComponent(id)}/open`, { method: "POST", body: "{}" });
      replaceProject(response.project);
      showToast("Ordner im Dateimanager geöffnet.");
    } catch (error) { showToast(error.message, true); }
  }

  async function detachFolder(id) {
    try {
      const response = await api(`/api/projects/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify({ folderPath: null }) });
      replaceProject(response.project);
      showToast("Verknüpfung gelöst. Der Ordner bleibt auf der Festplatte.");
    } catch (error) { showToast(error.message, true); }
  }

  async function assignFolder(id, path, closeafter = true) {
    try {
      const response = await api(`/api/projects/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify({ folderPath: path }) });
      replaceProject(response.project);
      if (closeafter) closeModal();
      showToast("Projektordner verknüpft.");
    } catch (error) { showToast(error.message, true); }
  }

  async function createBackup() {
    try {
      const result = await api("/api/backups", { method: "POST", body: "{}" });
      await openBackupsModal();
      showToast(result.backupCreated ? "Sicherung erstellt." : "Noch keine Bibliothek zum Sichern vorhanden.");
    } catch (error) { showToast(error.message, true); }
  }

  async function restoreBackup(name) {
    if (!window.confirm("Die aktuelle Bibliothek wird durch diese Sicherung ersetzt. Vorher wird automatisch eine Sicherung des jetzigen Stands erstellt. Fortfahren?")) return;
    try {
      const result = await api("/api/backups/restore", { method: "POST", body: JSON.stringify({ backup: name }) });
      state.projects = visibleProjects(result.projects || []);
      closeModal();
      render();
      showToast("Sicherung wiederhergestellt.");
    } catch (error) { showToast(error.message, true); }
  }

  document.addEventListener("click", async (event) => {
    const viewButton = event.target.closest("[data-view]");
    if (viewButton) {
      state.filter = "all";
      state.favoritesOnly = false;
      render();
      return;
    }
    const filterButton = event.target.closest("[data-filter]");
    if (filterButton) {
      state.filter = filterButton.dataset.filter;
      state.favoritesOnly = state.filter === "favorites";
      render();
      return;
    }
    const layoutButton = event.target.closest("[data-layout]");
    if (layoutButton) {
      state.layout = layoutButton.dataset.layout;
      savePreference("control-center-layout", state.layout);
      render();
      return;
    }

    const button = event.target.closest("[data-action]");
    if (!button) return;
    const { action, id } = button.dataset;
    if (action === "new-project") openNewProjectModal();
    else if (action === "edit-project") openEditModal(id);
    else if (action === "delete-project") await deleteProject(id);
    else if (action === "toggle-favorite") await toggleFavorite(id);
    else if (action === "ensure-folder") await ensureFolder(id);
    else if (action === "system-open") await openInFileManager(id);
    else if (action === "detach-folder") await detachFolder(id);
    else if (action === "pick-folder") await openFolderPicker(id);
    else if (action === "backups") await openBackupsModal();
    else if (action === "create-backup") await createBackup();
    else if (action === "restore-backup") await restoreBackup(button.dataset.name);
    else if (action === "picker-goto") await browseTo(button.dataset.path);
    else if (action === "picker-up") { if (pickerParent) await browseTo(pickerParent); }
    else if (action === "picker-choose") {
      if (button.dataset.id) await assignFolder(button.dataset.id, pickerPath);
      else await saveWorkspacePathFromPicker();
    }
    else if (action === "open-command") openCommandPalette();
    else if (action === "command-run") await runCommand(button.dataset.kind, id);
    else if (action === "show-about") showAbout();
    else if (action === "close-modal") closeModal();
    else if (action === "backdrop-close" && event.target.classList.contains("modal-backdrop")) closeModal();
    else if (action === "open-project") openProject(id);
    else if (action === "choose-workspace") await chooseWorkspace();
    else if (action === "export") exportLibrary();
    else if (action === "import") importInput.click();
    else if (action === "retry") await loadLibrary();
  });

  async function saveWorkspacePathFromPicker() {
    try {
      const result = await api("/api/workspace", { method: "POST", body: JSON.stringify({ path: pickerPath }) });
      state.workspace = result.workspace;
      state.workspaceConfigured = true;
      (result.warnings || []).forEach((warning) => showToast(warning, true));
      closeModal();
      await refreshWorkspaceProjects();
    } catch (error) { showToast(error.message, true); }
  }

  document.addEventListener("submit", async (event) => {
    if (event.target.id === "new-project-form") {
      event.preventDefault();
      await createProject(event.target);
    } else if (event.target.id === "workspace-form") {
      event.preventDefault();
      await saveWorkspacePath(event.target);
    } else if (event.target.id === "edit-project-form") {
      event.preventDefault();
      const id = event.target.querySelector('[data-action="pick-folder"]')?.dataset.id;
      if (id) await saveProject(event.target, id);
    }
  });

  document.addEventListener("change", (event) => {
    if (event.target.id === "sort-select") {
      state.sort = event.target.value;
      savePreference("control-center-sort", state.sort);
      render();
    }
  });

  let searchTimer = 0;
  searchInput.addEventListener("input", () => {
    window.clearTimeout(searchTimer);
    const value = searchInput.value;
    searchTimer = window.setTimeout(() => {
      state.query = value;
      render();
    }, 140);
  });
  importInput.addEventListener("change", () => { if (importInput.files?.[0]) importLibrary(importInput.files[0]); });

  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); openCommandPalette(); return; }
    const palette = document.getElementById("command-search");
    if (palette && ["ArrowDown", "ArrowUp"].includes(event.key)) {
      const options = [...document.querySelectorAll("#command-results .command-item")];
      if (options.length) {
        event.preventDefault();
        const current = options.findIndex((option) => option.classList.contains("is-selected"));
        const offset = event.key === "ArrowDown" ? 1 : -1;
        options.forEach((option) => option.classList.remove("is-selected"));
        options[(current + offset + options.length) % options.length].classList.add("is-selected");
      }
    } else if (palette && event.key === "Enter") {
      const selected = document.querySelector("#command-results .command-item.is-selected");
      if (selected) { event.preventDefault(); selected.click(); }
    } else if (event.key === "Escape") {
      if (modalRoot.firstElementChild) closeModal();
      else if (document.activeElement === searchInput && searchInput.value) {
        searchInput.value = "";
        state.query = "";
        render();
      }
    } else if (event.key === "/" && document.activeElement !== searchInput && !modalRoot.firstElementChild && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) {
      event.preventDefault();
      searchInput.focus();
    } else if (event.key === "Tab" && modalRoot.firstElementChild) {
      const focusable = [...modalRoot.querySelectorAll("button, input, select, textarea, [href]")].filter((item) => !item.disabled && item.offsetParent !== null);
      if (focusable.length) {
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
    }
  });

  state.layout = loadPreference("control-center-layout", "grid") === "list" ? "list" : "grid";
  state.sort = SORT_OPTIONS.some(([key]) => key === loadPreference("control-center-sort", "updated")) ? loadPreference("control-center-sort", "updated") : "updated";
  loadLibrary();
})();
