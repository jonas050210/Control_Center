/* Control Center: local project library, notes, folders, and JSON backup. */
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
    lock: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 5v2"/></svg>',
  };

  const state = { projects: [], query: "", filter: "all", layout: "grid", readOnly: false, workspace: "", workspaceConfigured: false, loading: true, error: "" };

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  }

  function icon(name) { return ICONS[name] || ICONS.hub; }

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
        try {
          const legacy = localStorage.getItem("control-center-projects-v1");
          const oldProjects = legacy ? JSON.parse(legacy) : null;
          if (Array.isArray(oldProjects) && oldProjects.length) {
            await api("/api/projects/import", {
              method: "POST",
              body: JSON.stringify({ format: "control-center-library", version: 1, projects: oldProjects }),
            });
            localStorage.setItem("control-center-projects-v1-backup", legacy);
            localStorage.removeItem("control-center-projects-v1");
          }
        } catch (migrationError) {
          console.warn("Alte Browser-Bibliothek wurde nicht automatisch übernommen.", migrationError);
        }
      }
      const result = await api("/api/projects");
      state.projects = Array.isArray(result.projects) ? result.projects.filter((project) => !project.isDemo || project.folderPath || project.hasFolder) : [];
    } catch (error) {
      state.error = error.message || "Control Center konnte die lokale Bibliothek nicht laden.";
    } finally {
      state.loading = false;
      render();
    }
  }

  function statusClass(project) {
    return ["idea", "project", "prototype", "reference", "active", "paused", "archived", "template"].includes(project.status) ? project.status : "prototype";
  }

  function projectCard(project) {
    const tags = Array.isArray(project.tags) ? project.tags : [];
    const blob = [project.title, project.description, project.category, ...tags].join(" ").toLowerCase();
    const hasFolder = Boolean(project.folderPath || project.hasFolder);
    const openLabel = state.readOnly ? "Nur am eigenen PC" : hasFolder ? "Projekt öffnen" : "Ordner fehlt";
    return `
      <article class="project-card" data-kind="${escapeHtml(project.status || "prototype")}" data-search="${escapeHtml(blob)}">
        <div class="project-top">
          <div class="project-icon ${escapeHtml(project.color || "lime")}">${icon(project.icon)}</div>
          <span class="project-state ${statusClass(project)}">${escapeHtml(project.statusLabel || "Projekt")}</span>
        </div>
        <h3 class="project-title">${escapeHtml(project.title)}</h3>
        <p class="project-description">${escapeHtml(project.description || (hasFolder ? "Der Projektordner enthält die Dateien dieses Projekts." : "Projektordner auswählen, um den echten Inhalt zu öffnen."))}</p>
        <div class="project-tags">${tags.map((tag) => `<span class="project-tag">${escapeHtml(tag)}</span>`).join("")}</div>
        <div class="project-footer">
          <span class="project-updated">${icon("folder")} ${escapeHtml(hasFolder ? "Projektordner verbunden" : "Noch keinem Ordner zugeordnet")}</span>
          <button class="project-open" type="button" data-action="open-project" data-id="${escapeHtml(project.id)}" ${state.readOnly || !hasFolder ? "disabled" : ""}>${escapeHtml(openLabel)} ${icon("arrow")}</button>
        </div>
      </article>`;
  }

  function pageHeading(eyebrow, title, description, action = "") {
    return `<div class="page-heading"><div><div class="eyebrow">${escapeHtml(eyebrow)}</div><h1>${escapeHtml(title)}</h1><p>${escapeHtml(description)}</p></div>${action}</div>`;
  }

  function writeActions() {
    if (state.readOnly) return `<span class="readonly-chip">${icon("lock")} Ordnerzugriff nur lokal am PC</span>`;
    return `<div class="heading-actions"><button class="button small" type="button" data-action="import">${icon("upload")} Import</button><button class="button small" type="button" data-action="export">${icon("download")} Export</button><button class="button primary" type="button" data-action="new-project" ${state.workspaceConfigured ? "" : "disabled title=\"Verbinde zuerst deinen Desktop-Projektordner\""}>${icon("plus")} Neues Projekt</button></div>`;
  }

  function workspacePanel() {
    const workspace = state.workspace || "";
    return `
      <section class="workspace-panel">
        <div class="workspace-panel-heading"><div class="workspace-panel-icon">${icon("folder")}</div><div><span class="resume-kicker">DEIN PROJEKTORDNER</span><h2>Ordner mit deinen Projekten verbinden</h2><p>Wähle den Ordner auf deinem Desktop, in dem die einzelnen Projektordner liegen. Die vorhandenen Dateien werden nicht verändert.</p></div></div>
        ${state.readOnly ? `<div class="workspace-readonly-note">Die Arena-Vorschau kann nicht auf deinen Desktop zugreifen. Starte Control Center auf deinem PC, um den lokalen Projektordner zu verbinden.</div>` : `<form id="workspace-form" class="workspace-form"><label for="workspace-path">Hauptordner</label><div class="workspace-input-row"><input id="workspace-path" name="path" type="text" value="${escapeHtml(workspace)}" placeholder="z. B. C:\\Users\\Du\\Desktop\\Meine Projekte" required><button class="button" type="button" data-action="choose-workspace">${icon("folder")} Ordner wählen</button><button class="button primary" type="submit">Verbinden &amp; einlesen ${icon("arrow")}</button></div><small>Jeder direkte Unterordner wird als ein vorhandenes Projekt erkannt. Es werden keine neuen Projektdateien erstellt.</small></form>`}
      </section>`;
  }

  function statsPanel() {
    const ideaCount = state.projects.filter((project) => project.status === "idea").length;
    const folderCount = state.projects.filter((project) => project.folderPath || project.hasFolder).length;
    return `
      <div class="overview-stats library-stats" aria-label="Bibliotheksübersicht">
        <div class="overview-stat"><div class="stat-icon">${icon("folder")}</div><div class="stat-copy"><strong>${state.projects.length}</strong><span>Projekte in der Bibliothek</span></div></div>
        <div class="overview-stat"><div class="stat-icon violet">${icon("spark")}</div><div class="stat-copy"><strong>${ideaCount}</strong><span>Ideen</span></div></div>
        <div class="overview-stat"><div class="stat-icon blue">${icon("check")}</div><div class="stat-copy"><strong>${folderCount}</strong><span>Verbunden mit Projektordner</span></div></div>
      </div>`;
  }

  function filterProjects(items) {
    let filtered = [...items];
    if (state.filter === "ideas") filtered = filtered.filter((project) => project.status === "idea");
    if (state.filter === "projects") filtered = filtered.filter((project) => project.status !== "idea");
    if (state.filter === "missing") filtered = filtered.filter((project) => !project.folderPath && !project.hasFolder);
    if (state.query.trim()) {
      const query = state.query.trim().toLowerCase();
      filtered = filtered.filter((project) => [project.title, project.description, project.category, ...(project.tags || [])].join(" ").toLowerCase().includes(query));
    }
    return filtered;
  }

  function renderEmpty(title, description, action = "") {
    return `<div class="empty-state"><div class="empty-icon">${icon("folder")}</div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(description)}</p>${action}</div>`;
  }

  function renderLibrary() {
    const items = filterProjects(state.projects);
    const filters = `<div class="filter-pills" role="group" aria-label="Bibliothek filtern"><button class="filter-pill ${state.filter === "all" ? "active" : ""}" type="button" data-filter="all">Alle</button><button class="filter-pill ${state.filter === "projects" ? "active" : ""}" type="button" data-filter="projects">Projekte</button><button class="filter-pill ${state.filter === "ideas" ? "active" : ""}" type="button" data-filter="ideas">Ideen</button><button class="filter-pill ${state.filter === "missing" ? "active" : ""}" type="button" data-filter="missing">Ordner fehlen</button></div>`;
    const emptyAction = state.readOnly ? "" : state.workspaceConfigured ? `<button class="button small" type="button" data-action="new-project">${icon("plus")} Eintrag hinzufügen</button>` : `<button class="button small" type="button" data-action="choose-workspace">${icon("folder")} Desktop-Ordner wählen</button>`;
    const emptyTitle = state.query ? "Keine Treffer" : state.filter === "ideas" ? "Noch keine Ideen" : state.filter === "missing" ? "Keine nicht zugeordneten Projekte" : "Bibliothek ist leer";
    const emptyDescription = state.query ? "Passe die Suche an oder verbinde den Ordner mit deinen vorhandenen Projekten." : state.filter === "missing" ? "Alle sichtbaren Einträge sind verbunden. Weitere Ordner kannst du oben einlesen." : "Verbinde oben deinen Projektordner oder füge einen Bibliothekseintrag hinzu.";
    return `
      ${pageHeading("WORKSPACE", "Meine Bibliothek", "Hier sind deine Projekte und Ideen. Ein Projekt öffnen zeigt seinen echten Ordner in einem neuen Browser-Tab.", writeActions())}
      ${workspacePanel()}
      ${statsPanel()}
      <section class="section">
        <div class="section-head"><div><div class="section-title"><h2>Projekte und Ideen</h2><span>${items.length} ${items.length === 1 ? "Eintrag" : "Einträge"}</span></div><p class="section-description">Projektkarten öffnen den echten Inhalt des zugeordneten Ordners.</p></div><div class="section-tools">${filters}<div class="view-toggle" role="group" aria-label="Ansicht wechseln"><button type="button" class="${state.layout === "grid" ? "active" : ""}" data-layout="grid" aria-label="Rasteransicht" title="Rasteransicht">${icon("grid")}</button><button type="button" class="${state.layout === "list" ? "active" : ""}" data-layout="list" aria-label="Listenansicht" title="Listenansicht">${icon("list")}</button></div></div>
        </div>
        ${items.length ? `<div class="project-grid ${state.layout === "list" ? "list-mode" : ""}">${items.map(projectCard).join("")}</div>` : renderEmpty(emptyTitle, emptyDescription, emptyAction)}
      </section>
      <div class="preview-note">${icon("info")}<span><b>Projekt öffnen:</b> Wenn ein Projektordner eine <code>index.html</code> enthält, wird die Projektseite geladen. Sonst öffnet sich eine Dateiansicht des Ordners. Die Arena-Vorschau selbst hat keinen Zugriff auf deinen Desktop.</span></div>`;
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
      localPill.querySelector("b").textContent = state.readOnly ? "Desktop nicht verfügbar" : "Lokale Projektbibliothek";
      localPill.querySelector("small").textContent = state.readOnly ? "Starte die App auf deinem PC" : "Projektdateien bleiben bei dir";
    }
    page.innerHTML = renderPage();
  }

  function showToast(message, isError = false) {
    const toast = document.createElement("div");
    toast.className = `toast ${isError ? "error" : ""}`;
    toast.innerHTML = `${icon(isError ? "info" : "check")}<span></span>`;
    toast.querySelector("span").textContent = message;
    toastStack.appendChild(toast);
    window.setTimeout(() => toast.remove(), 3800);
  }

  function openNewProjectModal(defaults = {}) {
    if (state.readOnly) return showToast("Neue Projekte können nur direkt am Server-PC angelegt werden.", true);
    if (!state.workspaceConfigured) return showToast("Verbinde zuerst den Desktop-Hauptordner, in dem deine Projekte liegen.", true);
    modalRoot.innerHTML = `
      <div class="modal-backdrop" data-action="backdrop-close">
        <section class="modal" role="dialog" aria-modal="true" aria-labelledby="new-project-title">
          <div class="modal-head"><div><h2 id="new-project-title">Neues Projekt</h2><p>Erfasse eine Idee und lege den zugehörigen Ordner gleich mit an.</p></div><button class="icon-button" type="button" data-action="close-modal" aria-label="Dialog schließen">${icon("close")}</button></div>
          <form class="modal-body" id="new-project-form">
            <div class="form-field"><label for="project-name">Projektname</label><input id="project-name" name="name" maxlength="80" required placeholder="z. B. Neon Arena" value="${escapeHtml(defaults.title || "")}"></div>
            <div class="form-field"><label for="project-description">Kurzbeschreibung</label><textarea id="project-description" name="description" maxlength="500" placeholder="Worum geht es bei der Idee?">${escapeHtml(defaults.description || "")}</textarea></div>
            <div class="form-field"><label for="project-category">Bereich</label><select id="project-category" name="category"><option>Spielidee</option><option>KI & Experimente</option><option>Tool</option><option>Allgemein</option></select></div>
            <label class="checkbox-field"><input type="checkbox" name="createFolder" checked><span><b>Projektordner gleich mit anlegen</b><small>Legt ihn im verbundenen Hauptordner zusammen mit deinen anderen Projekten an.</small></span></label>
            <div class="modal-actions"><button class="button" type="button" data-action="close-modal">Abbrechen</button><button class="button primary" type="submit">${icon("plus")} Projekt hinzufügen</button></div>
          </form>
        </section>
      </div>`;
    document.getElementById("project-name")?.focus();
  }

  function commandData(query = "") {
    const commands = [
      ...(!state.readOnly ? [
        { kind: "action", id: "connect-workspace", title: "Desktop-Projektordner verbinden", detail: "Vorhandene Ordner einlesen", icon: "folder", terms: "desktop ordner verbinden sync einlesen" },
        ...(state.workspaceConfigured ? [{ kind: "action", id: "new-project", title: "Eintrag hinzufügen", detail: "Idee oder Projekt in der Bibliothek erfassen", icon: "plus", terms: "create new idee hinzufügen" }] : []),
        { kind: "action", id: "export", title: "Bibliothek exportieren", detail: "JSON-Sicherung herunterladen", icon: "download", terms: "backup sichern export" },
        { kind: "action", id: "import", title: "Sicherung importieren", detail: "JSON-Bibliothek wiederherstellen", icon: "upload", terms: "restore import" },
      ] : []),
    ];
    const normalized = query.trim().toLowerCase();
    const matchingCommands = commands.filter((item) => `${item.title} ${item.detail} ${item.terms}`.toLowerCase().includes(normalized));
    const matchingProjects = state.projects.filter((project) => !normalized || `${project.title} ${project.description} ${project.category}`.toLowerCase().includes(normalized)).slice(0, normalized ? 8 : 4);
    return { commands: matchingCommands, projects: matchingProjects };
  }

  function renderCommandResults(query = "") {
    const results = document.getElementById("command-results");
    if (!results) return;
    const { commands, projects: matches } = commandData(query);
    const commandRows = commands.map((item) => `<button class="command-item" type="button" data-action="command-run" data-kind="${item.kind}" data-id="${escapeHtml(item.id)}"><span class="command-icon">${icon(item.icon)}</span><span class="command-copy"><b>${escapeHtml(item.title)}</b><small>${escapeHtml(item.detail)}</small></span><kbd>↵</kbd></button>`).join("");
    const projectRows = matches.map((project) => `<button class="command-item" type="button" data-action="command-run" data-kind="project" data-id="${escapeHtml(project.id)}"><span class="command-icon ${escapeHtml(project.color || "lime")}">${icon(project.icon)}</span><span class="command-copy"><b>${escapeHtml(project.title)}</b><small>${escapeHtml(project.folderPath || project.hasFolder ? "Projektordner verbunden" : "Ordner fehlt")}</small></span><kbd>↵</kbd></button>`).join("");
    results.innerHTML = `${commandRows ? `<div class="command-group-label">AKTIONEN</div>${commandRows}` : ""}${projectRows ? `<div class="command-group-label">PROJEKTE</div>${projectRows}` : ""}${!commandRows && !projectRows ? `<div class="command-empty">Keine passenden Aktionen oder Projekte.</div>` : ""}`;
    results.querySelector(".command-item")?.classList.add("is-selected");
  }

  function openCommandPalette() {
    if (state.loading || modalRoot.firstElementChild) return;
    modalRoot.innerHTML = `<div class="modal-backdrop command-backdrop" data-action="backdrop-close"><section class="command-modal" role="dialog" aria-modal="true" aria-labelledby="command-title"><h2 class="sr-only" id="command-title">Schnellaktionen</h2><label class="command-input-row">${icon("search")}<input id="command-search" type="search" autocomplete="off" placeholder="Projekt oder Aktion suchen…" aria-label="Schnellaktionen suchen"><kbd>ESC</kbd></label><div class="command-results" id="command-results"></div><div class="command-footer"><span><kbd>↑</kbd><kbd>↓</kbd> navigieren</span><span><kbd>↵</kbd> auswählen</span><span><kbd>Esc</kbd> schließen</span></div></section></div>`;
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
    }
  }

  function showAbout() {
    modalRoot.innerHTML = `
      <div class="modal-backdrop" data-action="backdrop-close"><section class="modal" role="dialog" aria-modal="true" aria-labelledby="about-title">
        <div class="modal-head"><div><h2 id="about-title">Control Center</h2><p>Lokales Projekt-Hub — keine Cloud.</p></div><button class="icon-button" type="button" data-action="close-modal" aria-label="Dialog schließen">${icon("close")}</button></div>
        <div class="modal-body"><div class="preview-note" style="margin:0">${icon("info")}<span>Control Center zeigt deine vorhandenen Projekte aus dem lokalen Hauptordner. Beim Öffnen lädt es den echten Projektinhalt in einem neuen Tab; Dateien und Ordner bleiben auf deinem PC. Die Arena-Vorschau selbst kann nicht auf deinen Desktop zugreifen.</span></div><div class="modal-actions"><button class="button primary" type="button" data-action="close-modal">Verstanden</button></div></div>
      </section></div>`;
  }

  function closeModal() { modalRoot.innerHTML = ""; }

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
      const { folderPath, hasFolder, lastOpened, ...portable } = project;
      return { ...portable, folderPath: null, hasFolder: false, lastOpened: null, isDemo: false };
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
      state.projects = (result.projects || []).filter((project) => !project.isDemo || project.folderPath || project.hasFolder);
      let folderMessage = "";
      if (state.workspace) {
        try {
          const scanned = await api("/api/workspace/scan", { method: "POST", body: "{}" });
          state.projects = (scanned.projects || []).filter((project) => !project.isDemo || project.folderPath || project.hasFolder);
          folderMessage = ` ${scanned.linked} Ordner erneut verbunden.`;
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

  function openProject(id) {
    const project = state.projects.find((item) => item.id === id);
    if (!project) return;
    if (state.readOnly) return showToast("Projektdateien kannst du nur lokal auf dem PC öffnen, auf dem sie liegen.", true);
    if (!project.folderPath && !project.hasFolder) return showToast("Verbinde zuerst den Hauptordner mit deinen Projektordnern.", true);
    const projectUrl = `http://127.0.0.1:${window.location.port}/project/${encodeURIComponent(id)}/`;
    const projectTab = window.open(projectUrl, "_blank");
    if (!projectTab) {
      showToast("Der Browser hat den neuen Tab blockiert. Erlaube Pop-ups für Control Center und versuche es erneut.", true);
      return;
    }
    try { projectTab.opener = null; } catch (_) { /* tab isolation is best-effort */ }
  }

  async function refreshWorkspaceProjects() {
    const result = await api("/api/workspace/scan", { method: "POST", body: "{}" });
    state.projects = Array.isArray(result.projects) ? result.projects.filter((project) => !project.isDemo || project.folderPath || project.hasFolder) : [];
    render();
    showToast(`${result.added} vorhandene Ordner als Projekte erkannt; ${result.linked} Einträge verbunden.`);
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
      await refreshWorkspaceProjects();
    } catch (error) {
      showToast(`${error.message} Du kannst den Pfad auch direkt in das Feld eintragen.`, true);
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

  document.addEventListener("click", async (event) => {
    const viewButton = event.target.closest("[data-view]");
    if (viewButton) {
      state.filter = "all";
      render();
      return;
    }
    const filterButton = event.target.closest("[data-filter]");
    if (filterButton) { state.filter = filterButton.dataset.filter; render(); return; }
    const layoutButton = event.target.closest("[data-layout]");
    if (layoutButton) { state.layout = layoutButton.dataset.layout; render(); return; }

    const button = event.target.closest("[data-action]");
    if (!button) return;
    const { action, id } = button.dataset;
    if (action === "new-project") openNewProjectModal();
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

  document.addEventListener("submit", async (event) => {
    if (event.target.id === "new-project-form") {
      event.preventDefault();
      await createProject(event.target);
    } else if (event.target.id === "workspace-form") {
      event.preventDefault();
      await saveWorkspacePath(event.target);
    }
  });

  searchInput.addEventListener("input", () => { state.query = searchInput.value; render(); });
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
    } else if (event.key === "Escape" && modalRoot.firstElementChild) closeModal();
  });

  loadLibrary();
})();
