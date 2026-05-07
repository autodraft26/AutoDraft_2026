const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const howToUseBtn = document.getElementById("howToUseBtn");

const serverSelect = document.getElementById("serverSelect");
const serverDropdown = document.getElementById("serverDropdown");
const serverDropdownBtn = document.getElementById("serverDropdownBtn");
const serverDropdownMenu = document.getElementById("serverDropdownMenu");
const serverModeHint = document.getElementById("serverModeHint");
const serverModelSelect = document.getElementById("serverModelSelect");
const serverSetKeyBtn = document.getElementById("serverSetKeyBtn");
const serverAddToggleBtn = document.getElementById("serverAddToggleBtn");
const floatingActionHost = document.getElementById("floatingActionHost");
const serverKeyPopup = document.getElementById("serverKeyPopup");
const closeServerKeyPopupBtn = document.getElementById("closeServerKeyPopupBtn");
const closeAddServerPopupBtn = document.getElementById("closeAddServerPopupBtn");
const addServerPanel = document.getElementById("addServerPanel");
const metricPreferenceSelect = document.getElementById("metricPreferenceSelect");
const serverQuantizationSelect = document.getElementById("serverQuantizationSelect");
const draftQuantizationSelect = document.getElementById("draftQuantizationSelect");
const probeBtn = document.getElementById("probeBtn");
const addServerBtn = document.getElementById("addServerBtn");
const selectedServerApiKey = document.getElementById("selectedServerApiKey");
const updateServerKeyBtn = document.getElementById("updateServerKeyBtn");
const newServerName = document.getElementById("newServerName");
const newServerHost = document.getElementById("newServerHost");
const newServerPort = document.getElementById("newServerPort");
const recommendationHint = document.getElementById("recommendationHint");
const jumpBestEfficiencyBtn = document.getElementById("jumpBestEfficiencyBtn");
const jumpParetoBtn = document.getElementById("jumpParetoBtn");
const jumpFastestBtn = document.getElementById("jumpFastestBtn");
const costSlider = document.getElementById("costSlider");
const costValue = document.getElementById("costValue");
const constraintSlider = document.getElementById("constraintSlider");
const constraintValue = document.getElementById("constraintValue");
const constraintHint = document.getElementById("constraintHint");
const costControlGroup = document.getElementById("costControlGroup");
const constraintTargetSelect = document.getElementById("constraintTargetSelect");
const constraintTargetControlGroup = document.getElementById("constraintTargetControlGroup");
const constraintControlGroup = document.getElementById("constraintControlGroup");
const tpsConstraintSlider = document.getElementById("tpsConstraintSlider");
const tpsConstraintValue = document.getElementById("tpsConstraintValue");
const tpsConstraintHint = document.getElementById("tpsConstraintHint");
const tpsConstraintControlGroup = document.getElementById("tpsConstraintControlGroup");
const maxTokensInput = document.getElementById("maxTokensInput");
const algorithmSelect = document.getElementById("algorithmSelect");
const draftModelSelect = document.getElementById("draftModelSelect");
const proactiveDrafting = document.getElementById("proactiveDrafting");
const modeSelect = document.getElementById("modeSelect");
const benchmarkDatasetSelect = document.getElementById("benchmarkDatasetSelect");
const objectiveModeSelect = document.getElementById("objectiveModeSelect");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const shutdownTargetBtn = document.getElementById("shutdownTargetBtn");
const sessionBadge = document.getElementById("sessionBadge");
const activityBanner = document.getElementById("activityBanner");
const inputRowEl = document.querySelector(".input-row");
const diagPhase = document.getElementById("diagPhase");
const diagChatReady = document.getElementById("diagChatReady");
const diagChatExitCode = document.getElementById("diagChatExitCode");
const diagLastLine = document.getElementById("diagLastLine");
const diagLastError = document.getElementById("diagLastError");
const treeMiniBox = document.getElementById("treeMiniBox");
const treeMiniCanvas = document.getElementById("treeMiniCanvas");
const tradeoffCanvas = document.getElementById("tradeoffCanvas");
const tradeoffHint = document.getElementById("tradeoffHint");
const refreshReferenceBtn = document.getElementById("refreshReferenceBtn");
const refreshReferenceDetailedBtn = document.getElementById("refreshReferenceDetailedBtn");
const uiHelpContent = document.getElementById("uiHelpContent");
const helpRefreshBtn = document.getElementById("helpRefreshBtn");
const fillTemplateExternalBtn = document.getElementById("fillTemplateExternalBtn");
const fillTemplateGroqBtn = document.getElementById("fillTemplateGroqBtn");
const fillTemplateHfBtn = document.getElementById("fillTemplateHfBtn");
const fillTemplateLocalBtn = document.getElementById("fillTemplateLocalBtn");
const clearTemplateBtn = document.getElementById("clearTemplateBtn");
const topControlsBox = document.querySelector(".top-controls");
const topTradeoffBox = document.querySelector(".top-tradeoff");
let pendingControl = null;
let selectedControl = null;
let pendingTimer = null;
let controlsLocked = false;
let tokenColoringEnabled = false;
let streamingAiBubble = null;
let lastTreeStats = null;
let feasibleConstraintMin = null;
let feasibleConstraintMax = null;
let feasibleTpsMin = null;
let feasibleTpsMax = null;
let tradeoffCurveBlend = null;
let tradeoffCurveConstraint = null;
let serverCandidates = [];
let serverCatalogSignature = "";
let serverProbeRows = [];
let serverProbeCurves = [];
let serverProbeCurveMode = "cost_sensitivity";
let recommendationSummary = {
  fastest: null,
  best_efficiency: null,
  pareto_optimal_ids: [],
};
let recommendationHoverKind = null;
let tradeoffHoverPointUid = null;
let tradeoffDisplayMode = "reference"; // "reference" | "probe" | "auto"
let latestChatReady = false;
let latestPhase = "idle";
let latestPhaseLabel = "idle";
let currentComposerTheme = "theme-stop";
const probeCurvePalette = [
  "#1f4f89",
  "#7c3aed",
  "#0f7a2b",
  "#b42318",
  "#0e7490",
  "#92400e",
  "#4b5563",
  "#d97706",
];

const HOW_TO_USE_HTML = `
  <div class="guide-content">
    <h3>Basic Workflow</h3>
    <ol>
      <li>Select which server to use, as well as the server model and the draft model.</li>
      <li>Choose the algorithm, mode (in most cases, select Chat), and the maximum number of new tokens.</li>
      <li>Select the objective and metric preference, then adjust the cost sensitivity.</li>
      <li>Click <strong>Start</strong> to launch the LLM.</li>
      <li>When the chat input box turns green, the system is ready. You can then continue by entering your prompt or question.</li>
    </ol>

    <hr class="guide-divider" />

    <h3>Cost–Throughput Trade-off Selection</h3>
    <ol>
      <li>
        When you click <strong>Start</strong>, a graph is displayed in the selectable trade-off panel if the corresponding reference and profile already exist.
        If they do not exist, click <strong>Profile LLM</strong> to run profiling and generate the graph.
      </li>
      <li>
        In the selectable trade-off panel, choose the point that matches your desired cost-throughput trade-off.
        You may also select one of the predefined options below:
        <ul>
          <li><strong>Best Efficiency</strong></li>
          <li><strong>Pareto-optimal</strong></li>
          <li><strong>Fastest</strong></li>
        </ul>
      </li>
    </ol>

    <hr class="guide-divider" />

    <h3>Description of Each Selection Box</h3>

    <h4>1. Algorithm</h4>
    <ul class="guide-list-detail">
      <li>
        <strong>AutoDraft</strong>
        <div class="guide-desc">
          Our proposed algorithm. It runs the draft model on the user device and the target model on the server, and dynamically adjusts tree depth, tree width, and the number of nodes passed to the target model.
        </div>
      </li>
      <li>
        <strong>Server-only SD</strong>
        <div class="guide-desc">
          A dynamic tree-based speculative decoding method in which both the draft model and the target model are executed on the server.
        </div>
      </li>
      <li>
        <strong>Server-only AR</strong>
        <div class="guide-desc">
          A standard auto-regressive decoding method in which only the target model is executed on the server.
        </div>
      </li>
      <li>
        <strong>OPT Tree</strong>
        <div class="guide-desc">
          A collaborative speculative decoding method in which the draft model runs on the user device and the target model runs on the server, while the tree structure is optimized.
        </div>
      </li>
      <li>
        <strong>Fixed Tree</strong>
        <div class="guide-desc">
          A collaborative speculative decoding method with fixed tree depth, width, and nnodes.
        </div>
      </li>
    </ul>

    <h4>2. Mode</h4>
    <ul class="guide-list-detail">
      <li>
        <strong>Chat</strong>
        <div class="guide-desc">Interactive mode like ChatGPT or Gemini.</div>
      </li>
      <li>
        <strong>Benchmark</strong>
        <div class="guide-desc">Dataset-based performance evaluation mode.</div>
      </li>
    </ul>

    <h4>3. Max New Tokens</h4>
    <ul class="guide-list-detail">
      <li>
        <strong>Warning</strong>
        <div class="guide-desc">
          If this value is too large, OOM can occur due to KV cache growth. Tune it based on the VRAM of the device running the draft model.
        </div>
      </li>
    </ul>
  </div>
`;

function resetTradeoffViewAfterStop() {
  tradeoffCurveBlend = null;
  tradeoffCurveConstraint = null;
  feasibleConstraintMin = null;
  feasibleConstraintMax = null;
  feasibleTpsMin = null;
  feasibleTpsMax = null;
  serverProbeRows = [];
  serverProbeCurves = [];
  recommendationSummary = {
    fastest: null,
    best_efficiency: null,
    pareto_optimal_ids: [],
  };
  tradeoffHoverPointUid = null;
  recommendationHoverKind = null;
  tradeoffDisplayMode = "reference";
  updateConstraintSliderVisual();
  updateRecommendationHint();
  drawTradeoffChart();
}
const lastSeriesPoint = {
  gpu_energy: null,
  draft_cost: null,
  target_cost: null,
  throughput: null,
};
const allDraftModelOptions = draftModelSelect
  ? Array.from(draftModelSelect.options).map((opt) => ({ value: opt.value, label: opt.textContent || opt.value }))
  : [];
let activePopup = null; // null | "key" | "add"

// Default value when serverSelect is missing
const getServerValue = () => serverSelect ? serverSelect.value : "RTX Pro 6000";

function getUiHelpText(mode) {
  if (mode === "benchmark") {
    return [
      "[Benchmark Mode]",
      "- Use Start/Stop to run or stop benchmark sessions.",
      "- Shutdown Target fully terminates the remote autodraft_target server process.",
      "- Compare trade-offs by changing objective mode and sliders.",
      "- Rebuild the local reference trade-off with Profile LLM.",
      "- Check runtime status by typing status or /status.",
      "",
      "[Server and Recommendation]",
      "- Select Server, Server Model, and Metric Preference, then click Profile Servers.",
      "- Server list refreshes automatically after Add Server or Set Key.",
      "- Profile Servers overlays all server Profile LLM curves in one chart.",
      "- Jump to suggested candidates with Fastest / Best Efficiency / Pareto buttons.",
      "- Click a chart point to sync selection immediately.",
    ].join("\n");
  }
  return [
    "[Chat Mode]",
    "- Ask a question and send to generate a hybrid speculative response.",
    "- Use Start/Stop to control the chat session.",
    "- Shutdown Target fully terminates the remote autodraft_target server process.",
    "- Re-open command help by typing help or /help.",
    "",
    "[How to Add a Server]",
    "1) Open Add Server (Manual)",
    "   - Server Name: display name",
    "   - IP Address: server host/IP",
    "   - Port: target socket port",
    "2) Click Add Server",
    "",
    "[Profiling and Recommendation]",
    "- Profile Servers: aggregate Profile LLM curves across all servers in one chart.",
    "- Priority order: active runtime -> reference cache -> short probe fallback.",
    "- Probe mode depends on server protocol: Server-only API (openai_chat_completions) or Hybrid-capable Target (autodraft_target).",
    "- Metric Preference: switch recommendation metric (cost/energy).",
    "- Recommendation tags: Fastest / Best Efficiency / Pareto-Optimal.",
    "- Non-recommended points are dimmed in the chart.",
  ].join("\n");
}

function renderUiHelp() {
  if (!uiHelpContent) return;
  const mode = (modeSelect && modeSelect.value) ? modeSelect.value : "chat";
  uiHelpContent.textContent = getUiHelpText(mode);
}

function resetAddServerForm() {
  if (!newServerName || !newServerHost || !newServerPort) return;
  newServerName.value = "";
  newServerHost.value = "";
  newServerPort.value = "";
}

function setPopup(mode) {
  if (mode === "add") {
    resetAddServerForm();
  }
  activePopup = mode;
  const showAny = !!mode;
  if (floatingActionHost) floatingActionHost.classList.toggle("hidden", !showAny);
  if (serverKeyPopup) serverKeyPopup.classList.toggle("hidden", mode !== "key");
  if (addServerPanel) addServerPanel.classList.toggle("hidden", mode !== "add");
  if (serverAddToggleBtn) {
    const addActive = mode === "add";
    serverAddToggleBtn.classList.toggle("active", addActive);
    serverAddToggleBtn.textContent = "Add Server";
  }
  if (serverSetKeyBtn) {
    serverSetKeyBtn.classList.toggle("active", mode === "key");
  }
  if (showAny) {
    const focusEl = mode === "key" ? selectedServerApiKey : newServerName;
    if (focusEl && typeof focusEl.focus === "function") {
      window.setTimeout(() => focusEl.focus(), 0);
    }
  }
}

function fillServerFormTemplate(kind) {
  if (!newServerName || !newServerHost || !newServerPort) {
    return;
  }
  if (kind === "local") {
    newServerName.value = "Local Target";
    newServerHost.value = "192.168.0.12";
    newServerPort.value = "26001";
    setActivity("Filled local target template. Update host and port if needed.", "success");
    return;
  }
  resetAddServerForm();
  setActivity("Cleared the Add Server form.", "idle");
}

function getSelectedServerSpec() {
  if (!serverSelect) return null;
  const sid = serverSelect.value;
  return serverCandidates.find((s) => String(s.server_id) === String(sid)) || null;
}

function getServerOptionLabel(spec) {
  if (!spec) return "No server";
  const keyState = spec.requires_api_key && !spec.has_api_key ? " (key required)" : "";
  return `${spec.name}${keyState}`;
}

function updateServerModeHint() {
  if (!serverModeHint) return;
  const spec = getSelectedServerSpec();
  if (!spec) {
    serverModeHint.textContent = "Mode: -";
    return;
  }
  if (String(spec.protocol) === "autodraft_target") {
    const isBridge = !!(spec.metadata && spec.metadata.bridge_external);
    serverModeHint.textContent = isBridge
      ? "Mode: Hybrid-capable Target (Bridge)"
      : "Mode: Hybrid-capable Target";
    return;
  }
  serverModeHint.textContent = "Mode: Server-only API (openai_chat_completions)";
}

function enforceAlgorithmPolicyByServer(notify = false) {
  if (!algorithmSelect) return false;
  const spec = getSelectedServerSpec();
  const apiServerOnly = !!(spec && String(spec.protocol) === "openai_chat_completions");
  Array.from(algorithmSelect.options || []).forEach((opt) => {
    if (!apiServerOnly) {
      opt.disabled = false;
      return;
    }
    // External Chat API: only server-only autoregressive is executable.
    opt.disabled = opt.value !== "Server-Only-AR";
  });
  let changed = false;
  if (apiServerOnly && algorithmSelect.value !== "Server-Only-AR") {
    algorithmSelect.value = "Server-Only-AR";
    if (proactiveDrafting) proactiveDrafting.checked = false;
    changed = true;
    if (notify) {
      setActivity(
        "External Chat API supports Server-only AR only (no verify/logits/KV signals for hybrid/server-only-SD).",
        "warn"
      );
    }
  }
  return changed;
}

function closeServerDropdown() {
  if (serverDropdownMenu) serverDropdownMenu.classList.add("hidden");
}

function renderServerDropdown() {
  if (!serverDropdownBtn || !serverDropdownMenu || !serverSelect) return;
  const selectedSpec = getSelectedServerSpec();
  serverDropdownBtn.textContent = getServerOptionLabel(selectedSpec);
  serverDropdownBtn.disabled = !Array.isArray(serverCandidates) || !serverCandidates.length;
  serverDropdownMenu.innerHTML = "";
  if (!Array.isArray(serverCandidates) || !serverCandidates.length) {
    updateServerModeHint();
    return;
  }

  const currentId = String(serverSelect.value || "");
  serverCandidates.forEach((s) => {
    const sid = String(s.server_id || "");
    const row = document.createElement("div");
    row.className = "server-dropdown-row";

    const pickBtn = document.createElement("button");
    pickBtn.type = "button";
    pickBtn.className = "server-dropdown-item";
    if (sid === currentId) pickBtn.classList.add("active");
    pickBtn.textContent = getServerOptionLabel(s);
    pickBtn.addEventListener("click", () => {
      serverSelect.value = sid;
      closeServerDropdown();
      serverSelect.dispatchEvent(new Event("change"));
    });

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "server-dropdown-remove";
    removeBtn.textContent = "x";
    removeBtn.title = "Remove this server";
    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify({ type: "server_remove", server_id: sid }));
    });

    row.appendChild(pickBtn);
    row.appendChild(removeBtn);
    serverDropdownMenu.appendChild(row);
  });
  updateServerModeHint();
}

function getServerCatalogSignature(rows) {
  try {
    if (!Array.isArray(rows)) return "";
    return JSON.stringify(
      rows.map((r) => ({
        server_id: r.server_id,
        name: r.name,
        endpoint: r.endpoint,
        protocol: r.protocol,
        default_model_id: r.default_model_id,
        has_api_key: r.has_api_key,
        models: Array.isArray(r.models) ? r.models.map((m) => m.model_id) : [],
      }))
    );
  } catch (_) {
    return String(Date.now());
  }
}

const TARGET_DRAFT_COMPATIBILITY = {
  "meta-llama/llama-3.3-70b-instruct": [
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.2-1b-instruct",
  ],
  "qwen/qwen2.5-32b-instruct": [
    "qwen/qwen2.5-3b-instruct",
    "qwen/qwen2.5-1.5b-instruct",
  ],
  "qwen/qwen2.5-14b-instruct": [
    "qwen/qwen2.5-1.5b-instruct",
    "qwen/qwen2.5-0.5b-instruct",
  ],
  "qwen/qwen3-32b": [
    "qwen/qwen3-0.6b",
  ],
  "qwen/qwen3-14b": [
    "qwen/qwen3-0.6b",
  ],
};

function normalizeModelKey(modelId) {
  return String(modelId || "").trim().toLowerCase();
}

function canonicalModelKey(modelId) {
  const raw = normalizeModelKey(modelId).split(":")[0];
  if (!raw) return "";
  if (raw.includes("llama-3.3-70b-instruct")) return "meta-llama/llama-3.3-70b-instruct";
  if (raw.includes("llama-3.2-3b-instruct")) return "meta-llama/llama-3.2-3b-instruct";
  if (raw.includes("llama-3.2-1b-instruct")) return "meta-llama/llama-3.2-1b-instruct";
  if (raw.includes("qwen2.5-32b-instruct")) return "qwen/qwen2.5-32b-instruct";
  if (raw.includes("qwen2.5-14b-instruct")) return "qwen/qwen2.5-14b-instruct";
  if (raw.includes("qwen2.5-3b-instruct")) return "qwen/qwen2.5-3b-instruct";
  if (raw.includes("qwen2.5-1.5b-instruct")) return "qwen/qwen2.5-1.5b-instruct";
  if (raw.includes("qwen2.5-0.5b-instruct")) return "qwen/qwen2.5-0.5b-instruct";
  if (raw.includes("qwen3-32b")) return "qwen/qwen3-32b";
  if (raw.includes("qwen3-14b")) return "qwen/qwen3-14b";
  if (raw.includes("qwen3-0.6b")) return "qwen/qwen3-0.6b";
  return raw;
}

function isDraftCompatible(targetModelId, draftModelId) {
  const targetKey = canonicalModelKey(targetModelId);
  const draftKey = canonicalModelKey(draftModelId);
  if (!targetKey || !draftKey) return false;
  const allow = TARGET_DRAFT_COMPATIBILITY[targetKey];
  if (Array.isArray(allow) && allow.length) {
    return allow.includes(draftKey);
  }
  return false;
}

function getPreferredDraftCandidates(targetModelId) {
  const targetKey = canonicalModelKey(targetModelId);
  if (!targetKey) return [];
  const allow = TARGET_DRAFT_COMPATIBILITY[targetKey];
  return Array.isArray(allow) ? allow : [];
}

function getDefaultTargetQuantization(modelId) {
  const key = canonicalModelKey(modelId);
  if (!key) return "none";
  return (key.includes("70b") || key.includes("72b")) ? "8bit" : "none";
}

function getDefaultDraftQuantization(modelId) {
  const key = canonicalModelKey(modelId);
  if (!key) return "none";
  if (key.includes("8b")) return "4bit";
  if (key.includes("7b")) return "8bit";
  return "none";
}

function applyModelBasedQuantizationDefaults() {
  if (serverQuantizationSelect && serverModelSelect) {
    const targetQ = getDefaultTargetQuantization(serverModelSelect.value);
    if (["none", "8bit", "4bit"].includes(targetQ)) {
      serverQuantizationSelect.value = targetQ;
    }
  }
  if (draftQuantizationSelect && draftModelSelect) {
    const draftQ = getDefaultDraftQuantization(draftModelSelect.value);
    if (["none", "8bit", "4bit"].includes(draftQ)) {
      draftQuantizationSelect.value = draftQ;
    }
  }
}

function refreshDraftModelCompatibility(preferTargetDefault = false) {
  if (!draftModelSelect || !allDraftModelOptions.length) return;
  const selectedSpec = getSelectedServerSpec();
  const selectedTargetModel = serverModelSelect ? serverModelSelect.value : "";
  const selectedDraft = draftModelSelect.value;
  const shouldFilter = selectedSpec && String(selectedSpec.protocol) === "autodraft_target";
  let allowed = allDraftModelOptions;
  if (shouldFilter) {
    allowed = allDraftModelOptions.filter((m) => isDraftCompatible(selectedTargetModel, m.value));
  }
  draftModelSelect.innerHTML = "";
  allowed.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.value;
    opt.textContent = m.label;
    draftModelSelect.appendChild(opt);
  });
  const hasSelected = allowed.some((m) => String(m.value) === String(selectedDraft));
  let nextDraft = "";
  if (preferTargetDefault) {
    const preferredCanon = getPreferredDraftCandidates(selectedTargetModel);
    for (const canon of preferredCanon) {
      const found = allowed.find((m) => canonicalModelKey(m.value) === canon);
      if (found) {
        nextDraft = found.value;
        break;
      }
    }
  }
  if (!nextDraft && hasSelected) {
    nextDraft = selectedDraft;
  }
  if (!nextDraft && allowed.length) {
    nextDraft = allowed[0].value;
  }
  if (nextDraft) {
    draftModelSelect.value = nextDraft;
  }
  applyModelBasedQuantizationDefaults();
}

function syncServerModelSelect(preferredModelId = null) {
  if (!serverModelSelect) return;
  const spec = getSelectedServerSpec();
  serverModelSelect.innerHTML = "";
  const modelsRaw = spec && Array.isArray(spec.models) ? spec.models : [];
  const uniqueMap = new Map();
  modelsRaw.forEach((m) => {
    const key = String((m && m.model_id) || "").trim().toLowerCase();
    if (!key || uniqueMap.has(key)) return;
    uniqueMap.set(key, m);
  });
  const models = Array.from(uniqueMap.values());
  if (!spec || !models.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No model";
    serverModelSelect.appendChild(opt);
    return;
  }
  const selected = preferredModelId || spec.default_model_id || models[0].model_id;
  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = String(m.model_id);
    opt.textContent = String(m.label || m.model_id);
    if (String(m.model_id) === String(selected)) {
      opt.selected = true;
    }
    serverModelSelect.appendChild(opt);
  });
}

function renderServerCatalog(preferredServerId = null, preferredModelId = null) {
  if (!serverSelect) return;
  const prevServer = preferredServerId || serverSelect.value;
  const prevModel = preferredModelId || (serverModelSelect ? serverModelSelect.value : null);
  serverSelect.innerHTML = "";
  if (!Array.isArray(serverCandidates) || !serverCandidates.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No server";
    serverSelect.appendChild(opt);
    syncServerModelSelect();
    renderServerDropdown();
    updateServerModeHint();
    return;
  }
  serverCandidates.forEach((s) => {
    const opt = document.createElement("option");
    opt.value = String(s.server_id);
    const keyState = s.requires_api_key && !s.has_api_key ? " (key required)" : "";
    opt.textContent = `${s.name}${keyState}`;
    if (String(s.server_id) === String(prevServer)) {
      opt.selected = true;
    }
    serverSelect.appendChild(opt);
  });
  if (!serverSelect.value) serverSelect.selectedIndex = 0;
  syncServerModelSelect(prevModel);
  refreshDraftModelCompatibility();
  enforceAlgorithmPolicyByServer(false);
  renderServerDropdown();
  updateServerModeHint();
}

function requestServerCatalog() {
  if (ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "server_catalog_request" }));
}

function updateRecommendationHint() {
  if (!recommendationHint) return;
  [jumpFastestBtn, jumpBestEfficiencyBtn, jumpParetoBtn].forEach((btn) => {
    if (btn) btn.classList.remove("active");
  });
  const points = getActiveTradeoffPoints();
  const isProbe = points.length > 0 && (points[0].source === "probe" || points[0].source === "probe_curve");
  const summary = isProbe ? recommendationSummary : computeReferenceRecommendations(points);
  const fastest = summary && summary.fastest;
  const best = summary && summary.best_efficiency;
  const paretoCount = (summary && summary.pareto_optimal_ids || []).length;
  if (!fastest && !best && !paretoCount) {
    recommendationHint.textContent = isProbe
      ? "Run profiling to compute recommendations."
      : "Measure reference to build trade-off recommendations.";
    return;
  }
  if (isProbe) {
    recommendationHint.textContent = `Fastest: ${fastest ? `${fastest.server_id}/${fastest.model_id}` : "-"} | Best-eff: ${best ? `${best.server_id}/${best.model_id}` : "-"} | Pareto: ${paretoCount}`;
  } else {
    recommendationHint.textContent = `Fastest / Best-eff / Pareto computed on current reference curve (${paretoCount} pareto points).`;
  }
}

function flashButtonActive(btn) {
  if (!btn) return;
  btn.classList.add("active");
  window.setTimeout(() => {
    btn.classList.remove("active");
    updateRecommendationHint();
  }, 220);
}

function jumpToRecommendation(kind) {
  const points = getActiveTradeoffPoints();
  if (!points.length) return;
  const isProbe = points[0].source === "probe" || points[0].source === "probe_curve";
  const summary = isProbe ? recommendationSummary : computeReferenceRecommendations(points);
  let target = null;
  if (kind === "fastest" && summary.fastest) {
    target = summary.fastest;
  } else if (kind === "best_efficiency" && summary.best_efficiency) {
    target = summary.best_efficiency;
  } else if (kind === "pareto") {
    const first = (summary.pareto_optimal_ids || [])[0];
    if (first) target = first;
  }
  if (!target) return;

  let targetPoint = null;
  if (typeof target === "string") {
    targetPoint = points.find((p) => p.uid === target) || null;
  } else if (target.uid) {
    targetPoint = points.find((p) => p.uid === target.uid) || null;
  } else if (target.server_id && target.model_id) {
    const serverModelUid = `${String(target.server_id)}::${String(target.model_id)}`;
    const selectorNow = getCurrentSelectorValue();
    const candidates = points.filter(
      (p) => String(p.server_id || "") === String(target.server_id) && String(p.model_id || "") === String(target.model_id)
    );
    if (candidates.length) {
      targetPoint = candidates.reduce((best, p) => {
        const dBest = Math.abs(Number(best.selector || 0) - selectorNow);
        const dNow = Math.abs(Number(p.selector || 0) - selectorNow);
        return dNow < dBest ? p : best;
      }, candidates[0]);
    } else {
      targetPoint = points.find((p) => p.uid === serverModelUid) || null;
    }
  }

  if (isProbe) {
    const sid = targetPoint ? String(targetPoint.server_id || "") : String(target.server_id || "");
    const mid = targetPoint ? String(targetPoint.model_id || "") : String(target.model_id || "");
    if (serverSelect) serverSelect.value = String(sid);
    syncServerModelSelect(mid);
    if (serverModelSelect) serverModelSelect.value = String(mid);
    renderServerDropdown();
  }
  if (targetPoint && Number.isFinite(Number(targetPoint.selector))) {
    const objectiveMode = (objectiveModeSelect && objectiveModeSelect.value === "constraint") ? "constraint" : "balanced";
    if (objectiveMode === "constraint") {
      if (getConstraintTarget() === "tps" && tpsConstraintSlider) {
        tpsConstraintSlider.value = String(targetPoint.selector);
        if (tpsConstraintValue) tpsConstraintValue.textContent = Number(tpsConstraintSlider.value).toFixed(1);
        updateTpsConstraintHint();
      } else if (constraintSlider) {
        constraintSlider.value = String(targetPoint.selector);
        clampConstraintToFeasible();
        if (constraintValue) constraintValue.textContent = Number(constraintSlider.value).toFixed(1);
      }
    } else if (costSlider) {
      costSlider.value = String(Number(targetPoint.selector).toFixed(2));
      if (costValue) costValue.textContent = Number(costSlider.value).toFixed(2);
    }
  }

  drawTradeoffChart();
  if (ws.readyState === WebSocket.OPEN) sendSettings();
}

function getRecommendationTarget(kind) {
  if (kind === "fastest" && recommendationSummary.fastest) {
    return recommendationSummary.fastest;
  }
  if (kind === "best_efficiency" && recommendationSummary.best_efficiency) {
    return recommendationSummary.best_efficiency;
  }
  if (kind === "pareto") {
    const first = (recommendationSummary.pareto_optimal_ids || [])[0];
    if (first && first.includes("::")) {
      const [server_id, model_id] = first.split("::");
      return { server_id, model_id };
    }
  }
  return null;
}

function getRecommendationHoverUid() {
  const target = getRecommendationTarget(recommendationHoverKind);
  if (!target) return null;
  if (target.uid) return String(target.uid);
  return `${String(target.server_id)}::${String(target.model_id)}`;
}

function getRecommendationHoverUids() {
  if (!recommendationHoverKind) return new Set();
  const points = getActiveTradeoffPoints();
  const isProbe = points.length > 0 && (points[0].source === "probe" || points[0].source === "probe_curve");
  const isProbeCurve = points.length > 0 && points[0].source === "probe_curve";
  const summary = isProbe ? recommendationSummary : computeReferenceRecommendations(points);
  if (recommendationHoverKind === "pareto") {
    const base = new Set((summary && summary.pareto_optimal_ids) || []);
    if (!isProbeCurve) return base;
    const out = new Set();
    base.forEach((id) => {
      points.forEach((p) => {
        const key = `${String(p.server_id || "")}::${String(p.model_id || "")}`;
        if (key === id && p.uid) out.add(String(p.uid));
      });
    });
    return out;
  }
  let single = null;
  const target = recommendationHoverKind === "fastest"
    ? (summary && summary.fastest)
    : (summary && summary.best_efficiency);
  if (target) {
    single = target.uid ? String(target.uid) : `${String(target.server_id)}::${String(target.model_id)}`;
  }
  if (!single) return new Set();
  if (!isProbeCurve) return new Set([single]);
  const out = new Set();
  points.forEach((p) => {
    const key = `${String(p.server_id || "")}::${String(p.model_id || "")}`;
    if (key === single && p.uid) out.add(String(p.uid));
  });
  return out;
}

function getCurrentSelectorValue() {
  const objectiveMode = (objectiveModeSelect && objectiveModeSelect.value === "constraint") ? "constraint" : "balanced";
  if (objectiveMode === "constraint") {
    const constraintTarget = (constraintTargetSelect && constraintTargetSelect.value === "tps") ? "tps" : "metric";
    return constraintTarget === "tps"
      ? Number(tpsConstraintSlider ? tpsConstraintSlider.value : 0)
      : Number(constraintSlider ? constraintSlider.value : 0);
  }
  return Number(costSlider ? costSlider.value : 0);
}

function getConstraintTarget() {
  return (constraintTargetSelect && constraintTargetSelect.value === "tps") ? "tps" : "metric";
}

function computeReferenceRecommendations(points) {
  if (!Array.isArray(points) || !points.length) {
    return { fastest: null, best_efficiency: null, pareto_optimal_ids: [] };
  }
  const valid = points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && p.y > 0 && p.x > 0);
  if (!valid.length) {
    return { fastest: null, best_efficiency: null, pareto_optimal_ids: [] };
  }
  let fastest = valid[0];
  let best = valid[0];
  valid.forEach((p) => {
    if (p.y > fastest.y) fastest = p;
    if (p.x < best.x || (p.x === best.x && p.y > best.y)) best = p;
  });
  const pareto = valid.filter((a) => !valid.some((b) => (
    b !== a &&
    b.x <= a.x &&
    b.y >= a.y &&
    (b.x < a.x || b.y > a.y)
  )));
  return {
    fastest: fastest ? { uid: fastest.uid } : null,
    best_efficiency: best ? { uid: best.uid } : null,
    pareto_optimal_ids: pareto.map((p) => p.uid),
  };
}

function getTradeoffLayout(points, width, height) {
  const pad = { left: 102, right: 30, top: 20, bottom: 62 };
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const xr = Math.max(1e-9, maxX - minX);
  const yr = Math.max(1e-9, maxY - minY);
  const xToPx = (x) => pad.left + ((x - minX) / xr) * (width - pad.left - pad.right);
  const yToPx = (y) => height - pad.bottom - ((y - minY) / yr) * (height - pad.top - pad.bottom);
  return { pad, minX, maxX, minY, maxY, xr, yr, xToPx, yToPx };
}

function findNearestTradeoffPoint(points, mouseX, mouseY, width, height) {
  if (!Array.isArray(points) || !points.length) return null;
  const layout = getTradeoffLayout(points, width, height);
  let nearest = null;
  let nearestDistSq = Number.POSITIVE_INFINITY;
  points.forEach((p) => {
    const dx = mouseX - layout.xToPx(p.x);
    const dy = mouseY - layout.yToPx(p.y);
    const d = dx * dx + dy * dy;
    if (d < nearestDistSq) {
      nearestDistSq = d;
      nearest = p;
    }
  });
  if (!nearest) return null;
  if (nearestDistSq > (16 * 16)) return null;
  return nearest;
}

// Enable/disable Cost Sensitivity and Proactive Drafting depending on the algorithm
function updateControlsState() {
  enforceAlgorithmPolicyByServer(false);
  const algo = algorithmSelect.value;
  const isObjectiveAlgo = algo === "AutoDraft" || algo === "Server-Only" || algo === "Server-Only-AR";
  const allowsProactive = algo === "AutoDraft";
  if (!isObjectiveAlgo && objectiveModeSelect) {
    objectiveModeSelect.value = "balanced";
  }
  const objectiveMode = (objectiveModeSelect && objectiveModeSelect.value === "constraint") ? "constraint" : "balanced";
  const constraintTarget = getConstraintTarget();
  
  modeSelect.disabled = controlsLocked;
  objectiveModeSelect.disabled = controlsLocked || !isObjectiveAlgo;
  algorithmSelect.disabled = controlsLocked;
  draftModelSelect.disabled = controlsLocked;
  if (constraintTargetSelect) constraintTargetSelect.disabled = !isObjectiveAlgo || objectiveMode !== "constraint";

  // cost/proactive/constraintcan be adjusted in real time during a session
  costSlider.disabled = !isObjectiveAlgo || objectiveMode === "constraint";
  if (constraintSlider) constraintSlider.disabled = !isObjectiveAlgo || objectiveMode !== "constraint" || constraintTarget !== "metric";
  if (tpsConstraintSlider) tpsConstraintSlider.disabled = !isObjectiveAlgo || objectiveMode !== "constraint" || constraintTarget !== "tps";
  proactiveDrafting.disabled = !allowsProactive;
  const showConstraint = isObjectiveAlgo && objectiveMode === "constraint";
  if (costControlGroup) costControlGroup.style.display = showConstraint ? "none" : "";
  if (constraintTargetControlGroup) constraintTargetControlGroup.style.display = showConstraint ? "" : "none";
  if (constraintControlGroup) constraintControlGroup.style.display = showConstraint && constraintTarget === "metric" ? "" : "none";
  if (tpsConstraintControlGroup) tpsConstraintControlGroup.style.display = showConstraint && constraintTarget === "tps" ? "" : "none";
  updateConstraintHint();
  updateTpsConstraintHint();
  updateConstraintSliderVisual();
  
  // Cost Sensitivity section (title and slider row)
  const costTitle = costSlider.closest(".side-panel")?.querySelector(".panel-title");
  const costSliderRow = costSlider.closest(".slider-row");
  
  // Proactive Drafting section
  const proactiveSection = proactiveDrafting.closest(".checkbox-title");
  
  // Find the Cost Sensitivity title among all panel-title elements
  const allTitles = document.querySelectorAll(".panel-title");
  let costTitleElement = null;
  for (let i = 0; i < allTitles.length; i++) {
    if (allTitles[i].textContent.trim() === "Cost Sensitivity") {
      costTitleElement = allTitles[i];
      break;
    }
  }
  
  // Add/remove the disabled class
  if (costTitleElement) {
    if (isObjectiveAlgo) {
      costTitleElement.classList.remove("disabled");
    } else {
      costTitleElement.classList.add("disabled");
    }
  }
  
  if (costSliderRow) {
    if (isObjectiveAlgo) {
      costSliderRow.classList.remove("disabled");
    } else {
      costSliderRow.classList.add("disabled");
    }
  }
  
  if (proactiveSection) {
    if (allowsProactive) {
      proactiveSection.classList.remove("disabled");
    } else {
      proactiveSection.classList.add("disabled");
    }
  }
  drawTradeoffChart();
}

function updateConstraintHint() {
  if (!constraintHint) return;
  if (getConstraintTarget() !== "metric") {
    constraintHint.textContent = "Choose Metric Budget to activate this constraint.";
    return;
  }
  const hasFeasible =
    Number.isFinite(feasibleConstraintMin) &&
    Number.isFinite(feasibleConstraintMax) &&
    feasibleConstraintMin <= feasibleConstraintMax;
  if (!hasFeasible) {
    constraintHint.textContent = "Feasible range unavailable. Run/refresh reference cache first.";
    return;
  }
  constraintHint.textContent = `Feasible range: ${feasibleConstraintMin.toFixed(2)} ~ ${feasibleConstraintMax.toFixed(2)} ($/1M tok)`;
}

function updateTpsConstraintHint() {
  if (!tpsConstraintHint || !tpsConstraintSlider) return;
  if (getConstraintTarget() !== "tps") {
    tpsConstraintHint.textContent = "Choose Minimum Throughput to activate this constraint.";
    return;
  }
  const minTps = Number(tpsConstraintSlider.value || 0);
  const hasFeasible =
    Number.isFinite(feasibleTpsMin) &&
    Number.isFinite(feasibleTpsMax) &&
    feasibleTpsMin <= feasibleTpsMax;
  const suffix = hasFeasible
    ? ` Feasible range: ${feasibleTpsMin.toFixed(1)} ~ ${feasibleTpsMax.toFixed(1)} tok/s.`
    : "";
  tpsConstraintHint.textContent = minTps > 0
    ? `Candidates with predicted throughput below ${minTps.toFixed(1)} tok/s are treated as infeasible.${suffix}`
    : `0 disables the TPS floor.${suffix}`;
}

function updateConstraintSliderVisual() {
  if (!constraintSlider) return;
  if (getConstraintTarget() !== "metric") {
    constraintSlider.style.background = "";
    return;
  }
  const sliderMin = Number(constraintSlider.min || 1);
  const sliderMax = Number(constraintSlider.max || 30);
  const hasFeasible =
    Number.isFinite(feasibleConstraintMin) &&
    Number.isFinite(feasibleConstraintMax) &&
    feasibleConstraintMin <= feasibleConstraintMax;
  if (!hasFeasible) {
    constraintSlider.style.background = "";
    updateConstraintHint();
    return;
  }
  const leftPct = Math.max(0, Math.min(100, ((feasibleConstraintMin - sliderMin) / (sliderMax - sliderMin)) * 100));
  const rightPct = Math.max(0, Math.min(100, ((feasibleConstraintMax - sliderMin) / (sliderMax - sliderMin)) * 100));
  constraintSlider.style.background = `linear-gradient(to right, #d1d5db 0%, #d1d5db ${leftPct}%, #9cc0f0 ${leftPct}%, #9cc0f0 ${rightPct}%, #d1d5db ${rightPct}%, #d1d5db 100%)`;
  updateConstraintHint();
}

function clampConstraintToFeasible() {
  if (!constraintSlider) return;
  if (getConstraintTarget() !== "metric") return;
  if (!Number.isFinite(feasibleConstraintMin) || !Number.isFinite(feasibleConstraintMax)) return;
  let v = Number(constraintSlider.value);
  if (!Number.isFinite(v)) return;
  v = Math.max(feasibleConstraintMin, Math.min(feasibleConstraintMax, v));
  constraintSlider.value = String(v);
  if (constraintValue) constraintValue.textContent = Number(v).toFixed(1);
}

function getActiveTradeoffPoints() {
  const objectiveMode = (objectiveModeSelect && objectiveModeSelect.value === "constraint") ? "constraint" : "balanced";
  const sourceRef = objectiveMode === "constraint" ? tradeoffCurveConstraint : tradeoffCurveBlend;
  const selectorNow = getCurrentSelectorValue();

  const probePoints = (Array.isArray(serverProbeRows) && serverProbeRows.length)
    ? serverProbeRows
      .filter((p) => p && p.ok)
      .map((p, idx) => {
        const x = Number(
          p.selected_metric_per_1m !== undefined && p.selected_metric_per_1m !== null
            ? p.selected_metric_per_1m
            : p.metric_per_1m
        );
        const y = Number(p.throughput_tps);
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
        const server_id = String(p.server_id || "");
        const model_id = String(p.model_id || "");
        return {
          x,
          y,
          selector: Number.isFinite(selectorNow) ? selectorNow : idx,
          source: "probe",
          server_id,
          model_id,
          uid: `${server_id}::${model_id}`,
          recommended: !!p.recommended,
          tags: Array.isArray(p.recommendation_tags) ? p.recommendation_tags : [],
        };
      })
      .filter(Boolean)
    : [];

  const probeCurvePoints = (Array.isArray(serverProbeCurves) && serverProbeCurves.length)
    ? serverProbeCurves
      .flatMap((series, sidx) => {
        const server_id = String(series && series.server_id ? series.server_id : "");
        const model_id = String(series && series.model_id ? series.model_id : "");
        const seriesUid = `${server_id}::${model_id}`;
        const pointRows = Array.isArray(series && series.points) ? series.points : [];
        const points = pointRows
          .map((p, pidx) => {
            const x = Number(p.metric_per_1m);
            const y = Number(p.throughput_tps);
            const selector = Number(p.selector_value);
            if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(selector)) return null;
            return {
              x,
              y,
              selector,
              source: "probe_curve",
              series_uid: seriesUid,
              series_idx: sidx,
              server_id,
              model_id,
              uid: `${seriesUid}::${pidx}`,
              recommended: false,
              tags: [],
            };
          })
          .filter(Boolean);
        const currentSelector = getCurrentSelectorValue();
        const nearest = points.length
          ? points.reduce((best, now) => (
            Math.abs(now.selector - currentSelector) < Math.abs(best.selector - currentSelector) ? now : best
          ), points[0])
          : null;
        const recKey = `${server_id}::${model_id}`;
        const isFast = recommendationSummary && recommendationSummary.fastest
          && `${String(recommendationSummary.fastest.server_id)}::${String(recommendationSummary.fastest.model_id)}` === recKey;
        const isBest = recommendationSummary && recommendationSummary.best_efficiency
          && `${String(recommendationSummary.best_efficiency.server_id)}::${String(recommendationSummary.best_efficiency.model_id)}` === recKey;
        const isPareto = Array.isArray(recommendationSummary && recommendationSummary.pareto_optimal_ids)
          && recommendationSummary.pareto_optimal_ids.includes(recKey);
        if (nearest) {
          nearest.recommended = isFast || isBest || isPareto;
          if (isFast) nearest.tags.push("fastest");
          if (isBest) nearest.tags.push("best_efficiency");
          if (isPareto) nearest.tags.push("pareto_optimal");
        }
        return points;
      })
    : [];

  const referencePoints = Array.isArray(sourceRef)
    ? sourceRef
    .map((p, idx) => {
      const x = Number(p.cost_per_1m);
      const y = Number(p.tps);
      const selector = Number(p.selector_value);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(selector)) return null;
      return { x, y, selector, source: "reference", uid: `reference::${selector}::${idx}` };
    })
    .filter(Boolean)
    : [];

  if (tradeoffDisplayMode === "probe") {
    if (probeCurvePoints.length) return probeCurvePoints;
    return probePoints.length ? probePoints : referencePoints;
  }
  if (tradeoffDisplayMode === "reference") {
    return referencePoints.length ? referencePoints : probePoints;
  }
  if (probeCurvePoints.length) return probeCurvePoints;
  return probePoints.length ? probePoints : referencePoints;
}

function drawTradeoffChart() {
  if (!tradeoffCanvas) return;
  const ctx = tradeoffCanvas.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  // Read stable layout size from CSS box.
  const width = Math.max(1, Math.floor(tradeoffCanvas.clientWidth));
  const height = Math.max(1, Math.floor(tradeoffCanvas.clientHeight));
  const targetW = Math.floor(width * dpr);
  const targetH = Math.floor(height * dpr);
  // Resize backing store only when needed to avoid reflow thrash.
  if (tradeoffCanvas.width !== targetW || tradeoffCanvas.height !== targetH) {
    tradeoffCanvas.width = targetW;
    tradeoffCanvas.height = targetH;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);

  const points = getActiveTradeoffPoints();
  if (!points.length) {
    if (tradeoffHint) tradeoffHint.textContent = "Trade-off curve unavailable (reference cache needed).";
    ctx.fillStyle = "#7b8794";
    ctx.font = "19px system-ui";
    ctx.fillText("No trade-off points", 10, 20);
    return;
  }
  if (tradeoffHint) {
    tradeoffHint.textContent = (points[0].source === "probe" || points[0].source === "probe_curve")
      ? "Click point to select server/model."
      : "Click on graph to sync selector.";
  }

  const { pad, minX, maxX, minY, maxY, xToPx, yToPx } = getTradeoffLayout(points, width, height);

  // y-grid + y-axis tick labels
  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#6b7280";
  ctx.font = "20px system-ui";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const yy = pad.top + (i * (height - pad.top - pad.bottom)) / 4;
    const yv = maxY - (i * (maxY - minY)) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(width - pad.right, yy);
    ctx.stroke();
    ctx.fillText(yv.toFixed(2), pad.left - 8, yy + 5);
  }
  ctx.textAlign = "start";

  // x-grid lines
  for (let i = 0; i <= 4; i++) {
    const xx = pad.left + (i * (width - pad.left - pad.right)) / 4;
    ctx.beginPath();
    ctx.moveTo(xx, pad.top);
    ctx.lineTo(xx, height - pad.bottom);
    ctx.stroke();
  }

  const sorted = [...points].sort((a, b) => a.x - b.x);
  const isProbeCurveView = points.length > 0 && points[0].source === "probe_curve";
  if (isProbeCurveView) {
    const seriesMap = new Map();
    points.forEach((p) => {
      const key = String(p.series_uid || `${p.server_id || ""}::${p.model_id || ""}`);
      if (!seriesMap.has(key)) seriesMap.set(key, []);
      seriesMap.get(key).push(p);
    });
    let seriesIdx = 0;
    for (const [, group] of seriesMap.entries()) {
      const linePts = [...group].sort((a, b) => Number(a.selector) - Number(b.selector));
      const color = probeCurvePalette[seriesIdx % probeCurvePalette.length];
      seriesIdx += 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      linePts.forEach((p, idx) => {
        const px = xToPx(p.x), py = yToPx(p.y);
        if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }
  } else {
    ctx.strokeStyle = "#2f6fab";
    ctx.lineWidth = 2;
    ctx.beginPath();
    sorted.forEach((p, idx) => {
      const px = xToPx(p.x), py = yToPx(p.y);
      if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }

  const objectiveMode = (objectiveModeSelect && objectiveModeSelect.value === "constraint") ? "constraint" : "balanced";
  const currentSelector = objectiveMode === "constraint"
    ? getCurrentSelectorValue()
    : Number(costSlider ? costSlider.value : 0);
  let bestIdx = 0;
  let bestDist = Number.POSITIVE_INFINITY;
  sorted.forEach((p, i) => {
    const d = Math.abs(p.selector - currentSelector);
    if (d < bestDist) { bestDist = d; bestIdx = i; }
  });

  sorted.forEach((p, i) => {
    const px = xToPx(p.x), py = yToPx(p.y);
    const recommendationHoverUids = getRecommendationHoverUids();
    const hasTag = Array.isArray(p.tags)
      && (
        (recommendationHoverKind === "fastest" && p.tags.includes("fastest"))
        || (recommendationHoverKind === "best_efficiency" && p.tags.includes("best_efficiency"))
        || (recommendationHoverKind === "pareto" && p.tags.includes("pareto_optimal"))
      );
    const isRecHovered = !!((p.uid && recommendationHoverUids.has(p.uid)) || hasTag);
    const isPointHovered = !!(tradeoffHoverPointUid && p.uid && p.uid === tradeoffHoverPointUid);
    const emphasize = isRecHovered || isPointHovered;
    ctx.beginPath();
    if (p.source === "probe_curve") {
      ctx.fillStyle = probeCurvePalette[Number(p.series_idx || 0) % probeCurvePalette.length];
      if (!p.recommended) {
        ctx.globalAlpha = 0.78;
      }
    } else if (p.source === "probe") {
      if (!p.recommended) {
        ctx.fillStyle = "rgba(31,79,137,0.28)";
      } else if (p.tags.includes("fastest")) {
        ctx.fillStyle = "#b42318";
      } else if (p.tags.includes("best_efficiency")) {
        ctx.fillStyle = "#0f7a2b";
      } else {
        ctx.fillStyle = "#7c3aed";
      }
    } else {
      ctx.fillStyle = (i === bestIdx) ? "#b42318" : "#1f4f89";
    }
    const baseRadius =
      p.source === "probe_curve"
        ? (p.recommended ? 5.0 : 2.8)
        : (p.source === "probe" ? (p.recommended ? 4.6 : 3.1) : (i === bestIdx ? 4.5 : 3.2));
    const radius = emphasize ? baseRadius + 2.6 : baseRadius;
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1.0;
    if (emphasize) {
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 1.6;
      ctx.stroke();
    }
  });

  // x-axis tick labels
  ctx.fillStyle = "#6b7280";
  ctx.font = "20px system-ui";
  ctx.textAlign = "center";
  for (let i = 0; i <= 4; i++) {
    const xv = minX + (i * (maxX - minX)) / 4;
    const xx = pad.left + (i * (width - pad.left - pad.right)) / 4;
    ctx.fillText(xv.toFixed(2), xx, height - pad.bottom + 22);
  }
  ctx.textAlign = "start";

  // centered axis titles
  ctx.fillStyle = "#374151";
  ctx.font = "22px system-ui";
  const xLabel = "cost ($ / 1M tok)";
  const xw = ctx.measureText(xLabel).width;
  ctx.fillText(xLabel, (width - xw) / 2, height - 8);
  ctx.save();
  ctx.translate(30, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const yLabel = "throughput (tok/s)";
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();

  const hovered = points.find((p) => p.uid && tradeoffHoverPointUid && p.uid === tradeoffHoverPointUid) || null;
  if (hovered) {
    const hpX = xToPx(hovered.x);
    const hpY = yToPx(hovered.y);
    ctx.save();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = "rgba(107,114,128,0.65)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(hpX, pad.top);
    ctx.lineTo(hpX, height - pad.bottom);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad.left, hpY);
    ctx.lineTo(width - pad.right, hpY);
    ctx.stroke();
    ctx.restore();

    const objectiveMode = (objectiveModeSelect && objectiveModeSelect.value === "constraint") ? "constraint" : "balanced";
    const hoverMode = String(hovered && hovered.source) === "probe_curve"
      ? String(serverProbeCurveMode || "cost_sensitivity")
      : (objectiveMode === "constraint" ? "constraint" : "cost_sensitivity");
    const selectorLabel = Number.isFinite(Number(hovered.selector))
      ? (hoverMode === "constraint"
        ? (getConstraintTarget() === "tps"
          ? `min-tps: ${Number(hovered.selector).toFixed(2)}`
          : `metric-budget: ${Number(hovered.selector).toFixed(2)}`)
        : `cost-sensitivity: ${Number(hovered.selector).toFixed(2)}`)
      : "cost-sensitivity: -";
    const valueLabel = `cost: ${hovered.x.toFixed(2)} | throughput: ${hovered.y.toFixed(2)}`;
    ctx.fillStyle = "#6b7280";
    ctx.font = "16px system-ui";
    ctx.fillText(selectorLabel, pad.left + 8, pad.top + 18);
    ctx.fillText(valueLabel, pad.left + 8, pad.top + 38);
  }
}

function syncTopStripPanelHeights() {
  if (!topControlsBox || !topTradeoffBox) return;
  const targetHeight = Math.max(1, Math.floor(topControlsBox.getBoundingClientRect().height));
  const current = parseFloat(topTradeoffBox.style.height || "0") || 0;
  if (Math.abs(current - targetHeight) > 0.5) {
    topTradeoffBox.style.height = `${targetHeight}px`;
  }
}

function maybeAddSeriesPoint(key, chart, value) {
  if (!chart || value === undefined || value === null) return;
  const num = Number(value);
  if (!Number.isFinite(num)) return;
  const prev = lastSeriesPoint[key];
  if (prev !== null && Math.abs(prev - num) < 1e-12) return;
  lastSeriesPoint[key] = num;
  chart.addDataPoint(num);
}

function wsUrl() {
  const proto = (location.protocol === "https:") ? "wss" : "ws";
  return `${proto}://${location.host}/ws`;
}

const ws = new WebSocket(wsUrl());

function sanitizeModelText(rawText) {
  let text = String(rawText || "");
  // Remove common EOS/special control tokens emitted by some runtimes.
  text = text.replace(/<\|eot_id\|>|<\|end_of_text\|>|<\|eom_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|<\/s>|<s>|<eos>|\[eos\]/gi, "");
  // Remove standalone EOS marker text.
  text = text.replace(/\bEOS\b/gi, "");
  // Remove tag-like fragments such as "<color red>".
  text = text.replace(/<[^>\n]+>/g, "");
  // Keep line breaks but normalize excessive spaces.
  text = text.replace(/[ \t]{2,}/g, " ");
  return text.trim();
}

function sanitizeTokenTracePiece(rawText) {
  let text = String(rawText || "");
  // Keep piece-level spacing exactly as generated; only strip explicit control tokens.
  text = text.replace(/<\|eot_id\|>|<\|end_of_text\|>|<\|eom_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|<\/s>|<s>|<eos>|\[eos\]/gi, "");
  text = text.replace(/\bEOS\b/gi, "");
  return text;
}

function renderTokenTraceContent(container, tokenTrace, targetText = "") {
  container.innerHTML = "";
  if (!Array.isArray(tokenTrace) || tokenTrace.length === 0) return false;
  const target = String(targetText || "");
  let built = "";
  tokenTrace.forEach((t) => {
    if (target && built.length >= target.length) return;
    const span = document.createElement("span");
    const origin = String((t && t.origin) || "");
    let piece = sanitizeTokenTracePiece(String((t && t.text) || ""));
    if (!piece) return;
    if (target) {
      // sanitizeModelText() trims leading spaces, while token pieces may start with one.
      if (!built.length) piece = piece.replace(/^\s+/, "");
      const remainingLen = target.length - built.length;
      if (remainingLen <= 0) return;
      if (piece.length > remainingLen) piece = piece.slice(0, remainingLen);
      if (!piece) return;
    }
    if (origin === "draft_accept") span.className = "tok-draft";
    else if (origin === "server_new") span.className = "tok-server";
    else if (origin === "proactive_accept") span.className = "tok-proactive";
    span.textContent = piece;
    container.appendChild(span);
    built += piece;
  });
  // Do not require exact full-string match; tokenization boundaries can differ
  // slightly from final sanitized text. If we rendered any token spans, keep coloring.
  return built.length > 0;
}

function ensureBubbleContentNode(div) {
  if (!div) return null;
  let content = div.querySelector(":scope > .bubble-content");
  if (content) return content;
  content = document.createElement("div");
  content.className = "bubble-content";
  const metrics = div.querySelector(":scope > .bubble-metrics");
  if (metrics) div.insertBefore(content, metrics);
  else div.appendChild(content);
  return content;
}

function renderTreeSketch(stats) {
  if (!treeMiniBox) return;
  if (!stats || !Array.isArray(stats.depth_widths)) {
    treeMiniBox.textContent = "-";
    return;
  }
  const treeDepth = Number(stats.tree_depth || 0);
  const nodes = Number(stats.final_nnodes || 0);
  const edges = Math.max(0, nodes - 1);
  treeMiniBox.textContent = [
    `step=${Number(stats.step || 0)}`,
    `depth=${treeDepth}`,
    `nodes=${nodes}`,
    `edges≈${edges}`,
  ].join("\n");
}

function drawMiniTree(stats) {
  if (stats && typeof stats === "object") {
    lastTreeStats = stats;
  } else {
    stats = lastTreeStats;
  }
  if (!treeMiniCanvas) return;
  const ctx = treeMiniCanvas.getContext("2d");
  if (!ctx) return;

  const rect = treeMiniCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  treeMiniCanvas.width = Math.floor(width * dpr);
  treeMiniCanvas.height = Math.floor(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const widths = (stats && Array.isArray(stats.depth_widths))
    ? stats.depth_widths.map((v) => Math.max(0, Number(v) || 0))
    : [];
  const parent = (stats && Array.isArray(stats.tree_parent))
    ? stats.tree_parent.map((v) => Number(v))
    : [];
  const hasParentTree = parent.length > 0;
  if (!hasParentTree && !widths.length) {
    ctx.fillStyle = "#888";
    ctx.font = "21px system-ui";
    ctx.fillText("no tree data", 8, 18);
    return;
  }

  const maxLevels = Number.POSITIVE_INFINITY;
  const maxNodesPerLevel = Number.POSITIVE_INFINITY;
  const acceptLength = Math.max(0, Number((stats && stats.accept_length) || 0));
  const yTop = 20;
  const yBottom = height - 22;
  const xLeft = 18;
  const xRight = width - 18;
  const origin = String((stats && stats.tree_origin) || "draft_accept");
  let rootFill = "#111111";
  let acceptedFill = "#1d4ed8";
  let nonAcceptedFill = "#93c5fd";
  let acceptedEdgeColor = "#1d4ed8";
  let nonAcceptedEdgeColor = "#93c5fd";
  let labelColor = "#1e3a8a";
  if (origin === "proactive_accept") {
    rootFill = "#1d4ed8";
    acceptedFill = "#1d8f2f";
    nonAcceptedFill = "#86efac";
    acceptedEdgeColor = "#1d8f2f";
    nonAcceptedEdgeColor = "#86efac";
    labelColor = "#166a23";
  } else if (origin !== "draft_accept") {
    // Fallback for unknown origins.
    rootFill = "#111111";
    acceptedFill = "#111111";
    nonAcceptedFill = "#9ca3af";
    acceptedEdgeColor = "#111111";
    nonAcceptedEdgeColor = "#9ca3af";
    labelColor = "#374151";
  }
  let levelNodeIndices = [];
  let edges = [];
  let acceptedSet = new Set();

  if (hasParentTree) {
    const n = parent.length;
    const roots = [];
    const children = Array.from({ length: n }, () => []);
    for (let i = 0; i < n; i++) {
      const p = Number(parent[i]);
      if (!Number.isFinite(p) || p < 0 || p >= n || p === i) {
        roots.push(i);
      } else {
        children[p].push(i);
        edges.push([p, i]);
      }
    }
    const depth = Array(n).fill(null);
    const q = [];
    roots.forEach((r) => {
      depth[r] = 1;
      q.push(r);
    });
    while (q.length) {
      const cur = q.shift();
      const d = depth[cur] || 1;
      const kids = children[cur] || [];
      kids.forEach((k) => {
        const nd = d + 1;
        if (depth[k] === null || nd < depth[k]) {
          depth[k] = nd;
          q.push(k);
        }
      });
    }
    for (let i = 0; i < n; i++) {
      if (depth[i] === null) depth[i] = 1;
    }
    const maxDepthRaw = depth.reduce((m, v) => Math.max(m, Number(v) || 1), 1);
    const shownDepth = Math.min(maxLevels, 1 + maxDepthRaw); // +1 for synthetic root
    levelNodeIndices = Array.from({ length: shownDepth }, () => []);
    // depth 0: synthetic root
    levelNodeIndices[0] = [-1];
    for (let i = 0; i < n; i++) {
      const d = Math.max(1, Math.min(shownDepth - 1, Number(depth[i]) || 1));
      levelNodeIndices[d].push(i);
    }
    // synthetic root -> all depth1 roots
    roots.forEach((r) => edges.push([-1, r]));
    const acceptedNodes = (stats && Array.isArray(stats.accepted_node_ids)) ? stats.accepted_node_ids : [];
    acceptedSet = new Set([ -1 ]);
    if (acceptedNodes.length) {
      acceptedNodes.forEach((v) => {
        const nid = Number(v);
        if (Number.isFinite(nid) && nid >= 0 && nid < n) acceptedSet.add(nid);
      });
      // If upstream mapping failed, don't show broken highlights.
      if (acceptedSet.size <= 1) {
        for (let d = 1; d <= acceptLength; d++) {
          (levelNodeIndices[d] || []).forEach((nid) => acceptedSet.add(nid));
        }
      }
    } else {
      // fallback when accepted path is unavailable
      for (let d = 1; d <= acceptLength; d++) {
        (levelNodeIndices[d] || []).forEach((nid) => acceptedSet.add(nid));
      }
    }
  } else {
    // Fallback: depth-width approximation only.
    const shownWidths = [1, ...widths.slice(0, Math.max(0, maxLevels - 1))];
    levelNodeIndices = shownWidths.map((w, d) => {
      const count = Math.min(maxNodesPerLevel, Math.max(1, w));
      return Array.from({ length: count }, (_, i) => (d === 0 ? -1 : (d * 1000 + i)));
    });
    for (let d = 0; d < levelNodeIndices.length - 1; d++) {
      const cur = levelNodeIndices[d];
      const nxt = levelNodeIndices[d + 1];
      cur.forEach((cid, i) => {
        const j = Math.round((i / Math.max(1, cur.length - 1)) * Math.max(0, nxt.length - 1));
        edges.push([cid, nxt[j]]);
      });
    }
    acceptedSet = new Set();
    for (let d = 0; d < levelNodeIndices.length; d++) {
      if (d === 0 || d <= acceptLength) {
        levelNodeIndices[d].forEach((nid) => acceptedSet.add(nid));
      }
    }
  }

  const shownDepth = levelNodeIndices.length;
  const dy = shownDepth > 1 ? (yBottom - yTop) / (shownDepth - 1) : 0;
  const maxLevelWidth = Math.max(
    1,
    ...levelNodeIndices.map((ids) => (Array.isArray(ids) ? ids.length : 0))
  );
  const denseX = (xRight - xLeft) / Math.max(2, maxLevelWidth);
  const denseY = (yBottom - yTop) / Math.max(2, shownDepth);
  const nodeRadius = Math.max(2.6, Math.min(6.2, Math.min(denseX * 0.42, denseY * 0.38)));
  const edgeWidth = Math.max(0.9, Math.min(1.8, nodeRadius * 0.40));
  const acceptedEdgeWidth = Math.max(edgeWidth + 0.7, Math.min(3.0, nodeRadius * 0.72));
  const labelFontPx = Math.max(11, Math.min(22, Math.round(Math.min(denseY * 0.55, 22))));
  const minorFontPx = Math.max(10, labelFontPx - 1);
  const levelLabelStride = shownDepth > 24 ? Math.ceil(shownDepth / 20) : 1;
  const posByNode = new Map();
  const levelMeta = [];
  for (let d = 0; d < shownDepth; d++) {
    const ids = levelNodeIndices[d] || [];
    const shownCount = Math.max(1, Math.min(maxNodesPerLevel, ids.length || 1));
    const y = yTop + d * dy;
    const xs = [];
    for (let i = 0; i < shownCount; i++) {
      const x = shownCount > 1 ? xLeft + (i * (xRight - xLeft)) / (shownCount - 1) : (xLeft + xRight) / 2;
      xs.push(x);
      const nid = ids[i];
      if (nid !== undefined) posByNode.set(nid, { x, y, depth: d });
    }
    levelMeta.push({ y, shownCount, realCount: ids.length });
  }

  edges.forEach(([from, to]) => {
    const a = posByNode.get(from);
    const b = posByNode.get(to);
    if (!a || !b) return;
    const acceptedEdge = acceptedSet.has(from) && acceptedSet.has(to);
    ctx.lineWidth = acceptedEdge ? acceptedEdgeWidth : edgeWidth;
    ctx.strokeStyle = acceptedEdge ? acceptedEdgeColor : nonAcceptedEdgeColor;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });

  for (let d = 0; d < shownDepth; d++) {
    const ids = levelNodeIndices[d] || [];
    const realCount = (levelMeta[d] && levelMeta[d].realCount) ? levelMeta[d].realCount : 0;
    for (let i = 0; i < Math.min(ids.length, maxNodesPerLevel); i++) {
      const nid = ids[i];
      const pos = posByNode.get(nid);
      if (!pos) continue;
      const isRootNode = pos.depth === 0 || nid === -1;
      const isAccepted = acceptedSet.has(nid);
      ctx.beginPath();
      ctx.fillStyle = isRootNode ? rootFill : (isAccepted ? acceptedFill : nonAcceptedFill);
      ctx.arc(pos.x, pos.y, nodeRadius, 0, Math.PI * 2);
      ctx.fill();
    }
    const showLabel = d === 0 || d === shownDepth - 1 || d % levelLabelStride === 0;
    const isAcceptedLevel = d === 0 || d <= acceptLength;
    if (showLabel) {
      ctx.fillStyle = isAcceptedLevel ? labelColor : "#7b8794";
      ctx.font = `${labelFontPx}px system-ui`;
      const label = d === 0 ? "root:1" : `d${d}:${realCount}`;
      ctx.fillText(label, 4, (levelMeta[d] ? levelMeta[d].y : (yTop + d * dy)) - Math.max(3, nodeRadius + 1));
    }
    if (realCount > maxNodesPerLevel) {
      ctx.fillStyle = "#666";
      ctx.font = `${minorFontPx}px system-ui`;
      ctx.fillText(`+${realCount - maxNodesPerLevel}`, width - 52, (levelMeta[d] ? levelMeta[d].y : (yTop + d * dy)) - 5);
    }
  }

  ctx.fillStyle = "#4b5563";
  ctx.font = `${minorFontPx}px system-ui`;
  ctx.fillText(`accept_length=${acceptLength}`, 6, height - 6);
}

function setBubbleContent(div, text, tokenTrace = null) {
  const cleanText = sanitizeModelText(text);
  div.dataset.rawText = cleanText;
  if (tokenTrace) {
    div.dataset.tokenTrace = JSON.stringify(tokenTrace);
  } else {
    delete div.dataset.tokenTrace;
  }
  const content = ensureBubbleContentNode(div);
  if (!content) return;
  if (div.classList.contains("ai") && tokenColoringEnabled && tokenTrace) {
    const rendered = renderTokenTraceContent(content, tokenTrace, cleanText);
    if (!rendered) content.textContent = cleanText;
  } else {
    content.textContent = cleanText;
  }
}

function setBubbleMetrics(div, stats) {
  if (!div || !div.classList.contains("ai")) return;
  const old = div.querySelector(".bubble-metrics");
  if (old) old.remove();
  if (!stats || typeof stats !== "object") return;
  const tps = Number(stats.throughput);
  const draft = Number(stats.draft_cost);
  const target = Number(stats.target_cost);
  const comm = Number(stats.communication_cost || 0);
  const energy = Number(stats.gpu_energy);
  if (![tps, draft, target, energy].every(Number.isFinite)) return;
  const total = (Number.isFinite(comm) ? comm : 0) + draft + target;
  const m = document.createElement("div");
  m.className = "bubble-metrics";
  m.textContent = `throughput ${tps.toFixed(2)} tok/s | draft ${draft.toFixed(3)} $/1M | target ${target.toFixed(3)} $/1M | total ${total.toFixed(3)} $/1M | draft energy ${energy.toFixed(3)} kWh/1M`;
  div.appendChild(m);
}

function appendBubble(role, text, tokenTrace = null) {
  const div = document.createElement("div");
  div.className = `bubble ${role === "user" ? "user" : "ai"}`;
  setBubbleContent(div, text, tokenTrace);
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function appendHtmlBubble(role, html) {
  const div = document.createElement("div");
  div.className = `bubble ${role === "user" ? "user" : "ai"}`;
  if (role !== "user") div.classList.add("guide-bubble");
  const content = ensureBubbleContentNode(div);
  if (content) content.innerHTML = String(html || "");
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function recolorAllAiBubbles() {
  const bubbles = messagesEl.querySelectorAll(".bubble.ai");
  bubbles.forEach((b) => {
    const raw = b.dataset.rawText || b.textContent || "";
    const content = ensureBubbleContentNode(b);
    if (!content) return;
    let trace = null;
    if (b.dataset.tokenTrace) {
      try {
        trace = JSON.parse(b.dataset.tokenTrace);
      } catch (_) {
        trace = null;
      }
    }
    if (tokenColoringEnabled && trace) {
      const rendered = renderTokenTraceContent(content, trace, raw);
      if (!rendered) content.textContent = raw;
    } else {
      content.textContent = raw;
    }
  });
}

function sendSettings() {
  let maxNewTokens = parseInt(maxTokensInput.value, 10);
  if (!Number.isFinite(maxNewTokens) || maxNewTokens < 1) maxNewTokens = 512;
  maxNewTokens = Math.min(4096, maxNewTokens);
  maxTokensInput.value = String(maxNewTokens);
  const payload = {
    type: "settings",
    server: getServerValue(),
    selected_server_id: serverSelect ? serverSelect.value : null,
    selected_model_id: serverModelSelect ? serverModelSelect.value : null,
    target_quantization: serverQuantizationSelect ? serverQuantizationSelect.value : "none",
    draft_quantization: draftQuantizationSelect ? draftQuantizationSelect.value : "none",
    cost: parseFloat(costSlider.value),
    max_new_tokens: maxNewTokens,
    metric_preference: metricPreferenceSelect ? metricPreferenceSelect.value : "total_cost",
    constraint_target: getConstraintTarget(),
    metric_constraint_per_1m_token: constraintSlider ? parseFloat(constraintSlider.value) : 14.0,
    min_tps_constraint: tpsConstraintSlider ? parseFloat(tpsConstraintSlider.value) : 0.0,
    objective_selection_mode: (objectiveModeSelect && objectiveModeSelect.value === "constraint") ? "constraint" : "blend",
    algorithm: algorithmSelect.value,
    draft_model_path: draftModelSelect.value,
    benchmark_dataset: benchmarkDatasetSelect ? benchmarkDatasetSelect.value : "mt_bench",
    proactive_drafting: proactiveDrafting.checked,
    online_profile_update: true,
    online_profile_lr: 0.05,
    mode: modeSelect.value,
  };
  ws.send(JSON.stringify(payload));
}

function sendControl(action) {
  const payload = {
    type: "control",
    action,
    mode: modeSelect.value,
  };
  ws.send(JSON.stringify(payload));
}

function setSessionBadge(state, text) {
  const normalized = state || "idle";
  const modeClass = (modeSelect.value === "chat") ? "chat" : "benchmark";
  sessionBadge.classList.remove("idle", "running", "stopped", "error", "warn", "chat", "benchmark");
  sessionBadge.classList.add(normalized);
  sessionBadge.classList.add(modeClass);
  sessionBadge.textContent = text || normalized;
}

function setActivity(text, level = "idle") {
  activityBanner.classList.remove("idle", "running", "success", "warn", "error");
  activityBanner.classList.add(level);
  activityBanner.textContent = text;
}

function setControlBusy(isBusy) {
  startBtn.disabled = isBusy;
  stopBtn.disabled = isBusy;
  if (shutdownTargetBtn) shutdownTargetBtn.disabled = isBusy;
  if (isBusy) {
    startBtn.textContent = "Starting...";
    stopBtn.textContent = "Stopping...";
    if (shutdownTargetBtn) shutdownTargetBtn.textContent = "Shutting down...";
  } else {
    startBtn.textContent = "Start";
    stopBtn.textContent = "Stop";
    if (shutdownTargetBtn) shutdownTargetBtn.textContent = "Shutdown";
  }
  updateControlSelectionState();
}

function clearPendingControl() {
  pendingControl = null;
  if (pendingTimer) {
    clearTimeout(pendingTimer);
    pendingTimer = null;
  }
  setControlBusy(false);
  updateControlSelectionState();
}

function setControlsLocked(locked) {
  controlsLocked = !!locked;
  updateControlsState();
}

function startPendingControl(action, modeText) {
  pendingControl = action;
  selectedControl = action;
  updateControlSelectionState();
  if (pendingTimer) clearTimeout(pendingTimer);
  pendingTimer = setTimeout(() => {
    if (pendingControl) {
      setSessionBadge("warn", `No Ack (${modeText})`);
      setActivity(`No control_ack received for ${action}. Check Diagnostics / backend logs.`, "warn");
      clearPendingControl();
    }
  }, 6000);
}

function updateControlSelectionState() {
  startBtn.classList.toggle("active", selectedControl === "start");
  stopBtn.classList.toggle("active", selectedControl === "stop");
  if (shutdownTargetBtn) {
    shutdownTargetBtn.classList.toggle("active", selectedControl === "shutdown_target");
  }
  updateComposerThemeByControlSelection();
}

function updateComposerThemeByControlSelection() {
  const themeClasses = ["theme-start", "theme-stop", "theme-shutdown"];
  const phaseLower = String(latestPhase || "").toLowerCase();
  const isReferencePhase = phaseLower.startsWith("reference");
  const isStopIntent =
    selectedControl === "stop" || selectedControl === "refresh_reference";
  const isShutdownIntent = selectedControl === "shutdown_target";
  const shouldShowReadyTheme =
    latestChatReady &&
    !isReferencePhase &&
    !isStopIntent &&
    !isShutdownIntent;
  const nextTheme =
    isReferencePhase
      ? "theme-stop"
      : isShutdownIntent
      ? "theme-shutdown"
      : isStopIntent
        ? "theme-stop"
        : shouldShowReadyTheme
          ? "theme-start"
          : null;
  if (nextTheme) {
    currentComposerTheme = nextTheme;
  }
  const themeToApply = currentComposerTheme || "theme-stop";
  activityBanner.classList.remove(...themeClasses);
  if (inputRowEl) inputRowEl.classList.remove(...themeClasses);
  activityBanner.classList.add(themeToApply);
  if (inputRowEl) inputRowEl.classList.add(themeToApply);
}

function updateDiagnostics(msg) {
  diagPhase.textContent = msg.phase || "-";
  diagChatReady.textContent = String(!!msg.chat_ready);
  diagChatExitCode.textContent = (msg.chat_exit_code === null || msg.chat_exit_code === undefined) ? "-" : String(msg.chat_exit_code);
  diagLastLine.textContent = msg.last_line || "-";
  diagLastError.textContent = msg.last_error || "-";
}

function isChatInputReady() {
  if (modeSelect.value !== "chat") return true;
  return ws.readyState === WebSocket.OPEN && !!latestChatReady;
}

function sendChat() {
  if (!isChatInputReady()) {
    setActivity("Chat runtime is not ready yet. Wait until Ready (Chat).", "warn");
    return;
  }
  const text = inputEl.value.trim();
  if (!text) return;

  appendBubble("user", text);
  if (modeSelect.value === "chat") {
    streamingAiBubble = appendBubble("ai", "");
  } else {
    streamingAiBubble = null;
  }
  inputEl.value = "";
  inputEl.style.height = "auto";

  let maxNewTokens = parseInt(maxTokensInput.value, 10);
  if (!Number.isFinite(maxNewTokens) || maxNewTokens < 1) maxNewTokens = 512;
  maxNewTokens = Math.min(4096, maxNewTokens);
  maxTokensInput.value = String(maxNewTokens);

  const payload = {
    type: "chat",
    message: text,
    server: getServerValue(),
    selected_server_id: serverSelect ? serverSelect.value : null,
    selected_model_id: serverModelSelect ? serverModelSelect.value : null,
    target_quantization: serverQuantizationSelect ? serverQuantizationSelect.value : "none",
    draft_quantization: draftQuantizationSelect ? draftQuantizationSelect.value : "none",
    cost: parseFloat(costSlider.value),
    max_new_tokens: maxNewTokens,
    metric_preference: metricPreferenceSelect ? metricPreferenceSelect.value : "total_cost",
    constraint_target: getConstraintTarget(),
    metric_constraint_per_1m_token: constraintSlider ? parseFloat(constraintSlider.value) : 14.0,
    min_tps_constraint: tpsConstraintSlider ? parseFloat(tpsConstraintSlider.value) : 0.0,
    objective_selection_mode: (objectiveModeSelect && objectiveModeSelect.value === "constraint") ? "constraint" : "blend",
    algorithm: algorithmSelect.value,
    benchmark_dataset: benchmarkDatasetSelect ? benchmarkDatasetSelect.value : "mt_bench",
    proactive_drafting: proactiveDrafting.checked,
    online_profile_update: true,
    online_profile_lr: 0.05,
    mode: modeSelect.value,
  };

  ws.send(JSON.stringify(payload));
}

ws.addEventListener("open", () => {
  latestChatReady = false;
  latestPhase = "idle";
  latestPhaseLabel = "idle";
  currentComposerTheme = "theme-stop";
  updateComposerThemeByControlSelection();
  setSessionBadge("idle", "Idle");
  setActivity("Connected. Ready.", "success");
  sendSettings();
  requestServerCatalog();
  updateControlsState(); // Set the initial state
  renderUiHelp();
});

ws.addEventListener("error", (error) => {
  console.error("WebSocket error:", error);
  setSessionBadge("error", "Socket Error");
  setActivity("WebSocket error. Check server/console logs.", "error");
});

ws.addEventListener("close", () => {
  console.log("WebSocket closed");
  latestChatReady = false;
  latestPhase = "idle";
  latestPhaseLabel = "idle";
  currentComposerTheme = "theme-stop";
  updateComposerThemeByControlSelection();
  setControlsLocked(false);
  setSessionBadge("stopped", "Disconnected");
  setActivity("Disconnected from backend.", "warn");
});

ws.addEventListener("message", (ev) => {
  try {
    const msg = JSON.parse(ev.data);

    if (msg.type === "chat_partial") {
      const partialReply = String(msg.reply || "");
      const partialTrace = msg.token_trace || null;
      if (!streamingAiBubble) {
        streamingAiBubble = appendBubble("ai", partialReply, partialTrace);
      } else {
        setBubbleContent(streamingAiBubble, partialReply, partialTrace);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
      const ps = (msg && msg.stats) ? msg.stats : null;
      if (ps) {
        renderTreeSketch(ps);
        drawMiniTree(ps);
        maybeAddSeriesPoint("gpu_energy", gpuEnergyChart, ps.gpu_energy);
        maybeAddSeriesPoint("draft_cost", draftCostChart, ps.draft_cost);
        maybeAddSeriesPoint("target_cost", targetCostChart, ps.target_cost);
        maybeAddSeriesPoint("throughput", throughputChart, ps.throughput);
      }
    } else if (msg.type === "chat_reply") {
      const finalReply = String(msg.reply || "");
      const finalTrace = msg.token_trace || null;
      const finalStats = msg.final_stats || null;
      if (finalStats && typeof finalStats === "object") {
        maybeAddSeriesPoint("gpu_energy", gpuEnergyChart, finalStats.gpu_energy);
        maybeAddSeriesPoint("draft_cost", draftCostChart, finalStats.draft_cost);
        maybeAddSeriesPoint("target_cost", targetCostChart, finalStats.target_cost);
        maybeAddSeriesPoint("throughput", throughputChart, finalStats.throughput);
      }
      if (Array.isArray(msg.reference_tradeoff_curve_cs0_1)) {
        tradeoffCurveBlend = msg.reference_tradeoff_curve_cs0_1.map((r) => ({
          cost_per_1m: Number(r.predicted_metric_per_1m_token),
          tps: Number(r.predicted_tps),
          selector_value: Number(r.cost_sensitivity),
        }));
      }
      if (Array.isArray(msg.reference_tradeoff_curve_by_constraint)) {
        tradeoffCurveConstraint = msg.reference_tradeoff_curve_by_constraint.map((r) => ({
          cost_per_1m: Number(r.predicted_metric_per_1m_token),
          tps: Number(r.predicted_tps),
          selector_value: Number(
            getConstraintTarget() === "tps"
              ? (r.min_tps_constraint !== undefined ? r.min_tps_constraint : r.predicted_tps)
              : r.metric_constraint_per_1m_token
          ),
        }));
      }
      const fr = msg.feasible_constraint_range_per_1m;
      if (fr && Number.isFinite(Number(fr.min)) && Number.isFinite(Number(fr.max))) {
        feasibleConstraintMin = Number(fr.min);
        feasibleConstraintMax = Number(fr.max);
      }
      const ft = msg.feasible_tps_range;
      if (ft && Number.isFinite(Number(ft.min)) && Number.isFinite(Number(ft.max))) {
        feasibleTpsMin = Number(ft.min);
        feasibleTpsMax = Number(ft.max);
      }
      if (!streamingAiBubble) {
        const b = appendBubble("ai", finalReply, finalTrace);
        setBubbleMetrics(b, finalStats);
      } else {
        setBubbleContent(streamingAiBubble, finalReply, finalTrace);
        setBubbleMetrics(streamingAiBubble, finalStats);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
      streamingAiBubble = null;
      drawTradeoffChart();
    } else if (msg.type === "control_ack") {
      clearPendingControl();
      appendBubble("ai", msg.message);
      const t = (msg.message || "").toLowerCase();
      const ackAction = String(msg.action || "").toLowerCase();
      const modeText = modeSelect.value === "chat" ? "Chat" : "Benchmark";
      if (ackAction === "start") {
        if (t.includes("failed") || t.includes("error")) {
          setControlsLocked(false);
          setSessionBadge("error", `Start Failed (${modeText})`);
          setActivity(msg.message, "error");
        } else {
          setControlsLocked(true);
          setSessionBadge("running", `Running (${modeText})`);
          setActivity(`Started ${modeText.toLowerCase()} session.`, "success");
        }
      } else if (ackAction === "stop") {
        setControlsLocked(false);
        resetTradeoffViewAfterStop();
        setSessionBadge("stopped", `Stopped (${modeText})`);
        setActivity(msg.message || `Stopped ${modeText.toLowerCase()} session.`, "warn");
      } else if (ackAction === "shutdown_target") {
        setControlsLocked(false);
        resetTradeoffViewAfterStop();
        if (t.includes("failed") || t.includes("error")) {
          setSessionBadge("error", "Target Shutdown Failed");
          setActivity(msg.message, "error");
        } else {
          setSessionBadge("stopped", "Target Shutdown");
          setActivity(msg.message || "Target shutdown requested.", "warn");
        }
      } else if (ackAction === "refresh_reference") {
        if (t.includes("refreshing reference")) {
          tradeoffDisplayMode = "reference";
          setSessionBadge("running", `Refreshing Reference (${modeText})`);
          setActivity(msg.message, "running");
        } else if (t.includes("reference cache refreshed")) {
          tradeoffDisplayMode = "reference";
          setSessionBadge("running", `Ready (${modeText})`);
          setActivity(msg.message, "success");
          selectedControl = latestChatReady ? "start" : null;
          updateControlSelectionState();
          drawTradeoffChart();
        } else if (t.includes("reference refresh failed")) {
          setSessionBadge("error", `Reference Refresh Failed (${modeText})`);
          setActivity(msg.message, "error");
          selectedControl = latestChatReady ? "start" : null;
          updateControlSelectionState();
        } else {
          setActivity(msg.message, "idle");
        }
      } else if (t.includes("started")) {
        setControlsLocked(true);
        setSessionBadge("running", `Running (${modeText})`);
        setActivity(`Started ${modeText.toLowerCase()} session.`, "success");
      } else if (t.includes("stopped") || t.includes("stopping")) {
        setControlsLocked(false);
        setSessionBadge("stopped", `Stopped (${modeText})`);
        setActivity(`Stopped ${modeText.toLowerCase()} session.`, "warn");
      } else if (t.includes("start")) {
        setControlsLocked(true);
        setSessionBadge("running", `Starting (${modeText})...`);
        setActivity(msg.message, "running");
      } else if (t.includes("refreshing reference")) {
        setSessionBadge("running", `Refreshing Reference (${modeText})`);
        setActivity(msg.message, "running");
      } else if (t.includes("reference cache refreshed")) {
        setSessionBadge("running", `Ready (${modeText})`);
        setActivity(msg.message, "success");
      } else if (t.includes("reference refresh failed")) {
        setSessionBadge("error", `Reference Refresh Failed (${modeText})`);
        setActivity(msg.message, "error");
      }
    } else if (msg.type === "error") {
      streamingAiBubble = null;
      clearPendingControl();
      setControlsLocked(false);
      const err = msg.message || "Unknown backend error";
      appendBubble("ai", `Error: ${err}`);
      setSessionBadge("error", "Backend Error");
      setActivity(`Backend error: ${err}`, "error");
    } else if (msg.type === "server_catalog") {
      if (Array.isArray(msg.servers)) {
        const sig = getServerCatalogSignature(msg.servers);
        if (sig !== serverCatalogSignature) {
          const prevServer = serverSelect ? serverSelect.value : null;
          const prevModel = serverModelSelect ? serverModelSelect.value : null;
          serverCandidates = msg.servers;
          serverCatalogSignature = sig;
          renderServerCatalog(prevServer, prevModel);
        }
      }
      updateRecommendationHint();
    } else if (msg.type === "server_add_result") {
      if (msg.ok) {
        setActivity("Server added.", "success");
        resetAddServerForm();
        setPopup(null);
        requestServerCatalog();
      } else {
        setActivity("Failed to add server.", "error");
      }
    } else if (msg.type === "server_update_key_result") {
      if (msg.ok) {
        setActivity("API key updated.", "success");
        requestServerCatalog();
      } else {
        setActivity("Failed to update API key.", "error");
      }
    } else if (msg.type === "server_remove_result") {
      if (msg.ok) {
        setActivity("Server removed.", "warn");
        requestServerCatalog();
      } else {
        setActivity("Cannot remove server.", "error");
      }
    } else if (msg.type === "probe_status") {
      if (msg.running) setActivity("Profiling server candidates...", "running");
    } else if (msg.type === "probe_result") {
      tradeoffDisplayMode = "probe";
      if (Array.isArray(msg.rows)) serverProbeRows = msg.rows;
      if (Array.isArray(msg.curves)) serverProbeCurves = msg.curves;
      if (typeof msg.curve_mode === "string") serverProbeCurveMode = msg.curve_mode;
      if (msg.summary && typeof msg.summary === "object") recommendationSummary = msg.summary;
      updateRecommendationHint();
      drawTradeoffChart();
      setActivity("Profiling complete.", "success");
    } else if (msg.type === "recommendation_result") {
      if (Array.isArray(serverProbeRows) && serverProbeRows.length) tradeoffDisplayMode = "probe";
      if (Array.isArray(msg.rows)) serverProbeRows = msg.rows;
      if (Array.isArray(msg.curves)) serverProbeCurves = msg.curves;
      if (typeof msg.curve_mode === "string") serverProbeCurveMode = msg.curve_mode;
      if (msg.summary && typeof msg.summary === "object") recommendationSummary = msg.summary;
      updateRecommendationHint();
      drawTradeoffChart();
    } else if (msg.type === "stats") {
      updateDiagnostics(msg);
      if (Array.isArray(msg.server_candidates)) {
        const sig = getServerCatalogSignature(msg.server_candidates);
        if (sig !== serverCatalogSignature) {
          const prevServer = serverSelect ? serverSelect.value : null;
          const prevModel = serverModelSelect ? serverModelSelect.value : null;
          serverCandidates = msg.server_candidates;
          serverCatalogSignature = sig;
          renderServerCatalog(prevServer, prevModel);
        }
      }
      if (Array.isArray(msg.server_probe_rows)) {
        serverProbeRows = msg.server_probe_rows;
      }
      if (Array.isArray(msg.server_probe_curves)) {
        serverProbeCurves = msg.server_probe_curves;
      }
      if (typeof msg.server_probe_curve_mode === "string") {
        serverProbeCurveMode = msg.server_probe_curve_mode;
      }
      if (msg.recommendations && typeof msg.recommendations === "object") {
        recommendationSummary = msg.recommendations;
      }
      if (metricPreferenceSelect && typeof msg.metric_preference === "string") {
        metricPreferenceSelect.value = msg.metric_preference;
      }
      if (benchmarkDatasetSelect && typeof msg.benchmark_dataset === "string") {
        benchmarkDatasetSelect.value = msg.benchmark_dataset;
      }
      if (constraintTargetSelect && typeof msg.constraint_target === "string") {
        constraintTargetSelect.value = (msg.constraint_target === "tps") ? "tps" : "metric";
      }
      if (constraintSlider && msg.metric_constraint_per_1m_token !== undefined && msg.metric_constraint_per_1m_token !== null) {
        const metricBudget = Number(msg.metric_constraint_per_1m_token);
        if (Number.isFinite(metricBudget)) {
          constraintSlider.value = String(metricBudget);
          if (constraintValue) constraintValue.textContent = metricBudget.toFixed(1);
        }
      }
      if (tpsConstraintSlider && msg.min_tps_constraint !== undefined && msg.min_tps_constraint !== null) {
        const minTps = Number(msg.min_tps_constraint);
        if (Number.isFinite(minTps)) {
          tpsConstraintSlider.value = String(minTps);
          if (tpsConstraintValue) tpsConstraintValue.textContent = minTps.toFixed(1);
          updateTpsConstraintHint();
        }
      }
      updateRecommendationHint();
      const constraintTarget = getConstraintTarget();
      if (Array.isArray(msg.reference_tradeoff_curve_cs0_1)) {
        tradeoffCurveBlend = msg.reference_tradeoff_curve_cs0_1.map((r) => ({
          cost_per_1m: Number(r.predicted_metric_per_1m_token),
          tps: Number(r.predicted_tps),
          selector_value: Number(r.cost_sensitivity),
        }));
      }
      if (Array.isArray(msg.reference_tradeoff_curve_by_constraint)) {
        tradeoffCurveConstraint = msg.reference_tradeoff_curve_by_constraint.map((r) => ({
          cost_per_1m: Number(r.predicted_metric_per_1m_token),
          tps: Number(r.predicted_tps),
          selector_value: Number(
            constraintTarget === "tps"
              ? (r.min_tps_constraint !== undefined ? r.min_tps_constraint : r.predicted_tps)
              : r.metric_constraint_per_1m_token
          ),
        }));
      } else if (Array.isArray(msg.reference_constraint_anchor_curve)) {
        // Fallback for older/newer caches without interpolated constraint trade-off curve
        tradeoffCurveConstraint = msg.reference_constraint_anchor_curve.map((r) => ({
          cost_per_1m: Number(
            r.predicted_metric_per_1m_token !== undefined
              ? r.predicted_metric_per_1m_token
              : r.predicted_objective_per_1m_token
          ),
          tps: Number(r.predicted_tps),
          selector_value: Number(
            constraintTarget === "tps"
              ? (r.min_tps_constraint !== undefined ? r.min_tps_constraint : r.predicted_tps)
              : r.metric_constraint_per_1m_token
          ),
        }));
      }
      const fr = msg.feasible_constraint_range_per_1m;
      if (fr && Number.isFinite(Number(fr.min)) && Number.isFinite(Number(fr.max))) {
        feasibleConstraintMin = Number(fr.min);
        feasibleConstraintMax = Number(fr.max);
      } else if (fr === null) {
        feasibleConstraintMin = null;
        feasibleConstraintMax = null;
      }
      const ft = msg.feasible_tps_range;
      if (ft && Number.isFinite(Number(ft.min)) && Number.isFinite(Number(ft.max))) {
        feasibleTpsMin = Number(ft.min);
        feasibleTpsMax = Number(ft.max);
      } else if (ft === null) {
        feasibleTpsMin = null;
        feasibleTpsMax = null;
      }
      clampConstraintToFeasible();
      updateTpsConstraintHint();
      updateConstraintSliderVisual();
      drawTradeoffChart();
      // Update stats data
      const isChat = modeSelect.value === "chat";
      const isRunning = isChat ? !!msg.chat_running : !!msg.running;
      latestChatReady = !!msg.chat_ready;
      latestPhase = String(msg.phase || "");
      latestPhaseLabel = String(msg.phase_label || "");
      updateComposerThemeByControlSelection();
      const phase = (msg.phase || "").toLowerCase();
      const phaseLabel = msg.phase_label || "";
      const isBootingPhase =
        phase.startsWith("chat_") ||
        phase === "starting" ||
        phase === "loading_model" ||
        phase === "warmup" ||
        phase.startsWith("reference");

      // Release the lock immediately when the runtime actually stops, even if the Stop request ACK is delayed.
      if (pendingControl === "stop" && !isRunning && !isBootingPhase) {
        clearPendingControl();
        setControlsLocked(false);
      }

      // Do not overwrite Idle from stats while waiting immediately after a start/stop request.
      if (pendingControl) {
        if (phaseLabel && phaseLabel.toLowerCase() !== "idle") {
          const modeText = isChat ? "Chat" : "Benchmark";
          setSessionBadge("running", `${phaseLabel} (${modeText})`);
          setActivity(`${phaseLabel} (${modeText})`, "running");
        }
      } else if ((isRunning || isBootingPhase) && phaseLabel) {
        const modeText = isChat ? "Chat" : "Benchmark";
        if (phase.includes("error")) {
          setSessionBadge("error", `${phaseLabel} (${modeText})`);
          setActivity(`${phaseLabel} (${modeText})`, "error");
        } else {
          setSessionBadge("running", `${phaseLabel} (${modeText})`);
          setActivity(`${phaseLabel} (${modeText})`, "running");
        }
      } else if (!isRunning && phaseLabel) {
        const modeText = isChat ? "Chat" : "Benchmark";
        if (phase.includes("error")) {
          setControlsLocked(false);
          setSessionBadge("error", `${phaseLabel} (${modeText})`);
          setActivity(`${phaseLabel} (${modeText})`, "error");
        } else {
          setControlsLocked(false);
          setSessionBadge("stopped", `${phaseLabel} (${modeText})`);
          setActivity(`${phaseLabel} (${modeText})`, "warn");
        }
      }
      maybeAddSeriesPoint("gpu_energy", gpuEnergyChart, msg.gpu_energy);
      maybeAddSeriesPoint("draft_cost", draftCostChart, msg.draft_cost);
      maybeAddSeriesPoint("target_cost", targetCostChart, msg.target_cost);
      maybeAddSeriesPoint("throughput", throughputChart, msg.throughput);
    }
  } catch (error) {
    console.error("Error parsing message:", error, ev.data);
    setSessionBadge("error", "Parse Error");
    setActivity("Message parse error.", "error");
  }
});

if (tradeoffCanvas) {
  tradeoffCanvas.addEventListener("mousemove", (e) => {
    const points = getActiveTradeoffPoints();
    if (!points.length) return;
    const rect = tradeoffCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const nearest = findNearestTradeoffPoint(points, x, y, rect.width, rect.height);
    const nextUid = nearest && nearest.uid ? nearest.uid : null;
    if (tradeoffHoverPointUid !== nextUid) {
      tradeoffHoverPointUid = nextUid;
      drawTradeoffChart();
    }
  });
  tradeoffCanvas.addEventListener("mouseleave", () => {
    if (tradeoffHoverPointUid !== null) {
      tradeoffHoverPointUid = null;
      drawTradeoffChart();
    }
  });
  tradeoffCanvas.addEventListener("click", (e) => {
    const points = getActiveTradeoffPoints();
    if (!points.length) return;
    const rect = tradeoffCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const best = findNearestTradeoffPoint(points, x, y, rect.width, rect.height) || points[0];
    if (best.source === "probe" || best.source === "probe_curve") {
      if (serverSelect) serverSelect.value = String(best.server_id);
      syncServerModelSelect(best.model_id);
      if (serverModelSelect) serverModelSelect.value = String(best.model_id);
      renderServerDropdown();
      if (Number.isFinite(Number(best.selector))) {
        if (serverProbeCurveMode === "constraint") {
          if (getConstraintTarget() === "tps" && tpsConstraintSlider) {
            tpsConstraintSlider.value = String(best.selector);
            if (tpsConstraintValue) tpsConstraintValue.textContent = Number(tpsConstraintSlider.value).toFixed(1);
            updateTpsConstraintHint();
          } else if (constraintSlider) {
            constraintSlider.value = String(best.selector);
            clampConstraintToFeasible();
            if (constraintValue) constraintValue.textContent = Number(constraintSlider.value).toFixed(1);
          }
        } else if (costSlider) {
          costSlider.value = String(Number(best.selector).toFixed(2));
          if (costValue) costValue.textContent = Number(costSlider.value).toFixed(2);
        }
      }
      drawTradeoffChart();
      if (ws.readyState === WebSocket.OPEN) sendSettings();
      return;
    }
    const objectiveMode = (objectiveModeSelect && objectiveModeSelect.value === "constraint") ? "constraint" : "balanced";
    if (objectiveMode === "constraint") {
      if (getConstraintTarget() === "tps" && tpsConstraintSlider) {
        tpsConstraintSlider.value = String(best.selector);
        if (tpsConstraintValue) tpsConstraintValue.textContent = Number(tpsConstraintSlider.value).toFixed(1);
        updateTpsConstraintHint();
      } else if (constraintSlider) {
        constraintSlider.value = String(best.selector);
        clampConstraintToFeasible();
        constraintValue.textContent = Number(constraintSlider.value).toFixed(1);
      }
    } else {
      costSlider.value = String(Number(best.selector).toFixed(2));
      costValue.textContent = Number(costSlider.value).toFixed(2);
    }
    drawTradeoffChart();
    if (ws.readyState === WebSocket.OPEN) sendSettings();
  });
}

sendBtn.addEventListener("click", sendChat);
if (howToUseBtn) {
  howToUseBtn.addEventListener("click", () => {
    appendHtmlBubble("ai", HOW_TO_USE_HTML);
    setActivity("Displayed usage guide in chat.", "success");
  });
}

// Enter sends; Shift+Enter inserts a newline
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!isChatInputReady()) return;
    sendChat();
  }
});

// Auto-resize textarea height
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = (inputEl.scrollHeight) + "px";
});

// Display slider UI and send it to Python
costSlider.addEventListener("input", () => {
  if (!costSlider.disabled) {
    costValue.textContent = Number(costSlider.value).toFixed(2);
    drawTradeoffChart();
    if (ws.readyState === WebSocket.OPEN) sendSettings();
  }
});
if (constraintSlider) {
  constraintSlider.addEventListener("input", () => {
    if (!constraintSlider.disabled) {
      clampConstraintToFeasible();
      constraintValue.textContent = Number(constraintSlider.value).toFixed(1);
      drawTradeoffChart();
      if (ws.readyState === WebSocket.OPEN) sendSettings();
    }
  });
  constraintSlider.addEventListener("change", () => {
    if (!constraintSlider.disabled && ws.readyState === WebSocket.OPEN) {
      clampConstraintToFeasible();
      sendSettings();
    }
  });
}
if (constraintTargetSelect) {
  constraintTargetSelect.addEventListener("change", () => {
    tradeoffCurveConstraint = null;
    updateControlsState();
    drawTradeoffChart();
    if (ws.readyState === WebSocket.OPEN) sendSettings();
  });
}
if (tpsConstraintSlider) {
  tpsConstraintSlider.addEventListener("input", () => {
    if (!tpsConstraintSlider.disabled) {
      if (tpsConstraintValue) tpsConstraintValue.textContent = Number(tpsConstraintSlider.value).toFixed(1);
      updateTpsConstraintHint();
      if (ws.readyState === WebSocket.OPEN) sendSettings();
    }
  });
  tpsConstraintSlider.addEventListener("change", () => {
    if (!tpsConstraintSlider.disabled && ws.readyState === WebSocket.OPEN) {
      updateTpsConstraintHint();
      sendSettings();
    }
  });
}
costSlider.addEventListener("change", () => {
  if (!costSlider.disabled && ws.readyState === WebSocket.OPEN) {
    sendSettings();
  }
});

if (serverSelect) {
  serverSelect.addEventListener("change", () => {
    syncServerModelSelect();
    refreshDraftModelCompatibility(true);
    enforceAlgorithmPolicyByServer(true);
    renderServerDropdown();
    updateServerModeHint();
    updateControlsState();
    if (ws.readyState === WebSocket.OPEN) sendSettings();
  });
}
if (serverDropdownBtn && serverDropdownMenu) {
  serverDropdownBtn.addEventListener("click", () => {
    if (serverDropdownBtn.disabled) return;
    serverDropdownMenu.classList.toggle("hidden");
  });
}
if (serverDropdown) {
  document.addEventListener("click", (e) => {
    if (!serverDropdown.contains(e.target)) {
      closeServerDropdown();
    }
  });
}
if (serverModelSelect) {
  serverModelSelect.addEventListener("change", () => {
    refreshDraftModelCompatibility(true);
    if (ws.readyState === WebSocket.OPEN) sendSettings();
  });
}
if (metricPreferenceSelect) {
  metricPreferenceSelect.addEventListener("change", () => {
    if (ws.readyState !== WebSocket.OPEN) return;
    sendSettings();
    ws.send(JSON.stringify({
      type: "recommendation_request",
      metric_preference: metricPreferenceSelect.value,
    }));
  });
}
if (serverQuantizationSelect) {
  serverQuantizationSelect.addEventListener("change", () => {
    if (ws.readyState === WebSocket.OPEN) sendSettings();
  });
}
if (draftQuantizationSelect) {
  draftQuantizationSelect.addEventListener("change", () => {
    if (ws.readyState === WebSocket.OPEN) sendSettings();
  });
}
if (serverSetKeyBtn) {
  serverSetKeyBtn.addEventListener("click", () => {
    setPopup(activePopup === "key" ? null : "key");
  });
}
if (serverAddToggleBtn) {
  serverAddToggleBtn.addEventListener("click", () => {
    setPopup(activePopup === "add" ? null : "add");
  });
}
if (closeServerKeyPopupBtn) {
  closeServerKeyPopupBtn.addEventListener("click", () => setPopup(null));
}
if (closeAddServerPopupBtn) {
  closeAddServerPopupBtn.addEventListener("click", () => setPopup(null));
}
if (floatingActionHost) {
  floatingActionHost.addEventListener("click", (e) => {
    if (e.target === floatingActionHost) {
      setPopup(null);
    }
  });
}
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && activePopup) {
    setPopup(null);
    return;
  }
  if (e.key === "Escape") {
    closeServerDropdown();
  }
});
if (probeBtn) {
  probeBtn.addEventListener("click", () => {
    if (ws.readyState !== WebSocket.OPEN) return;
    tradeoffDisplayMode = "probe";
    flashButtonActive(probeBtn);
    ws.send(JSON.stringify({
      type: "probe_start",
      metric_preference: metricPreferenceSelect ? metricPreferenceSelect.value : "total_cost",
      benchmark_dataset: benchmarkDatasetSelect ? benchmarkDatasetSelect.value : "mt_bench",
      model_overrides: serverSelect && serverModelSelect && serverSelect.value
        ? { [serverSelect.value]: serverModelSelect.value }
        : {},
      measured_runs: 1,
      warmup_runs: 1,
      timeout_s: 25.0,
      max_tokens: 64,
    }));
  });
}
if (addServerBtn) {
  addServerBtn.addEventListener("click", () => {
    if (ws.readyState !== WebSocket.OPEN) return;
    const name = newServerName ? newServerName.value.trim() : "";
    const host = newServerHost ? newServerHost.value.trim() : "";
    const portRaw = newServerPort ? newServerPort.value.trim() : "";
    const port = Number.parseInt(portRaw, 10);
    if (!name || !host || !Number.isInteger(port) || port < 1 || port > 65535) {
      setActivity("Enter valid Server Name, IP Address, and Port.", "error");
      return;
    }
    const payload = {
      type: "server_add",
      name,
      host,
      port,
    };
    ws.send(JSON.stringify(payload));
  });
}
if (updateServerKeyBtn) {
  updateServerKeyBtn.addEventListener("click", () => {
    if (ws.readyState !== WebSocket.OPEN || !serverSelect || !serverSelect.value || !selectedServerApiKey) return;
    const key = selectedServerApiKey.value.trim();
    if (!key) return;
    ws.send(JSON.stringify({ type: "server_update_key", server_id: serverSelect.value, api_key: key }));
    selectedServerApiKey.value = "";
    setPopup(null);
  });
}
if (jumpFastestBtn) {
  jumpFastestBtn.addEventListener("click", () => {
    flashButtonActive(jumpFastestBtn);
    jumpToRecommendation("fastest");
  });
  jumpFastestBtn.addEventListener("mouseenter", () => {
    recommendationHoverKind = "fastest";
    drawTradeoffChart();
  });
  jumpFastestBtn.addEventListener("mouseleave", () => {
    recommendationHoverKind = null;
    drawTradeoffChart();
  });
}
if (jumpBestEfficiencyBtn) {
  jumpBestEfficiencyBtn.addEventListener("click", () => {
    flashButtonActive(jumpBestEfficiencyBtn);
    jumpToRecommendation("best_efficiency");
  });
  jumpBestEfficiencyBtn.addEventListener("mouseenter", () => {
    recommendationHoverKind = "best_efficiency";
    drawTradeoffChart();
  });
  jumpBestEfficiencyBtn.addEventListener("mouseleave", () => {
    recommendationHoverKind = null;
    drawTradeoffChart();
  });
}
if (jumpParetoBtn) {
  jumpParetoBtn.addEventListener("click", () => {
    flashButtonActive(jumpParetoBtn);
    jumpToRecommendation("pareto");
  });
  jumpParetoBtn.addEventListener("mouseenter", () => {
    recommendationHoverKind = "pareto";
    drawTradeoffChart();
  });
  jumpParetoBtn.addEventListener("mouseleave", () => {
    recommendationHoverKind = null;
    drawTradeoffChart();
  });
}

algorithmSelect.addEventListener("change", () => {
  if (controlsLocked) return;
  updateControlsState(); // Update control state when the algorithm changes
  if (ws.readyState === WebSocket.OPEN) sendSettings();
});

draftModelSelect.addEventListener("change", () => {
  if (controlsLocked) return;
  if (ws.readyState === WebSocket.OPEN) sendSettings();
});

modeSelect.addEventListener("change", () => {
  if (controlsLocked) return;
  updateControlsState();
  renderUiHelp();
  const modeText = modeSelect.value === "chat" ? "Chat" : "Benchmark";
  setSessionBadge("idle", `Idle (${modeText})`);
  setActivity(`Mode changed to ${modeText}.`, "idle");
  if (ws.readyState === WebSocket.OPEN) sendSettings();
});

if (benchmarkDatasetSelect) {
  benchmarkDatasetSelect.addEventListener("change", () => {
    if (controlsLocked) return;
    if (ws.readyState === WebSocket.OPEN) sendSettings();
  });
}

if (helpRefreshBtn) {
  helpRefreshBtn.addEventListener("click", () => {
    renderUiHelp();
    setActivity("Help content refreshed for current mode.", "success");
  });
}
if (fillTemplateExternalBtn) {
  fillTemplateExternalBtn.addEventListener("click", () => fillServerFormTemplate("external"));
}
if (fillTemplateGroqBtn) {
  fillTemplateGroqBtn.addEventListener("click", () => fillServerFormTemplate("groq"));
}
if (fillTemplateHfBtn) {
  fillTemplateHfBtn.addEventListener("click", () => fillServerFormTemplate("hf"));
}
if (fillTemplateLocalBtn) {
  fillTemplateLocalBtn.addEventListener("click", () => fillServerFormTemplate("local"));
}
if (clearTemplateBtn) {
  clearTemplateBtn.addEventListener("click", () => fillServerFormTemplate("clear"));
}

if (objectiveModeSelect) {
  objectiveModeSelect.addEventListener("change", () => {
    if (controlsLocked) return;
    updateControlsState();
    if (ws.readyState === WebSocket.OPEN) {
      sendSettings();
      ws.send(JSON.stringify({
        type: "recommendation_request",
        metric_preference: metricPreferenceSelect ? metricPreferenceSelect.value : "total_cost",
      }));
    }
  });
}

maxTokensInput.addEventListener("change", () => {
  let v = parseInt(maxTokensInput.value, 10);
  if (!Number.isFinite(v) || v < 1) v = 512;
  v = Math.min(4096, v);
  maxTokensInput.value = String(v);
  if (ws.readyState === WebSocket.OPEN) sendSettings();
});

proactiveDrafting.addEventListener("change", () => {
  if (!proactiveDrafting.disabled && ws.readyState === WebSocket.OPEN) {
    sendSettings();
  }
});

startBtn.addEventListener("click", () => {
  const modeText = modeSelect.value === "chat" ? "Chat" : "Benchmark";
  if (ws.readyState !== WebSocket.OPEN) {
    setSessionBadge("error", "Not Connected");
    setActivity("Cannot start: websocket is not connected.", "error");
    return;
  }
  setSessionBadge("running", `Starting (${modeText})...`);
  setActivity(`Sending start command (${modeText})...`, "running");
  setControlBusy(true);
  startPendingControl("start", modeText);
  sendSettings();
  sendControl("start");
});

stopBtn.addEventListener("click", () => {
  const modeText = modeSelect.value === "chat" ? "Chat" : "Benchmark";
  if (ws.readyState !== WebSocket.OPEN) {
    setSessionBadge("error", "Not Connected");
    setActivity("Cannot stop: websocket is not connected.", "error");
    return;
  }
  setSessionBadge("stopped", `Stopping (${modeText})...`);
  setActivity(`Sending stop command (${modeText})...`, "warn");
  setControlBusy(true);
  startPendingControl("stop", modeText);
  sendControl("stop");
});

if (shutdownTargetBtn) {
  shutdownTargetBtn.addEventListener("click", () => {
    if (ws.readyState !== WebSocket.OPEN) {
      setSessionBadge("error", "Not Connected");
      setActivity("Cannot shutdown target: websocket is not connected.", "error");
      return;
    }
    setSessionBadge("warn", "Shutting Down Target...");
    setActivity("Sending shutdown_target command...", "warn");
    setControlBusy(true);
    startPendingControl("shutdown_target", "Target");
    sendControl("shutdown_target");
  });
}

function triggerProfileRefresh(detailedProfile) {
  if (ws.readyState !== WebSocket.OPEN) {
    setSessionBadge("error", "Not Connected");
    setActivity("Cannot refresh: websocket is not connected.", "error");
    return;
  }
  // Ensure backend uses the latest UI selections (server/model/quantization)
  // before starting reference refresh.
  sendSettings();
  selectedControl = "refresh_reference";
  updateControlSelectionState();
  tradeoffDisplayMode = "reference";
  if (detailedProfile) {
    if (refreshReferenceDetailedBtn) flashButtonActive(refreshReferenceDetailedBtn);
    setActivity("Refreshing reference cache with detailed profiling...", "running");
  } else {
    if (refreshReferenceBtn) flashButtonActive(refreshReferenceBtn);
    setActivity("Refreshing reference cache...", "running");
  }
  ws.send(JSON.stringify({ type: "refresh_reference", detailed_profile: !!detailedProfile }));
}

if (refreshReferenceBtn) {
  refreshReferenceBtn.addEventListener("click", () => {
    triggerProfileRefresh(false);
  });
}

if (refreshReferenceDetailedBtn) {
  refreshReferenceDetailedBtn.addEventListener("click", () => {
    triggerProfileRefresh(true);
  });
}

// Set state on initial load
updateControlsState();
renderUiHelp();
setPopup(null);
refreshDraftModelCompatibility();
renderServerDropdown();
syncTopStripPanelHeights();
drawMiniTree(null);
drawTradeoffChart();
const tokenColoringToggle = document.getElementById("tokenColoringToggle");
if (tokenColoringToggle) {
  tokenColoringEnabled = !!tokenColoringToggle.checked;
  tokenColoringToggle.addEventListener("change", () => {
    tokenColoringEnabled = !!tokenColoringToggle.checked;
    recolorAllAiBubbles();
  });
}

// ========== Stats Graph-related code ==========
class RealTimeChart {
  constructor(canvasId, color = '#1f4f89') {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.color = color;
    this.data = [];
    this.maxDataPoints = 50; // Maximum number of displayed data points
    this.padding = { top: 5, right: 5, bottom: 15, left: 5 };
    
    // Set canvas size
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }
  
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * window.devicePixelRatio;
    this.canvas.height = rect.height * window.devicePixelRatio;
    this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    this.draw();
  }
  
  addDataPoint(value) {
    this.data.push(value);
    if (this.data.length > this.maxDataPoints) {
      this.data.shift(); // Remove old data
    }
    this.draw();
  }
  
  draw() {
    const width = this.canvas.width / window.devicePixelRatio;
    const height = this.canvas.height / window.devicePixelRatio;
    const ctx = this.ctx;
    
    // Clear background
    ctx.clearRect(0, 0, width, height);
    
    if (this.data.length === 0) return;
    
    // Calculate data range
    const min = Math.min(...this.data, 0);
    const max = Math.max(...this.data, 1) || 1;
    const range = max - min || 1;
    
    const chartWidth = width - this.padding.left - this.padding.right;
    const chartHeight = height - this.padding.top - this.padding.bottom;
    
    // Draw grid lines
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = this.padding.top + (chartHeight / 4) * i;
      ctx.beginPath();
      ctx.moveTo(this.padding.left, y);
      ctx.lineTo(width - this.padding.right, y);
      ctx.stroke();
    }
    
    // Draw data line
    if (this.data.length > 1) {
      ctx.strokeStyle = this.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      this.data.forEach((value, index) => {
        const x = this.padding.left + (chartWidth / (this.data.length - 1)) * index;
        const y = this.padding.top + chartHeight - ((value - min) / range) * chartHeight;
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.stroke();
      
      // Draw data points
      ctx.fillStyle = this.color;
      this.data.forEach((value, index) => {
        const x = this.padding.left + (chartWidth / (this.data.length - 1)) * index;
        const y = this.padding.top + chartHeight - ((value - min) / range) * chartHeight;
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fill();
      });
    }
    
    // Display latest value
    if (this.data.length > 0) {
      const latestValue = this.data[this.data.length - 1];
      ctx.fillStyle = '#333';
      ctx.font = '21px system-ui';
      ctx.textAlign = 'right';
      ctx.fillText(latestValue.toFixed(2), width - this.padding.right, this.padding.top + 12);
    }
  }
}

// Graph instance (initialize after DOM load)
let gpuEnergyChart, draftCostChart, targetCostChart, throughputChart;

// Initialize graphs after DOM load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initCharts);
} else {
  initCharts();
}

function initCharts() {
  gpuEnergyChart = new RealTimeChart('gpuEnergyChart', '#e74c3c');
  draftCostChart = new RealTimeChart('draftCostChart', '#3498db');
  targetCostChart = new RealTimeChart('targetCostChart', '#2ecc71');
  throughputChart = new RealTimeChart('throughputChart', '#f39c12');
}

window.addEventListener("resize", () => {
  syncTopStripPanelHeights();
  drawTradeoffChart();
  drawMiniTree(null);
});

if (typeof ResizeObserver !== "undefined" && topControlsBox) {
  const topStripResizeObserver = new ResizeObserver(() => {
    syncTopStripPanelHeights();
    drawTradeoffChart();
  });
  topStripResizeObserver.observe(topControlsBox);
}
