(function () {
  const PG_BRAND = {
    name: "PrivaGen™",
    bbd: "Barren Business Development",
    product: "a Barren Business Development Product",
  };

  const $ = (id) => document.getElementById(id);
  const folderCounts = $("folderCounts");
  const statusLine = $("statusLine");
  const progressBar = $("progressBar");
  const progressPct = $("progressPct");
  const liveLog = $("liveLog");
  const batchSizeInput = $("batchSize");
  const batchSizeNum = $("batchSizeNum");
  const btnStart = $("btnStart");
  const btnStop = $("btnStop");
  const btnRefresh = $("btnRefresh");
  const btnLogs = $("btnLogs");
  const readyBadge = $("readyBadge");
  const pipelineSection = $("pipelineSection");
  const setupTerminal = $("setupTerminal");
  const btnInstallDeps = $("btnInstallDeps");
  const btnRefreshEnv = $("btnRefreshEnv");

  const pathFields = {
    input_raw: "pathInputRaw",
    final_clean: "pathFinalClean",
    quarantine: "pathQuarantine",
    manual_review: "pathManualReview",
    reports: "pathReports",
    logs: "pathLogs",
    temp_processed: "pathTemp",
  };

  let totalTarget = 0;
  let lastProcessed = 0;
  let logLines = [];
  let setupLines = [];
  let pipelineReady = false;
  let installRunning = false;
  let datasetConfig = { source_mode: "local", local_path: "input_raw" };
  // Image-Only scope: operator must explicitly confirm before Start is enabled.
  let imageOnlyConfirmed = false;

  function appendLog(line) {
    const ts = new Date().toISOString().slice(11, 19);
    logLines.push(`[${ts}] ${line}`);
    if (logLines.length > 500) logLines = logLines.slice(-500);
    liveLog.textContent = logLines.join("\n");
    liveLog.scrollTop = liveLog.scrollHeight;
  }

  function appendSetup(line, isErr) {
    const ts = new Date().toISOString().slice(11, 19);
    setupLines.push(`[${ts}] ${line}`);
    if (setupLines.length > 2000) setupLines = setupLines.slice(-2000);
    setupTerminal.textContent = setupLines.join("\n");
    setupTerminal.scrollTop = setupTerminal.scrollHeight;
    if (isErr) appendLog(`setup: ${line}`);
  }

  function setDot(id, state) {
    const el = $(id);
    if (!el) return;
    el.classList.remove("ok", "fail", "warn");
    if (state === "ok") el.classList.add("ok");
    else if (state === "warn") el.classList.add("warn");
    else if (state === "fail") el.classList.add("fail");
  }

  function setInstallProgress(pct, label) {
    const bar = $("installProgressBar");
    const lbl = $("installPhaseLabel");
    const pctEl = $("installPhasePct");
    if (bar) bar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    if (lbl && label) lbl.textContent = label;
    if (pctEl) pctEl.textContent = pct >= 0 ? `${Math.round(pct)}%` : "—";
  }

  function setPipelineRunAllowed(canStart) {
    pipelineReady = !!canStart;
    if (pipelineSection) {
      pipelineSection.classList.remove("pg-disabled");
      pipelineSection.removeAttribute("aria-disabled");
    }
    const banner = $("pipelineLimitedBanner");
    if (banner) {
      if (canStart) banner.classList.add("pg-hidden");
      else banner.classList.remove("pg-hidden");
    }
    updateStartEnabled();
  }

  // Single source of truth for the Start button: env ready AND Image-Only confirmed
  // AND a valid run scope + batch size (and no install in progress).
  function updateStartEnabled() {
    if (!btnStart) return;
    if (installRunning) {
      btnStart.disabled = true;
      return;
    }
    btnStart.disabled = !(pipelineReady && imageOnlyConfirmed && runScopeValid());
  }

  function currentBatchSize() {
    const v = parseInt((batchSizeInput && batchSizeInput.value) || "32", 10);
    if (!Number.isFinite(v)) return 32;
    return Math.max(8, Math.min(128, v));
  }

  // Returns { runType: "pilot"|"full", batchSize, maxImages }.
  // Pilot: maxImages = explicit N images, or (N batches × batch size) when batches set.
  function getRunSettings() {
    const batchSize = currentBatchSize();
    const full = !!($("scopeFull") && $("scopeFull").checked);
    if (full) return { runType: "full", batchSize, maxImages: 0 };
    const batchesEl = $("pilotBatches");
    const imagesEl = $("pilotImages");
    const nBatches = parseInt((batchesEl && batchesEl.value) || "", 10);
    let maxImages;
    if (Number.isFinite(nBatches) && nBatches > 0) {
      maxImages = nBatches * batchSize;
    } else {
      maxImages = parseInt((imagesEl && imagesEl.value) || "", 10);
    }
    return { runType: "pilot", batchSize, maxImages };
  }

  function runScopeValid() {
    const s = getRunSettings();
    if (s.batchSize < 8 || s.batchSize > 128) return false;
    if (s.runType === "pilot") return Number.isFinite(s.maxImages) && s.maxImages >= 1;
    return true;
  }

  function updateRunSummary() {
    const el = $("runSummary");
    if (!el) return;
    const s = getRunSettings();
    if (s.runType === "full") {
      el.textContent = `Full dataset run — will process all available images in batches of ${s.batchSize}.`;
    } else if (Number.isFinite(s.maxImages) && s.maxImages >= 1) {
      const nb = Math.ceil(s.maxImages / s.batchSize);
      el.textContent = `Pilot run — will process the first ${s.maxImages.toLocaleString()} images in batches of ${s.batchSize} (~${nb} batch${nb === 1 ? "" : "es"}).`;
    } else {
      el.textContent = "Pilot run — enter the number of images (or batches) to process.";
    }
  }

  function confirmImageOnlyMode() {
    imageOnlyConfirmed = true;
    const msg = $("modeConfirmMsg");
    const req = $("modeRequiredMsg");
    if (msg) {
      msg.textContent = "Image-Only workflow locked. Videos will be automatically excluded and ignored.";
      msg.classList.remove("pg-hidden");
    }
    if (req) req.classList.add("pg-hidden");
    const panel = $("processingModePanel");
    if (panel) panel.classList.add("pg-mode-panel--confirmed");
    updateStartEnabled();
    appendLog("Processing mode confirmed: Images Only (video support deferred).");
  }

  function updateReportLinks() {
    const q = statsQuery();
    const base = (path) => `/api/reports/${path}${q || ""}`;
    const csv = $("linkCsv");
    const pdf = $("linkPdf");
    if (csv) csv.href = base("master_summary.csv");
    if (pdf) pdf.href = base("master_summary.pdf");
  }

  function applyEnvironment(env) {
    if (!env) return;
    const badge = $("overallReadinessBadge");
    const summary = $("readinessSummary");
    const warnings = $("envWarnings");

    const readiness = env.readiness || "not_ready";
    const label = env.readiness_label || readiness;

    if (badge) {
      badge.textContent = label;
      badge.className = "pg-badge";
      if (readiness === "ready_gpu") badge.classList.add("pg-badge--ready-gpu");
      else if (readiness === "ready_cpu") badge.classList.add("pg-badge--ready-cpu");
      else badge.classList.add("pg-badge--not-ready");
    }
    if (summary) summary.textContent = label;

    setDot("dotRequirements", env.requirements_ok ? "ok" : "fail");
    setDot(
      "dotGpu",
      env.cuda_available ? "ok" : env.torch_available ? "warn" : "fail"
    );
    const paddle = (env.components || {}).paddleocr || {};
    setDot("dotPaddle", paddle.ok ? "ok" : env.requirements_ok ? "fail" : "fail");
    const iopaint = (env.components || {}).iopaint || {};
    setDot("dotIopaint", iopaint.ok ? "ok" : env.requirements_ok ? "warn" : "fail");
    const dp2 = (env.components || {}).deep_privacy2 || {};
    setDot("dotDp2", dp2.ok ? "ok" : env.requirements_ok ? "warn" : "fail");

    const conda = env.conda || {};
    setDot("dotConda", conda.active ? "ok" : "warn");
    const condaLabel = $("condaLabel");
    if (condaLabel) {
      condaLabel.textContent = conda.env_name
        ? `Conda: ${conda.env_name}`
        : `Python: ${(env.python_executable || "").split(/[/\\]/).pop() || "system"}`;
    }

    if (warnings) {
      const warns = env.warnings || [];
      if (env.compute_message && !warns.includes(env.compute_message)) {
        warns.unshift(env.compute_message);
      }
      if (warns.length) {
        warnings.classList.remove("pg-hidden");
        warnings.innerHTML = warns.map((w) => `<div>⚠ ${w}</div>`).join("");
      } else {
        warnings.classList.add("pg-hidden");
        warnings.innerHTML = "";
      }
    }

    const canRun = readiness === "ready_gpu" || readiness === "ready_cpu";
    setPipelineRunAllowed(canRun && !installRunning);

    if (canRun) {
      setInstallProgress(100, "Environment ready");
    } else if (!installRunning) {
      setInstallProgress(0, "Waiting for install");
    }
  }

  async function fetchEnvironment(refresh) {
    try {
      const url = refresh ? "/api/environment/check" : "/api/environment";
      const r = await fetch(url, refresh ? { method: "POST" } : {});
      const j = await r.json();
      installRunning = !!j.install_running;
      applyEnvironment(j.environment);
      if (btnInstallDeps) btnInstallDeps.disabled = installRunning;
      return j;
    } catch (e) {
      appendSetup(`environment fetch error: ${e}`, true);
      return null;
    }
  }

  function statsQuery() {
    const p = new URLSearchParams();
    for (const [key, elId] of Object.entries(pathFields)) {
      const v = $(elId).value.trim();
      if (v) p.set(key, v);
    }
    const q = p.toString();
    return q ? `?${q}` : "";
  }

  async function refreshStats() {
    try {
      const r = await fetch(`/api/stats${statsQuery()}`);
      const j = await r.json();
      if (j.environment) applyEnvironment(j.environment);
      if (j.pipeline_ready !== undefined) {
        setPipelineRunAllowed(!!j.pipeline_ready && !installRunning);
      }
      const c = j.counts || {};
      const modeTag =
        j.stats_mode === "light"
          ? ' <span class="pg-section-desc">(light scan)</span>'
          : "";
      folderCounts.innerHTML = [
        `input_raw (ready): <span class="pg-accent-text">${c.input_raw ?? 0}</span>${modeTag}`,
        `quarantine (retry): <span style="color: var(--pg-warn)">${c.quarantine ?? 0}</span>`,
        `final_clean (done): <span class="pg-accent-text">${c.final_clean ?? 0}</span>`,
        `manual_review: <span style="color: var(--pg-danger)">${c.manual_review ?? 0}</span>`,
      ].join("<br/>");
      if (j.message && !pipelineReady) {
        folderCounts.innerHTML += `<br/><span class="pg-section-desc">${j.message}</span>`;
      }
      const sec = j.security || {};
      const gpu = j.gpu || {};
      const snap = gpu.snapshot || {};
      $("secLevel").textContent = sec.level || "standard";
      $("secCrypt").textContent = sec.crypt_enabled ? "enabled" : "off";
      $("gpuDevice").textContent = gpu.device || "—";
      $("gpuVram").textContent =
        snap.allocated_mb != null ? String(snap.allocated_mb) : snap.cuda_available ? "0" : "n/a";
      $("manifestEntries").textContent = String((j.manifest || {}).entries ?? "—");
      const banner = $("cpuFallbackBanner");
      if (banner) {
        const msg = j.user_message || (j.compute_profile || {}).user_message;
        if (j.cpu_fallback && msg) {
          banner.textContent = msg;
          banner.classList.remove("pg-hidden");
        } else {
          banner.classList.add("pg-hidden");
        }
      }
      totalTarget = Math.max(totalTarget, (c.input_raw || 0) + (c.quarantine || 0));
    } catch (e) {
      appendLog(`stats error: ${e}`);
    }
  }

  async function refreshStatus() {
    try {
      const r = await fetch("/api/status");
      const j = await r.json();
      if (j.pipeline_ready !== undefined) setPipelineRunAllowed(!!j.pipeline_ready && !installRunning);
      const st = j.status || "idle";
      const stopping = j.stop_requested && st === "running";
      if (st === "running") {
        statusLine.textContent = stopping
          ? "Stopping — will exit after the current batch completes."
          : "Running — pipeline is active.";
        readyBadge.textContent = stopping ? "Stopping…" : "Busy";
        readyBadge.className = "pg-badge " + (stopping ? "pg-badge--stopping" : "pg-badge--busy");
        btnStart.disabled = true;
        btnStop.disabled = false;
      } else {
        statusLine.textContent = pipelineReady
          ? "Idle — ready to start."
          : "Install dependencies in Setup Environment first.";
        readyBadge.textContent = pipelineReady ? "Pipeline ready" : "Not ready";
        readyBadge.className = "pg-badge " + (pipelineReady ? "pg-badge--ready-gpu" : "pg-badge--not-ready");
        updateStartEnabled();
        btnStop.disabled = true;
        if (j.last_error) appendLog(`last error: ${j.last_error}`);
      }
      if (j.live) applyLiveStatus(j.live);
    } catch (e) {
      appendLog(`status error: ${e}`);
    }
  }

  function setProgress(processed, hint) {
    const denom = Math.max(hint || 0, processed, totalTarget, 1);
    const pct = Math.min(100, Math.round((100 * processed) / denom));
    progressBar.style.width = `${pct}%`;
    progressPct.textContent = `${pct}%`;
  }

  function formatEta(sec) {
    if (sec == null || !Number.isFinite(Number(sec))) return "—";
    const s = Math.max(0, Math.floor(Number(sec)));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const r = s % 60;
    if (h) return `${h}h ${m}m`;
    if (m) return `${m}m ${r}s`;
    return `${r}s`;
  }

  function applyLiveStatus(live) {
    if (!live) return;
    const badge = $("liveStatusBadge");
    const st = live.status || "idle";
    if (badge) {
      badge.textContent = st === "running" ? "Running" : st === "error" ? "Error" : "Idle";
      badge.className = "live-badge rounded-full px-3 py-1 text-xs font-semibold border ";
      if (st === "running") badge.classList.add("live-running");
      else if (st === "error") badge.classList.add("live-error");
      else badge.classList.add("live-idle");
    }
    const detected = live.total_detected ?? 0;
    const tgt = live.total_target || totalTarget || 0;
    const proc = live.processed ?? 0;
    if ($("liveTotalDetected")) {
      $("liveTotalDetected").textContent = String(detected || tgt || "—");
    }
    if ($("liveProcessedLine")) {
      $("liveProcessedLine").textContent =
        tgt > 0 ? `${proc} / ${tgt}` : proc > 0 ? `${proc} / ?` : `0 / —`;
    }
    const curBatch = live.current_batch || 0;
    const tbe = live.total_batches_estimate || 0;
    if ($("liveBatchLine")) {
      $("liveBatchLine").textContent =
        curBatch && tbe ? `Batch ${curBatch} of ${tbe}` : curBatch ? `Batch ${curBatch}` : "—";
    }
    if ($("liveBatchSize")) {
      const bs = live.current_batch_size;
      $("liveBatchSize").textContent =
        bs != null && bs > 0 ? `Batch size: ${bs} images` : "Batch size: —";
    }
    const pct = live.progress_pct ?? 0;
    if ($("liveProgressBar")) $("liveProgressBar").style.width = `${Math.min(100, pct)}%`;
    if ($("liveProgressPct")) $("liveProgressPct").textContent = `${pct.toFixed(1)}%`;
    if ($("liveProgressLabel")) {
      $("liveProgressLabel").textContent =
        tgt > 0 ? `Overall progress (${proc} / ${tgt} images)` : "Overall progress";
    }
    const ips = live.images_per_sec;
    if ($("liveThroughput")) {
      $("liveThroughput").textContent =
        ips != null && ips > 0 ? `${Number(ips).toFixed(2)} img/s` : "—";
    }
    if ($("liveEta")) $("liveEta").textContent = `ETA: ${formatEta(live.eta_sec)}`;
    const modeEl = $("liveComputeMode");
    const explEl = $("liveComputeExplanation");
    const panel = $("liveComputePanel");
    const mode = live.compute_mode || "—";
    if (modeEl) modeEl.textContent = mode;
    if (panel) {
      panel.classList.remove("mode-gpu", "mode-cpu");
      if (live.cpu_fallback || String(mode).toLowerCase().includes("cpu")) {
        panel.classList.add("mode-cpu");
        if (badge) badge.classList.add("live-cpu");
      } else if (String(mode).toLowerCase().includes("gpu")) {
        panel.classList.add("mode-gpu");
      }
    }
    if (explEl) {
      const expl = live.mode_explanation || "";
      if (expl) {
        explEl.textContent = expl;
        explEl.classList.remove("pg-hidden");
      } else {
        explEl.classList.add("pg-hidden");
      }
    }
    setProgress(proc, tgt || detected);
    if (tgt > 0) totalTarget = Math.max(totalTarget, tgt);
    lastProcessed = proc;
    $("mProcessed").textContent = String(proc);
    if (live.success_rate != null) {
      $("mSuccess").textContent = `${(live.success_rate * 100).toFixed(1)}%`;
    }
    if (live.images_per_sec != null && live.images_per_sec > 0) {
      $("mRate").textContent = `${Number(live.images_per_sec).toFixed(2)} img/s`;
    }
    $("mEta").textContent = formatEta(live.eta_sec);
  }

  function datasetPayload() {
    const mode = $("datasetModeB2")?.checked ? "b2" : "local";
    const batchRaw = parseInt($("datasetB2BatchSize")?.value, 10);
    return {
      source_mode: mode,
      local_path: ($("datasetLocalPath")?.value || "input_raw").trim(),
      b2_remote_path: ($("datasetB2Path")?.value || "").trim(),
      b2_ingest_on_start: !!$("datasetB2IngestOnStart")?.checked,
      b2_export_remote_path: ($("datasetB2ExportPath")?.value || "").trim(),
      b2_export_on_complete: !!$("datasetB2ExportOnComplete")?.checked,
      b2_transfer_batch_size: Number.isFinite(batchRaw) && batchRaw > 0 ? batchRaw : 32,
      sync_to_input_raw: true,
    };
  }

  function applyB2Overview(b2) {
    if (!b2) return;
    if ($("b2ReadonlyBucket")) {
      $("b2ReadonlyBucket").textContent = b2.readonly_bucket || "(set B2_READONLY_BUCKET)";
    }
    if ($("b2WriteBucket")) {
      $("b2WriteBucket").textContent = b2.write_bucket || "(set B2_WRITE_BUCKET)";
    }
    if (b2.ingest_remote_path && $("datasetB2Path") && !$("datasetB2Path").value) {
      $("datasetB2Path").value = b2.ingest_remote_path;
    }
    if (b2.export_remote_path && $("datasetB2ExportPath") && !$("datasetB2ExportPath").value) {
      $("datasetB2ExportPath").value = b2.export_remote_path;
    }
    if ($("datasetB2BatchSize") && b2.transfer_batch_size) {
      $("datasetB2BatchSize").value = String(b2.transfer_batch_size);
    }
  }

  async function loadB2Overview() {
    try {
      const r = await fetch("/api/b2/overview");
      const j = await r.json();
      if (j.b2) {
        b2OverviewCache = j.b2;
        applyB2Overview(j.b2);
      }
      if (j.ui) {
        if ($("datasetB2ExportPath") && j.ui.b2_export_remote_path) {
          $("datasetB2ExportPath").value = j.ui.b2_export_remote_path;
        }
        if ($("datasetB2ExportOnComplete")) {
          $("datasetB2ExportOnComplete").checked = !!j.ui.b2_export_on_complete;
        }
      }
      if ($("datasetModeB2")?.checked) scheduleRclonePreview();
    } catch (e) {
      appendLog(`b2 overview: ${e}`);
    }
  }

  let b2OverviewCache = null;
  let rclonePreviewTimer = null;

  function rclonePreviewQuery() {
    const p = new URLSearchParams();
    const ingest = ($("datasetB2Path")?.value || "").trim();
    const exportP = ($("datasetB2ExportPath")?.value || "").trim();
    const batch = parseInt($("datasetB2BatchSize")?.value, 10);
    if (ingest) p.set("ingest_path", ingest);
    if (exportP) p.set("export_path", exportP);
    if (Number.isFinite(batch) && batch > 0) p.set("batch_size", String(batch));
    const ro = b2OverviewCache?.readonly_bucket;
    const wr = b2OverviewCache?.write_bucket;
    if (ro) p.set("readonly_bucket", ro);
    if (wr) p.set("write_bucket", wr);
    if ($("enableCrypt")?.checked) p.set("crypt_enabled", "1");
    if ($("testMode")?.checked) p.set("dry_run", "1");
    return p.toString();
  }

  function formatRcloneReference(ref) {
    if (!ref) return "(no reference)";
    const lines = [];
    lines.push("═══════════════════════════════════════════════════════════");
    lines.push(`  ${PG_BRAND.name} — ${PG_BRAND.product}`);
    lines.push("  Preview Rclone Commands (Backblaze B2)");
    lines.push("═══════════════════════════════════════════════════════════\n");

    const sec = ref.security_summary || {};
    if (sec.ingest) {
      lines.push("▶ INGEST — READ-ONLY (original data, copy only)");
      lines.push(`  ${sec.ingest}\n`);
    }
    if (sec.export) {
      lines.push("▶ EXPORT — WRITE BUCKET (anonymized products only)");
      lines.push(`  ${sec.export}\n`);
    }
    if (ref.placeholders_active) {
      lines.push("ℹ Placeholders (<source-bucket>, <ingest-path>, …) update as you fill fields above.\n");
    }

    const cmd = ref.commands || {};
    const ingestKeys = [
      "ingest_list",
      "ingest_copy_all",
      "ingest_copy_per_batch",
      "ingest_note",
      "write_config",
    ];
    const exportKeys = [
      "export_copy_all",
      "export_copy_per_batch",
      "export_verify_check",
      "export_note",
      "buyer_decrypt_crypt",
    ];

    lines.push("── INGEST COMMANDS (b2-readonly → input_raw/) ──\n");
    ingestKeys.forEach((k) => {
      if (!cmd[k]) return;
      lines.push(`[${k}]`);
      lines.push(String(cmd[k]));
      lines.push("");
    });

    lines.push("── EXPORT COMMANDS (final_clean/ → b2-write) ──\n");
    exportKeys.forEach((k) => {
      if (!cmd[k]) return;
      lines.push(`[${k}]`);
      lines.push(String(cmd[k]));
      lines.push("");
    });

    if (ref.paths) {
      lines.push("── RESOLVED PATHS (live) ──");
      Object.entries(ref.paths).forEach(([k, v]) => lines.push(`  ${k}: ${v}`));
    }
    return lines.join("\n");
  }

  async function refreshRclonePreview() {
    if (!$("datasetModeB2")?.checked) return;
    const pane = $("rcloneCommandsPane");
    const hint = $("rclonePlaceholderHint");
    try {
      const qs = rclonePreviewQuery();
      const r = await fetch(`/api/b2/commands?${qs}`);
      const j = await r.json();
      if (!r.ok || !j.reference) {
        if (pane) pane.textContent = j.error || `Preview unavailable (HTTP ${r.status})`;
        return;
      }
      if (pane) pane.textContent = formatRcloneReference(j.reference);
      if (hint) {
        hint.classList.toggle("pg-hidden", !j.reference.placeholders_active);
      }
      const sum = $("rcloneSecuritySummary");
      if (sum && j.reference.security_summary) {
        const s = j.reference.security_summary;
        sum.innerHTML =
          `<strong>Read-only ingest:</strong> ${s.ingest || ""}` +
          `<br /><strong>Write export:</strong> ${s.export || ""}`;
      }
    } catch (e) {
      if (pane) pane.textContent = `Preview error: ${e}`;
    }
  }

  // Safety card: render the ACTUAL generated copy-only commands (single source of
  // truth = /api/b2/commands). Always shown (not gated behind B2 selection) so the
  // operator has full, unambiguous review of exactly what will run.
  function formatSafetyCommands(ref) {
    if (!ref) return "(no reference)";
    const cmd = ref.commands || {};
    const out = [];
    out.push("# INGEST — READ-ONLY / COPY-ONLY (source is only read + copied; never move/sync/delete)");
    if (cmd.ingest_list) out.push(cmd.ingest_list);
    if (cmd.ingest_copy_all) out.push(cmd.ingest_copy_all);
    out.push("");
    out.push("# EXPORT — COPY anonymized outputs to a SEPARATE destination bucket (write key)");
    if (cmd.export_copy_all) out.push(cmd.export_copy_all);
    if (cmd.export_verify_check) {
      out.push("");
      out.push("# VERIFY — checksum compare local -> remote (read-only check)");
      out.push(cmd.export_verify_check);
    }
    if (ref.placeholders_active) {
      out.push("");
      out.push("# NOTE: <…> placeholders resolve once your source/dest buckets and paths are configured.");
    }
    return out.join("\n");
  }

  async function refreshSafetyRclone() {
    const pane = $("safetyRclonePane");
    if (!pane) return;
    try {
      const r = await fetch(`/api/b2/commands?${rclonePreviewQuery()}`);
      const j = await r.json();
      if (!r.ok || !j.reference) {
        pane.textContent = j.error || `Live commands unavailable (HTTP ${r.status})`;
        return;
      }
      pane.textContent = formatSafetyCommands(j.reference);
    } catch (e) {
      pane.textContent = `Live commands error: ${e}`;
    }
  }

  function scheduleRclonePreview() {
    if (rclonePreviewTimer) clearTimeout(rclonePreviewTimer);
    rclonePreviewTimer = setTimeout(() => {
      refreshRclonePreview();
      refreshSafetyRclone();
    }, 200);
  }

  async function previewRcloneCommands() {
    await refreshRclonePreview();
  }

  function applyDatasetConfig(cfg) {
    if (!cfg) return;
    datasetConfig = cfg;
    const mode = (cfg.source_mode || "local").toLowerCase();
    if ($("datasetModeLocal")) $("datasetModeLocal").checked = mode !== "b2";
    if ($("datasetModeB2")) $("datasetModeB2").checked = mode === "b2";
    if ($("datasetLocalPath")) $("datasetLocalPath").value = cfg.local_path || "input_raw";
    if ($("datasetB2Path")) $("datasetB2Path").value = cfg.b2_remote_path || "";
    if ($("datasetB2IngestOnStart")) $("datasetB2IngestOnStart").checked = !!cfg.b2_ingest_on_start;
    if ($("datasetB2ExportPath")) $("datasetB2ExportPath").value = cfg.b2_export_remote_path || "";
    if ($("datasetB2ExportOnComplete")) $("datasetB2ExportOnComplete").checked = !!cfg.b2_export_on_complete;
    if ($("datasetB2BatchSize") && cfg.b2_transfer_batch_size) {
      $("datasetB2BatchSize").value = String(cfg.b2_transfer_batch_size);
    }
    toggleDatasetPanels();
    if ((cfg.source_mode || "").toLowerCase() === "b2") scheduleRclonePreview();
    const scan = cfg.last_scan;
    if (scan && typeof scan === "object") {
      updateDatasetScanUI(scan);
    }
  }

  function toggleDatasetPanels() {
    const b2 = $("datasetModeB2")?.checked;
    if ($("datasetLocalPanel")) $("datasetLocalPanel").classList.toggle("pg-hidden", !!b2);
    if ($("datasetB2Panel")) $("datasetB2Panel").classList.toggle("pg-hidden", !b2);
    if (b2) {
      scheduleRclonePreview();
    } else if ($("rcloneCommandsPane")) {
      $("rcloneCommandsPane").textContent = "Select Backblaze B2 above to load command preview…";
    }
  }

  function updateDatasetScanUI(scan) {
    const n = scan.image_count ?? 0;
    if ($("datasetTotalImages")) $("datasetTotalImages").textContent = String(n);
    if ($("liveTotalDetected")) $("liveTotalDetected").textContent = String(n);
    totalTarget = Math.max(totalTarget, n);
    if ($("datasetScanMeta")) {
      const parts = [];
      if (scan.source_mode) parts.push(scan.source_mode);
      if (scan.path_relative || scan.path_display) {
        parts.push(scan.path_relative || scan.path_display);
      } else if (scan.remote_display) parts.push(scan.remote_display);
      if (scan.error) parts.push(`error: ${scan.error}`);
      $("datasetScanMeta").textContent = parts.join(" · ") || (scan.ok ? "Scan OK" : "Scan failed");
    }
  }

  async function loadDatasetConfig() {
    try {
      const r = await fetch("/api/dataset/config");
      const j = await r.json();
      if (j.config) applyDatasetConfig(j.config);
    } catch (e) {
      appendLog(`dataset config load: ${e}`);
    }
  }

  async function saveDatasetConfig() {
    try {
      const body = {
        ...datasetPayload(),
        persist_yaml: !!$("datasetPersistYaml")?.checked,
      };
      const r = await fetch("/api/dataset/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const j = await r.json();
      if (!r.ok) {
        appendLog(`save dataset failed: ${j.error || r.status}`);
        return;
      }
      applyDatasetConfig(j.config);
      appendLog("Dataset configuration saved.");
    } catch (e) {
      appendLog(`save dataset error: ${e}`);
    }
  }

  async function scanDataset() {
    try {
      if ($("datasetScanMeta")) $("datasetScanMeta").textContent = "Scanning…";
      const r = await fetch("/api/dataset/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(datasetPayload()),
      });
      const j = await r.json();
      if (j.config) applyDatasetConfig(j.config);
      if (j.scan) updateDatasetScanUI(j.scan);
      appendLog(
        `dataset scan: ${j.scan?.image_count ?? 0} images` +
          (j.scan?.error ? ` (${j.scan.error})` : "")
      );
    } catch (e) {
      appendLog(`scan dataset error: ${e}`);
    }
  }

  if ($("datasetModeLocal")) $("datasetModeLocal").addEventListener("change", toggleDatasetPanels);
  if ($("datasetModeB2")) $("datasetModeB2").addEventListener("change", toggleDatasetPanels);
  ["datasetB2Path", "datasetB2ExportPath", "datasetB2BatchSize"].forEach((id) => {
    const el = $(id);
    if (el) {
      el.addEventListener("input", scheduleRclonePreview);
      el.addEventListener("change", scheduleRclonePreview);
    }
  });
  if ($("datasetB2IngestOnStart")) $("datasetB2IngestOnStart").addEventListener("change", scheduleRclonePreview);
  if ($("datasetB2ExportOnComplete")) $("datasetB2ExportOnComplete").addEventListener("change", scheduleRclonePreview);
  if ($("enableCrypt")) $("enableCrypt").addEventListener("change", scheduleRclonePreview);
  if ($("btnScanDataset")) $("btnScanDataset").addEventListener("click", scanDataset);
  if ($("btnSaveDataset")) $("btnSaveDataset").addEventListener("click", saveDatasetConfig);
  if ($("btnPreviewRclone")) $("btnPreviewRclone").addEventListener("click", previewRcloneCommands);
  if ($("btnBrowseFolder")) {
    $("btnBrowseFolder").addEventListener("click", () => $("datasetFolderPicker")?.click());
  }
  if ($("datasetFolderPicker")) {
    $("datasetFolderPicker").addEventListener("change", (ev) => {
      const files = ev.target.files;
      if (!files || !files.length) return;
      const rel = files[0].webkitRelativePath || "";
      const top = rel.split("/")[0] || rel.split("\\")[0];
      if (top && $("datasetLocalPath")) {
        $("datasetLocalPath").value = top;
        appendLog(`Browse hint: using folder name "${top}" — verify path on disk.`);
      }
    });
  }

  function syncBatchSize(source) {
    let v = parseInt((source === "num" ? batchSizeNum.value : batchSizeInput.value) || "32", 10);
    if (!Number.isFinite(v)) v = 32;
    v = Math.max(8, Math.min(128, v));
    if (batchSizeInput) batchSizeInput.value = String(Math.round(v / 8) * 8 || 8);
    if (batchSizeNum) batchSizeNum.value = String(v);
    updateRunSummary();
    updateStartEnabled();
  }
  if (batchSizeInput) batchSizeInput.addEventListener("input", () => syncBatchSize("range"));
  if (batchSizeNum) batchSizeNum.addEventListener("input", () => syncBatchSize("num"));

  function applyRunScopeVisibility() {
    const full = !!($("scopeFull") && $("scopeFull").checked);
    const pilot = $("pilotFields");
    const note = $("fullScopeNote");
    if (pilot) pilot.classList.toggle("pg-hidden", full);
    if (note) note.classList.toggle("pg-hidden", !full);
    updateRunSummary();
    updateStartEnabled();
  }
  ["scopePilot", "scopeFull"].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener("change", applyRunScopeVisibility);
  });
  ["pilotImages", "pilotBatches"].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener("input", () => { updateRunSummary(); updateStartEnabled(); });
  });
  document.querySelectorAll(".pg-chip[data-n]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const n = btn.getAttribute("data-n");
      const scopePilot = $("scopePilot");
      if (scopePilot) scopePilot.checked = true;
      const batchesEl = $("pilotBatches");
      if (batchesEl) batchesEl.value = "";
      const imagesEl = $("pilotImages");
      if (imagesEl) imagesEl.value = n;
      applyRunScopeVisibility();
    });
  });
  updateRunSummary();

  btnRefresh.addEventListener("click", () => {
    refreshStats();
    refreshMonitoring();
    appendLog("Folder stats and monitoring refreshed.");
  });

  for (const elId of Object.values(pathFields)) {
    const el = $(elId);
    if (el) {
      el.addEventListener("input", scheduleStatsRefresh);
      el.addEventListener("change", scheduleStatsRefresh);
    }
  }

  if (btnRefreshEnv) {
    btnRefreshEnv.addEventListener("click", async () => {
      appendSetup("Re-checking environment…");
      await fetchEnvironment(true);
    });
  }

  if (btnInstallDeps) {
    btnInstallDeps.addEventListener("click", async () => {
      if (installRunning) return;
      setupLines = [];
      setupTerminal.textContent = "";
      appendSetup("Starting one-click install (pip install -r requirements.txt)…");
      setInstallProgress(5, "Starting install…");
      installRunning = true;
      setPipelineRunAllowed(false);
      btnInstallDeps.disabled = true;
      try {
        const r = await fetch("/api/environment/install", { method: "POST" });
        const j = await r.json();
        if (!r.ok) {
          appendSetup(`Install start failed: ${j.error || r.status}`, true);
          installRunning = false;
          btnInstallDeps.disabled = false;
          await fetchEnvironment(false);
        }
      } catch (e) {
        appendSetup(`Install request error: ${e}`, true);
        installRunning = false;
        btnInstallDeps.disabled = false;
      }
    });
  }

  const modeImagesOnly = $("modeImagesOnly");
  if (modeImagesOnly) {
    modeImagesOnly.addEventListener("change", () => {
      if (modeImagesOnly.checked) confirmImageOnlyMode();
    });
  }

  btnStart.addEventListener("click", async () => {
    if (!imageOnlyConfirmed) {
      appendLog("Cannot start — select and confirm 'Images Only' processing mode first.");
      return;
    }
    if (!pipelineReady) {
      appendLog("Cannot start — environment not ready. Use Setup Environment.");
      return;
    }
    logLines = [];
    lastProcessed = 0;
    totalTarget = 0;
    await refreshStats();
    const paths = {};
    for (const [key, elId] of Object.entries(pathFields)) {
      const v = $(elId).value.trim();
      if (v) paths[key] = v;
    }
    const run = getRunSettings();
    if (!runScopeValid()) {
      appendLog("Cannot start — set a valid run scope (pilot image/batch count) and batch size.");
      return;
    }
    const body = {
      processing_mode: "images_only",
      run_type: run.runType,
      batch_size: run.batchSize,
      paths,
      dataset: datasetPayload(),
      security_level: $("securityLevel").value,
      gpu_device: $("gpuDeviceSelect").value,
      enable_crypt: $("enableCrypt").checked,
      test_mode: $("testMode").checked,
      force_gpu: $("forceGpu").checked,
    };
    if (run.runType === "pilot" && run.maxImages > 0) body.max_images = run.maxImages;
    try {
      const r = await fetch("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const j = await r.json();
      if (!r.ok) {
        appendLog(`start failed: ${j.error || r.status}`);
        if (j.environment) applyEnvironment(j.environment);
        return;
      }
      appendLog(`started with overrides: ${JSON.stringify(j.overrides)}`);
      refreshStatus();
    } catch (e) {
      appendLog(`start error: ${e}`);
    }
  });

  btnStop.addEventListener("click", async () => {
    try {
      const r = await fetch("/api/stop", { method: "POST" });
      const j = await r.json();
      appendLog(j.message || "stop sent");
      refreshStatus();
    } catch (e) {
      appendLog(`stop error: ${e}`);
    }
  });

  btnLogs.addEventListener("click", async () => {
    try {
      const r = await fetch(`/api/logs/latest${statsQuery()}`);
      const j = await r.json();
      appendLog(`--- log tail: ${j.path} ---`);
      (j.lines || []).forEach((line) => appendLog(line));
    } catch (e) {
      appendLog(`logs error: ${e}`);
    }
  });

  const socket = io({ transports: ["websocket", "polling"] });
  socket.on("connect", () => appendLog(`socket connected (${socket.id})`));
  socket.on("connect_error", (err) => appendLog(`socket error: ${err.message || err}`));
  socket.on("disconnect", (reason) => appendLog(`socket disconnected: ${reason}`));

  socket.on("environment_status", (data) => {
    if (data.environment) applyEnvironment(data.environment);
    installRunning = !!data.install_running;
    if (btnInstallDeps) btnInstallDeps.disabled = installRunning;
  });

  socket.on("install_status", (data) => {
    const msg = data.message || "";
    appendSetup(msg);
    const phase = data.phase || "";
    const phasePct = {
      start: 10,
      conda: 12,
      bootstrap: 18,
      torch: 35,
      pillow: 45,
      pip_install: 55,
      verify: 85,
      gpu_test: 92,
    };
    if (phasePct[phase] != null) setInstallProgress(phasePct[phase], msg);
  });

  socket.on("install_output", (data) => {
    if (data.line) appendSetup(data.line);
    setInstallProgress(60, "Installing packages…");
  });

  socket.on("install_error", (data) => {
    appendSetup(`ERROR: ${data.error || "unknown"}`, true);
    setInstallProgress(-1, "Install error");
  });

  socket.on("install_complete", async (data) => {
    installRunning = false;
    if (btnInstallDeps) btnInstallDeps.disabled = false;
    appendSetup(data.message || "Install complete.");
    if (data.detail) appendSetup(data.detail);
    setInstallProgress(data.ok ? 100 : 0, data.ok ? "Complete" : "Failed");
    await fetchEnvironment(true);
  });

  socket.on("pipeline_status_update", (live) => applyLiveStatus(live));
  socket.on("progress_tick", (data) => applyLiveStatus(data));
  socket.on("batch_start", (data) => {
    applyLiveStatus(data);
    appendLog(`batch ${data.batch_index ?? data.current_batch} start n=${data.n ?? data.current_batch_size}`);
  });
  socket.on("batch_complete", (data) => {
    applyLiveStatus(data);
  });
  socket.on("pipeline_complete", (data) => {
    applyLiveStatus(data);
  });

  socket.on("pipeline_event", (ev) => {
    const t = ev.type || "event";
    if (t === "pipeline_start") {
      appendLog(`pipeline_start root=${ev.project_root || ""}`);
      if (ev.cpu_fallback && ev.user_message) {
        appendLog(ev.user_message);
        const banner = $("cpuFallbackBanner");
        if (banner) {
          banner.textContent = ev.user_message;
          banner.classList.remove("pg-hidden");
        }
      }
    } else if (t === "models_warmed") {
      const comp = ev.components || (ev.meta || {}).component_activation || {};
      appendLog(`models warmed — components: ${JSON.stringify(comp)}`);
      if (ev.cpu_fallback && ev.user_message) {
        appendLog(ev.user_message);
        const banner = $("cpuFallbackBanner");
        if (banner) {
          banner.textContent = ev.user_message;
          banner.classList.remove("pg-hidden");
        }
      }
    } else if (t === "wave_start") {
      appendLog(`wave ${ev.wave} pending=${ev.pending}`);
      totalTarget = Math.max(totalTarget, ev.pending || 0);
    } else if (t === "batch_start") {
      appendLog(`batch ${ev.batch_index} n=${ev.n} images: ${(ev.images || []).join(", ")}`);
    } else if (t === "batch_complete") {
      applyLiveStatus(ev);
      lastProcessed = ev.processed_this_run ?? lastProcessed;
      $("mProcessed").textContent = String(lastProcessed);
      const sr = ev.success_rate;
      $("mSuccess").textContent = typeof sr === "number" ? `${(sr * 100).toFixed(1)}%` : "—";
      if (ev.eta_sec != null && Number.isFinite(ev.eta_sec)) {
        const m = Math.floor(ev.eta_sec / 60);
        const s = Math.floor(ev.eta_sec % 60);
        $("mEta").textContent = `${m}m ${s}s`;
      } else {
        $("mEta").textContent = "—";
      }
      $("mRate").textContent =
        ev.images_per_sec != null && ev.images_per_sec > 0
          ? `${ev.images_per_sec.toFixed(2)} img/s`
          : "—";
      if (ev.quarantine_rate != null) {
        $("mQuarantine").textContent = `${(ev.quarantine_rate * 100).toFixed(1)}%`;
      }
      if (ev.gpu) {
        $("gpuVram").textContent =
          ev.gpu.allocated_mb != null ? String(ev.gpu.allocated_mb) : "—";
      }
      setProgress(lastProcessed, ev.total_hint);
      appendLog(
        `batch ${ev.batch_index} complete routed=${ev.routed} QA rows=${ev.qa_rows}`
      );
    } else if (t === "pipeline_complete") {
      appendLog(`pipeline_complete stopped_early=${ev.stopped_early}`);
      setProgress(100, 100);
      refreshStats();
      refreshStatus();
    } else if (t === "pipeline_error") {
      appendLog(`ERROR ${ev.message}`);
      refreshStatus();
    } else if (t === "hello") {
      /* ignore */
    } else {
      appendLog(`${t} ${JSON.stringify(ev)}`);
    }
  });

  async function refreshMonitoring() {
    try {
      const r = await fetch(`/api/monitoring${statsQuery()}`);
      const j = await r.json();
      const lines = (j.recent_batches || []).map(
        (e) =>
          `batch ${e.batch_index ?? "?"} ${e.images_per_sec?.toFixed?.(2) ?? e.images_per_sec} img/s`
      );
      const pane = $("monitoringPane");
      if (!pane) return;
      if (lines.length) {
        pane.textContent = lines.join("\n");
      } else {
        pane.textContent = j.message || "(no events yet)";
      }
    } catch (e) {
      /* ignore */
    }
  }

  let statsDebounce = null;
  function scheduleStatsRefresh() {
    updateReportLinks();
    if (statsDebounce) clearTimeout(statsDebounce);
    statsDebounce = setTimeout(refreshStats, 400);
  }

  setInterval(refreshStatus, 2000);
  setInterval(refreshMonitoring, 5000);
  setInterval(() => fetchEnvironment(false), 15000); // GET /api/environment — light check only

  const btnRefreshSafetyCmds = $("btnRefreshSafetyCmds");
  if (btnRefreshSafetyCmds) btnRefreshSafetyCmds.addEventListener("click", refreshSafetyRclone);

  // Make every UI section collapsible: click a section header (or press Enter/Space) to
  // expand/collapse its body. Purely presentational; controls/buttons inside headers
  // still work (clicks on them do not toggle). Runs once; safe to re-invoke.
  function makeSectionsCollapsible() {
    document.querySelectorAll(".pg-section").forEach((section) => {
      if (section.dataset.collapsible === "1") return;
      const header =
        section.querySelector(":scope > .pg-section-head") ||
        section.querySelector(":scope > .pg-section-title");
      if (!header) return;
      section.dataset.collapsible = "1";

      // Wrap everything after the header into a single collapsible body element.
      const body = document.createElement("div");
      body.className = "pg-section-body";
      let node = header.nextSibling;
      while (node) {
        const next = node.nextSibling;
        body.appendChild(node);
        node = next;
      }
      section.appendChild(body);

      // Caret shown at the start of the section title.
      const titleHost = header.classList.contains("pg-section-title")
        ? header
        : header.querySelector(".pg-section-title") || header;
      const caret = document.createElement("span");
      caret.className = "pg-collapse-caret";
      caret.setAttribute("aria-hidden", "true");
      caret.textContent = "\u25be"; // ▾
      titleHost.insertBefore(caret, titleHost.firstChild);

      header.classList.add("pg-collapsible-header");
      header.setAttribute("role", "button");
      header.setAttribute("tabindex", "0");
      header.setAttribute("aria-expanded", "true");

      const insideControl = (t) => t.closest && t.closest("button, a, input, select, textarea, label");
      const toggle = () => {
        const collapsed = section.classList.toggle("pg-collapsed");
        caret.textContent = collapsed ? "\u25b8" : "\u25be"; // ▸ / ▾
        header.setAttribute("aria-expanded", String(!collapsed));
      };
      header.addEventListener("click", (e) => {
        if (insideControl(e.target)) return;
        toggle();
      });
      header.addEventListener("keydown", (e) => {
        if ((e.key === "Enter" || e.key === " ") && !insideControl(e.target)) {
          e.preventDefault();
          toggle();
        }
      });
    });
  }
  makeSectionsCollapsible();

  fetchEnvironment(false).then(() => {
    loadDatasetConfig();
    loadB2Overview();
    toggleDatasetPanels();
    updateReportLinks();
    refreshStats();
    refreshStatus();
    refreshMonitoring();
    refreshSafetyRclone();
  });
})();
