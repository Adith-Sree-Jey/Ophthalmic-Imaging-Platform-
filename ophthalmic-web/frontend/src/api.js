function resolveBaseUrl() {
  const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
  if (configuredBaseUrl) {
    return configuredBaseUrl.replace(/\/+$/, "");
  }

  if (typeof window !== "undefined") {
    const { hostname, protocol } = window.location;
    const isPrivateNetworkHost =
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname === "0.0.0.0" ||
      /^192\.168\.\d{1,3}\.\d{1,3}$/.test(hostname) ||
      /^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(hostname) ||
      /^172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}$/.test(hostname) ||
      /^[a-z0-9-]+$/i.test(hostname);
    const isLocalBrowser =
      hostname === "localhost" || hostname === "127.0.0.1" || hostname === "0.0.0.0";

    if (isLocalBrowser && protocol !== "file:") {
      return "http://127.0.0.1:8000";
    }

    if (isPrivateNetworkHost && protocol !== "file:") {
      return `${protocol}//${hostname}:8000`;
    }
  }

  return "/api";
}

const BASE_URL = resolveBaseUrl();
const REQUEST_RETRY_MS = 1200;
const JOB_POLL_INTERVAL_MS = 3000;

function getToken() {
  return localStorage.getItem("token") || localStorage.getItem("access_token");
}

function authHeaders(extraHeaders = {}) {
  const token = getToken();
  return token
    ? { ...extraHeaders, Authorization: `Bearer ${token}` }
    : extraHeaders;
}

async function readErrorMessage(res, fallback) {
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") return data.detail;
    if (typeof data?.message === "string") return data.message;
    return JSON.stringify(data);
  } catch {
    try {
      const text = await res.text();
      return text || fallback;
    } catch {
      return fallback;
    }
  }
}

function handle401(res) {
  if (res.status === 401) {
    localStorage.removeItem("token");
    localStorage.removeItem("access_token");
    localStorage.removeItem("username");
    localStorage.removeItem("role");
    window.dispatchEvent(new Event("ophthalmic-imaging:auth-expired"));
    window.location.replace("/login");
    throw new Error("Session expired. Please log in again.");
  }
}

function delay(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function fetchWithRetry(url, options, retries = 1) {
  try {
    return await fetch(url, options);
  } catch (error) {
    if (retries <= 0) {
      throw error;
    }
    await delay(REQUEST_RETRY_MS);
    return fetchWithRetry(url, options, retries - 1);
  }
}

function resolveApiPath(path) {
  if (!path) return BASE_URL;
  if (/^https?:\/\//i.test(path)) return path;
  if (path.startsWith("/")) return `${BASE_URL}${path}`;
  return `${BASE_URL}/${path}`;
}

async function fetchJobStatus(jobId) {
  let res;
  try {
    res = await fetchWithRetry(
      `${BASE_URL}/jobs/${encodeURIComponent(jobId)}`,
      {
        method: "GET",
        headers: authHeaders(),
      },
      1
    );
  } catch {
    throw new Error("Job status polling could not reach the backend. Check the API connection and make sure the backend is running.");
  }

  handle401(res);
  const payload = await res.json();
  if (!res.ok) {
    throw new Error(payload?.detail || payload?.error || "Job status polling failed");
  }
  return payload;
}

async function fetchJobResult(resultPath) {
  let res;
  try {
    res = await fetchWithRetry(
      resolveApiPath(resultPath),
      {
        method: "GET",
        headers: authHeaders(),
      },
      1
    );
  } catch {
    throw new Error("Job result download could not reach the backend. Check the API connection and make sure the backend is running.");
  }

  handle401(res);
  const payload = await res.json();
  if (!res.ok) {
    throw new Error(payload?.detail || payload?.error || "Job result retrieval failed");
  }
  return payload;
}

async function waitForJob(jobId, onJobStatus) {
  while (true) {
    const statusPayload = await fetchJobStatus(jobId);
    if (typeof onJobStatus === "function") {
      onJobStatus(statusPayload.status, statusPayload);
    }

    if (statusPayload.status === "done") {
      return statusPayload;
    }
    if (statusPayload.status === "error") {
      throw new Error(statusPayload.error || "Background job failed");
    }

    await delay(JOB_POLL_INTERVAL_MS);
  }
}

async function submitJobForm(
  endpoint,
  formData,
  { networkErrorMessage, failureMessage, onJobStatus } = {}
) {
  let res;
  try {
    res = await fetchWithRetry(
      `${BASE_URL}${endpoint}`,
      {
        method: "POST",
        headers: authHeaders(),
        body: formData,
      },
      1
    );
  } catch {
    throw new Error(networkErrorMessage || "The request could not reach the backend. Check the API connection and make sure the backend is running.");
  }

  handle401(res);
  const payload = await res.json();
  if (!res.ok) {
    throw new Error(payload?.detail || payload?.error || failureMessage || "Request failed");
  }

  if (!payload?.job_id) {
    return payload;
  }

  if (typeof onJobStatus === "function") {
    onJobStatus(payload.status || "pending", payload);
  }

  const finalStatus = await waitForJob(payload.job_id, onJobStatus);
  if (!finalStatus?.result_path) {
    throw new Error("Background job finished without a result path.");
  }
  return fetchJobResult(finalStatus.result_path);
}

export async function login(username, password) {
  let res;
  try {
    res = await fetch(`${BASE_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
  } catch {
    throw new Error("Login could not reach the backend. Check that the API server is running and refresh the page.");
  }

  if (!res.ok) throw new Error(await readErrorMessage(res, "Login failed"));
  return res.json();
}

export async function detectImageTypes(files) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  let res;
  try {
    res = await fetchWithRetry(
      `${BASE_URL}/detect-image-types`,
      {
        method: "POST",
        headers: authHeaders(),
        body: formData,
      },
      1
    );
  } catch {
    throw new Error("Auto-detect could not reach the backend. Check the API connection and make sure the backend is running.");
  }

  handle401(res);
  if (!res.ok) throw new Error(await readErrorMessage(res, "Detection failed"));
  return res.json();
}

export async function classifyImage({
  files,
  types,
  caseId,
  patientName,
  mriNumber,
  eyeSide,
  onJobStatus,
}) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  types.forEach((type) => formData.append("types", type));
  formData.append("case_id", caseId || "UNKNOWN");
  formData.append("patient_name", patientName || "");
  formData.append("mri_number", mriNumber || "");
  formData.append("eye_side", eyeSide || "UNKNOWN");

  return submitJobForm("/classify", formData, {
    networkErrorMessage: "Classification could not reach the backend. Check the API connection and make sure the backend is running.",
    failureMessage: "Classification failed",
    onJobStatus,
  });
}

export async function segmentRetinaImage({
  file,
  threshold,
  caseId,
  patientName,
  mriNumber,
  eyeSide,
  onJobStatus,
}) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("threshold", String(threshold ?? 0.5));
  formData.append("case_id", caseId || "UNKNOWN");
  formData.append("patient_name", patientName || "");
  formData.append("mri_number", mriNumber || "");
  formData.append("eye_side", eyeSide || "UNKNOWN");

  return submitJobForm("/retina-segmentation", formData, {
    networkErrorMessage: "Retina segmentation could not reach the backend. Check the API connection and make sure the backend is running.",
    failureMessage: "Retina segmentation failed",
    onJobStatus,
  });
}

export async function fetchRetinaProbabilityPreview({
  file,
  caseId,
  patientName,
  mriNumber,
  eyeSide,
  onJobStatus,
}) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("case_id", caseId || "UNKNOWN");
  formData.append("patient_name", patientName || "");
  formData.append("mri_number", mriNumber || "");
  formData.append("eye_side", eyeSide || "UNKNOWN");

  return submitJobForm("/retina/probability", formData, {
    networkErrorMessage: "Retina probability preview could not reach the backend. Check the API connection and make sure the backend is running.",
    failureMessage: "Retina probability preview failed",
    onJobStatus,
  });
}

export async function classifyGlaucomaImage({
  file,
  caseId,
  patientName,
  mriNumber,
  eyeSide,
  onJobStatus,
}) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("case_id", caseId || "UNKNOWN");
  formData.append("patient_name", patientName || "");
  formData.append("mri_number", mriNumber || "");
  formData.append("eye_side", eyeSide || "UNKNOWN");

  return submitJobForm("/glaucoma/predict", formData, {
    networkErrorMessage: "Glaucoma classification could not reach the backend. Check the API connection and make sure the backend is running.",
    failureMessage: "Glaucoma classification failed",
    onJobStatus,
  });
}

export async function analyzeGlaucomaReport({
  file,
  patientId,
  eyeSide,
  patientName,
  mriNumber,
  caseId,
  onJobStatus,
}) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("patient_id", patientId || "");
  formData.append("eye_side", eyeSide || "");
  formData.append("patient_name", patientName || "");
  formData.append("mri_number", mriNumber || "");
  formData.append("case_id", caseId || "");

  return submitJobForm("/glaucoma/report", formData, {
    networkErrorMessage: "Glaucoma analysis could not reach the backend. Check the API connection and make sure the backend is running.",
    failureMessage: "Glaucoma analysis failed",
    onJobStatus,
  });
}

export async function generateGlaucomaReport({
  file,
  patientId,
  eyeSide,
  patientName,
  mriNumber,
  caseId,
  onJobStatus,
}) {
  if (!(file instanceof File)) {
    throw new Error("Report generation requires the uploaded fundus image file.");
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("patient_id", patientId || "");
  formData.append("eye_side", eyeSide || "");
  formData.append("patient_name", patientName || "");
  formData.append("mri_number", mriNumber || "");
  formData.append("case_id", caseId || "");

  return submitJobForm("/glaucoma/report", formData, {
    networkErrorMessage: "Glaucoma report generation could not reach the backend. Check the API connection and make sure the backend is running.",
    failureMessage: "Glaucoma report generation failed",
    onJobStatus,
  });
}

export async function downloadGlaucomaReportPdf(pdfPath) {
  const res = await fetch(
    `${BASE_URL}/glaucoma/report/download?pdf_path=${encodeURIComponent(pdfPath)}`,
    {
      method: "GET",
      headers: authHeaders(),
    }
  );
  handle401(res);
  if (!res.ok) throw new Error(await readErrorMessage(res, "Glaucoma report download failed"));

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const fileName = pdfPath?.split("/").pop() || "glaucoma-report.pdf";
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export async function classify(images, caseId) {
  return classifyImage({
    files: images.map((img) => img.file),
    types: images.map((img) => img.assignedType),
    caseId,
    patientName: "",
    mriNumber: "",
    eyeSide: "UNKNOWN",
  });
}

export async function downloadBatchReport(results, caseId) {
  const res = await fetch(`${BASE_URL}/report-batch`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(results),
  });
  handle401(res);
  if (!res.ok) throw new Error(await readErrorMessage(res, "Batch report download failed"));

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const hasMultipleResults = Array.isArray(results) && results.length > 1;
  link.href = url;
  link.download = caseId
    ? `${caseId}${hasMultipleResults ? "-both-eyes" : ""}.pdf`
    : hasMultipleResults
      ? "ophthalmic-imaging-batch-report.pdf"
      : "ophthalmic-imaging-report.pdf";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export async function fetchPatientHistory(mriNumber) {
  const res = await fetch(`${BASE_URL}/history/${encodeURIComponent(mriNumber)}`, {
    headers: authHeaders(),
  });
  handle401(res);
  if (!res.ok) throw new Error(await readErrorMessage(res, "Failed to load patient history"));
  return res.json();
}

export async function fetchLongitudinal(mriNumber, eyeSide) {
  const res = await fetch(
    `${BASE_URL}/longitudinal/${encodeURIComponent(mriNumber)}/${encodeURIComponent(eyeSide)}`,
    { headers: authHeaders() }
  );
  handle401(res);
  if (!res.ok) throw new Error(await readErrorMessage(res, "Failed to load longitudinal data"));
  return res.json();
}

export async function fetchAllPatients() {
  const res = await fetch(`${BASE_URL}/patients`, {
    headers: authHeaders(),
  });
  handle401(res);
  if (!res.ok) throw new Error(await readErrorMessage(res, "Failed to load patients"));
  return res.json();
}

export async function fetchWorkspaceStatus() {
  const res = await fetch(`${BASE_URL}/workspace-status`, {
    headers: authHeaders(),
  });
  handle401(res);
  if (!res.ok) throw new Error(await readErrorMessage(res, "Failed to load workspace status"));
  return res.json();
}

export async function fetchOverviewStats() {
  const res = await fetch(`${BASE_URL}/stats/overview`, {
    headers: authHeaders(),
  });
  handle401(res);
  if (!res.ok) throw new Error(await readErrorMessage(res, "Failed to load overview stats"));
  return res.json();
}

export async function fetchModuleActivity(period = "6months") {
  const res = await fetch(`${BASE_URL}/stats/module-activity?period=${encodeURIComponent(period)}`, {
    headers: authHeaders(),
  });
  handle401(res);
  if (!res.ok) throw new Error(await readErrorMessage(res, "Failed to load module activity"));
  return res.json();
}

async function apiRequest(path, options = {}, fallback = "Request failed") {
  const res = await fetch(resolveApiPath(path), {
    ...options,
    headers: authHeaders(options.headers || {}),
  });
  handle401(res);
  if (!res.ok) throw new Error(await readErrorMessage(res, fallback));
  return res;
}

async function apiJson(path, options = {}, fallback = "Request failed") {
  const res = await apiRequest(path, options, fallback);
  return res.json();
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

// Cases
export const createCase = (data) =>
  apiJson("/cases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  }, "Failed to create case");

export const getCases = (filters = {}) =>
  apiJson(`/cases?${new URLSearchParams(Object.entries(filters).filter(([, value]) => value != null && value !== "")).toString()}`, {}, "Failed to load cases");

export const getCase = (caseId) =>
  apiJson(
    `/cases/${encodeURIComponent(caseId)}`,
    {
      cache: "no-store",
      headers: { "Cache-Control": "no-cache" },
    },
    "Failed to load case"
  );

export const updateCase = (caseId, data) =>
  apiJson(`/cases/${encodeURIComponent(caseId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  }, "Failed to update case");

export const assignReviewer = (caseId, reviewerId) =>
  apiJson(`/cases/${encodeURIComponent(caseId)}/assign`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviewer_id: reviewerId }),
  }, "Failed to assign reviewer");

// Review actions
export const approveResult = (caseId, module, notes) =>
  apiJson(`/cases/${encodeURIComponent(caseId)}/results/${encodeURIComponent(module)}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ notes }),
  }, "Failed to approve result");

export const rejectResult = (caseId, module, reason) =>
  apiJson(`/cases/${encodeURIComponent(caseId)}/results/${encodeURIComponent(module)}/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reason }),
  }, "Failed to reject result");

export const overrideResult = (caseId, module, grade, notes) =>
  apiJson(`/cases/${encodeURIComponent(caseId)}/results/${encodeURIComponent(module)}/override`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clinician_grade: grade, clinician_notes: notes }),
  }, "Failed to override result");

// Report
export const approveReport = (caseId) =>
  apiJson(`/cases/${encodeURIComponent(caseId)}/report/approve`, { method: "POST" }, "Failed to approve report");

export const rejectReport = (caseId, reason) =>
  apiJson(`/cases/${encodeURIComponent(caseId)}/report/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reason }),
  }, "Failed to reject report");

export async function downloadReport(caseId) {
  const res = await apiRequest(`/cases/${encodeURIComponent(caseId)}/report/download`, { method: "GET" }, "Report download failed");
  const blob = await res.blob();
  downloadBlob(blob, `${caseId || "ophthalmic-imaging-case"}.pdf`);
}

// Queue
export const getReviewQueue = () => apiJson("/review-queue", {}, "Failed to load review queue");
export const getQueueStats = () => apiJson("/review-queue/stats", {}, "Failed to load review queue stats");

// Audit
export const getAuditLog = (filters = {}) =>
  apiJson(`/audit-log?${new URLSearchParams(Object.entries(filters).filter(([, value]) => value != null && value !== "")).toString()}`, {}, "Failed to load audit log");

export async function exportAuditLog(filters = {}) {
  const query = new URLSearchParams(Object.entries(filters).filter(([, value]) => value != null && value !== "")).toString();
  const res = await apiRequest(`/audit-log/export${query ? `?${query}` : ""}`, { method: "GET" }, "Failed to export audit log");
  const blob = await res.blob();
  downloadBlob(blob, "audit-log.csv");
}

// Thresholds
export const getThresholds = (module) => apiJson(`/thresholds/${encodeURIComponent(module)}`, {}, "Failed to load thresholds");
export const updateThreshold = (module, data) =>
  apiJson(`/thresholds/${encodeURIComponent(module)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  }, "Failed to update threshold");
export const getThresholdHistory = (module) => apiJson(`/thresholds/history/${encodeURIComponent(module)}`, {}, "Failed to load threshold history");

// Calibration
export const getCalibrationStatus = (module) => apiJson(`/admin/calibration/${encodeURIComponent(module)}/status`, {}, "Failed to load calibration status");
export const getReliabilityDiagram = (module) => apiJson(`/admin/calibration/reliability-diagram/${encodeURIComponent(module)}`, {}, "Failed to load reliability diagram");
export const fitCalibration = (module, data) =>
  apiJson(`/admin/calibration/${encodeURIComponent(module)}/fit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  }, "Failed to fit calibration");

// Validation
export const getValidationReport = (module) => apiJson(`/validation-report/${encodeURIComponent(module)}`, {}, "Failed to load validation report");
export async function downloadValidationReportPdf(module) {
  const res = await apiRequest(`/validation-report/${encodeURIComponent(module)}/pdf`, { method: "GET" }, "Failed to download validation report PDF");
  const blob = await res.blob();
  downloadBlob(blob, `${module}-validation.pdf`);
}

// Performance (admin)
export const getPerformanceOverview = (filters = {}) =>
  apiJson(`/admin/performance/overview?${new URLSearchParams(Object.entries(filters).filter(([, value]) => value != null && value !== "")).toString()}`, {}, "Failed to load performance overview");
export const getOverrideAnalysis = (filters = {}) =>
  apiJson(`/admin/performance/override-analysis?${new URLSearchParams(Object.entries(filters).filter(([, value]) => value != null && value !== "")).toString()}`, {}, "Failed to load override analysis");
export const getCalibrationDrift = (filters = {}) =>
  apiJson(`/admin/performance/calibration-drift?${new URLSearchParams(Object.entries(filters).filter(([, value]) => value != null && value !== "")).toString()}`, {}, "Failed to load calibration drift");
export const getReviewerAgreement = (filters = {}) =>
  apiJson(`/admin/performance/reviewer-agreement?${new URLSearchParams(Object.entries(filters).filter(([, value]) => value != null && value !== "")).toString()}`, {}, "Failed to load reviewer agreement");
export const getThroughput = (filters = {}) =>
  apiJson(`/admin/performance/throughput?${new URLSearchParams(Object.entries(filters).filter(([, value]) => value != null && value !== "")).toString()}`, {}, "Failed to load throughput data");
