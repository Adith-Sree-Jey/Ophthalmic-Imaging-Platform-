import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import DashboardHeader from "../components/DashboardHeader";
import DashboardOverview from "../components/DashboardOverview";
import UploadPanel from "../components/UploadPanel";
import ResultsPanel from "../components/ResultsPanel";
import GlaucomaClassificationPanel from "../components/GlaucomaClassificationPanel";
import GlaucomaClassificationResults from "../components/GlaucomaClassificationResults";
import RetinaSegmentationPanel from "../components/RetinaSegmentationPanel";
import RetinaSegmentationResults from "../components/RetinaSegmentationResults";
import {
  analyzeGlaucomaReport,
  classifyImage,
  fetchModuleActivity,
  fetchOverviewStats,
  fetchRetinaProbabilityPreview,
  fetchWorkspaceStatus,
} from "../api";

const EYES = ["OD", "OS"];
const MODALITIES = ["anterior_segment", "red_glow", "slit_lamp"];
const HOME_VIEW = "Platform Home";
const GLAUCOMA_MODULE = "Glaucoma";
const CATARACT_MODULE = "Cataract";
const RETINA_MODULE = "Retina Segmentation";
const RETINA_OVERLAY_ALPHA = 0.48;

function resolveRetinaBaseUrl() {
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
    const isLocalBrowser = hostname === "localhost" || hostname === "127.0.0.1" || hostname === "0.0.0.0";

    if (isLocalBrowser && protocol !== "file:") {
      return "http://127.0.0.1:8000";
    }

    if (isPrivateNetworkHost && protocol !== "file:") {
      return `${protocol}//${hostname}:8000`;
    }
  }

  return "/api";
}

const RETINA_BASE_URL = resolveRetinaBaseUrl();

function isEyeReady(images) {
  if (!images.length || images.some((img) => img.isDetecting)) return false;
  const counts = Object.fromEntries(MODALITIES.map((modality) => [modality, 0]));
  for (const image of images) {
    if (counts[image.assignedType] != null) counts[image.assignedType] += 1;
  }
  return counts.anterior_segment === 1 && MODALITIES.every((modality) => counts[modality] <= 1);
}

function getRetinaToken() {
  return localStorage.getItem("token") || localStorage.getItem("access_token");
}

function retinaAuthHeaders() {
  const token = getRetinaToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function handleRetina401(res) {
  if (res.status === 401) {
    localStorage.removeItem("token");
    localStorage.removeItem("access_token");
    localStorage.removeItem("username");
    window.location.replace("/login");
    throw new Error("Session expired. Please log in again.");
  }
}

function stripDataUrl(dataUrl) {
  return dataUrl.replace(/^data:image\/png;base64,/, "");
}

function loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Unable to decode retina image."));
    image.src = src;
  });
}

async function decodeProbabilityMap(base64Png) {
  const image = await loadImageElement(`data:image/png;base64,${base64Png}`);
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Probability map canvas context is unavailable.");
  }

  canvas.width = image.width;
  canvas.height = image.height;
  context.drawImage(image, 0, 0);
  const { data } = context.getImageData(0, 0, image.width, image.height);
  const grayscale = new Uint8ClampedArray(image.width * image.height);

  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    grayscale[p] = data[i];
  }

  return { width: image.width, height: image.height, probabilityPixels: grayscale };
}

async function buildOriginalImageCache(file, width, height) {
  const objectUrl = URL.createObjectURL(file);

  try {
    const image = await loadImageElement(objectUrl);
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Original image canvas context is unavailable.");
    }

    canvas.width = width;
    canvas.height = height;
    context.drawImage(image, 0, 0, width, height);
    const imageData = context.getImageData(0, 0, width, height);

    return {
      originalPixels: new Uint8ClampedArray(imageData.data),
      originalImageBase64: stripDataUrl(canvas.toDataURL("image/png")),
    };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function buildRetinaDisplayResult(cacheEntry, threshold) {
  const thresholdValue = Math.round(threshold * 255);
  const totalPixels = cacheEntry.width * cacheEntry.height;
  const overlayPixels = new Uint8ClampedArray(cacheEntry.originalPixels);
  const maskPixels = new Uint8ClampedArray(totalPixels * 4);

  for (let i = 0; i < totalPixels; i += 1) {
    const pixelIndex = i * 4;
    const isVessel = cacheEntry.probabilityPixels[i] > thresholdValue;

    if (isVessel) {
      maskPixels[pixelIndex] = 255;
      maskPixels[pixelIndex + 1] = 255;
      maskPixels[pixelIndex + 2] = 255;
      maskPixels[pixelIndex + 3] = 255;
      overlayPixels[pixelIndex] = Math.round(overlayPixels[pixelIndex] * (1 - RETINA_OVERLAY_ALPHA));
      overlayPixels[pixelIndex + 1] = Math.round(
        overlayPixels[pixelIndex + 1] * (1 - RETINA_OVERLAY_ALPHA) + 200 * RETINA_OVERLAY_ALPHA
      );
      overlayPixels[pixelIndex + 2] = Math.round(
        overlayPixels[pixelIndex + 2] * (1 - RETINA_OVERLAY_ALPHA) + 80 * RETINA_OVERLAY_ALPHA
      );
      overlayPixels[pixelIndex + 3] = 255;
    } else {
      maskPixels[pixelIndex] = 0;
      maskPixels[pixelIndex + 1] = 0;
      maskPixels[pixelIndex + 2] = 0;
      maskPixels[pixelIndex + 3] = 255;
      overlayPixels[pixelIndex + 3] = 255;
    }
  }

  const overlayCanvas = document.createElement("canvas");
  const overlayContext = overlayCanvas.getContext("2d");
  const maskCanvas = document.createElement("canvas");
  const maskContext = maskCanvas.getContext("2d");
  if (!overlayContext || !maskContext) {
    throw new Error("Retina result canvas context is unavailable.");
  }

  overlayCanvas.width = cacheEntry.width;
  overlayCanvas.height = cacheEntry.height;
  maskCanvas.width = cacheEntry.width;
  maskCanvas.height = cacheEntry.height;
  overlayContext.putImageData(new ImageData(overlayPixels, cacheEntry.width, cacheEntry.height), 0, 0);
  maskContext.putImageData(new ImageData(maskPixels, cacheEntry.width, cacheEntry.height), 0, 0);

  const downloadStem = cacheEntry.caseId && cacheEntry.caseId !== "UNKNOWN"
    ? cacheEntry.caseId
    : `${cacheEntry.eyeSide.toLowerCase()}_retina`;

  return {
    module: "retina_segmentation",
    case_id: cacheEntry.caseId,
    patient_name: cacheEntry.patientName,
    mri_number: cacheEntry.mriNumber,
    eye_side: cacheEntry.eyeSide,
    filename: cacheEntry.filename,
    threshold,
    result_summary: "Segmentation complete",
    ground_truth_available: false,
    original_image_base64: cacheEntry.originalImageBase64,
    overlay_image_base64: stripDataUrl(overlayCanvas.toDataURL("image/png")),
    mask_base64: stripDataUrl(maskCanvas.toDataURL("image/png")),
    mask_filename: `${downloadStem}_vessel_mask_${threshold.toFixed(2)}.png`,
  };
}

async function fetchRetinaProbability({ file, caseId, patientName, mriNumber, eyeSide, onJobStatus }) {
  return fetchRetinaProbabilityPreview({ file, caseId, patientName, mriNumber, eyeSide, onJobStatus });
}

function getPageTitle(selectedModule) {
  if (selectedModule === HOME_VIEW) return HOME_VIEW;
  if (selectedModule === RETINA_MODULE) return "Retina Segmentation";
  return selectedModule;
}

function getJobLoadingLabel(baseLabel, status) {
  if (status === "pending") return `${baseLabel}: queued...`;
  if (status === "running") return `${baseLabel}: processing...`;
  return baseLabel;
}

function ModuleCanvas({ children, isDarkMode }) {
  return (
    <div
      className={`h-full overflow-hidden rounded-[28px] border shadow-[0_16px_40px_rgba(15,23,42,0.06)] ${
        isDarkMode ? "border-[#263041] bg-[#161D27]" : "border-[#D9E2EE] bg-white"
      }`}
    >
      {children}
    </div>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const token = localStorage.getItem("token") || localStorage.getItem("access_token");
  const username = localStorage.getItem("username") || "Clinician";

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingLabel, setLoadingLabel] = useState("Analysing...");
  const [error, setError] = useState(null);
  const [glaucomaResult, setGlaucomaResult] = useState(null);
  const [glaucomaLoading, setGlaucomaLoading] = useState(false);
  const [glaucomaLoadingLabel, setGlaucomaLoadingLabel] = useState("Running Classification...");
  const [glaucomaError, setGlaucomaError] = useState(null);
  const [retinaResult, setRetinaResult] = useState(null);
  const [retinaLoading, setRetinaLoading] = useState(false);
  const [retinaLoadingLabel, setRetinaLoadingLabel] = useState("Running Segmentation...");
  const [retinaError, setRetinaError] = useState(null);
  const [retinaThreshold, setRetinaThreshold] = useState(0.5);
  const [retinaProbabilityByEye, setRetinaProbabilityByEye] = useState({ OD: null, OS: null });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [selectedView, setSelectedView] = useState(HOME_VIEW);
  const [moduleStatus, setModuleStatus] = useState({
    glaucoma: { available: false, status: "Checking model", reason: null },
  });
  const [overviewStats, setOverviewStats] = useState({});
  const [activityData, setActivityData] = useState([]);
  const [activityPeriod, setActivityPeriod] = useState("6months");
  const [overviewLoading, setOverviewLoading] = useState(true);
  const [overviewError, setOverviewError] = useState("");
  const [patientName, setPatientName] = useState("");
  const [mriNumber, setMriNumber] = useState("");
  const [caseId, setCaseId] = useState("");
  const [eyeSide, setEyeSide] = useState("OD");
  const [imagesByEye, setImagesByEye] = useState({ OD: [], OS: [] });
  const [fundusByEye, setFundusByEye] = useState({ OD: null, OS: null });
  const [glaucomaFundusByEye, setGlaucomaFundusByEye] = useState({ OD: null, OS: null });
  const [isDarkMode, setIsDarkMode] = useState(() => {
    if (typeof window === "undefined") return false;
    return localStorage.getItem("ophthalmic-imaging-theme") === "dark";
  });

  useEffect(() => {
    if (!token) {
      navigate("/login", { replace: true });
    }
  }, [token, navigate]);

  useEffect(() => {
    let active = true;

    fetchWorkspaceStatus()
      .then((statusPayload) => {
        if (!active) return;
        setModuleStatus((prev) => ({ ...prev, ...statusPayload }));
      })
      .catch(() => {
        if (!active) return;
        setModuleStatus((prev) => ({
          ...prev,
          glaucoma: {
            available: false,
            status: "Model not found",
            reason: "Unable to confirm glaucoma model availability.",
          },
        }));
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!token) return;

    let active = true;
    setOverviewLoading(true);
    setOverviewError("");

    Promise.all([fetchOverviewStats(), fetchModuleActivity(activityPeriod)])
      .then(([statsPayload, activityPayload]) => {
        if (!active) return;
        setOverviewStats(statsPayload || {});
        setActivityData(Array.isArray(activityPayload) ? activityPayload : []);
      })
      .catch((err) => {
        if (!active) return;
        setOverviewError(err.message || "Unable to load dashboard overview.");
      })
      .finally(() => {
        if (active) setOverviewLoading(false);
      });

    return () => {
      active = false;
    };
  }, [activityPeriod, token]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    localStorage.setItem("ophthalmic-imaging-theme", isDarkMode ? "dark" : "light");
  }, [isDarkMode]);

  useEffect(() => {
    setRetinaProbabilityByEye((prev) => {
      let changed = false;
      const next = { ...prev };

      EYES.forEach((eye) => {
        const currentImage = fundusByEye[eye];
        const cached = prev[eye];
        if (!currentImage && cached) {
          next[eye] = null;
          changed = true;
          return;
        }
        if (currentImage && cached && cached.fileId !== currentImage.id) {
          next[eye] = null;
          changed = true;
        }
      });

      return changed ? next : prev;
    });
  }, [fundusByEye]);

  useEffect(() => {
    const availableEyes = EYES.filter((eye) => retinaProbabilityByEye[eye]);
    if (!availableEyes.length) {
      setRetinaResult(null);
      return;
    }

    try {
      const nextResults = {};
      availableEyes.forEach((eye) => {
        nextResults[eye] = buildRetinaDisplayResult(retinaProbabilityByEye[eye], retinaThreshold);
      });
      setRetinaResult(nextResults);
    } catch (err) {
      setRetinaError(err.message || "Failed to refresh the retina preview.");
    }
  }, [retinaProbabilityByEye, retinaThreshold]);

  const handleAnalyse = async () => {
    const eyesWithImages = EYES.filter((eye) => imagesByEye[eye].length > 0);
    const invalidEyes = eyesWithImages.filter((eye) => !isEyeReady(imagesByEye[eye]));
    const readyEyes = eyesWithImages.filter((eye) => isEyeReady(imagesByEye[eye]));

    if (!readyEyes.length) {
      setError("Add one complete set of images for at least one eye before running classification.");
      return;
    }
    if (invalidEyes.length) {
      setError(`Complete or clear the image set for ${invalidEyes.join(" and ")} before running classification.`);
      return;
    }

    setLoading(true);
    setLoadingLabel("Analysing...");
    setError(null);
    setResult(null);

    try {
      const resultEntries = await Promise.all(
        readyEyes.map(async (eye) => [
          eye,
          await classifyImage({
            files: imagesByEye[eye].map((img) => img.file),
            types: imagesByEye[eye].map((img) => img.assignedType),
            caseId: caseId || "UNKNOWN",
            patientName: patientName.trim(),
            mriNumber: mriNumber.trim(),
            eyeSide: eye,
            onJobStatus: (status) => setLoadingLabel(getJobLoadingLabel("Cataract analysis", status)),
          }),
        ])
      );
      setResult(Object.fromEntries(resultEntries));
    } catch (err) {
      setError(err.message || "Analysis failed. Please try again.");
    } finally {
      setLoading(false);
      setLoadingLabel("Analysing...");
    }
  };

  const handleRetinaSegmentation = async () => {
    const readyEyes = EYES.filter((eye) => Boolean(fundusByEye[eye]));
    if (!readyEyes.length) {
      setRetinaError("Add one fundus image for at least one eye before running segmentation.");
      return;
    }

    setRetinaLoading(true);
    setRetinaLoadingLabel("Running Segmentation...");
    setRetinaError(null);
    setRetinaResult(null);

    try {
      const probabilityEntries = await Promise.all(
        readyEyes.map(async (eye) => {
          const currentFile = fundusByEye[eye];
          const response = await fetchRetinaProbability({
            file: currentFile.file,
            caseId: caseId || "UNKNOWN",
            patientName: patientName.trim(),
            mriNumber: mriNumber.trim(),
            eyeSide: eye,
            onJobStatus: (status) => setRetinaLoadingLabel(getJobLoadingLabel("Retina segmentation", status)),
          });

          const decodedProbability = await decodeProbabilityMap(response.probability_map);
          const originalImageCache = await buildOriginalImageCache(
            currentFile.file,
            decodedProbability.width,
            decodedProbability.height
          );

          return [
            eye,
            {
              fileId: currentFile.id,
              filename: currentFile.file.name,
              caseId: caseId || "UNKNOWN",
              patientName: patientName.trim(),
              mriNumber: mriNumber.trim(),
              eyeSide: eye,
              width: decodedProbability.width,
              height: decodedProbability.height,
              probabilityPixels: decodedProbability.probabilityPixels,
              originalPixels: originalImageCache.originalPixels,
              originalImageBase64: originalImageCache.originalImageBase64,
            },
          ];
        })
      );

      setRetinaProbabilityByEye((prev) => ({ ...prev, ...Object.fromEntries(probabilityEntries) }));
    } catch (err) {
      setRetinaError(err.message || "Retina segmentation failed. Please try again.");
    } finally {
      setRetinaLoading(false);
      setRetinaLoadingLabel("Running Segmentation...");
    }
  };

  const handleGlaucomaClassification = async () => {
    const readyEyes = EYES.filter((eye) => Boolean(glaucomaFundusByEye[eye]));
    if (!readyEyes.length) {
      setGlaucomaError("Add one fundus image for at least one eye before running classification.");
      return;
    }

    setGlaucomaLoading(true);
    setGlaucomaLoadingLabel("Running Classification...");
    setGlaucomaError(null);
    setGlaucomaResult(null);

    try {
      const resultEntries = await Promise.all(
        readyEyes.map(async (eye) => [
          eye,
          await analyzeGlaucomaReport({
   		 file: glaucomaFundusByEye[eye].file,
   		 patientId: caseId || mriNumber.trim() || patientName.trim() || "UNKNOWN",
   		 eyeSide: eye,
 	         patientName: patientName.trim(),
    		 mriNumber: mriNumber.trim(),
    		 caseId: caseId || "",
    		 onJobStatus: (status) => setGlaucomaLoadingLabel(getJobLoadingLabel("Glaucoma analysis", status)),
	}),
        ])
      );
      setGlaucomaResult(Object.fromEntries(resultEntries));
    } catch (err) {
      setGlaucomaError(err.message || "Glaucoma classification failed. Please try again.");
    } finally {
      setGlaucomaLoading(false);
      setGlaucomaLoadingLabel("Running Classification...");
    }
  };

  return (
    <div className={`flex h-screen overflow-hidden font-['DM Sans',system-ui,sans-serif] ${isDarkMode ? "bg-[#11161F]" : "bg-[#EAF0F6]"}`}>
      <Navbar
        variant="sidebar"
        collapsed={sidebarCollapsed}
        onToggleSidebar={() => setSidebarCollapsed((current) => !current)}
        selectedView={selectedView}
        onSelectView={setSelectedView}
        moduleStatus={moduleStatus}
      />

      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <DashboardHeader
          title={getPageTitle(selectedView)}
          username={username}
          isDarkMode={isDarkMode}
          onToggleTheme={() => setIsDarkMode((current) => !current)}
        />

        <main className="flex-1 overflow-hidden p-5">
          {selectedView === HOME_VIEW ? (
            <DashboardOverview
              stats={overviewStats}
              activityData={activityData}
              activityPeriod={activityPeriod}
              onPeriodChange={setActivityPeriod}
              loading={overviewLoading}
              error={overviewError}
              searchQuery=""
              isDarkMode={isDarkMode}
              onOpenHistory={(preselectedMri) => {
                if (preselectedMri) {
                  navigate("/history", { state: { mriNumber: preselectedMri } });
                  return;
                }
                navigate("/history");
              }}
            />
          ) : selectedView === CATARACT_MODULE ? (
            <ModuleCanvas isDarkMode={isDarkMode}>
              <div className="flex h-full min-w-0 overflow-hidden">
                <UploadPanel
                  isDarkMode={isDarkMode}
                  patientName={patientName}
                  setPatientName={setPatientName}
                  mriNumber={mriNumber}
                  setMriNumber={setMriNumber}
                  caseId={caseId}
                  setCaseId={setCaseId}
                  eyeSide={eyeSide}
                  setEyeSide={setEyeSide}
                  imagesByEye={imagesByEye}
                  setImagesByEye={setImagesByEye}
                  loading={loading}
                  loadingLabel={loadingLabel}
                  result={result}
                  onResult={setResult}
                  onError={setError}
                  onAnalyse={handleAnalyse}
                />
                <ResultsPanel
                  isDarkMode={isDarkMode}
                  patientName={patientName}
                  mriNumber={mriNumber}
                  caseId={caseId}
                  result={result}
                  loading={loading}
                  loadingLabel={loadingLabel}
                  error={error}
                />
              </div>
            </ModuleCanvas>
          ) : selectedView === GLAUCOMA_MODULE ? (
            <ModuleCanvas isDarkMode={isDarkMode}>
              <div className="flex h-full min-w-0 overflow-hidden">
                <GlaucomaClassificationPanel
                  isDarkMode={isDarkMode}
                  patientName={patientName}
                  setPatientName={setPatientName}
                  mriNumber={mriNumber}
                  setMriNumber={setMriNumber}
                  caseId={caseId}
                  setCaseId={setCaseId}
                  eyeSide={eyeSide}
                  setEyeSide={setEyeSide}
                  fundusByEye={glaucomaFundusByEye}
                  setFundusByEye={setGlaucomaFundusByEye}
                  loading={glaucomaLoading}
                  loadingLabel={glaucomaLoadingLabel}
                  error={glaucomaError}
                  onResult={setGlaucomaResult}
                  onError={setGlaucomaError}
                  onClassify={handleGlaucomaClassification}
                />
                <GlaucomaClassificationResults
                  isDarkMode={isDarkMode}
                  patientName={patientName}
                  mriNumber={mriNumber}
                  caseId={caseId}
                  result={glaucomaResult}
                  loading={glaucomaLoading}
                  loadingLabel={glaucomaLoadingLabel}
                  error={glaucomaError}
                  fundusByEye={glaucomaFundusByEye}
                />
              </div>
            </ModuleCanvas>
          ) : (
            <ModuleCanvas isDarkMode={isDarkMode}>
              <div className="flex h-full min-w-0 overflow-hidden">
                <RetinaSegmentationPanel
                  isDarkMode={isDarkMode}
                  patientName={patientName}
                  setPatientName={setPatientName}
                  mriNumber={mriNumber}
                  setMriNumber={setMriNumber}
                  caseId={caseId}
                  setCaseId={setCaseId}
                  eyeSide={eyeSide}
                  setEyeSide={setEyeSide}
                  fundusByEye={fundusByEye}
                  setFundusByEye={setFundusByEye}
                  threshold={retinaThreshold}
                  setThreshold={setRetinaThreshold}
                  loading={retinaLoading}
                  loadingLabel={retinaLoadingLabel}
                  result={retinaResult}
                  onResult={setRetinaResult}
                  onError={setRetinaError}
                  onSegment={handleRetinaSegmentation}
                />
                <RetinaSegmentationResults
                  isDarkMode={isDarkMode}
                  patientName={patientName}
                  mriNumber={mriNumber}
                  caseId={caseId}
                  result={retinaResult}
                  loading={retinaLoading}
                  loadingLabel={retinaLoadingLabel}
                  error={retinaError}
                />
              </div>
            </ModuleCanvas>
          )}
        </main>
      </div>
    </div>
  );
}
