import { useCallback, useMemo, useRef, useState } from "react";
import PatientInfoFields from "./PatientInfoFields";
import { detectImageTypes, downloadBatchReport } from "../api";

const EYES = ["OD", "OS"];
const TYPE_OPTIONS = ["anterior_segment", "red_glow", "slit_lamp", "unknown"];
const MODALITIES = ["anterior_segment", "red_glow", "slit_lamp"];
const DETECTABLE_TYPES = new Set(TYPE_OPTIONS);

const TYPE_META = {
  anterior_segment: {
    label: "Anterior Segment",
    badge: "bg-blue-100 text-blue-700 border border-blue-300",
    dot: "bg-blue-500",
  },
  red_glow: {
    label: "Red Glow",
    badge: "bg-red-100 text-red-700 border border-red-300",
    dot: "bg-red-500",
  },
  slit_lamp: {
    label: "Slit Lamp",
    badge: "bg-amber-100 text-amber-700 border border-amber-300",
    dot: "bg-amber-500",
  },
  unknown: {
    label: "Unknown",
    badge: "bg-gray-100 text-gray-500 border border-gray-300",
    dot: "bg-gray-400",
  },
};

function normalizeDetectedType(value) {
  return DETECTABLE_TYPES.has(value) ? value : "unknown";
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(objectUrl);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Unable to read image for local auto-detect."));
    };
    image.src = objectUrl;
  });
}

async function inferTypeFromImageFile(file) {
  try {
    const image = await loadImageFromFile(file);
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) return "unknown";

    const width = 128;
    const height = 128;
    canvas.width = width;
    canvas.height = height;
    context.drawImage(image, 0, 0, width, height);

    const { data } = context.getImageData(0, 0, width, height);
    let brightnessSum = 0;
    let brightnessSqSum = 0;
    let redDominanceSum = 0;
    let darkPixels = 0;

    const pixelCount = width * height;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i] / 255;
      const g = data[i + 1] / 255;
      const b = data[i + 2] / 255;
      const brightness = (r + g + b) / 3;

      brightnessSum += brightness;
      brightnessSqSum += brightness * brightness;
      redDominanceSum += r - (g + b) / 2;

      if (brightness < 0.16) {
        darkPixels += 1;
      }
    }

    const meanBrightness = brightnessSum / pixelCount;
    const variance = Math.max(brightnessSqSum / pixelCount - meanBrightness * meanBrightness, 0);
    const brightnessStd = Math.sqrt(variance);
    const redDominance = redDominanceSum / pixelCount;
    const darkFraction = darkPixels / pixelCount;

    if (meanBrightness > 0.22 || (meanBrightness > 0.18 && brightnessStd > 0.16 && darkFraction < 0.62)) {
      return "anterior_segment";
    }

    if (darkFraction > 0.58 && redDominance > 0.02) {
      return "red_glow";
    }

    if (darkFraction > 0.52) {
      return "slit_lamp";
    }

    return "unknown";
  } catch {
    return "unknown";
  }
}

function inferTypeFromFilename(filename) {
  const text = (filename || "").toLowerCase();

  if (
    text.includes("anterior") ||
    text.includes("ant seg") ||
    text.includes("anterior_segment") ||
    text.includes("anterior segment")
  ) {
    return "anterior_segment";
  }

  if (
    text.includes("red_glow") ||
    text.includes("red glow") ||
    text.includes("retroillumination") ||
    text.includes("retro illumination")
  ) {
    return "red_glow";
  }

  if (
    text.includes("slit_lamp") ||
    text.includes("slit lamp") ||
    text.includes("slitlamp")
  ) {
    return "slit_lamp";
  }

  return "unknown";
}

async function resolveDetectedType(file, detection) {
  const apiType = normalizeDetectedType(detection?.detected_type);
  if (apiType !== "unknown") {
    return {
      assignedType: apiType,
      reasoning: detection?.reasoning || "Auto-detected from image content.",
      confidence: detection?.confidence || "",
    };
  }

  const filenameType = inferTypeFromFilename(file?.name);
  if (filenameType !== "unknown") {
    return {
      assignedType: filenameType,
      reasoning: "Auto-detected from filename pattern.",
      confidence: "medium",
    };
  }

  const imageType = await inferTypeFromImageFile(file);
  if (imageType !== "unknown") {
    return {
      assignedType: imageType,
      reasoning: "Auto-detected from local image analysis.",
      confidence: "medium",
    };
  }

  return {
    assignedType: "unknown",
    reasoning: detection?.reasoning || "Unable to auto-detect this image type.",
    confidence: detection?.confidence || "",
  };
}

function Spinner({ size = 14 }) {
  return (
    <svg
      className="animate-spin shrink-0"
      style={{ width: size, height: size }}
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

function getEyeMetrics(images) {
  const counts = Object.fromEntries(MODALITIES.map((modality) => [modality, 0]));
  for (const image of images) {
    if (counts[image.assignedType] != null) counts[image.assignedType] += 1;
  }
  const hasDuplicate = MODALITIES.some((modality) => counts[modality] > 1);
  const stillDetecting = images.some((img) => img.isDetecting);
  const isReady = images.length > 0 && counts.anterior_segment === 1 && !hasDuplicate && !stillDetecting;
  return { counts, hasDuplicate, stillDetecting, isReady };
}

function ImageQueueRow({ img, onRemove, onTypeChange, isDarkMode = false }) {
  const meta = TYPE_META[img.assignedType] || TYPE_META.unknown;

  return (
    <div className={`flex gap-2 p-2 rounded-lg border mb-2 items-start ${isDarkMode ? "border-[#334155] bg-[#1B2431]" : "border-[#E2E8F0] bg-white"}`}>
      <img
        src={img.previewUrl}
        alt={img.file.name}
        className={`w-12 h-12 object-cover rounded shrink-0 border ${isDarkMode ? "border-[#334155]" : "border-[#E2E8F0]"}`}
      />

      <div className="flex-1 min-w-0">
        <p className={`text-[11px] font-medium truncate ${isDarkMode ? "text-[#E2E8F0]" : "text-[#3A4258]"}`}>{img.file.name}</p>
        <div className="mt-1">
          {img.isDetecting ? (
            <span className={`inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border ${isDarkMode ? "text-[#A8B6CC] bg-[#202938] border-[#334155]" : "text-[#5A6478] bg-[#F4F6F9] border-[#D0D7E2]"}`}>
              <Spinner size={10} />
              Detecting...
            </span>
          ) : (
            <span className={`inline-flex items-center gap-1.5 text-[10px] font-semibold px-2 py-0.5 rounded-full ${meta.badge}`}>
              <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${meta.dot}`} />
              {meta.label}
            </span>
          )}
        </div>

        {!img.isDetecting && img.reasoning && (
          <p className={`text-[10px] mt-0.5 truncate ${isDarkMode ? "text-[#8FA0B6]" : "text-[#9AA3B5]"}`} title={img.reasoning}>
            {img.reasoning}
          </p>
        )}

        {!img.isDetecting && (
          <select
            value={img.assignedType}
            onChange={(e) => onTypeChange(img.id, e.target.value)}
            className={`mt-1.5 w-full text-[11px] border rounded px-1.5 py-1 transition-colors focus:outline-none ${isDarkMode ? "border-[#334155] text-[#E2E8F0] bg-[#161D27] focus:border-[#8FB2FF]" : "border-[#D0D7E2] text-[#1A1A2E] bg-white focus:border-[#0A2342]"}`}
          >
            {TYPE_OPTIONS.map((type) => (
              <option key={type} value={type}>
                {TYPE_META[type].label}
              </option>
            ))}
          </select>
        )}
      </div>

      <button
        onClick={() => onRemove(img.id)}
        className={`transition-colors text-sm mt-0.5 shrink-0 px-0.5 ${isDarkMode ? "text-[#64748B] hover:text-[#FCA5A5]" : "text-[#B0BEC5] hover:text-red-500"}`}
        title="Remove"
      >
        x
      </button>
    </div>
  );
}

export default function UploadPanel({
  isDarkMode = false,
  patientName,
  setPatientName,
  mriNumber,
  setMriNumber,
  caseId,
  setCaseId,
  eyeSide,
  setEyeSide,
  imagesByEye,
  setImagesByEye,
  loading,
  loadingLabel = "Analysing...",
  result,
  onResult,
  onError,
  onAnalyse,
}) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);
  const currentImages = imagesByEye[eyeSide] || [];
  const currentMetrics = getEyeMetrics(currentImages);
  const totalImages = EYES.reduce((sum, eye) => sum + (imagesByEye[eye]?.length || 0), 0);
  const readyEyes = EYES.filter((eye) => getEyeMetrics(imagesByEye[eye] || []).isReady);
  const invalidEyes = EYES.filter((eye) => {
    const images = imagesByEye[eye] || [];
    return images.length > 0 && !getEyeMetrics(images).isReady;
  });
  const canRun = readyEyes.length > 0 && invalidEyes.length === 0 && !loading;

  const downloadableResults = useMemo(() => {
    if (!result || typeof result !== "object") return [];
    const resultsToReport = [];
    if (result.OD) resultsToReport.push(result.OD);
    if (result.OS) resultsToReport.push(result.OS);
    return resultsToReport;
  }, [result]);
  const hasCombinedDownload = downloadableResults.length > 1;
  const eyeStats = useMemo(
    () =>
      Object.fromEntries(
        EYES.map((eye) => {
          const images = imagesByEye[eye] || [];
          const metrics = getEyeMetrics(images);
          const label = images.length === 0 ? "Empty" : metrics.isReady ? "Ready" : "Incomplete";
          return [eye, { count: images.length, label }];
        })
      ),
    [imagesByEye]
  );

  const addFiles = useCallback(
    async (fileList) => {
      const targetEye = eyeSide;
      const validFiles = Array.from(fileList).filter(
        (file) => file.type === "image/jpeg" || file.type === "image/png"
      );
      if (!validFiles.length) return;

      const pending = validFiles.map((file) => ({
        id: crypto.randomUUID(),
        file,
        previewUrl: URL.createObjectURL(file),
        isDetecting: true,
        assignedType: "unknown",
        confidence: "",
        reasoning: "",
      }));

      onResult(null);
      onError(null);
      setImagesByEye((prev) => ({
        ...prev,
        [targetEye]: [...(prev[targetEye] || []), ...pending],
      }));

      try {
        const detected = await detectImageTypes(validFiles);
        const resolvedById = Object.fromEntries(
          await Promise.all(
            pending.map(async (pendingImage, idx) => [
              pendingImage.id,
              await resolveDetectedType(validFiles[idx], detected[idx]),
            ])
          )
        );
        setImagesByEye((prev) => ({
          ...prev,
          [targetEye]: (prev[targetEye] || []).map((img) => {
            const resolved = resolvedById[img.id];
            if (!resolved) return img;
            return {
              ...img,
              isDetecting: false,
              assignedType: resolved.assignedType,
              confidence: resolved.confidence,
              reasoning: resolved.reasoning,
            };
          }),
        }));
      } catch (err) {
        const resolvedById = Object.fromEntries(
          await Promise.all(
            pending.map(async (pendingImage, idx) => [
              pendingImage.id,
              await resolveDetectedType(validFiles[idx], null),
            ])
          )
        );
        const unresolvedCount = Object.values(resolvedById).filter(
          (resolved) => resolved.assignedType === "unknown"
        ).length;
        setImagesByEye((prev) => ({
          ...prev,
          [targetEye]: (prev[targetEye] || []).map((img) => {
            const resolved = resolvedById[img.id];
            if (!resolved) return img;
            return {
              ...img,
              isDetecting: false,
              assignedType: resolved.assignedType,
              confidence: resolved.confidence,
              reasoning: resolved.reasoning,
            };
          }),
        }));
        if (unresolvedCount > 0) {
          onError(err.message || "Image type detection failed.");
        }
      }
    },
    [eyeSide, onError, onResult, setImagesByEye]
  );

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    addFiles(e.dataTransfer.files);
  };

  const handleRemove = (id) => {
    onResult(null);
    setImagesByEye((prev) => {
      const image = (prev[eyeSide] || []).find((item) => item.id === id);
      if (image) URL.revokeObjectURL(image.previewUrl);
      return {
        ...prev,
        [eyeSide]: (prev[eyeSide] || []).filter((item) => item.id !== id),
      };
    });
  };

  const handleTypeChange = (id, newType) => {
    onResult(null);
    setImagesByEye((prev) => ({
      ...prev,
      [eyeSide]: (prev[eyeSide] || []).map((img) =>
        img.id === id ? { ...img, assignedType: newType } : img
      ),
    }));
  };

  const handleClearImages = () => {
    Object.values(imagesByEye).flat().forEach((img) => URL.revokeObjectURL(img.previewUrl));
    setImagesByEye({ OD: [], OS: [] });
    setPatientName("");
    setMriNumber("");
    setCaseId("");
    setEyeSide("OD");
    onResult(null);
    onError(null);
  };

  const handleAnalyse = async () => {
    if (!canRun) return;
    await onAnalyse();
  };

  const handleDownloadPdf = async () => {
    const trimmedPatientName = patientName.trim();
    const trimmedMriNumber = mriNumber.trim();
    const trimmedCaseId = caseId.trim();
    const enrichReportResult = (eyeResult) => ({
      ...eyeResult,
      patient_name: trimmedPatientName || eyeResult?.patient_name || "",
      mri_number: trimmedMriNumber || eyeResult?.mri_number || "",
      case_id: trimmedCaseId || eyeResult?.case_id || eyeResult?.filename || "UNKNOWN",
    });

    const resultsToReport = [];
    if (result?.OD) resultsToReport.push(enrichReportResult(result.OD));
    if (result?.OS) resultsToReport.push(enrichReportResult(result.OS));

    if (resultsToReport.length === 0) {
      onError("No classification results available to download.");
      return;
    }

    try {
      const downloadName =
        trimmedCaseId ||
        trimmedMriNumber ||
        resultsToReport[0]?.case_id ||
        resultsToReport[0]?.filename ||
        "ophthalmic-imaging";

      await downloadBatchReport(resultsToReport, downloadName);
    } catch (err) {
      onError(err.message || "Failed to download report.");
    }
  };

  return (
    <div className={`w-[340px] shrink-0 border-r flex flex-col overflow-y-auto ${isDarkMode ? "bg-[#161D27] border-[#263041]" : "bg-white border-[#D0D7E2]"}`}>
      <div className="bg-[#0A2342] text-white px-4 py-3 shrink-0">
        <span className="font-semibold text-sm">Patient Information</span>
      </div>

      <PatientInfoFields
        patientName={patientName}
        setPatientName={setPatientName}
        mriNumber={mriNumber}
        setMriNumber={setMriNumber}
        caseId={caseId}
        setCaseId={setCaseId}
        eyeSide={eyeSide}
        setEyeSide={setEyeSide}
        eyeStats={eyeStats}
        isDarkMode={isDarkMode}
      />

      <div className={`border-t mx-4 shrink-0 ${isDarkMode ? "border-[#263041]" : "border-[#D0D7E2]"}`} />

      <div className="px-4 pt-3 pb-2 flex-1 flex flex-col">
        <div className="flex items-center justify-between mb-3 shrink-0">
          <p className={`text-xs font-semibold uppercase tracking-wide ${isDarkMode ? "text-[#D7E3F4]" : "text-[#0A2342]"}`}>
            Clinical Images - {eyeSide}
          </p>
          {totalImages > 0 && (
            <span className={`text-[10px] px-2 py-0.5 rounded-full ${isDarkMode ? "bg-[#202938] text-[#A8B6CC]" : "bg-[#F4F6F9] text-[#5A6478]"}`}>
              {totalImages} total
            </span>
          )}
        </div>

        <div
          onClick={() => inputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          className={`
            border-2 border-dashed rounded-xl cursor-pointer transition-all shrink-0
            flex flex-col items-center justify-center gap-2 py-6 px-4 text-center
            ${dragOver
              ? isDarkMode
                ? "border-[#6B8AF7] bg-[#1E293B]"
                : "border-[#0A2342] bg-[#EBF0FA]"
              : isDarkMode
                ? "border-[#334155] bg-[#161D27] hover:border-[#6B8AF7] hover:bg-[#1B2431]"
                : "border-[#C8D0DC] hover:border-[#0A2342] hover:bg-[#F4F6F9]"}
          `}
        >
          <div className={`w-11 h-11 rounded-full flex items-center justify-center transition-colors ${dragOver ? (isDarkMode ? "bg-[#243247]" : "bg-[#D6E0F5]") : (isDarkMode ? "bg-[#202938]" : "bg-[#F0F2F6]")}`}>
            <svg className={`w-6 h-6 ${isDarkMode ? "text-[#D7E3F4]" : "text-[#0A2342]"}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
          </div>
          <div>
            <p className={`text-sm font-semibold ${isDarkMode ? "text-[#F8FAFC]" : "text-[#0A2342]"}`}>
              {dragOver ? `Release to add ${eyeSide} images` : `Drop ${eyeSide} exam images here`}
            </p>
            <p className={`text-[11px] mt-0.5 ${isDarkMode ? "text-[#A8B6CC]" : "text-[#5A6478]"}`}>
              or click to browse - JPEG / PNG
            </p>
            <p className={`text-[10px] mt-1 ${isDarkMode ? "text-[#7C8CA5]" : "text-[#9AA3B5]"}`}>
              Upload up to 3 images for {eyeSide}. Switch eyes to manage the other side separately.
            </p>
          </div>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          multiple
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.length) addFiles(e.target.files);
            e.target.value = "";
          }}
        />

        {currentImages.length > 0 && (
          <div className="mt-3 flex-1">
            <div className="flex gap-2 mb-2 flex-wrap">
              {MODALITIES.map((type) => (
                <span
                  key={type}
                  className={`inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full ${TYPE_META[type].badge}`}
                >
                  <span className={`w-1.5 h-1.5 rounded-full ${TYPE_META[type].dot}`} />
                  {TYPE_META[type].label}
                </span>
              ))}
            </div>

            {currentImages.map((img) => (
              <ImageQueueRow
                key={img.id}
                img={img}
                isDarkMode={isDarkMode}
                onRemove={handleRemove}
                onTypeChange={handleTypeChange}
              />
            ))}

            {!currentMetrics.stillDetecting && currentMetrics.counts.anterior_segment === 0 && (
              <div className="flex items-start gap-2 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 mt-1">
                <span className="text-amber-500 text-sm shrink-0 mt-0.5">!</span>
                <p className="text-[11px] text-amber-700">
                  No <strong>Anterior Segment</strong> assigned for {eyeSide}. Required for classification.
                </p>
              </div>
            )}

            {!currentMetrics.stillDetecting && currentMetrics.hasDuplicate && (
              <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg px-3 py-2 mt-1">
                <span className="text-red-500 text-sm shrink-0 mt-0.5">!</span>
                <p className="text-[11px] text-red-700">
                  Duplicate type assigned within {eyeSide}. Each modality may appear only once per eye.
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      <div className={`border-t mx-4 shrink-0 ${isDarkMode ? "border-[#263041]" : "border-[#D0D7E2]"}`} />

      <div className="px-4 py-4 shrink-0">
        <button
          onClick={handleAnalyse}
          disabled={!canRun}
          className="w-full bg-[#0A2342] text-white py-2.5 rounded text-sm font-semibold hover:bg-[#1A4A7A] disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
        >
          {loading && <Spinner size={16} />}
          {loading ? loadingLabel : readyEyes.length > 1 ? "Run Classification for Both Eyes" : "Run Classification"}
        </button>

        <button
          onClick={handleClearImages}
          disabled={loading || totalImages === 0}
          className={`w-full border py-2 rounded text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed transition-colors mt-2 ${
            isDarkMode
              ? "border-[#334155] text-[#A8B6CC] hover:bg-[#1B2431]"
              : "border-[#D0D7E2] text-[#5A6478] hover:bg-[#F4F6F9]"
          }`}
        >
          Clear Images
        </button>

        <button
          onClick={handleDownloadPdf}
          disabled={downloadableResults.length === 0}
          className={`w-full border py-2 rounded text-sm font-semibold transition-colors mt-2 disabled:cursor-not-allowed ${
            isDarkMode
              ? "border-[#334155] text-[#D7E3F4] hover:bg-[#1B2431] disabled:border-[#334155] disabled:text-[#8FA0B6]"
              : "border-[#0A2342] text-[#0A2342] hover:bg-[#F4F6F9] disabled:border-[#D0D7E2] disabled:text-[#5A6478]"
          }`}
        >
          {hasCombinedDownload ? "Download Combined Report PDF" : "Download Report PDF"}
        </button>

        {!canRun && totalImages === 0 && (
          <p className={`text-[10px] text-center mt-2 ${isDarkMode ? "text-[#7C8CA5]" : "text-[#9AA3B5]"}`}>
            Add images for OD and/or OS, then run classification once the eye set is complete.
          </p>
        )}
      </div>
    </div>
  );
}
