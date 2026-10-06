/*
Codebase map used for this integration:
- `src/App.jsx` routes authenticated users into `Dashboard.jsx`; module switching happens inside the dashboard rather than separate pages.
- `src/pages/Dashboard.jsx` currently renders Cataract (`UploadPanel`/`ResultsPanel`) and Retina (`RetinaSegmentationPanel`/`RetinaSegmentationResults`) based on `selectedModule`.
- `src/components/Navbar.jsx` contains the sidebar entry where Glaucoma was marked "Coming soon".
- `src/components/PatientInfoFields.jsx` is the shared patient form for Patient Name, MRI Number, UID, and Eye Side.
- `src/components/UploadPanel.jsx` is the closest UI pattern for button styling and left-column workspace layout.
- `src/components/RetinaSegmentationPanel.jsx` shows the single-image-per-eye upload pattern Glaucoma should follow.
- `backend/main.py` exposes the authenticated FastAPI endpoints used by these panels.
- `backend/database.py` persists history records with a `module` field so Glaucoma entries can appear beside Cataract and Retina.
*/

import { useRef, useState } from "react";
import PatientInfoFields from "./PatientInfoFields";

const EYES = ["OD", "OS"];

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

export default function GlaucomaClassificationPanel({
  isDarkMode = false,
  patientName,
  setPatientName,
  mriNumber,
  setMriNumber,
  caseId,
  setCaseId,
  eyeSide,
  setEyeSide,
  fundusByEye,
  setFundusByEye,
  loading,
  loadingLabel = "Running Classification...",
  error,
  onResult,
  onError,
  onClassify,
}) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);
  const currentImage = fundusByEye[eyeSide] || null;
  const totalImages = EYES.reduce((sum, eye) => sum + (fundusByEye[eye] ? 1 : 0), 0);
  const readyEyes = EYES.filter((eye) => Boolean(fundusByEye[eye]));
  const canRun = readyEyes.length > 0 && !loading;

  const eyeStats = EYES.reduce((stats, eye) => {
    stats[eye] = {
      count: fundusByEye[eye] ? 1 : 0,
      label: fundusByEye[eye] ? "Ready" : "Empty",
    };
    return stats;
  }, {});

  const addFile = (fileList) => {
    const validFile = Array.from(fileList).find(
      (file) => file.type === "image/jpeg" || file.type === "image/png"
    );
    if (!validFile) return;

    onResult(null);
    onError(null);
    setFundusByEye((prev) => {
      const existing = prev[eyeSide];
      if (existing?.previewUrl) {
        URL.revokeObjectURL(existing.previewUrl);
      }
      return {
        ...prev,
        [eyeSide]: {
          id: crypto.randomUUID(),
          file: validFile,
          previewUrl: URL.createObjectURL(validFile),
        },
      };
    });
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    addFile(event.dataTransfer.files);
  };

  const handleRemove = () => {
    onResult(null);
    onError(null);
    setFundusByEye((prev) => {
      const existing = prev[eyeSide];
      if (existing?.previewUrl) {
        URL.revokeObjectURL(existing.previewUrl);
      }
      return { ...prev, [eyeSide]: null };
    });
  };

  const handleClearImages = () => {
    Object.values(fundusByEye).forEach((item) => {
      if (item?.previewUrl) {
        URL.revokeObjectURL(item.previewUrl);
      }
    });
    setFundusByEye({ OD: null, OS: null });
    setPatientName("");
    setMriNumber("");
    setCaseId("");
    setEyeSide("OD");
    onResult(null);
    onError(null);
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
            FUNDUS IMAGE - {eyeSide}
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
          onDragOver={(event) => {
            event.preventDefault();
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
                ? "border-[#334155] hover:border-[#6B8AF7] hover:bg-[#1B2431]"
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
              {dragOver ? `Release to add ${eyeSide} fundus image` : `Drop ${eyeSide} fundus image here`}
            </p>
            <p className={`text-[11px] mt-0.5 ${isDarkMode ? "text-[#A8B6CC]" : "text-[#5A6478]"}`}>
              or click to browse - JPEG / PNG
            </p>
            <p className={`text-[10px] mt-1 ${isDarkMode ? "text-[#7C8CA5]" : "text-[#9AA3B5]"}`}>
              One fundus image per eye side. Switch eyes to manage the other side separately.
            </p>
          </div>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          className="hidden"
          onChange={(event) => {
            if (event.target.files?.length) addFile(event.target.files);
            event.target.value = "";
          }}
        />

        {currentImage && (
          <div className={`mt-3 rounded-lg border p-3 ${isDarkMode ? "border-[#334155] bg-[#1B2431]" : "border-[#E2E8F0] bg-white"}`}>
            <div className="flex gap-3 items-start">
              <img
                src={currentImage.previewUrl}
                alt={currentImage.file.name}
                className="h-16 w-16 rounded-lg object-cover border border-[#E2E8F0] shrink-0"
              />
              <div className="flex-1 min-w-0">
                <p className={`text-sm font-semibold truncate ${isDarkMode ? "text-[#F8FAFC]" : "text-[#1A1A2E]"}`}>{currentImage.file.name}</p>
                <p className={`text-[11px] mt-1 ${isDarkMode ? "text-[#A8B6CC]" : "text-[#5A6478]"}`}>
                  Ready for glaucoma classification on {eyeSide}.
                </p>
                <button
                  type="button"
                  onClick={handleRemove}
                  className="mt-2 text-[11px] font-semibold text-red-600 hover:text-red-700 transition-colors"
                >
                  Remove image
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className={`border-t mx-4 shrink-0 ${isDarkMode ? "border-[#263041]" : "border-[#D0D7E2]"}`} />

      <div className="px-4 py-4 shrink-0">
        <button
          onClick={onClassify}
          disabled={!canRun}
          className="w-full bg-[#0A2342] text-white py-2.5 rounded text-sm font-semibold hover:bg-[#1A4A7A] disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
        >
          {loading && <Spinner size={16} />}
          {loading ? loadingLabel : "Run Classification"}
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

        {error && (
          <p className="mt-3 text-sm text-red-600">
            {error}
          </p>
        )}

        {!error && totalImages === 0 && (
          <p className={`text-[10px] text-center mt-2 ${isDarkMode ? "text-[#7C8CA5]" : "text-[#9AA3B5]"}`}>
            Add a fundus image for OD and/or OS, then run classification for the selected eye set.
          </p>
        )}
      </div>
    </div>
  );
}
