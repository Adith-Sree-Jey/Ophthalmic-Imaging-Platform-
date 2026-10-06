function downloadMask(base64Data, filename) {
  const link = document.createElement("a");
  link.href = `data:image/png;base64,${base64Data}`;
  link.download = filename || "retina_vessel_mask.png";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function RetinaResultCard({ eyeLabel, result, patientName, mriNumber, caseId }) {
  const patientNameValue = result.patient_name || patientName || "-";
  const mriNumberValue = result.mri_number || mriNumber || "-";
  const caseIdValue = result.case_id || caseId || "-";
  const eyeSideValue = result.eye_side || eyeLabel;
  return (
    <div className="mb-6">
      <div className="flex items-center gap-2 mb-3">
        <span className="inline-flex items-center rounded-full bg-[#0A2342] px-3 py-1 text-xs font-bold text-white">
          {eyeSideValue}
        </span>
        <span className="text-[#5A6478] text-sm">Retina vessel result</span>
      </div>

      <div className="bg-white border border-[#D0D7E2] rounded px-5 py-4 mb-4 border-l-4 border-l-[#0A2342]">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-[#1A1A2E] text-lg font-bold">Retina Blood Vessel Segmentation</h2>
            <p className="text-sm font-semibold mt-0.5 text-[#0A2342]">
              {result.result_summary || "Segmentation complete"}
            </p>
            <p className="text-[#5A6478] text-sm mt-1">
              Threshold: {typeof result.threshold === "number" ? result.threshold.toFixed(2) : "-"}
            </p>
            <p className="text-[#5A6478] text-xs mt-1">
              Patient: {patientNameValue} | MRI: {mriNumberValue} | UID: {caseIdValue} | Eye: {eyeSideValue}
            </p>
          </div>
          <span className="text-white text-xs font-bold px-2 py-1 rounded shrink-0 bg-[#0A2342]">
            Retina
          </span>
        </div>
      </div>

      <div className="bg-white border border-[#D0D7E2] rounded mb-4 grid grid-cols-1 xl:grid-cols-2 divide-y xl:divide-y-0 xl:divide-x divide-[#D0D7E2]">
        <div className="flex flex-col">
          <div className="text-xs font-bold text-[#0A2342] uppercase tracking-wide text-center py-2 border-b border-[#D0D7E2]">
            Original Fundus Image
          </div>
          <div className="flex-1 flex items-center justify-center p-3">
            <img
              src={`data:image/png;base64,${result.original_image_base64}`}
              alt="Original fundus"
              className="w-full object-contain max-h-[340px] rounded"
            />
          </div>
        </div>

        <div className="flex flex-col">
          <div className="text-xs font-bold text-[#0A2342] uppercase tracking-wide text-center py-2 border-b border-[#D0D7E2]">
            Vessel Mask Overlay
          </div>
          <div className="flex-1 flex items-center justify-center p-3">
            <img
              src={`data:image/png;base64,${result.overlay_image_base64}`}
              alt="Retina vessel overlay"
              className="w-full object-contain max-h-[340px] rounded"
            />
          </div>
        </div>
      </div>

      <div className="bg-white border border-[#D0D7E2] rounded px-5 py-4 mb-4">
        <p className="text-sm font-medium text-[#0A2342]">Segmentation complete</p>
        <p className="mt-2 text-xs text-[#5A6478]">
          Population-level retina performance metrics are invalidated pending leakage-free re-evaluation.
        </p>

        <button
          type="button"
          onClick={() => downloadMask(result.mask_base64, result.mask_filename)}
          className="mt-4 inline-flex items-center justify-center border border-[#0A2342] text-[#0A2342] px-4 py-2 rounded text-sm font-semibold hover:bg-[#F4F6F9] transition-colors"
        >
          Download Predicted Mask
        </button>

        {result.db_error && (
          <div className="bg-red-50 border border-red-200 rounded px-4 py-3 mt-4">
            <p className="text-sm text-red-700">{result.db_error}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default function RetinaSegmentationResults({
  result,
  loading,
  loadingLabel = "Running retina segmentation...",
  error,
  patientName,
  mriNumber,
  caseId,
  isDarkMode = false,
}) {
  const eyeResults = result && typeof result === "object"
    ? Object.entries(result).filter(([, eyeResult]) => eyeResult)
    : [];

  if (!eyeResults.length && !loading && !error) {
    return (
      <div className={`flex flex-1 items-center justify-center overflow-y-auto p-10 ${isDarkMode ? "bg-[#161D27]" : "bg-white"}`}>
        <div className="max-w-md text-center">
          <div className={`mx-auto flex h-20 w-20 items-center justify-center rounded-[24px] ${isDarkMode ? "bg-[#202938]" : "bg-[#F4F7FB]"}`}>
            <svg className={`w-10 h-10 ${isDarkMode ? "text-[#64748B]" : "text-[#B0BEC5]"}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 7h16M4 12h16M4 17h16" />
            </svg>
          </div>
          <p className={`text-lg font-semibold mt-5 ${isDarkMode ? "text-[#F8FAFC]" : "text-[#1A1A2E]"}`}>No Retina segmentation results yet</p>
          <p className={`text-sm text-center mt-2 leading-6 ${isDarkMode ? "text-[#A8B6CC]" : "text-[#5A6478]"}`}>
            Upload one fundus image for OD, OS, or both, then run Retina segmentation to generate overlay previews and downloadable vessel masks.
          </p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className={`flex flex-1 items-center justify-center overflow-y-auto p-10 ${isDarkMode ? "bg-[#161D27]" : "bg-white"}`}>
        <div className="text-center">
          <svg className={`animate-spin w-10 h-10 mx-auto ${isDarkMode ? "text-[#D7E3F4]" : "text-[#0A2342]"}`} fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <p className={`text-sm mt-4 ${isDarkMode ? "text-[#F8FAFC]" : "text-[#1A1A2E]"}`}>{loadingLabel}</p>
        </div>
      </div>
    );
  }

  if (error && !eyeResults.length) {
    return (
      <div className={`flex flex-1 items-center justify-center overflow-y-auto p-10 ${isDarkMode ? "bg-[#161D27]" : "bg-white"}`}>
        <div className="text-center max-w-sm">
          <p className="text-red-600 text-sm font-medium">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex-1 overflow-y-auto p-6 ${isDarkMode ? "bg-[#161D27]" : "bg-white"}`}>
      {eyeResults.map(([eye, eyeResult]) => (
        <RetinaResultCard
          key={eye}
          eyeLabel={eye}
          result={eyeResult}
          patientName={patientName}
          mriNumber={mriNumber}
          caseId={caseId}
        />
      ))}
    </div>
  );
}
