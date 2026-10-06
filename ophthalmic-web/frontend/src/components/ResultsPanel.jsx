const GRADE_COLOURS = {
  NS1: "#1B7A3E",
  NS2: "#7A6000",
  NS3: "#B84B00",
  NS4: "#8B0000",
  Unknown: "#37474F",
};

const SEVERITY_LABELS = {
  NS1: "Minimal",
  NS2: "Mild",
  NS3: "Advanced",
  NS4: "Severe",
};

function ProbabilityBar({ label, value, isWinner, gradeColour, isDarkMode = false }) {
  const pct = Math.round((value || 0) * 100);
  const fillColour = isWinner ? gradeColour : "#B0BEC5";

  return (
    <div className="flex items-center gap-3 mb-2">
      <span className="text-sm font-bold w-8" style={{ color: GRADE_COLOURS[label] || "#5A6478" }}>
        {label}
      </span>
      <div className={`flex-1 rounded-full h-2.5 ${isDarkMode ? "bg-[#243244]" : "bg-[#EEEEEE]"}`}>
        <div
          className="rounded-full h-2.5 transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: fillColour }}
        />
      </div>
      <span className={`text-xs w-10 text-right ${isDarkMode ? "text-[#8FA0B6]" : "text-[#5A6478]"}`}>{pct}%</span>
    </div>
  );
}

function SingleResultPanel({ result, eyeLabel, patientName, mriNumber, caseId, isDarkMode = false }) {
  const grade = result.grade || "Unknown";
  const severity = SEVERITY_LABELS[grade] || "Unknown";
  const confidence = result.confidence;
  const confidencePct = confidence != null ? `${Math.round(confidence * 100)}%` : "N/A";
  const gradeColour = GRADE_COLOURS[grade] || GRADE_COLOURS.Unknown;
  const probabilities = result.probabilities || {};
  const attention = result.attention || {};
  const needsReview = result.needs_review || false;
  const reviewReason = result.review_reason || "";
  const patientNameValue = result.patient_name || patientName || "-";
  const mriNumberValue = result.mri_number || mriNumber || "-";
  const caseIdValue = result.case_id || caseId || result.filename || "-";
  const eyeSideValue = result.eye_side || eyeLabel;

  const dominantKey = Object.keys(attention).length > 0
    ? Object.entries(attention).sort((a, b) => b[1] - a[1])[0][0]
    : null;

  const modalities = [
    { key: "anterior", label: "Anterior Segment", b64Key: "anterior_segment_base64" },
    { key: "red_glow", label: "Red Glow", b64Key: "red_glow_base64" },
    { key: "slit_lamp", label: "Slit Lamp", b64Key: "slit_lamp_base64" },
  ];

  return (
    <div className="mb-6">
      <div className="flex items-center gap-2 mb-3">
        <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-bold text-white ${isDarkMode ? "bg-[#243248]" : "bg-[#0A2342]"}`}>
          {eyeSideValue}
        </span>
        <span className={`text-sm ${isDarkMode ? "text-[#8FA0B6]" : "text-[#5A6478]"}`}>Eye result</span>
      </div>

      <div
        className={`rounded px-5 py-4 mb-4 border border-l-4 ${isDarkMode ? "bg-[#1B2431] border-[#334155]" : "bg-white border-[#D0D7E2]"}`}
        style={{ borderLeftColor: gradeColour }}
      >
        <div className="flex items-start justify-between">
          <div>
            <h2 className={`text-lg font-bold ${isDarkMode ? "text-[#F8FAFC]" : "text-[#1A1A2E]"}`}>
              Nuclear Sclerosis Cataract - Grade {grade}
            </h2>
            <p className="text-sm font-semibold mt-0.5" style={{ color: gradeColour }}>
              {severity}
            </p>
            <p className={`text-sm mt-1 ${isDarkMode ? "text-[#A8B6CC]" : "text-[#5A6478]"}`}>Confidence: {confidencePct}</p>
            <p className={`text-xs mt-1 ${isDarkMode ? "text-[#8FA0B6]" : "text-[#5A6478]"}`}>
              Patient: {patientNameValue} | MRI: {mriNumberValue} | UID: {caseIdValue} | Eye: {eyeSideValue}
            </p>
          </div>
          <span className="text-white text-xs font-bold px-2 py-1 rounded shrink-0 ml-4" style={{ backgroundColor: gradeColour }}>
            {grade}
          </span>
        </div>
      </div>

      {needsReview && (
        <div className={`border-l-4 rounded px-4 py-3 mb-4 ${isDarkMode ? "bg-[#332815] border-[#F9A825]" : "bg-[#FFF8E1] border-[#F9A825]"}`}>
          <p className={`text-sm font-bold ${isDarkMode ? "text-[#F6C66D]" : "text-[#7A5000]"}`}>Clinical review recommended</p>
          {reviewReason && <p className={`text-sm mt-1 ${isDarkMode ? "text-[#F0D49A]" : "text-[#7A5000]"}`}>{reviewReason}</p>}
        </div>
      )}

      <div className={`rounded mb-4 grid grid-cols-3 divide-x border ${isDarkMode ? "bg-[#1B2431] border-[#334155] divide-[#334155]" : "bg-white border-[#D0D7E2] divide-[#D0D7E2]"}`}>
        {modalities.map(({ key, label, b64Key }) => {
          const b64 = result[b64Key];
          const attnPct = Math.round((attention[key] || 0) * 100);
          const isDominant = key === dominantKey;
          return (
            <div key={key} className="flex flex-col">
              <div className={`text-xs font-bold uppercase tracking-wide text-center py-2 border-b ${isDarkMode ? "text-[#D7E3F4] border-[#334155]" : "text-[#0A2342] border-[#D0D7E2]"}`}>
                {label}
              </div>
              <div className={`flex-1 flex items-center justify-center p-2 ${isDarkMode ? "bg-[#18212D]" : ""}`}>
                {b64 ? (
                  <img
                    src={`data:image/png;base64,${b64}`}
                    alt={label}
                    className="w-full object-contain max-h-44"
                  />
                ) : (
                  <span className={`text-xs italic ${isDarkMode ? "text-[#64748B]" : "text-[#B0BEC5]"}`}>Not provided</span>
                )}
              </div>
              <div className="px-3 pb-3 text-center">
                <p className={`text-xs mb-1 ${isDarkMode ? "text-[#8FA0B6]" : "text-[#5A6478]"}`}>Attention: {attnPct}%</p>
                <div className={`w-full rounded-full h-1.5 mb-1 ${isDarkMode ? "bg-[#243244]" : "bg-[#D0D7E2]"}`}>
                  <div className={`rounded-full h-1.5 transition-all ${isDarkMode ? "bg-[#8FB2FF]" : "bg-[#0A2342]"}`} style={{ width: `${attnPct}%` }} />
                </div>
                {isDominant && (
                  <span className={`inline-block text-[10px] font-bold px-1 rounded ${isDarkMode ? "bg-[#243248] text-[#D7E3F4]" : "bg-[#E3F2FD] text-[#0A2342]"}`}>
                    Primary
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {Object.keys(probabilities).length > 0 && (
        <div className={`rounded px-5 py-4 mb-4 border ${isDarkMode ? "bg-[#1B2431] border-[#334155]" : "bg-white border-[#D0D7E2]"}`}>
          <p className={`text-xs font-bold uppercase tracking-wide mb-3 ${isDarkMode ? "text-[#D7E3F4]" : "text-[#0A2342]"}`}>
            Classification Probabilities
          </p>
          {["NS1", "NS2", "NS3", "NS4"].map((candidateGrade) => (
            <ProbabilityBar
              key={candidateGrade}
              label={candidateGrade}
              value={probabilities[candidateGrade] || 0}
              isWinner={candidateGrade === grade}
              gradeColour={gradeColour}
              isDarkMode={isDarkMode}
            />
          ))}
        </div>
      )}

      {result.db_error && (
        <div className={`rounded px-4 py-3 mb-4 border ${isDarkMode ? "bg-[#341B1B] border-[#7F1D1D]" : "bg-red-50 border-red-200"}`}>
          <p className={`text-sm ${isDarkMode ? "text-[#FCA5A5]" : "text-red-700"}`}>{result.db_error}</p>
        </div>
      )}
    </div>
  );
}

export default function ResultsPanel({
  result,
  loading,
  loadingLabel = "Analysing Cataract image sets...",
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
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </div>
          <p className={`text-lg font-semibold mt-5 ${isDarkMode ? "text-[#F8FAFC]" : "text-[#1A1A2E]"}`}>No Cataract analysis results yet</p>
          <p className={`text-sm text-center mt-2 leading-6 ${isDarkMode ? "text-[#A8B6CC]" : "text-[#5A6478]"}`}>
            Upload a complete image set for OD, OS, or both, then run Cataract classification to see grades, probabilities, and clinical review flags here.
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
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
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
        <SingleResultPanel
          key={eye}
          eyeLabel={eye}
          result={eyeResult}
          patientName={patientName}
          mriNumber={mriNumber}
          caseId={caseId}
          isDarkMode={isDarkMode}
        />
      ))}
    </div>
  );
}
