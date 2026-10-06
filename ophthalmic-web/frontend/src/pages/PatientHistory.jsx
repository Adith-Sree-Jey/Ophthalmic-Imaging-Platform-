import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import {
  downloadGlaucomaReportPdf,
  fetchAllPatients,
  fetchLongitudinal,
  fetchPatientHistory,
} from "../api";

const GRADE_NUM = { NS1: 1, NS2: 2, NS3: 3, NS4: 4 };
const GRADE_COLOR = {
  NS1: "#22d3ee",
  NS2: "#86efac",
  NS3: "#fbbf24",
  NS4: "#f87171",
};
const GLAUCOMA_COLOR = {
  Glaucoma: "#DC2626",
  Non_Glaucoma: "#16A34A",
  Unknown: "#9ca3af",
};
const SEVERITY_LABEL = {
  NS1: "Minimal",
  NS2: "Mild",
  NS3: "Moderate",
  NS4: "Severe",
};
const MODULE_LABEL = {
  cataract: "Cataract",
  glaucoma: "Glaucoma",
  retina_segmentation: "Retina Segmentation",
};

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  const normalized = value <= 1 ? value * 100 : value;
  return `${normalized.toFixed(1)}%`;
}

function normalizeProbabilityPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return 0;
  return value <= 1 ? value * 100 : value;
}

function ProgressionChart({ data, eyeSide }) {
  if (!data?.length) return null;

  const width = 560;
  const height = 200;
  const pad = 40;
  const innerWidth = width - pad * 2;
  const innerHeight = height - pad * 2;

  const points = data.map((item, index) => ({
    x: pad + (data.length === 1 ? innerWidth / 2 : (index / (data.length - 1)) * innerWidth),
    y: pad + innerHeight - ((GRADE_NUM[item.grade] - 1) / 3) * innerHeight,
    grade: item.grade,
    date: item.visit_date ? item.visit_date.slice(0, 10) : "-",
    review: item.needs_review,
  }));

  return (
    <div className="mt-4">
      <h4 className="text-xs uppercase tracking-widest text-gray-400 mb-2">
        Grade Progression - {eyeSide === "OD" ? "Right Eye (OD)" : "Left Eye (OS)"}
      </h4>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full rounded-xl bg-gray-900 border border-gray-700">
        {[1, 2, 3, 4].map((gradeIndex) => {
          const y = pad + innerHeight - ((gradeIndex - 1) / 3) * innerHeight;
          return (
            <g key={gradeIndex}>
              <line x1={pad} y1={y} x2={width - pad} y2={y} stroke="#374151" strokeWidth="1" strokeDasharray="4,3" />
              <text x={pad - 6} y={y + 4} textAnchor="end" fontSize="11" fill="#9ca3af">
                NS{gradeIndex}
              </text>
            </g>
          );
        })}

        {points.length > 1 && (
          <polyline
            points={points.map((point) => `${point.x},${point.y}`).join(" ")}
            fill="none"
            stroke="#06b6d4"
            strokeWidth="2.5"
            strokeLinejoin="round"
          />
        )}

        {points.map((point, index) => (
          <g key={`${point.grade}-${point.date}-${index}`}>
            <circle cx={point.x} cy={point.y} r="7" fill={GRADE_COLOR[point.grade] || "#6b7280"} stroke="#111827" strokeWidth="2" />
            {point.review && (
              <circle cx={point.x} cy={point.y} r="10" fill="none" stroke="#f87171" strokeWidth="1.5" strokeDasharray="3,2" />
            )}
            <text x={point.x} y={height - 4} textAnchor="middle" fontSize="10" fill="#6b7280">
              {point.date}
            </text>
            <text x={point.x} y={point.y - 12} textAnchor="middle" fontSize="11" fontWeight="bold" fill={GRADE_COLOR[point.grade] || "#fff"}>
              {point.grade}
            </text>
          </g>
        ))}
      </svg>
      <p className="text-xs text-gray-500 mt-1">
        Dashed red ring marks visits flagged for review.
      </p>
    </div>
  );
}

function VisitCard({ visit }) {
  const moduleKey = visit.module || "cataract";
  const moduleLabel = MODULE_LABEL[moduleKey] || moduleKey;
  const isRetinaVisit = moduleKey === "retina_segmentation";
  const isGlaucomaVisit = moduleKey === "glaucoma";
  const displayLabel = isGlaucomaVisit ? (visit.predicted_class || visit.grade || "-") : (visit.grade || "-");
  const gradeColor = isGlaucomaVisit
    ? (GLAUCOMA_COLOR[displayLabel] || GLAUCOMA_COLOR.Unknown)
    : (GRADE_COLOR[visit.grade] || "#9ca3af");
  const probabilityEntries = Object.entries(visit.probabilities || {}).filter(([, probability]) => probability != null);

  const handleDownloadReport = async () => {
    if (!visit.pdf_path) return;
    try {
      await downloadGlaucomaReportPdf(visit.pdf_path);
    } catch (error) {
      window.alert(error.message || "Failed to download Glaucoma report.");
    }
  };

  return (
    <div className="bg-gray-800 rounded-xl p-4 border border-gray-700 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 flex-wrap">
          <span
            className="text-lg font-bold px-3 py-1 rounded-lg"
            style={{ color: gradeColor, background: `${gradeColor}22` }}
          >
            {displayLabel}
          </span>
          <span className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded-full">
            {visit.eye_side === "OD" ? "Right Eye (OD)" : visit.eye_side === "OS" ? "Left Eye (OS)" : visit.eye_side}
          </span>
          <span className="text-xs bg-cyan-950/60 text-cyan-300 border border-cyan-800 px-2 py-1 rounded-full">
            {moduleLabel}
          </span>
          {visit.needs_review && (
            <span className="text-xs bg-red-900/50 text-red-400 border border-red-700 px-2 py-1 rounded-full">
              Review
            </span>
          )}
        </div>
        <span className="text-xs text-gray-500">
          {visit.visit_date ? new Date(visit.visit_date).toLocaleDateString() : "-"}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs text-gray-400">
        <div>
          Confidence: <span className="text-white font-medium">{formatPercent(visit.confidence)}</span>
        </div>
        <div>
          {isRetinaVisit ? (
            <>Result: <span className="text-white font-medium">{visit.result_summary || "Segmentation complete"}</span></>
          ) : isGlaucomaVisit ? (
            <>Prediction: <span className="text-white font-medium">{displayLabel}</span></>
          ) : (
            <>Severity: <span className="text-white font-medium">{SEVERITY_LABEL[visit.grade] || "-"}</span></>
          )}
        </div>
        <div>
          UID: <span className="text-white">{visit.case_id || "-"}</span>
        </div>
        <div>
          Graded by: <span className="text-white">{visit.graded_by || "-"}</span>
        </div>
      </div>

      {isGlaucomaVisit && visit.report_generated && visit.pdf_path && (
        <div className="flex items-center gap-3 text-xs text-gray-400">
          <span>
            Report: <span className="text-white">{visit.report_timestamp ? new Date(visit.report_timestamp).toLocaleString() : "Generated"}</span>
          </span>
          <button
            type="button"
            onClick={handleDownloadReport}
            className="text-cyan-300 underline underline-offset-2 hover:text-white"
          >
            Download Report
          </button>
        </div>
      )}

      {probabilityEntries.length > 0 && !isRetinaVisit && (
        <div className="mt-1">
          <div className="text-xs text-gray-500 mb-1">Class probabilities</div>
          <div className="flex gap-2">
            {probabilityEntries.map(([label, probability]) => (
              <div key={label} className="flex-1 text-center">
                <div
                  className="h-1.5 rounded-full mb-1"
                  style={{
                    background: isGlaucomaVisit ? (GLAUCOMA_COLOR[label] || "#6b7280") : (GRADE_COLOR[label] || "#6b7280"),
                    width: `${Math.round(normalizeProbabilityPercent(probability))}%`,
                    minWidth: "4px",
                  }}
                />
                <div className="text-xs text-gray-400">{label}</div>
                <div className="text-xs text-white font-medium">{normalizeProbabilityPercent(probability).toFixed(0)}%</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {isRetinaVisit && visit.result_summary && (
        <p className="text-xs text-cyan-300 bg-cyan-900/20 border border-cyan-800 rounded-lg px-3 py-2 mt-1">
          {visit.result_summary}
        </p>
      )}

      {visit.recommendation && (
        <p className="text-xs text-cyan-300 bg-cyan-900/20 border border-cyan-800 rounded-lg px-3 py-2 mt-1">
          {visit.recommendation}
        </p>
      )}
    </div>
  );
}

export default function PatientHistory() {
  const navigate = useNavigate();
  const location = useLocation();
  const [patients, setPatients] = useState([]);
  const [selectedMRI, setSelectedMRI] = useState("");
  const [mriInput, setMriInput] = useState("");
  const [history, setHistory] = useState(null);
  const [longitudinalOD, setLongitudinalOD] = useState(null);
  const [longitudinalOS, setLongitudinalOS] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeEye, setActiveEye] = useState("OD");
  const [activeModule, setActiveModule] = useState("ALL");

  const username = localStorage.getItem("username") || "Clinician";

  useEffect(() => {
    fetchAllPatients()
      .then(setPatients)
      .catch(() => {});
  }, []);

  useEffect(() => {
    const preselectedMri = location.state?.mriNumber;
    if (!preselectedMri) return;
    setMriInput(preselectedMri);
    setSelectedMRI(preselectedMri);
    loadHistory(preselectedMri);
  }, [location.state]);

  async function loadHistory(mriNumber) {
    setLoading(true);
    setError("");
    setHistory(null);
    setLongitudinalOD(null);
    setLongitudinalOS(null);
    setActiveModule("ALL");

    try {
      const patientHistory = await fetchPatientHistory(mriNumber);
      setHistory(patientHistory);

      try {
        setLongitudinalOD(await fetchLongitudinal(mriNumber, "OD"));
      } catch {
        setLongitudinalOD(null);
      }

      try {
        setLongitudinalOS(await fetchLongitudinal(mriNumber, "OS"));
      } catch {
        setLongitudinalOS(null);
      }
    } catch (err) {
      setError(err.message || "Failed to load history");
    } finally {
      setLoading(false);
    }
  }

  function handleSearch(event) {
    event.preventDefault();
    if (!mriInput.trim()) return;
    const trimmed = mriInput.trim();
    setSelectedMRI(trimmed);
    loadHistory(trimmed);
  }

  const visitsToShow = history
    ? history.visits.filter(
      (visit) =>
        (activeEye === "ALL" || visit.eye_side === activeEye) &&
        (activeModule === "ALL" || (visit.module || "cataract") === activeModule)
    )
    : [];

  const chartData = activeEye === "OD" ? longitudinalOD?.progression : longitudinalOS?.progression;
  const patientLabel =
    history?.patient_name ||
    patients.find((patient) => patient.mri_number === selectedMRI)?.patient_name ||
    selectedMRI;
  const availableModules = history
    ? ["ALL", ...Array.from(new Set(history.visits.map((visit) => visit.module || "cataract")))]
    : ["ALL"];

  return (
    <div className="h-screen flex flex-col overflow-hidden font-['Inter',system-ui,sans-serif]">
      <Navbar username={username} />
      <div className="flex-1 overflow-y-auto bg-gray-950 text-white p-6">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <button
              type="button"
              onClick={() => navigate("/dashboard")}
              className="mb-4 inline-flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-900 px-4 py-2 text-sm font-medium text-cyan-300 transition hover:border-cyan-500 hover:text-white"
            >
              <span aria-hidden="true">&lt;-</span>
              Back to Dashboard
            </button>
            <h1 className="text-2xl font-bold text-cyan-300">Patient History</h1>
            <p className="text-gray-400 text-sm mt-1">
              Search by MRI number to review prior visits and grade progression by eye.
            </p>
          </div>

          <form onSubmit={handleSearch} className="flex gap-3 mb-6">
            <input
              type="text"
              placeholder="Enter MRI Number..."
              value={mriInput}
              onChange={(event) => setMriInput(event.target.value)}
              className="flex-1 bg-gray-800 border border-gray-600 rounded-xl px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500"
            />
            <button
              type="submit"
              className="bg-cyan-600 hover:bg-cyan-500 text-white font-semibold px-6 py-2.5 rounded-xl transition"
            >
              Search
            </button>
          </form>

          {patients.length > 0 && (
            <div className="mb-6">
              <p className="text-xs text-gray-500 mb-2 uppercase tracking-widest">All Patients</p>
              <div className="flex flex-wrap gap-2">
                {patients.map((patient) => (
                  <button
                    key={patient.id}
                    onClick={() => {
                      setMriInput(patient.mri_number);
                      setSelectedMRI(patient.mri_number);
                      loadHistory(patient.mri_number);
                    }}
                    className={`text-sm px-3 py-1.5 rounded-lg border transition ${
                      selectedMRI === patient.mri_number
                        ? "bg-cyan-600 border-cyan-500 text-white"
                        : "bg-gray-800 border-gray-600 text-gray-300 hover:border-gray-400"
                    }`}
                  >
                    {patient.patient_name || patient.mri_number}
                    <span className="ml-1.5 text-xs opacity-60">{patient.visit_count}v</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {loading && <p className="text-cyan-400 animate-pulse">Loading history...</p>}
          {error && <p className="text-red-400 bg-red-900/20 border border-red-800 rounded-xl px-4 py-3">{error}</p>}

          {history && !loading && (
            <div>
              <div className="flex items-center justify-between mb-4 gap-4 flex-wrap">
                <div>
                  <h2 className="text-lg font-bold">{patientLabel}</h2>
                  <p className="text-xs text-gray-500">
                    MRI: {selectedMRI} · {history.total_visits} visit{history.total_visits !== 1 ? "s" : ""}
                  </p>
                </div>

                <div className="flex gap-1 bg-gray-800 rounded-lg p-1">
                  {["OD", "OS", "ALL"].map((eye) => (
                    <button
                      key={eye}
                      onClick={() => setActiveEye(eye)}
                      className={`px-3 py-1 rounded-md text-sm font-medium transition ${
                        activeEye === eye
                          ? "bg-cyan-600 text-white"
                          : "text-gray-400 hover:text-white"
                      }`}
                    >
                      {eye}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex gap-1 bg-gray-800 rounded-lg p-1 mb-4 w-fit flex-wrap">
                {availableModules.map((moduleKey) => (
                  <button
                    key={moduleKey}
                    onClick={() => setActiveModule(moduleKey)}
                    className={`px-3 py-1 rounded-md text-sm font-medium transition ${
                      activeModule === moduleKey
                        ? "bg-cyan-600 text-white"
                        : "text-gray-400 hover:text-white"
                    }`}
                  >
                    {moduleKey === "ALL" ? "All Modules" : (MODULE_LABEL[moduleKey] || moduleKey)}
                  </button>
                ))}
              </div>

              {activeEye !== "ALL" && (activeModule === "ALL" || activeModule === "cataract") && chartData?.length > 0 && (
                <div className="mb-6 bg-gray-900 rounded-2xl p-4 border border-gray-700">
                  <ProgressionChart data={chartData} eyeSide={activeEye} />
                </div>
              )}

              {visitsToShow.length === 0 ? (
                <p className="text-gray-500 text-sm">No visits found for the selected filters.</p>
              ) : (
                <div className="flex flex-col gap-4">
                  {visitsToShow.map((visit) => (
                    <VisitCard key={visit.visit_id} visit={visit} />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
