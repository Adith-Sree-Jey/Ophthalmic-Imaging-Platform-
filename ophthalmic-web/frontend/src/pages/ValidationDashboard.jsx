import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { downloadValidationReportPdf, getReliabilityDiagram, getValidationReport } from "../api";

const MODULES = ["cataract", "glaucoma"];

export default function ValidationDashboard() {
  const navigate = useNavigate();
  const [module, setModule] = useState("cataract");
  const [report, setReport] = useState(null);
  const [bins, setBins] = useState([]);
  const [error, setError] = useState("");
  const role = localStorage.getItem("role");

  useEffect(() => {
    if (role !== "admin") return;
    Promise.all([getValidationReport(module), getReliabilityDiagram(module)])
      .then(([reportPayload, reliabilityPayload]) => {
        setReport(reportPayload);
        setBins(Array.isArray(reliabilityPayload) ? reliabilityPayload : []);
        setError("");
      })
      .catch((err) => setError(err.message || "Failed to load validation report."));
  }, [module, role]);

  if (role !== "admin") {
    return <RestrictedView />;
  }

  return (
    <div className="min-h-screen bg-[#0B1220] text-white font-['Inter',system-ui,sans-serif]">
      <Navbar />
      <div className="mx-auto max-w-7xl px-6 py-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-semibold">Validation Reports</h1>
            <p className="mt-2 text-sm text-slate-400">Standardized machine-readable validation summaries for each Ophthalmic Imaging module.</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => navigate("/dashboard")}
              className="rounded-full border border-white/10 px-4 py-2 text-sm text-slate-200 transition hover:bg-white/5"
            >
              Back to Dashboard
            </button>
            <button type="button" onClick={() => downloadValidationReportPdf(module)} className="rounded-full bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950">Download PDF</button>
          </div>
        </div>

        <div className="mb-6 flex gap-3">
          {MODULES.map((item) => (
            <button key={item} type="button" onClick={() => setModule(item)} className={`rounded-full px-4 py-2 text-sm ${module === item ? "bg-cyan-400 text-slate-950" : "bg-white/5 text-slate-300"}`}>{item}</button>
          ))}
        </div>

        {error && <div className="mb-4 rounded-2xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div>}
        {report ? (
          <>
            <div className="grid gap-4 md:grid-cols-3 xl:grid-cols-5">
              <Metric label="Accuracy" value={report.accuracy} />
              <Metric label="Sensitivity" value={report.sensitivity} />
              <Metric label="Specificity" value={report.specificity} />
              <Metric label="AUC" value={report.auc_roc} />
              <Metric label="F1" value={report.f1_score} />
              <Metric label="Kappa" value={report.cohen_kappa} />
              <Metric label="ECE Before" value={report.ece_before_calibration} />
              <Metric label="ECE After" value={report.ece_after_calibration} />
              <Metric label="Flag Rate" value={report.flag_rate} />
              <Metric label="Override Rate" value={report.override_rate} />
            </div>

            <div className="mt-8 grid gap-6 lg:grid-cols-2">
              <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                <h2 className="text-lg font-semibold">Reliability Diagram</h2>
                <svg viewBox="0 0 320 220" className="mt-4 w-full rounded-2xl bg-slate-950/40">
                  <line x1="40" y1="180" x2="280" y2="20" stroke="#475569" strokeDasharray="4 4" />
                  <line x1="40" y1="180" x2="280" y2="180" stroke="#475569" />
                  <line x1="40" y1="180" x2="40" y2="20" stroke="#475569" />
                  <polyline
                    fill="none"
                    stroke="#22d3ee"
                    strokeWidth="3"
                    points={bins.map((bin, index) => `${40 + index * 24},${180 - ((bin.fraction_positive || 0) * 160)}`).join(" ")}
                  />
                </svg>
              </div>
              <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                <h2 className="text-lg font-semibold">Confusion Matrix</h2>
                <div className="mt-4 grid gap-2" style={{ gridTemplateColumns: `repeat(${(report.class_labels?.length || 0) + 1}, minmax(0, 1fr))` }}>
                  <div />
                  {(report.class_labels || []).map((label) => <div key={`head-${label}`} className="text-center text-xs text-slate-400">{label}</div>)}
                  {(report.class_labels || []).map((rowLabel, rowIndex) => (
                    <FragmentRow key={rowLabel} label={rowLabel} values={report.confusion_matrix?.[rowIndex] || []} />
                  ))}
                </div>
              </div>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}

function RestrictedView() {
  return (
    <div className="min-h-screen bg-[#0B1220] text-white font-['Inter',system-ui,sans-serif]">
      <Navbar />
      <div className="mx-auto max-w-3xl px-6 py-16">
        <div className="rounded-3xl border border-amber-500/30 bg-amber-500/10 p-8 text-amber-100">Admin access is required to view validation reports.</div>
      </div>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 p-4">
      <div className="text-xs uppercase tracking-[0.18em] text-slate-400">{label}</div>
      <div className="mt-3 text-2xl font-semibold">{typeof value === "number" ? value.toFixed(3) : "-"}</div>
    </div>
  );
}

function FragmentRow({ label, values }) {
  return (
    <>
      <div className="self-center text-xs text-slate-400">{label}</div>
      {values.map((value, index) => (
        <div key={`${label}-${index}`} className="rounded-2xl bg-cyan-500/10 p-3 text-center text-sm text-white">{value}</div>
      ))}
    </>
  );
}
