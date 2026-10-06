import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import {
  getCalibrationDrift,
  getOverrideAnalysis,
  getPerformanceOverview,
  getReviewerAgreement,
  getThroughput,
} from "../api";

export default function PerformanceDashboard() {
  const navigate = useNavigate();
  const role = localStorage.getItem("role");
  const [overview, setOverview] = useState([]);
  const [drift, setDrift] = useState([]);
  const [overrideAnalysis, setOverrideAnalysis] = useState([]);
  const [agreement, setAgreement] = useState([]);
  const [throughput, setThroughput] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (role !== "admin") return;
    Promise.all([
      getPerformanceOverview(),
      getCalibrationDrift(),
      getOverrideAnalysis(),
      getReviewerAgreement(),
      getThroughput(),
    ])
      .then(([overviewPayload, driftPayload, overridePayload, agreementPayload, throughputPayload]) => {
        setOverview(overviewPayload || []);
        setDrift(driftPayload || []);
        setOverrideAnalysis(overridePayload || []);
        setAgreement(agreementPayload || []);
        setThroughput(throughputPayload || null);
        setError("");
      })
      .catch((err) => setError(err.message || "Failed to load performance data."));
  }, [role]);

  if (role !== "admin") {
    return (
      <div className="min-h-screen bg-[#0B1220] text-white font-['Inter',system-ui,sans-serif]">
        <Navbar />
        <div className="mx-auto max-w-3xl px-6 py-16">
          <div className="rounded-3xl border border-amber-500/30 bg-amber-500/10 p-8 text-amber-100">Admin access is required to view performance analytics.</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0B1220] text-white font-['Inter',system-ui,sans-serif]">
      <Navbar />
      <div className="mx-auto max-w-7xl px-6 py-8">
        <div className="mb-8">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h1 className="text-3xl font-semibold">Performance Dashboard</h1>
              <p className="mt-2 text-sm text-slate-400">Operational model monitoring across calibration, overrides, agreement, and throughput.</p>
            </div>
            <button
              type="button"
              onClick={() => navigate("/dashboard")}
              className="rounded-full border border-white/10 px-4 py-2 text-sm text-slate-200 transition hover:bg-white/5"
            >
              Back to Dashboard
            </button>
          </div>
        </div>

        {error && <div className="mb-4 rounded-2xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div>}

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
          {overview.map((item) => (
            <div key={item.module} className="rounded-3xl border border-white/10 bg-white/5 p-5">
              <div className="text-xs uppercase tracking-[0.18em] text-slate-400">{item.module}</div>
              <div className="mt-4 space-y-2 text-sm text-slate-300">
                <div>Accuracy: <span className="text-white">{item.accuracy?.toFixed(3)}</span></div>
                <div>ECE: <span className="text-white">{item.ece?.toFixed(3)}</span></div>
                <div>Flag Rate: <span className="text-white">{item.flag_rate?.toFixed(3)}</span></div>
                <div>Override Rate: <span className="text-white">{item.override_rate?.toFixed(3)}</span></div>
                <div>Avg Review Time: <span className="text-white">{item.avg_review_time?.toFixed(2)}h</span></div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <Panel title="Calibration Drift">
            {drift.map((item) => (
              <div key={`${item.module}-${item.week}`} className="flex items-center justify-between rounded-2xl bg-slate-950/40 px-4 py-3 text-sm">
                <span>{item.module} • Week {item.week}</span>
                <span className={item.alert ? "text-red-300" : "text-cyan-200"}>{item.ece?.toFixed(3)}</span>
              </div>
            ))}
          </Panel>
          <Panel title="Override Analysis">
            {overrideAnalysis.slice(0, 12).map((item, index) => (
              <div key={`${item.module}-${item.confidence_band}-${index}`} className="rounded-2xl bg-slate-950/40 px-4 py-3 text-sm text-slate-300">
                {item.module}: {item.model_grade || "-"} → {item.clinician_grade || "-"} ({item.confidence_band})
              </div>
            ))}
          </Panel>
          <Panel title="Reviewer Agreement">
            {agreement.length ? agreement.map((item) => (
              <div key={`${item.case_id}-${item.module}`} className="rounded-2xl bg-slate-950/40 px-4 py-3 text-sm text-slate-300">
                {item.module} • {item.case_id} • kappa {item.cohen_kappa?.toFixed(2)}
              </div>
            )) : <div className="text-sm text-slate-400">Not enough overlapping overrides yet.</div>}
          </Panel>
          <Panel title="Throughput">
            <div className="space-y-3 text-sm text-slate-300">
              <div>Avg time to first review: <span className="text-white">{throughput?.avg_time_to_first_review?.toFixed(2) || "0.00"}h</span></div>
              {(throughput?.cases_reviewed_per_reviewer || []).map((item) => (
                <div key={item.reviewer_id} className="rounded-2xl bg-slate-950/40 px-4 py-3">
                  Reviewer {item.reviewer_id}: {item.count} cases
                </div>
              ))}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function Panel({ title, children }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
      <h2 className="mb-4 text-lg font-semibold">{title}</h2>
      <div className="space-y-3">{children}</div>
    </div>
  );
}
