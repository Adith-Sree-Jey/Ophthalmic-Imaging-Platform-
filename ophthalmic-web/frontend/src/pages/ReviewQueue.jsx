import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { getQueueStats, getReviewQueue } from "../api";

const PRIORITY_STYLES = {
  critical: "border-red-500/60 bg-red-950/20",
  urgent: "border-amber-400/60 bg-amber-950/20",
  routine: "border-slate-700 bg-slate-900/50",
};

export default function ReviewQueue() {
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    const load = async () => {
      try {
        const [queue, queueStats] = await Promise.all([getReviewQueue(), getQueueStats()]);
        if (!active) return;
        setItems(Array.isArray(queue) ? queue : []);
        setStats(queueStats);
        setError("");
      } catch (err) {
        if (active) setError(err.message || "Failed to load review queue.");
      } finally {
        if (active) setLoading(false);
      }
    };

    load();
    const timer = window.setInterval(load, 30000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  return (
    <div className="min-h-screen bg-[#0B1220] text-white font-['Inter',system-ui,sans-serif]">
      <Navbar />
      <div className="mx-auto max-w-7xl px-6 py-8">
        <div className="mb-8 flex items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-semibold">Review Queue</h1>
            <p className="mt-2 text-sm text-slate-400">Pending and assigned cases for clinician review.</p>
          </div>
          <button
            type="button"
            onClick={() => navigate("/dashboard")}
            className="rounded-full border border-white/10 px-4 py-2 text-sm text-slate-200 transition hover:bg-white/5"
          >
            Back to Dashboard
          </button>
        </div>

        <div className="mb-8 grid gap-4 md:grid-cols-4">
          <StatCard label="Pending" value={stats?.pending_count ?? "-"} />
          <StatCard label="Under Review" value={stats?.under_review_count ?? "-"} />
          <StatCard label="Avg Review Time" value={typeof stats?.avg_review_time === "number" ? `${stats.avg_review_time.toFixed(1)}h` : "-"} />
          <StatCard label="Override Rate" value={stats?.override_rate_per_module ? `${Math.round(Object.values(stats.override_rate_per_module).reduce((sum, value) => sum + value, 0) / Math.max(Object.keys(stats.override_rate_per_module).length, 1) * 100)}%` : "-"} />
        </div>

        {error && <div className="mb-6 rounded-2xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div>}
        {loading ? <div className="text-sm text-slate-400">Loading queue...</div> : null}

        <div className="grid gap-4 lg:grid-cols-2">
          {items.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => navigate(`/cases/${item.id}/review`)}
              className={`rounded-3xl border p-5 text-left transition hover:-translate-y-0.5 hover:border-cyan-400/50 ${PRIORITY_STYLES[item.priority] || PRIORITY_STYLES.routine}`}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-lg font-semibold">{item.patient_name || "Unnamed patient"}</div>
                  <div className="mt-1 text-sm text-slate-400">MRI: {item.mri_number || "-"}</div>
                </div>
                <span className="rounded-full border border-white/10 px-3 py-1 text-xs uppercase tracking-[0.2em] text-slate-200">
                  {item.priority}
                </span>
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                {(item.module_types || []).map((module) => (
                  <span key={module} className="rounded-full bg-white/5 px-3 py-1 text-xs text-cyan-200">
                    {module}
                  </span>
                ))}
              </div>
              <div className="mt-5 grid grid-cols-2 gap-3 text-sm text-slate-300">
                <div>Waiting: {typeof item.age_hours === "number" ? `${item.age_hours.toFixed(1)}h` : "-"}</div>
                <div>Flags: {item.flag_count ?? 0}</div>
                <div>Status: {item.status}</div>
                <div>Reviewer: {item.assigned_reviewer || "Unassigned"}</div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
      <div className="text-xs uppercase tracking-[0.2em] text-slate-400">{label}</div>
      <div className="mt-3 text-3xl font-semibold text-white">{value}</div>
    </div>
  );
}
