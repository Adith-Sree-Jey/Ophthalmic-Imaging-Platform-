import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { getThresholdHistory, getThresholds, updateThreshold } from "../api";

const MODULES = ["cataract", "glaucoma", "retina"];

export default function ThresholdConfig() {
  const navigate = useNavigate();
  const role = localStorage.getItem("role");
  const [module, setModule] = useState("cataract");
  const [thresholds, setThresholds] = useState({});
  const [history, setHistory] = useState([]);
  const [form, setForm] = useState({ threshold_key: "", threshold_value: 0, priority_level: "routine", reason_template: "" });
  const [message, setMessage] = useState("");

  async function load() {
    const [thresholdPayload, historyPayload] = await Promise.all([getThresholds(module), getThresholdHistory(module)]);
    setThresholds(thresholdPayload || {});
    setHistory(historyPayload || []);
  }

  useEffect(() => {
    if (role === "admin") {
      load().catch((err) => setMessage(err.message || "Failed to load threshold configuration."));
    }
  }, [module, role]);

  if (role !== "admin") {
    return <div className="min-h-screen bg-[#0B1220] text-white"><Navbar /><div className="mx-auto max-w-3xl px-6 py-16"><div className="rounded-3xl border border-amber-500/30 bg-amber-500/10 p-8 text-amber-100">Admin access is required.</div></div></div>;
  }

  return (
    <div className="min-h-screen bg-[#0B1220] text-white font-['Inter',system-ui,sans-serif]">
      <Navbar />
      <div className="mx-auto max-w-6xl px-6 py-8">
        <div className="flex items-center justify-between gap-4">
          <h1 className="text-3xl font-semibold">Threshold Config</h1>
          <button
            type="button"
            onClick={() => navigate("/dashboard")}
            className="rounded-full border border-white/10 px-4 py-2 text-sm text-slate-200 transition hover:bg-white/5"
          >
            Back to Dashboard
          </button>
        </div>
        <div className="mt-6 flex gap-3">
          {MODULES.map((item) => (
            <button key={item} type="button" onClick={() => setModule(item)} className={`rounded-full px-4 py-2 text-sm ${module === item ? "bg-cyan-400 text-slate-950" : "bg-white/5 text-slate-300"}`}>{item}</button>
          ))}
        </div>
        {message && <div className="mt-4 rounded-2xl border border-cyan-500/40 bg-cyan-500/10 px-4 py-3 text-sm text-cyan-100">{message}</div>}
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
            <h2 className="text-lg font-semibold">Active Thresholds</h2>
            <div className="mt-4 space-y-3">
              {Object.values(thresholds).map((item) => (
                <button
                  key={item.threshold_key}
                  type="button"
                  onClick={() => setForm({ threshold_key: item.threshold_key, threshold_value: item.threshold_value, priority_level: item.priority_level, reason_template: item.reason_template })}
                  className="w-full rounded-2xl bg-slate-950/40 px-4 py-3 text-left text-sm text-slate-300"
                >
                  <div className="font-medium text-white">{item.threshold_key}</div>
                  <div className="mt-1">Value: {item.threshold_value} • Priority: {item.priority_level}</div>
                </button>
              ))}
            </div>
          </div>
          <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
            <h2 className="text-lg font-semibold">Update Threshold</h2>
            <div className="mt-4 space-y-3">
              <input value={form.threshold_key} onChange={(event) => setForm((prev) => ({ ...prev, threshold_key: event.target.value }))} placeholder="Threshold key" className="w-full rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3 text-sm" />
              <input value={form.threshold_value} onChange={(event) => setForm((prev) => ({ ...prev, threshold_value: Number(event.target.value) }))} placeholder="Threshold value" type="number" step="0.01" className="w-full rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3 text-sm" />
              <input value={form.priority_level} onChange={(event) => setForm((prev) => ({ ...prev, priority_level: event.target.value }))} placeholder="Priority" className="w-full rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3 text-sm" />
              <textarea value={form.reason_template} onChange={(event) => setForm((prev) => ({ ...prev, reason_template: event.target.value }))} placeholder="Reason template" className="h-28 w-full rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3 text-sm" />
              <button
                type="button"
                onClick={async () => {
                  await updateThreshold(module, form);
                  setMessage(`Updated ${form.threshold_key} for ${module}.`);
                  load();
                }}
                className="rounded-2xl bg-cyan-400 px-4 py-3 text-sm font-semibold text-slate-950"
              >
                Save Threshold
              </button>
            </div>
          </div>
        </div>
        <div className="mt-6 rounded-3xl border border-white/10 bg-white/5 p-5">
          <h2 className="text-lg font-semibold">History</h2>
          <div className="mt-4 space-y-3">
            {history.map((item) => (
              <div key={item.id} className="rounded-2xl bg-slate-950/40 px-4 py-3 text-sm text-slate-300">
                {item.updated_at} • {item.threshold_key} = {item.threshold_value} • {item.priority_level} • {item.is_active ? "active" : "archived"}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
