import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { fitCalibration, getCalibrationStatus, getReliabilityDiagram } from "../api";

const MODULES = ["cataract", "glaucoma", "retina"];

export default function CalibrationAdmin() {
  const navigate = useNavigate();
  const role = localStorage.getItem("role");
  const [module, setModule] = useState("cataract");
  const [status, setStatus] = useState(null);
  const [bins, setBins] = useState([]);
  const [logits, setLogits] = useState("[[2.1, 0.5], [0.4, 1.8]]");
  const [labels, setLabels] = useState("[0, 1]");
  const [message, setMessage] = useState("");

  async function load() {
    const [statusPayload, binsPayload] = await Promise.all([getCalibrationStatus(module), getReliabilityDiagram(module)]);
    setStatus(statusPayload);
    setBins(Array.isArray(binsPayload) ? binsPayload : []);
  }

  useEffect(() => {
    if (role === "admin") {
      load().catch((err) => setMessage(err.message || "Failed to load calibration data."));
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
          <h1 className="text-3xl font-semibold">Calibration</h1>
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
            <h2 className="text-lg font-semibold">Fit Temperature</h2>
            <textarea value={logits} onChange={(event) => setLogits(event.target.value)} className="mt-4 h-40 w-full rounded-2xl border border-white/10 bg-slate-950/40 p-4 text-sm" />
            <textarea value={labels} onChange={(event) => setLabels(event.target.value)} className="mt-4 h-24 w-full rounded-2xl border border-white/10 bg-slate-950/40 p-4 text-sm" />
            <button
              type="button"
              onClick={async () => {
                const payload = { logits: JSON.parse(logits), labels: JSON.parse(labels) };
                const result = await fitCalibration(module, payload);
                setMessage(`Applied temperature ${result.temperature.toFixed(3)} to ${module}.`);
                load();
              }}
              className="mt-4 rounded-2xl bg-cyan-400 px-4 py-3 text-sm font-semibold text-slate-950"
            >
              Fit Calibration
            </button>
          </div>
          <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
            <h2 className="text-lg font-semibold">Current Status</h2>
            <div className="mt-4 space-y-3 text-sm text-slate-300">
              <div>Calibrated: <span className="text-white">{String(status?.is_calibrated || false)}</span></div>
              <div>Temperature: <span className="text-white">{status?.temperature ?? "-"}</span></div>
              <div>ECE: <span className="text-white">{status?.ece ?? "-"}</span></div>
              <div>Samples: <span className="text-white">{status?.sample_count ?? "-"}</span></div>
            </div>
            <div className="mt-6 space-y-2">
              {bins.map((bin) => (
                <div key={`${bin.bin_start}-${bin.bin_end}`} className="rounded-2xl bg-slate-950/40 px-4 py-3 text-sm text-slate-300">
                  {bin.bin_start?.toFixed(1)}-{bin.bin_end?.toFixed(1)} • conf {bin.mean_confidence?.toFixed(2)} • acc {bin.fraction_positive?.toFixed(2)} • n {bin.count}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
