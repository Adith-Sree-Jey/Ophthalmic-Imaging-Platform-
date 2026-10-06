import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { exportAuditLog, getAuditLog } from "../api";

export default function AuditLog() {
  const navigate = useNavigate();
  const [filters, setFilters] = useState({ action: "", actor_id: "", case_id: "" });
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");

  async function loadAudit() {
    try {
      const payload = await getAuditLog({ ...filters, page_size: 10 });
      setRows(payload.items || []);
      setError("");
    } catch (err) {
      setError(err.message || "Failed to load audit log.");
    }
  }

  async function handleExport() {
    try {
      await exportAuditLog(filters);
      setError("");
    } catch (err) {
      setError(err.message || "Failed to export audit log.");
    }
  }

  useEffect(() => {
    loadAudit();
  }, []);

  return (
    <div className="min-h-screen bg-[#0B1220] text-white font-['Inter',system-ui,sans-serif]">
      <Navbar />
      <div className="mx-auto max-w-7xl px-6 py-8">
        <div className="mb-8 flex items-end justify-between gap-4">
          <div>
            <h1 className="text-3xl font-semibold">Audit Log</h1>
            <p className="mt-2 text-sm text-slate-400">Case lifecycle and clinician action history.</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => navigate("/dashboard")}
              className="rounded-full border border-white/10 px-4 py-2 text-sm text-slate-200 transition hover:bg-white/5"
            >
              Back to Dashboard
            </button>
            <button type="button" onClick={handleExport} className="rounded-full bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950">
              Export CSV
            </button>
          </div>
        </div>

        <div className="mb-6 grid gap-4 md:grid-cols-4">
          <input value={filters.action} onChange={(event) => setFilters((prev) => ({ ...prev, action: event.target.value }))} placeholder="Action type" className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm" />
          <input value={filters.actor_id} onChange={(event) => setFilters((prev) => ({ ...prev, actor_id: event.target.value }))} placeholder="Actor ID" className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm" />
          <input value={filters.case_id} onChange={(event) => setFilters((prev) => ({ ...prev, case_id: event.target.value }))} placeholder="Case ID" className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm" />
          <button type="button" onClick={loadAudit} className="rounded-2xl border border-white/10 px-4 py-3 text-sm">Apply Filters</button>
        </div>

        {error && <div className="mb-4 rounded-2xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div>}

        <div className="overflow-hidden rounded-3xl border border-white/10">
          <table className="min-w-full divide-y divide-white/10 text-sm">
            <thead className="bg-white/5 text-left text-slate-400">
              <tr>
                <th className="px-4 py-3">Timestamp</th>
                <th className="px-4 py-3">Actor</th>
                <th className="px-4 py-3">Action</th>
                <th className="px-4 py-3">Case</th>
                <th className="px-4 py-3">Diff</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10 bg-slate-950/30">
              {rows.map((row) => (
                <tr key={row.id}>
                  <td className="px-4 py-4 align-top">{row.timestamp}</td>
                  <td className="px-4 py-4 align-top">{row.actor_name || row.actor_id}</td>
                  <td className="px-4 py-4 align-top">{row.action}</td>
                  <td className="px-4 py-4 align-top">
                    {row.case_id ? <Link to={`/cases/${row.case_id}/review`} className="text-cyan-300 underline underline-offset-2">{row.case_id}</Link> : "-"}
                  </td>
                  <td className="px-4 py-4 align-top">
                    <details>
                      <summary className="cursor-pointer text-cyan-200">View</summary>
                      <pre className="mt-3 max-h-64 overflow-auto rounded-2xl bg-black/30 p-3 text-xs text-slate-300">
                        {JSON.stringify({ before: row.before_state, after: row.after_state }, null, 2)}
                      </pre>
                    </details>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
