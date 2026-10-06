import { useLayoutEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import Navbar from "../components/Navbar";
import {
  approveReport,
  approveResult,
  downloadReport,
  getCase,
  overrideResult,
  rejectReport,
  rejectResult,
} from "../api";

export default function CaseReview() {
  const { caseId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [activeModule, setActiveModule] = useState("cataract");
  const [overrideForm, setOverrideForm] = useState({ clinician_grade: "", clinician_notes: "" });
  const [rejectReason, setRejectReason] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const loadCase = async (shouldApply = () => true) => {
    setLoading(true);
    try {
      const payload = await getCase(caseId);
      if (!shouldApply()) return;
      setData(payload);
      const firstModule = payload?.module_results?.[0]?.module;
      if (firstModule) setActiveModule(firstModule);
      setError("");
    } catch (err) {
      if (!shouldApply()) return;
      setError(err.message || "Failed to load case.");
    } finally {
      if (shouldApply()) setLoading(false);
    }
  };

  useLayoutEffect(() => {
    let active = true;
    setData(null);
    setActiveModule("cataract");
    setError("");
    setLoading(true);
    loadCase(() => active);
    return () => {
      active = false;
    };
  }, [caseId]);

  const selected = data?.module_results?.find((item) => item.module === activeModule) || data?.module_results?.[0];
  const jobResult = selected?.job_result || {};

  async function runAction(action) {
    if (!selected) return;
    setStatus("Saving review...");
    try {
      if (action === "approve") await approveResult(caseId, selected.module, "");
      if (action === "reject") await rejectResult(caseId, selected.module, rejectReason || "Rejected by clinician");
      if (action === "override") await overrideResult(caseId, selected.module, overrideForm.clinician_grade, overrideForm.clinician_notes);
      await loadCase();
      setStatus("Review saved.");
    } catch (err) {
      setError(err.message || "Action failed.");
      setStatus("");
    }
  }

  async function handleReport(action) {
    setStatus("Updating report...");
    try {
      if (action === "approve") await approveReport(caseId);
      if (action === "reject") await rejectReport(caseId, "Requires re-review");
      if (action === "download") await downloadReport(caseId);
      await loadCase();
      setStatus(action === "download" ? "Report downloaded." : "Report updated.");
    } catch (err) {
      setError(err.message || "Report action failed.");
      setStatus("");
    }
  }

  return (
    <div className="min-h-screen bg-[#0B1220] text-white font-['Inter',system-ui,sans-serif]">
      <Navbar />
      <div className="mx-auto max-w-[1600px] px-6 py-6">

        {/* Header */}
        <div className="mb-5 flex items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold">Case Review</h1>
            <p className="mt-1 text-xs text-slate-400">Ophthalmic Imaging case {caseId}</p>
          </div>
          <div className="flex gap-3">
            <button type="button" onClick={() => navigate("/review-queue")} className="rounded-full border border-white/10 px-4 py-2 text-sm text-slate-200">Back</button>
            <button type="button" onClick={loadCase} className="rounded-full bg-cyan-500 px-4 py-2 text-sm font-medium text-slate-950">Refresh</button>
          </div>
        </div>

        {error && <div className="mb-4 rounded-2xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div>}
        {loading && <div role="status" className="mb-4 text-sm text-slate-400">Loading case...</div>}
        {status && <div className="mb-4 rounded-2xl border border-cyan-500/40 bg-cyan-500/10 px-4 py-3 text-sm text-cyan-100">{status}</div>}

        {/* 3-column grid */}
        <div className="grid gap-5" style={{ gridTemplateColumns: "260px minmax(0,1fr) 300px" }}>

          {/* LEFT — Patient Summary + Report Actions */}
          <div className="flex flex-col gap-4">
            <section className="rounded-3xl border border-white/10 bg-white/5 p-5">
              <h2 className="text-base font-semibold">Patient Summary</h2>
              <div className="mt-3 space-y-2 text-sm text-slate-300">
                <div>Name: <span className="font-medium text-white">{data?.patient_summary?.patient_name || "-"}</span></div>
                <div>MRI: <span className="font-medium text-white">{data?.patient_summary?.mri_number || "-"}</span></div>
                <div>Status: <span className="font-medium text-white">{data?.case?.status || "-"}</span></div>
                <div>Priority: <span className={`font-semibold ${data?.case?.priority === "urgent" ? "text-amber-400" : data?.case?.priority === "critical" ? "text-red-400" : "text-slate-300"}`}>{data?.case?.priority || "-"}</span></div>
                <div>Reviewer: <span className="font-medium text-white">{data?.case?.assigned_reviewer || "Unassigned"}</span></div>
                <div>Report: <span className="font-medium text-white">{data?.case?.report_status || "pending"}</span></div>
              </div>
            </section>

            <section className="rounded-3xl border border-white/10 bg-white/5 p-5">
              <h2 className="text-base font-semibold mb-3">Report Actions</h2>
              <div className="flex flex-col gap-2">
                <button type="button" onClick={() => handleReport("approve")} className="rounded-2xl bg-emerald-500 px-4 py-2.5 text-sm font-semibold text-slate-950">Approve Report</button>
                <button type="button" onClick={() => handleReport("reject")} className="rounded-2xl bg-amber-500 px-4 py-2.5 text-sm font-semibold text-slate-950">Reject Report</button>
                <button type="button" onClick={() => handleReport("download")} className="rounded-2xl border border-white/10 px-4 py-2.5 text-sm font-semibold text-white">Download PDF</button>
              </div>
            </section>
          </div>

          {/* CENTER — Module info + Images */}
          <section className="rounded-3xl border border-white/10 bg-white/5 p-5 overflow-y-auto max-h-[calc(100vh-160px)]">
            {/* Module tabs */}
            <div className="mb-4 flex flex-wrap gap-2">
              {(data?.module_results || []).map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActiveModule(item.module)}
                  className={`rounded-full px-4 py-1.5 text-sm font-medium ${activeModule === item.module ? "bg-cyan-400 text-slate-950" : "bg-white/5 text-slate-300 hover:bg-white/10"}`}
                >
                  {item.module}
                </button>
              ))}
            </div>

            {selected ? (
              <>
                {/* Metrics grid */}
                <div className="grid grid-cols-2 gap-3">
                  <InfoCard label="Model Prediction" value={selected.model_prediction?.predicted_class || selected.model_grade || "-"} />
                  <InfoCard label="Confidence" value={typeof selected.model_confidence === "number" ? `${Math.round(selected.model_confidence * 100)}%` : "-"} />
                  <InfoCard label="Model Grade" value={selected.model_grade || "-"} />
                  <InfoCard label="Review Status" value={selected.review_status || "-"} />
                </div>

                {/* Flag banner */}
                {selected.flagged_for_review && (
                  <div className="mt-3 rounded-2xl border border-amber-400/40 bg-amber-400/10 px-4 py-2.5 text-sm text-amber-100">
                    Flag: {selected.flag_reason || "Manual review required"}
                  </div>
                )}

                {/* Images */}
                <div className="mt-4 grid gap-3 grid-cols-3">
                  {selected.module === "cataract" ? (
    <div className="col-span-3 grid grid-cols-3 gap-3">
      {/* Column 1 - Anterior */}
      <div className="flex flex-col gap-3">
        {jobResult.anterior_segment_base64 && <ImagePanel title="Anterior Segment" image={jobResult.anterior_segment_base64} />}
        {jobResult.gradcam_heatmap_base64 && <ImagePanel title="Anterior Grad-CAM" image={jobResult.gradcam_heatmap_base64} />}
      </div>
      {/* Column 2 - Red Glow */}
      <div className="flex flex-col gap-3">
        {jobResult.red_glow_base64 && <ImagePanel title="Red Glow" image={jobResult.red_glow_base64} />}
        {jobResult.red_glow_gradcam_base64 && <ImagePanel title="Red Glow Grad-CAM" image={jobResult.red_glow_gradcam_base64} />}
      </div>
      {/* Column 3 - Slit Lamp */}
      <div className="flex flex-col gap-3">
        {jobResult.slit_lamp_base64 && <ImagePanel title="Slit Lamp" image={jobResult.slit_lamp_base64} />}
        {jobResult.slit_lamp_gradcam_base64 && <ImagePanel title="Slit Lamp Grad-CAM" image={jobResult.slit_lamp_gradcam_base64} />}
      </div>
    </div>
                  ) : selected.module === "glaucoma" ? (
                    <>
                      {jobResult.overlay_b64 && <ImagePanel title="Vessel Overlay" image={jobResult.overlay_b64} />}
                      {jobResult.gradcam_b64 && <ImagePanel title="Grad-CAM" image={jobResult.gradcam_b64} />}
                    </>
                  ) : selected.module === "retina" ? (
                    <>
                      {jobResult.probability_map && <ImagePanel title="Vessel Probability Map" image={jobResult.probability_map} />}
                      {jobResult.overlay_image_base64 && <ImagePanel title="Segmentation Overlay" image={jobResult.overlay_image_base64} />}
                    </>
                  ) : (
                    <>
                      <ImagePanel title="Grad-CAM" image={jobResult.gradcam_heatmap_base64 || jobResult.gradcam_b64} />
                      <ImagePanel title="Original" image={jobResult.anterior_segment_base64 || jobResult.overlay_b64 || jobResult.probability_map} />
                    </>
                  )}
                </div>
              </>
            ) : (
              <div className="text-sm text-slate-400">No module results available for this case yet.</div>
            )}
          </section>

          {/* RIGHT — Review Actions + Audit Trail */}
          <div className="flex flex-col gap-4 overflow-y-auto max-h-[calc(100vh-160px)]">

            {/* Review action buttons */}
            {selected && (
              <section className="rounded-3xl border border-white/10 bg-white/5 p-5">
                <h2 className="text-base font-semibold mb-3">Module Review</h2>
                <div className="flex gap-2 flex-wrap">
                  <button type="button" onClick={() => runAction("approve")} className="rounded-2xl bg-emerald-500 px-4 py-2 text-sm font-semibold text-slate-950">Approve</button>
                  <button type="button" onClick={() => runAction("override")} className="rounded-2xl bg-amber-500 px-4 py-2 text-sm font-semibold text-slate-950">Override</button>
                  <button type="button" onClick={() => runAction("reject")} className="rounded-2xl bg-red-500 px-4 py-2 text-sm font-semibold text-white">Reject</button>
                </div>

                {/* Override form */}
                <div className="mt-4">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Override</h3>
                  <input
                    value={overrideForm.clinician_grade}
                    onChange={(e) => setOverrideForm((prev) => ({ ...prev, clinician_grade: e.target.value }))}
                    placeholder="Clinician grade"
                    className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-white mb-2"
                  />
                  <textarea
                    value={overrideForm.clinician_notes}
                    onChange={(e) => setOverrideForm((prev) => ({ ...prev, clinician_notes: e.target.value }))}
                    placeholder="Override notes"
                    className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-white h-20"
                  />
                </div>

                {/* Reject form */}
                <div className="mt-4">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Reject</h3>
                  <textarea
                    value={rejectReason}
                    onChange={(e) => setRejectReason(e.target.value)}
                    placeholder="Rejection reason"
                    className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-white h-20"
                  />
                  <p className="mt-2 text-xs text-slate-500">Overridden results preserve the original model output and store clinician input separately.</p>
                </div>
              </section>
            )}

            {/* Audit Trail */}
            <section className="rounded-3xl border border-white/10 bg-white/5 p-5 flex-1">
              <h2 className="text-base font-semibold mb-3">Audit Trail</h2>
              <div className="space-y-3">
                {(data?.audit_trail || []).length === 0 ? (
                  <p className="text-sm text-slate-400">No audit entries yet.</p>
                ) : (
                  (data?.audit_trail || []).map((entry) => (
                    <div key={entry.id} className="rounded-2xl border border-white/10 bg-slate-950/40 p-3">
                      <div className="text-sm font-medium text-white">{entry.action}</div>
                      <div className="mt-1 text-xs text-slate-400">{entry.actor_name || `User ${entry.actor_id}`} • {entry.timestamp}</div>
                    </div>
                  ))
                )}
              </div>
            </section>
          </div>

        </div>
      </div>
    </div>
  );
}

function InfoCard({ label, value }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-slate-950/40 p-3">
      <div className="text-xs uppercase tracking-[0.15em] text-slate-500">{label}</div>
      <div className="mt-2 text-base font-semibold text-white">{value}</div>
    </div>
  );
}

function ImagePanel({ title, image }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-slate-950/40 p-3">
      <div className="mb-2 text-xs font-semibold text-slate-300">{title}</div>
      {image
        ? <img src={`data:image/png;base64,${image}`} alt={title} className="max-h-40 w-full rounded-xl object-contain" />
        : <div className="text-xs text-slate-500">No image available.</div>
      }
    </div>
  );
}
