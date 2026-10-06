import { useState } from "react";
import { downloadGlaucomaReportPdf } from "../api";

const RISK_STYLES = {
  low: "bg-[#E8F7EE] text-[#166534]",
  moderate: "bg-[#FFF4D6] text-[#92400E]",
  high: "bg-[#FDEAEA] text-[#B42318]",
  unknown: "bg-[#EEF2F7] text-[#526074]",
};

function formatMetric(value, digits = 2) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "-";
}

function formatLabel(value) {
  return String(value || "Unknown")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function cleanReportText(text) {
  if (!text || typeof text !== "string") return "";
  return (
    text
      .replace(/\*\*(.*?)\*\*/g, "$1")
      .replace(/\*(.*?)\*/g, "$1")
      .replace(/#{1,6}\s*/g, "")
      .replace(/^\s*\d+\.\s+/gm, "")
      .replace(/__/g, "")
      .replace(/^\s*[-*+]\s+/gm, "")
      .replace(/^\s*\d+\)\s+/gm, "")
      .trim() || ""
  );
}

function formatPercent(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${numeric.toFixed(2)}%` : "unknown confidence";
}

function getReportRawSource(report) {
  const candidates = [
    report?.raw_text,
    report?.rawText,
    report?.text,
    report?.report_text,
    report?.generated_text,
    report?.content,
  ];

  for (const candidate of candidates) {
    const cleaned = cleanReportText(candidate);
    if (cleaned) return cleaned;
  }

  return "";
}

function parseReportSectionsFromRaw(report) {
  const rawText = getReportRawSource(report);
  const sections = {
    findings: "",
    interpretation: "",
    recommendation: "",
    disclaimer: "",
  };

  if (!rawText) return sections;

  const headingMap = {
    findings: "findings",
    finding: "findings",
    summary: "findings",
    interpretation: "interpretation",
    assessment: "interpretation",
    recommendation: "recommendation",
    plan: "recommendation",
    disclaimer: "disclaimer",
    caution: "disclaimer",
    limitation: "disclaimer",
  };

  let currentKey = null;
  let sawHeading = false;

  for (const rawLine of rawText.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;

    const normalized = line
      .replace(/^[\W_]*\d*\.?\s*/, "")
      .replace(/:$/, "")
      .trim()
      .toLowerCase();

    if (headingMap[normalized]) {
      currentKey = headingMap[normalized];
      sawHeading = true;
      continue;
    }

    if (currentKey) {
      sections[currentKey] = `${sections[currentKey]} ${line}`.trim();
    }
  }

  if (sawHeading && Object.values(sections).some(Boolean)) {
    return sections;
  }

  sections.findings = rawText;
  return sections;
}

function buildFeatureFallbackSections(result) {
  const vesselFeatures = result?.vessel_features || {};
  const prediction = result?.prediction || result?.predicted_class || "Unknown";
  const confidence = formatPercent(result?.confidence);
  const cdr = Number(vesselFeatures.cdr);
  const vesselDensity = Number(vesselFeatures.vessel_density);
  const fractalDimension = Number(vesselFeatures.fractal_dimension);
  const meanTortuosity = Number(vesselFeatures.mean_tortuosity);
  const thinVesselRatio = Number(vesselFeatures.thin_vessel_ratio);
  const cdrInterpretation = vesselFeatures.cdr_interpretation || "no cup-to-disc interpretation was available";
  const riskLevel = String(result?.risk_level || vesselFeatures.risk_level || "unknown").toLowerCase();
  const sourceNote = result?.vessel_source === "estimated" ? " These vessel values are approximate image estimates." : "";

  const findingsParts = [];
  if (Number.isFinite(cdr)) {
    findingsParts.push(`Cup-to-disc ratio is ${cdr.toFixed(3)}, interpreted as ${cdrInterpretation}.`);
  }
  if (Number.isFinite(vesselDensity)) {
    findingsParts.push(`Vessel density is ${vesselDensity.toFixed(3)}.`);
  }
  if (Number.isFinite(fractalDimension)) {
    findingsParts.push(`Fractal dimension is ${fractalDimension.toFixed(3)}.`);
  }
  if (Number.isFinite(meanTortuosity)) {
    findingsParts.push(`Mean tortuosity is ${meanTortuosity.toFixed(4)}.`);
  }
  if (Number.isFinite(thinVesselRatio)) {
    findingsParts.push(`Thin vessel ratio is ${thinVesselRatio.toFixed(3)}.`);
  }

  return {
    findings: findingsParts.join(" ") || "",
    interpretation:
      findingsParts.length
        ? `These extracted biomarkers suggest a ${riskLevel} vascular and disc-risk profile. The classifier output of ${prediction} at ${confidence} should be treated as supporting context rather than the sole conclusion.${sourceNote}`
        : "",
    recommendation:
      findingsParts.length
        ? "Correlate these image-derived features with intraocular pressure, optic disc examination, OCT RNFL analysis, and visual field testing before making management decisions."
        : "",
    disclaimer:
      findingsParts.length
        ? "This is AI-assisted screening support based on image analysis and must not replace ophthalmologist review or a full clinical examination."
        : "",
  };
}

function getReportSectionText(result, key) {
  const report = result?.report || {};
  const structured = cleanReportText(report?.[key]);
  if (structured) return structured;

  const parsed = parseReportSectionsFromRaw(report);
  const parsedSection = cleanReportText(parsed[key]);
  if (parsedSection) return parsedSection;

  const fallback = buildFeatureFallbackSections(result);
  return cleanReportText(fallback[key]);
}

function getGroundingBadge(score) {
  if (!score || typeof score !== "string") {
    return { label: "Grounding unavailable", colorClass: "bg-[#EEF2F7] text-[#526074]" };
  }
  const pctMatch = score.match(/\((\d+)%\)/);
  const label = pctMatch ? `Grounded ${pctMatch[1]}%` : score;
  let colorClass;
  if (/^[45]\//.test(score)) {
    colorClass = "bg-[#E8F7EE] text-[#166534]";
  } else if (/^3\//.test(score)) {
    colorClass = "bg-[#FFF4D6] text-[#92400E]";
  } else {
    colorClass = "bg-[#FDEAEA] text-[#B42318]";
  }
  return { label, colorClass };
}

function ProbabilityBar({ label, value, winner }) {
  const pct = Math.max(0, Math.min(100, Number(value) || 0));
  const isGlaucoma = label.toLowerCase() === "glaucoma";
  const fillColor = winner ? (isGlaucoma ? "#B91C1C" : "#15803D") : "#B0BEC5";
  const labelColor = isGlaucoma ? "#B91C1C" : "#15803D";

  return (
    <div className="mb-3 flex items-center gap-3">
      <span className="min-w-[88px] text-sm font-bold" style={{ color: labelColor }}>
        {formatLabel(label)}
      </span>
      <div className="h-2.5 flex-1 rounded-full bg-[#EEEEEE]">
        <div
          className="h-2.5 rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: fillColor }}
        />
      </div>
      <span className="w-12 text-right text-xs text-[#5A6478]">{pct.toFixed(1)}%</span>
    </div>
  );
}

function FeatureRow({ label, value, digits = 2, normalRange = "—", highlight = null, note = null }) {
  const highlightClass =
    highlight === "red" ? "bg-[#FDEAEA]" : highlight === "amber" ? "bg-[#FFF4D6]" : "";

  return (
    <>
      <tr className={`border-b border-[#E6ECF3] ${highlightClass}`}>
        <td className="px-4 py-3 text-sm font-medium text-[#1A1A2E]">{label}</td>
        <td className="px-4 py-3 text-right text-sm text-[#5A6478]">{formatMetric(value, digits)}</td>
        <td className="px-4 py-3 text-right text-sm text-[#9AA3B5]">{normalRange}</td>
      </tr>
      {note && (
        <tr className={`border-b border-[#E6ECF3] ${highlightClass}`}>
          <td colSpan={3} className="px-4 pb-3 pt-0 text-xs italic text-[#9AA3B5]">{note}</td>
        </tr>
      )}
    </>
  );
}

function GlaucomaResultCard({ eyeLabel, result, patientName, mriNumber, caseId }) {
  const [downloadError, setDownloadError] = useState("");
  const [downloadLoading, setDownloadLoading] = useState(false);
  const prediction = result.prediction || result.predicted_class || "Unknown";
  const confidence = Number(result.confidence) || 0;
  const riskLevel = String(
    result.risk_level || result.vessel_features?.risk_level || "unknown"
  ).toLowerCase();
  const probabilities = result.probabilities || {};
  const vesselFeatures = result.vessel_features || {};
  const report = result.report || {};
  const patientInfo = result.patient_info || {};
  const eyeSideValue = patientInfo.eye_side || eyeLabel;
  const patientIdValue = patientInfo.patient_id || caseId || "-";
  const patientNameValue = patientName || "-";
  const mriNumberValue = mriNumber || "-";
  const bannerColor = prediction.toLowerCase() === "glaucoma" ? "#B91C1C" : "#15803D";
  const riskBadgeClass = RISK_STYLES[riskLevel] || RISK_STYLES.unknown;
  const vesselSourceSuffix = result.vessel_source === "estimated" ? " (estimated)" : "";
  const disclaimerText = getReportSectionText(result, "disclaimer");
  const cdrValue = Number(vesselFeatures.cdr);
  const cdrHighlight = cdrValue > 0.6 ? "red" : cdrValue > 0.5 ? "amber" : null;
  const groundingBadge = getGroundingBadge(report.grounding_score);
  const hasVesselData = Object.keys(vesselFeatures).length > 0;
  const pdfPath = result.pdf_path;

  const handleDownload = async () => {
    if (!pdfPath || downloadLoading) return;
    setDownloadError("");
    setDownloadLoading(true);
    try {
      await downloadGlaucomaReportPdf(pdfPath);
    } catch (error) {
      setDownloadError(error?.message || "Glaucoma report download failed.");
    } finally {
      setDownloadLoading(false);
    }
  };

  return (
    <div className="mb-6">
      <div className="mb-3 flex items-center gap-2">
        <span className="inline-flex items-center rounded-full bg-[#0A2342] px-3 py-1 text-xs font-bold text-white">
          {eyeSideValue}
        </span>
        <span className="text-sm text-[#5A6478]">Eye result</span>
      </div>

      {/* Classification banner */}
      <div
        className="mb-4 rounded border border-[#D0D7E2] border-l-4 bg-white px-5 py-4"
        style={{ borderLeftColor: bannerColor }}
      >
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-[#1A1A2E]">Glaucoma Screening Result</h2>
            <p className="mt-0.5 text-sm font-semibold" style={{ color: bannerColor }}>
              {prediction}
            </p>
            <p className="mt-1 text-sm text-[#5A6478]">Confidence: {confidence.toFixed(1)}%</p>
            <p className="mt-1 text-xs text-[#5A6478]">
              Patient: {patientNameValue} | MRI: {mriNumberValue} | UID: {patientIdValue} | Eye:{" "}
              {eyeSideValue}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span
              className="rounded px-3 py-1.5 text-sm font-bold text-white"
              style={{ backgroundColor: bannerColor }}
            >
              {prediction}
            </span>
            <span
              className={`rounded-full px-3 py-1 text-xs font-bold uppercase tracking-wide ${riskBadgeClass}`}
            >
              {formatLabel(riskLevel)} Risk
            </span>
          </div>
        </div>
      </div>

      {/* Classification probabilities */}
      <div className="mb-4 rounded border border-[#D0D7E2] bg-white px-5 py-4">
        <p className="mb-3 text-xs font-bold uppercase tracking-wide text-[#0A2342]">
          Classification Probabilities
        </p>
        {Object.entries(probabilities).map(([label, value]) => (
          <ProbabilityBar
            key={label}
            label={label}
            value={value}
            winner={label.toLowerCase() === prediction.toLowerCase()}
          />
        ))}
      </div>

      {/* Vessel biomarkers table — only when vessel_features is present */}
      {hasVesselData && (
        <div className="mb-4 rounded border border-[#D0D7E2] bg-white">
          <div className="flex items-center justify-between border-b border-[#E6ECF3] px-5 py-4">
            <div>
              <p className="text-xs font-bold uppercase tracking-wide text-[#0A2342]">
                Vessel Biomarkers{vesselSourceSuffix}
              </p>
              <p className="mt-1 text-xs text-[#5A6478]">
                Quantitative vessel summary from fundus analysis
              </p>
            </div>
            <span
              className={`rounded-full px-3 py-1 text-xs font-bold uppercase tracking-wide ${riskBadgeClass}`}
            >
              {formatLabel(riskLevel)}
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b border-[#E6ECF3] bg-[#F8FAFC]">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-semibold text-[#526074]">
                    Biomarker
                  </th>
                  <th className="px-4 py-2 text-right text-xs font-semibold text-[#526074]">
                    Value
                  </th>
                  <th className="px-4 py-2 text-right text-xs font-semibold text-[#526074]">
                    Normal Range
                  </th>
                </tr>
              </thead>
              <tbody>
                <FeatureRow
                  label="Cup-to-Disc Ratio (CDR)"
                  value={vesselFeatures.cdr}
                  normalRange="< 0.5"
                  highlight={cdrHighlight}
                  note={vesselFeatures.cdr_interpretation || null}
                />
                <FeatureRow
                  label="Vessel Density"
                  value={vesselFeatures.vessel_density}
                  normalRange="0.45–0.60"
                />
                <FeatureRow
                  label="Fractal Dimension"
                  value={vesselFeatures.fractal_dimension}
                  normalRange="1.65–1.75"
                />
                <FeatureRow label="Thin Vessel Ratio" value={vesselFeatures.thin_vessel_ratio} />
                <FeatureRow
                  label="Disc Region Density"
                  value={vesselFeatures.disc_region_density}
                />
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Clinical report — 4 sections with colored grounding badge */}
      <div className="mb-4 rounded border border-[#D0D7E2] bg-white px-5 py-4">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-xs font-bold uppercase tracking-wide text-[#0A2342]">
              Clinical Report
            </p>
            <p className="mt-1 text-xs text-[#5A6478]">
              Narrative grounded in extracted cup-disc and vascular biomarkers; classifier output is
              secondary context only.
            </p>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            <button
              type="button"
              onClick={handleDownload}
              disabled={downloadLoading || !pdfPath}
              className="rounded border border-[#0A2342] bg-white px-3 py-1.5 text-xs font-bold text-[#0A2342] transition hover:bg-[#F4F7FB] disabled:cursor-not-allowed disabled:opacity-60"
              title={pdfPath ? "Download glaucoma report PDF" : "PDF path is not available for this result"}
            >
              {downloadLoading ? "Downloading..." : "Download Report"}
            </button>
            {report.generation_time != null && (
              <span className="rounded-full bg-[#F4F6F9] px-3 py-1 text-xs font-semibold text-[#526074]">
                {Number(report.generation_time).toFixed(1)}s
              </span>
            )}
            <span className={`rounded-full px-3 py-1 text-xs font-bold ${groundingBadge.colorClass}`}>
              {groundingBadge.label}
            </span>
          </div>
        </div>

        {downloadError && (
          <div className="mb-3 rounded border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
            {downloadError}
          </div>
        )}

        <div className="grid gap-4">
          <div className="rounded border border-[#E6ECF3] bg-[#F8FAFC] p-4">
              <p className="text-sm font-bold text-[#1A1A2E]">Findings</p>
              <p className="mt-1 whitespace-pre-wrap text-sm leading-6 text-[#3A4258]">
              {getReportSectionText(result, "findings") || "Not available."}
              </p>
          </div>
          <div className="rounded border border-[#E6ECF3] bg-[#F8FAFC] p-4">
            <p className="text-sm font-bold text-[#1A1A2E]">Interpretation</p>
            <p className="mt-1 whitespace-pre-wrap text-sm leading-6 text-[#3A4258]">
              {getReportSectionText(result, "interpretation") || "Not available."}
            </p>
          </div>
          <div className="rounded border border-[#E6ECF3] bg-[#F8FAFC] p-4">
            <p className="text-sm font-bold text-[#1A1A2E]">Recommendation</p>
            <p className="mt-1 whitespace-pre-wrap text-sm leading-6 text-[#3A4258]">
              {getReportSectionText(result, "recommendation") || "Not available."}
            </p>
          </div>
          <div className="rounded border border-[#E6ECF3] bg-[#F8FAFC] p-4">
            <p className="text-sm font-bold text-[#1A1A2E]">Disclaimer</p>
            <p className="mt-1 whitespace-pre-wrap text-xs leading-6 text-[#9AA3B5]">
              {disclaimerText || "Not available."}
            </p>
          </div>
        </div>
      </div>

      {/* AI screening caution banner */}
      <div className="rounded border-l-4 border-[#F9A825] bg-[#FFF8E1] px-4 py-3">
        <p className="text-sm font-bold text-[#7A5000]">AI-assisted screening only</p>
        <p className="mt-1 text-sm leading-6 text-[#7A5000]">
          {disclaimerText} This model had 53% sensitivity on the held-out test set, so a normal
          result does not rule out glaucoma and ophthalmologist review remains essential.
        </p>
      </div>
    </div>
  );
}

export default function GlaucomaResult({
  result,
  loading,
  error,
  patientName,
  mriNumber,
  caseId,
  isDarkMode = false,
}) {
  const eyeResults =
    result && typeof result === "object"
      ? Object.entries(result).filter(([, eyeResult]) => eyeResult)
      : [];

  if (!eyeResults.length && !loading && !error) {
    return (
      <div
        className={`flex flex-1 items-center justify-center overflow-y-auto p-10 ${
          isDarkMode ? "bg-[#161D27]" : "bg-white"
        }`}
      >
        <div className="max-w-md text-center">
          <div
            className={`mx-auto flex h-20 w-20 items-center justify-center rounded-[24px] ${
              isDarkMode ? "bg-[#202938]" : "bg-[#F4F7FB]"
            }`}
          >
            <svg
              className={`h-10 w-10 ${isDarkMode ? "text-[#64748B]" : "text-[#B0BEC5]"}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1}
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1}
                d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
              />
            </svg>
          </div>
          <p
            className={`mt-5 text-lg font-semibold ${
              isDarkMode ? "text-[#F8FAFC]" : "text-[#1A1A2E]"
            }`}
          >
            No Glaucoma analysis results yet
          </p>
          <p
            className={`mt-2 text-center text-sm leading-6 ${
              isDarkMode ? "text-[#A8B6CC]" : "text-[#5A6478]"
            }`}
          >
            Upload one fundus image for OD, OS, or both, then run glaucoma analysis to review the
            prediction, vessel biomarkers, and grounded report here.
          </p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div
        className={`flex flex-1 items-center justify-center overflow-y-auto p-10 ${
          isDarkMode ? "bg-[#161D27]" : "bg-white"
        }`}
      >
        <div className="text-center">
          <svg
            className={`mx-auto h-10 w-10 animate-spin ${
              isDarkMode ? "text-[#D7E3F4]" : "text-[#0A2342]"
            }`}
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
          <p
            className={`mt-4 text-sm font-semibold ${
              isDarkMode ? "text-[#F8FAFC]" : "text-[#1A1A2E]"
            }`}
          >
            MedGemma is generating your clinical report.
          </p>
          <p className={`mt-1 text-xs ${isDarkMode ? "text-[#A8B6CC]" : "text-[#5A6478]"}`}>
            This takes 30–90 seconds.
          </p>
        </div>
      </div>
    );
  }

  if (error && !eyeResults.length) {
    return (
      <div
        className={`flex flex-1 items-center justify-center overflow-y-auto p-10 ${
          isDarkMode ? "bg-[#161D27]" : "bg-white"
        }`}
      >
        <div className="max-w-sm text-center">
          <p className="text-sm font-medium text-red-600">
            {typeof error === "string" ? error : "Report generation failed. Please try again."}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex-1 overflow-y-auto p-6 ${isDarkMode ? "bg-[#161D27]" : "bg-white"}`}>
      {eyeResults.map(([eye, eyeResult]) => (
        <GlaucomaResultCard
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
