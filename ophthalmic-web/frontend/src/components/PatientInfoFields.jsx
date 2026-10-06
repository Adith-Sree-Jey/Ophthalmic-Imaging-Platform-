import EyeSideSelector from "./EyeSideSelector";

export default function PatientInfoFields({
  patientName,
  setPatientName,
  mriNumber,
  setMriNumber,
  caseId,
  setCaseId,
  eyeSide,
  setEyeSide,
  eyeStats,
  isDarkMode = false,
}) {
  return (
    <div className="px-4 pt-4 pb-3 shrink-0">
      {[
        { label: "Patient Name", value: patientName, setter: setPatientName, ph: "e.g. Ravi Kumar" },
        { label: "MRI Number", value: mriNumber, setter: setMriNumber, ph: "e.g. MRI-2024-00891" },
        { label: "UID", value: caseId, setter: setCaseId, ph: "e.g. UID-2024-0001" },
      ].map(({ label, value, setter, ph }) => (
        <div className="mb-3" key={label}>
          <label className={`block text-xs font-semibold uppercase tracking-wide mb-1 ${isDarkMode ? "text-[#D7E3F4]" : "text-[#0A2342]"}`}>
            {label}
          </label>
          <input
            type="text"
            value={value}
            onChange={(e) => setter(e.target.value)}
            placeholder={ph}
            className={`w-full rounded px-3 py-2 text-sm transition-colors focus:outline-none ${
              isDarkMode
                ? "border border-[#334155] bg-[#1B2431] text-[#F8FAFC] placeholder:text-[#7C8CA5] focus:border-[#6B8AF7]"
                : "border border-[#D0D7E2] text-[#1A1A2E] focus:border-[#0A2342]"
            }`}
          />
        </div>
      ))}

      <EyeSideSelector value={eyeSide} onChange={setEyeSide} eyeStats={eyeStats} isDarkMode={isDarkMode} />
    </div>
  );
}
