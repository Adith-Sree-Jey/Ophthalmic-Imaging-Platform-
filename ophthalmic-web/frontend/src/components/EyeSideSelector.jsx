export default function EyeSideSelector({ value, onChange, eyeStats = {}, isDarkMode = false }) {
  const options = [
    { code: "OD", label: "OD", sublabel: "Right Eye" },
    { code: "OS", label: "OS", sublabel: "Left Eye" },
  ];

  return (
    <div className="mb-1">
      <label className={`block text-xs font-semibold uppercase tracking-wide mb-2 ${isDarkMode ? "text-[#D7E3F4]" : "text-[#0A2342]"}`}>
        Eye Side
      </label>
      <div className="grid grid-cols-2 gap-2">
        {options.map((option) => {
          const selected = value === option.code;
          const stats = eyeStats[option.code] || { count: 0, label: "Empty" };
          return (
            <button
              key={option.code}
              type="button"
              onClick={() => onChange(option.code)}
              className={`rounded-lg border px-3 py-2 text-left transition ${
                selected
                  ? isDarkMode
                    ? "border-[#6B8AF7] bg-[#1E293B] text-[#F8FAFC]"
                    : "border-[#0A2342] bg-[#EBF0FA] text-[#0A2342]"
                  : isDarkMode
                    ? "border-[#334155] bg-[#1B2431] text-[#A8B6CC] hover:border-[#6B8AF7]"
                    : "border-[#D0D7E2] bg-white text-[#5A6478] hover:border-[#0A2342]"
              }`}
            >
              <div className="text-sm font-bold">{option.label}</div>
              <div className="text-[11px] opacity-80">{option.sublabel}</div>
              <div className="text-[11px] mt-2 opacity-80">
                {stats.count} image{stats.count === 1 ? "" : "s"}
              </div>
              <div className="text-[11px] font-medium">{stats.label}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
