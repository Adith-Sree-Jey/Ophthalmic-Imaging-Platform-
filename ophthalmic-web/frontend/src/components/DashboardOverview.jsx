/*
Frontend structure map before this redesign:
- `src/App.jsx` provides routing for `/dashboard`, `/history`, and `/login`; authenticated workbench content is rendered by `src/pages/Dashboard.jsx`.
- `src/pages/Dashboard.jsx` currently owns the main two-column workbench layout, module switching state, and all per-module result/upload state.
- `src/components/Navbar.jsx` currently renders the dark sidebar shell and module/history navigation.
- Styling is handled with Tailwind CSS utility classes plus a small global stylesheet in `src/index.css`.
- No charting library existed originally; the dashboard redesign adds `recharts` for overview charts.
- Existing backend-exposed data available to the frontend included `/patients`, `/history/:mri_number`, `/longitudinal/:mri_number/:eye_side`, and `/workspace-status`; the redesign also uses new stats endpoints for overview and module activity.
*/

import {
  ArrowUpRight,
  CircleAlert,
  FileText,
  Grid2x2,
  MoveRight,
  Users,
} from "lucide-react";
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";

const PERIOD_OPTIONS = [
  { value: "1year", label: "1 Year" },
  { value: "6months", label: "6 Months" },
  { value: "3months", label: "3 Months" },
  { value: "1month", label: "1 Month" },
];

const MODULE_META = {
  glaucoma: { label: "Glaucoma", color: "#D84C4C", badge: "bg-[#FDEAEA] text-[#B42318]" },
  cataract: { label: "Cataract", color: "#377DFF", badge: "bg-[#EAF1FF] text-[#1747B7]" },
  retina_segmentation: { label: "Retina", color: "#00A389", badge: "bg-[#E7FBF6] text-[#007A68]" },
};

function formatModuleLabel(moduleKey) {
  return MODULE_META[moduleKey]?.label || moduleKey;
}

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  const normalized = value <= 1 ? value * 100 : value;
  return `${normalized.toFixed(1)}%`;
}

function formatTimeAgo(value) {
  if (!value) return "-";
  const utcValue = value.endsWith("Z") || value.includes("+") ? value : value + "Z";
  const date = new Date(utcValue);
  if (Number.isNaN(date.getTime())) return "-";
  const diffMs = Date.now() - date.getTime();
  const diffMinutes = Math.max(1, Math.floor(diffMs / 60000));
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  return `${Math.floor(diffHours / 24)}d ago`;
}

function StatCard({ icon: Icon, title, value, subtitle, tone = "default", isDarkMode = false }) {
  const iconTone =
    tone === "danger"
      ? isDarkMode
        ? "bg-[#3B2228] text-[#FF7B7B]"
        : "bg-[#FDEAEA] text-[#D84C4C]"
      : isDarkMode
        ? "bg-[#1D2A40] text-[#7EA8FF]"
        : "bg-[#EEF4FF] text-[#2553D6]";
  return (
    <div
      className={`rounded-[24px] border p-5 shadow-[0_12px_30px_rgba(15,23,42,0.05)] ${
        isDarkMode ? "border-[#2A3444] bg-[#161F2B]" : "border-[#E4EAF3] bg-white"
      }`}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className={`text-sm font-medium ${isDarkMode ? "text-[#9FAEC3]" : "text-[#7A879A]"}`}>{title}</p>
          <p
            className={`mt-4 text-[32px] font-semibold leading-none tracking-[-0.03em] ${
              isDarkMode ? "text-white" : "text-[#172033]"
            }`}
          >
            {value}
          </p>
          <p className={`mt-3 text-sm ${isDarkMode ? "text-[#7F8EA3]" : "text-[#8B97AB]"}`}>{subtitle}</p>
        </div>
        <div className={`inline-flex h-11 w-11 items-center justify-center rounded-2xl ${iconTone}`}>
          <Icon size={20} />
        </div>
      </div>
      <div className={`mt-4 flex items-center gap-2 text-xs font-medium ${isDarkMode ? "text-[#7CCF8A]" : "text-[#67A864]"}`}>
        <ArrowUpRight size={14} />
        Stable overview
      </div>
    </div>
  );
}

function EmptyList({ message, isDarkMode = false }) {
  return (
    <div
      className={`flex min-h-[180px] items-center justify-center rounded-[20px] border border-dashed p-6 text-center text-sm ${
        isDarkMode
          ? "border-[#314052] bg-[#111923] text-[#7F8EA3]"
          : "border-[#D8E0EC] bg-[#F8FAFD] text-[#8B97AB]"
      }`}
    >
      {message}
    </div>
  );
}

export default function DashboardOverview({
  stats,
  activityData,
  activityPeriod,
  onPeriodChange,
  loading,
  error,
  searchQuery,
  isDarkMode = false,
  onOpenHistory,
}) {
  const donutData = [
    {
      name: "Glaucoma",
      key: "glaucoma",
      value: activityData.reduce((sum, item) => sum + (item.glaucoma || 0), 0),
      color: MODULE_META.glaucoma.color,
    },
    {
      name: "Cataract",
      key: "cataract",
      value: activityData.reduce((sum, item) => sum + (item.cataract || 0), 0),
      color: MODULE_META.cataract.color,
    },
    {
      name: "Retina",
      key: "retina_segmentation",
      value: activityData.reduce((sum, item) => sum + (item.retina || 0), 0),
      color: MODULE_META.retina_segmentation.color,
    },
  ].filter((item) => item.value > 0);

  const normalizedQuery = searchQuery.trim().toLowerCase();
  const todayCases = (stats.today_cases || []).filter((item) => {
    if (!normalizedQuery) return true;
    return `${item.patient_name} ${item.mri_number} ${item.result}`.toLowerCase().includes(normalizedQuery);
  });
  const latestReports = (stats.latest_reports || []).filter((item) => {
    if (!normalizedQuery) return true;
    return `${item.patient_name} ${item.mri_number} ${item.result}`.toLowerCase().includes(normalizedQuery);
  });

  if (loading) {
    return (
      <div
        className={`flex h-full items-center justify-center rounded-[28px] border ${
          isDarkMode ? "border-[#2A3444] bg-[#121A24]" : "border-[#E2E8F0] bg-white"
        }`}
      >
        <div className="text-center">
          <div className="mx-auto h-10 w-10 animate-spin rounded-full border-4 border-[#D7E4FF] border-t-[#2553D6]" />
          <p className={`mt-4 text-sm font-medium ${isDarkMode ? "text-[#9FAEC3]" : "text-[#526074]"}`}>Loading dashboard overview...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center rounded-[28px] border border-[#F1C7C7] bg-[#FFF5F5] p-8">
        <div className="max-w-md text-center">
          <p className="text-base font-semibold text-[#9F1C1C]">Dashboard data could not be loaded</p>
          <p className="mt-2 text-sm text-[#B24141]">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`h-full overflow-y-auto rounded-[28px] border p-6 ${
        isDarkMode ? "border-[#2A3444] bg-[#121A24]" : "border-[#E2E8F0] bg-[#F5F8FC]"
      }`}
    >
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-4">
        <StatCard icon={Users} title="Total Patients" value={stats.total_patients || 0} subtitle="from history" isDarkMode={isDarkMode} />
        <StatCard icon={Grid2x2} title="This Week" value={stats.this_week || 0} subtitle="new cases" isDarkMode={isDarkMode} />
        <StatCard
          icon={CircleAlert}
          title="Needs Review"
          value={stats.needs_review || 0}
          subtitle="flagged cases"
          tone="danger"
          isDarkMode={isDarkMode}
        />
        <StatCard icon={FileText} title="Reports Gen." value={stats.reports_generated || 0} subtitle="total reports" isDarkMode={isDarkMode} />
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-[1.65fr_1fr]">
        <section
          className={`rounded-[24px] border p-5 shadow-[0_12px_30px_rgba(15,23,42,0.05)] ${
            isDarkMode ? "border-[#2A3444] bg-[#161F2B]" : "border-[#E4EAF3] bg-white"
          }`}
        >
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className={`text-lg font-semibold ${isDarkMode ? "text-white" : "text-[#172033]"}`}>Module Activity</p>
              <p className={`mt-1 text-sm ${isDarkMode ? "text-[#7F8EA3]" : "text-[#8B97AB]"}`}>Case volume by module across the selected time range</p>
            </div>
            <div className={`flex items-center rounded-full p-1 ${isDarkMode ? "bg-[#101823]" : "bg-[#F3F6FA]"}`}>
              {PERIOD_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => onPeriodChange?.(option.value)}
                  className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
                    activityPeriod === option.value
                      ? isDarkMode
                        ? "bg-[#243248] text-white shadow-sm"
                        : "bg-white text-[#172033] shadow-sm"
                      : isDarkMode
                        ? "text-[#7F8EA3] hover:text-white"
                        : "text-[#7A879A] hover:text-[#172033]"
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-6 h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={activityData}>
                <CartesianGrid vertical={false} stroke={isDarkMode ? "#243244" : "#EEF2F7"} />
                <XAxis dataKey="month" tickLine={false} axisLine={false} tick={{ fill: isDarkMode ? "#8FA0B6" : "#7A879A", fontSize: 12 }} />
                <YAxis tickLine={false} axisLine={false} tick={{ fill: isDarkMode ? "#8FA0B6" : "#7A879A", fontSize: 12 }} />
                <Tooltip
                  cursor={{ fill: isDarkMode ? "#1A2432" : "#F8FAFD" }}
                  contentStyle={{
                    borderRadius: 16,
                    borderColor: isDarkMode ? "#314052" : "#E4EAF3",
                    boxShadow: "0 12px 28px rgba(15,23,42,0.08)",
                    backgroundColor: isDarkMode ? "#111923" : "#FFFFFF",
                  }}
                  labelStyle={{ color: isDarkMode ? "#D8E2F0" : "#172033" }}
                />
                <Bar dataKey="glaucoma" stackId="a" fill={MODULE_META.glaucoma.color} radius={[6, 6, 0, 0]} />
                <Bar dataKey="cataract" stackId="a" fill={MODULE_META.cataract.color} radius={[6, 6, 0, 0]} />
                <Bar dataKey="retina" stackId="a" fill={MODULE_META.retina_segmentation.color} radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section
          className={`rounded-[24px] border p-5 shadow-[0_12px_30px_rgba(15,23,42,0.05)] ${
            isDarkMode ? "border-[#2A3444] bg-[#161F2B]" : "border-[#E4EAF3] bg-white"
          }`}
        >
          <p className={`text-lg font-semibold ${isDarkMode ? "text-white" : "text-[#172033]"}`}>Avg Diagnosis</p>
          <p className={`mt-1 text-sm ${isDarkMode ? "text-[#7F8EA3]" : "text-[#8B97AB]"}`}>Total case split by diagnostic module</p>
          <div className="mt-4 h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={donutData.length ? donutData : [{ name: "No Data", value: 1, color: "#DCE3EE" }]}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={70}
                  outerRadius={96}
                  paddingAngle={4}
                  strokeWidth={0}
                >
                  {(donutData.length ? donutData : [{ color: "#DCE3EE" }]).map((entry) => (
                    <Cell key={`${entry.name || "empty"}-${entry.color}`} fill={entry.color} />
                  ))}
                </Pie>
                <text x="50%" y="46%" textAnchor="middle" className={`${isDarkMode ? "fill-[#8FA0B6]" : "fill-[#8B97AB]"} text-[12px] font-medium`}>
                  Total Cases
                </text>
                <text x="50%" y="56%" textAnchor="middle" className={`${isDarkMode ? "fill-white" : "fill-[#172033]"} text-[28px] font-semibold`}>
                  {stats.total_cases || 0}
                </text>
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 space-y-3">
            {(donutData.length ? donutData : [
              { name: "Glaucoma", key: "glaucoma", value: 0, color: MODULE_META.glaucoma.color },
              { name: "Cataract", key: "cataract", value: 0, color: MODULE_META.cataract.color },
              { name: "Retina", key: "retina_segmentation", value: 0, color: MODULE_META.retina_segmentation.color },
            ]).map((item) => (
              <div key={item.key} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-3">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                  <span className={isDarkMode ? "text-[#C7D2E1]" : "text-[#526074]"}>{item.name}</span>
                </div>
                <span className={`font-semibold ${isDarkMode ? "text-white" : "text-[#172033]"}`}>{item.value}</span>
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6">
        <section
          className={`rounded-[24px] border p-5 shadow-[0_12px_30px_rgba(15,23,42,0.05)] ${
            isDarkMode ? "border-[#2A3444] bg-[#161F2B]" : "border-[#E4EAF3] bg-white"
          }`}
        >
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className={`text-lg font-semibold ${isDarkMode ? "text-white" : "text-[#172033]"}`}>Today's Cases</p>
              <p className={`mt-1 text-sm ${isDarkMode ? "text-[#7F8EA3]" : "text-[#8B97AB]"}`}>Recent diagnostic activity captured today</p>
            </div>
            <button
              type="button"
              onClick={() => onOpenHistory?.()}
              className={`inline-flex items-center gap-2 text-sm font-medium ${isDarkMode ? "text-[#8FB2FF]" : "text-[#2553D6]"}`}
            >
              View All
              <MoveRight size={14} />
            </button>
          </div>

          <div className="mt-5">
            {!todayCases.length ? (
              <EmptyList message="No cases run today" isDarkMode={isDarkMode} />
            ) : (
              <div className="space-y-3">
                {todayCases.slice(0, 5).map((item) => (
                  <button
                    key={`today-${item.visit_id}`}
                    type="button"
                    onClick={() => onOpenHistory?.(item.mri_number)}
                    className={`flex w-full items-center justify-between rounded-[18px] border px-4 py-3 text-left transition-colors ${
                      isDarkMode
                        ? "border-[#243244] bg-[#111923] hover:border-[#314052] hover:bg-[#162131]"
                        : "border-[#EDF1F6] bg-[#FBFCFE] hover:border-[#DCE4EE] hover:bg-white"
                    }`}
                  >
                    <div>
                      <p className={`text-sm font-semibold ${isDarkMode ? "text-white" : "text-[#172033]"}`}>{item.patient_name}</p>
                      <div className={`mt-1 flex items-center gap-2 text-xs ${isDarkMode ? "text-[#8FA0B6]" : "text-[#7A879A]"}`}>
                        <span className={`rounded-full px-2 py-1 font-medium ${MODULE_META[item.module]?.badge || "bg-[#EEF2F7] text-[#526074]"}`}>
                          {formatModuleLabel(item.module)}
                        </span>
                        <span>{item.eye_side || "UNKNOWN"}</span>
                        <span>{item.result || "-"}</span>
                      </div>
                    </div>
                    <p className={`text-xs font-medium ${isDarkMode ? "text-[#7F8EA3]" : "text-[#8B97AB]"}`}>{formatTimeAgo(item.timestamp)}</p>
                  </button>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
