import { NavLink } from "react-router-dom";
import {
  ClipboardCheck,
  ChevronLeft,
  ChevronRight,
  LayoutGrid,
  ScrollText,
} from "lucide-react";
import platformLogo from "../assets/platform-logo.svg";
import { CataractIcon, GlaucomaIcon, RetinaIcon } from "./icons";

const HOME_VIEW = "Platform Home";

function buildModules(moduleStatus) {
  const glaucomaAvailable = Boolean(moduleStatus?.glaucoma?.available);
  const glaucomaStatus = moduleStatus?.glaucoma?.status || (glaucomaAvailable ? "Checking model" : "Backend offline");

  return [
    { label: "Cataract", status: moduleStatus?.cataract?.status || "Checking model", icon: CataractIcon, disabled: false, kind: "module" },
    {
      label: "Glaucoma",
      status: glaucomaStatus,
      icon: GlaucomaIcon,
      disabled: !glaucomaAvailable,
      kind: "module",
    },
    { label: "Retina Segmentation", status: moduleStatus?.retina?.status || "Checking model", icon: RetinaIcon, disabled: false, kind: "module" },
  ];
}

function buildOperationsStats(modules) {
  const onlineCount = modules.filter((module) => !module.disabled).length;
  return {
    onlineCount,
    totalCount: modules.length,
    badgeLabel: onlineCount > 0 ? "LIVE" : "DOWN",
  };
}

function SidebarToggleButton({ collapsed, onClick }) {
  const Icon = collapsed ? ChevronRight : ChevronLeft;

  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex h-7 w-7 items-center justify-center rounded-full text-[rgba(255,255,255,0.58)] transition-colors hover:bg-[rgba(255,255,255,0.06)] hover:text-white"
      aria-label="Toggle sidebar"
      title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
    >
      <Icon size={16} strokeWidth={2} />
    </button>
  );
}

function SidebarRow({
  collapsed,
  active,
  disabled,
  icon: Icon,
  iconKind = "module",
  title,
  subtitle,
  onClick,
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      disabled={disabled}
      className={`group relative flex w-full items-center rounded-[22px] transition-all duration-200 ${
        collapsed ? "justify-center px-0 py-2.5" : "gap-3 px-4 py-3 text-left"
      } ${
        disabled
          ? "cursor-not-allowed text-[rgba(255,255,255,0.28)]"
          : active
            ? "bg-[#232A37] text-white shadow-[0_14px_28px_rgba(5,10,18,0.28)]"
            : "text-[rgba(255,255,255,0.78)] hover:bg-[rgba(255,255,255,0.04)] hover:text-white"
      }`}
    >
      {active && <span className="absolute left-0 top-2 h-[calc(100%-16px)] w-[3px] rounded-full bg-[#8197FF]" />}
      <span
        className={`flex shrink-0 items-center justify-center rounded-2xl ${
          collapsed ? "h-9 w-9" : "h-10 w-10"
        } ${
          active && iconKind === "lucide" ? "bg-[#1A2130]" : ""
        } ${disabled ? "opacity-45" : "opacity-100"}`}
      >
        {iconKind === "lucide" ? (
          <Icon
            size={collapsed ? 17 : 19}
            strokeWidth={1.9}
            className={active ? "text-[#9AB2FF]" : "text-[rgba(255,255,255,0.54)]"}
          />
        ) : (
          <Icon size={22} active={active && !disabled} />
        )}
      </span>
      {!collapsed && (
        <span className="min-w-0">
          <span className={`block text-[13px] leading-tight ${active ? "font-semibold text-white" : "font-medium"}`}>{title}</span>
          <span className={`mt-1 block text-[10px] leading-snug ${active ? "text-[rgba(255,255,255,0.62)]" : "text-[rgba(255,255,255,0.4)]"}`}>
            {subtitle}
          </span>
        </span>
      )}
    </button>
  );
}

export default function Navbar({
  variant = "topbar",
  collapsed = false,
  onToggleSidebar,
  selectedView,
  onSelectView,
  moduleStatus,
}) {
  const modules = buildModules(moduleStatus);
  const operations = buildOperationsStats(modules);
  const isAdmin = localStorage.getItem("role") === "admin";
  const reviewLinks = [
    { title: "Review Queue", href: "/review-queue", icon: ClipboardCheck, subtitle: "Clinician review queue" },
    ...(isAdmin
      ? [{ title: "Audit Log", href: "/audit-log", icon: ScrollText, subtitle: "Case activity history" }]
      : []),
  ];
  const topbarLinkClassName = ({ isActive }) =>
    [
      "text-sm px-3 py-1 rounded-full border transition-colors whitespace-nowrap",
      isActive
        ? "border-white/40 bg-white/12 text-white"
        : "border-white/20 text-white/70 hover:bg-white/10 hover:text-white",
    ].join(" ");

  if (variant === "sidebar") {
    return (
      <aside
        className={`shrink-0 border-r border-[rgba(255,255,255,0.06)] bg-[#0F1623] text-white flex h-screen min-h-0 flex-col transition-all duration-300 ${
          collapsed ? "w-[64px]" : "w-[220px]"
        }`}
        style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}
      >
        <div className="min-h-0 flex-1 overflow-hidden flex flex-col">
          <div className={`${collapsed ? "px-3 py-3" : "px-4 py-3"}`}>
            <div className={`flex items-start ${collapsed ? "flex-col items-center gap-3" : "justify-between gap-3"}`}>
              <div className={`flex items-center ${collapsed ? "justify-center" : "gap-3"}`}>
                <img
                  src={platformLogo}
                  alt="Ophthalmic Imaging logo"
                  className="h-8 w-8 shrink-0 rounded-xl bg-white object-contain p-1"
                  title="Ophthalmic Imaging"
                />
                {!collapsed && (
                  <div className="min-w-0">
                    <p className="truncate text-[14px] font-semibold leading-tight text-white">Ophthalmic Imaging</p>
                    <p className="mt-0.5 text-[11px] text-[rgba(255,255,255,0.5)]">Clinical AI Platform</p>
                  </div>
                )}
              </div>
              <SidebarToggleButton collapsed={collapsed} onClick={onToggleSidebar} />
            </div>
          </div>

          {!collapsed && (
            <div className="px-4 pb-2">
              <div className="rounded-[24px] border border-[rgba(255,255,255,0.06)] bg-[#151D2B] px-5 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.02)]">
                <p className="text-[11px] uppercase tracking-[0.26em] text-[#7D8BAA]">Operations</p>
                <div className="mt-4 flex items-end justify-between gap-3">
                  <div>
                    <p className="text-[18px] font-semibold leading-none text-white">
                      {operations.onlineCount}/{operations.totalCount}
                    </p>
                    <p className="mt-2 text-[11px] text-[rgba(255,255,255,0.68)]">Modules online</p>
                  </div>
                  <span className="inline-flex rounded-full border border-[#395A8C] bg-[#173052] px-3.5 py-1 text-[11px] font-semibold tracking-[0.16em] text-[#C8D8FF]">
                    {operations.badgeLabel}
                  </span>
                </div>
              </div>
            </div>
          )}

          <div className={collapsed ? "px-2 py-1.5" : "px-3 py-1.5"}>
            <SidebarRow
              collapsed={collapsed}
              active={selectedView === HOME_VIEW}
              disabled={false}
              icon={LayoutGrid}
              iconKind="lucide"
              title="Dashboard"
              subtitle="Overview and analytics"
              onClick={() => onSelectView?.(HOME_VIEW)}
            />
          </div>

          <div className={collapsed ? "flex-1 px-2 py-1.5" : "flex-1 px-3 py-1.5"}>
            <div className="flex h-full flex-col">
              {!collapsed && <p className="px-3 pb-1.5 text-[10px] uppercase tracking-[0.22em] text-[#6F7C99]">Modules</p>}
              <div className="flex flex-1 flex-col justify-evenly">
                {modules.map((module) => (
                  <SidebarRow
                    key={module.label}
                    collapsed={collapsed}
                    active={module.label === selectedView}
                    disabled={module.disabled}
                    icon={module.icon}
                    title={module.label}
                    subtitle={module.status}
                    onClick={() => {
                      if (!module.disabled) onSelectView?.(module.label);
                    }}
                  />
                ))}
              </div>
            </div>
          </div>

          <div className={collapsed ? "px-2 py-1.5" : "px-3 py-1.5"}>
            {!collapsed && <p className="px-3 pb-1.5 text-[10px] uppercase tracking-[0.22em] text-[#6F7C99]">Review</p>}
            <div className="space-y-0.5">
              {reviewLinks.map((item) => (
                <SidebarRow
                  key={item.title}
                  collapsed={collapsed}
                  active={false}
                  disabled={false}
                  icon={item.icon}
                  iconKind="lucide"
                  title={item.title}
                  subtitle={item.subtitle}
                  onClick={() => window.location.assign(item.href)}
                />
              ))}
            </div>
          </div>
        </div>

      </aside>
    );
  }

  return (
    <nav className="w-full h-[52px] bg-[#0A2342] flex items-center gap-4 px-4 font-['Inter',system-ui,sans-serif] select-none shrink-0">
      <div className="flex items-center gap-3">
        <img
          src={platformLogo}
          alt="Ophthalmic Imaging logo"
          className="w-8 h-8 rounded object-contain bg-white shrink-0"
        />
        <span className="text-white font-bold text-sm whitespace-nowrap">Ophthalmic Imaging</span>
        <span className="text-white/30 text-sm hidden sm:inline">|</span>
        <span className="text-white/60 text-sm hidden sm:inline whitespace-nowrap">Clinical AI Workbench</span>
      </div>

      <div className="hidden md:flex items-center gap-2">
        <NavLink to="/history" className={topbarLinkClassName}>
          Patient History
        </NavLink>
        <NavLink to="/review-queue" className={topbarLinkClassName}>
          Review Queue
        </NavLink>
        {isAdmin && (
          <NavLink to="/audit-log" className={topbarLinkClassName}>
            Audit Log
          </NavLink>
        )}
      </div>
    </nav>
  );
}
