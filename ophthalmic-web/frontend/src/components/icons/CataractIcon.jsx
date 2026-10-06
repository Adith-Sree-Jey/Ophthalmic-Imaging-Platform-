/*
Sidebar map before this icon extraction:
- Sidebar component file: `src/components/Navbar.jsx`.
- Existing module icons were rendered with Lucide React (`Eye`, `CircleDot`, `ScanLine`) via the `icon` field in `buildModules(...)`.
- Active module state is derived from the `selectedModule` prop and compared with each module label in `SidebarRow`.
- Active nav colors use `bg-[rgba(255,255,255,0.08)]` on the row plus a left accent bar `bg-[#8AA4FF]`; inactive rows use muted white text classes.
- The sidebar's current module icon size was `18px` (`<Icon size={18} ... />`) in both expanded and collapsed modes.
*/
export default function CataractIcon({ size = 28, active = false }) {
  const primary = active ? "#a8c4f0" : "#4a6fa5";
  const blue = active ? "#5b8fe8" : "#3a5a80";
  const accent = active ? "#e8955b" : "#6a4a35";
  const dim = active ? "#7aaad4" : "#3a5a70";
  const bgMid = "#0f1e38";

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 56 56"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Cataract module"
    >
      <ellipse cx="28" cy="30" rx="24" ry="16" fill="#1a2540" stroke="#2a3a5a" strokeWidth="0.6" />
      <path d="M10 22 Q18 10 28 8 Q38 10 46 22" fill={bgMid} stroke={blue} strokeWidth="1.2" strokeLinecap="round" />
      <path
        d="M12 24 Q20 14 28 12 Q36 14 44 24"
        fill="none"
        stroke={primary}
        strokeWidth="0.5"
        strokeDasharray="2 2"
        opacity="0.6"
      />
      <path d="M10 26 Q14 18 20 16 L21 28 Q16 28 10 30 Z" fill="#1e3560" stroke="#3a5a90" strokeWidth="0.6" />
      <path d="M46 26 Q42 18 36 16 L35 28 Q40 28 46 30 Z" fill="#1e3560" stroke="#3a5a90" strokeWidth="0.6" />
      <line x1="12" y1="25" x2="20" y2="20" stroke="#2a4a78" strokeWidth="0.5" />
      <line x1="12" y1="28" x2="20" y2="24" stroke="#2a4a78" strokeWidth="0.5" />
      <line x1="44" y1="25" x2="36" y2="20" stroke="#2a4a78" strokeWidth="0.5" />
      <line x1="44" y1="28" x2="36" y2="24" stroke="#2a4a78" strokeWidth="0.5" />
      <ellipse cx="28" cy="26" rx="10" ry="8" fill="#060c18" />
      <ellipse cx="28" cy="34" rx="13" ry="9" fill="none" stroke={dim} strokeWidth="0.9" />
      <ellipse cx="28" cy="34" rx="9" ry="6" fill={bgMid} stroke={blue} strokeWidth="0.6" />
      <ellipse cx="28" cy="34" rx="7" ry="4.5" fill="none" stroke={primary} strokeWidth="0.5" opacity="0.5" />
      <ellipse cx="28" cy="34" rx="4.5" ry="3" fill="none" stroke="#c4d8f8" strokeWidth="0.5" opacity="0.6" />
      <ellipse cx="28" cy="34" rx="5.5" ry="3.5" fill="#2a1a08" opacity="0.6" />
      <ellipse cx="28" cy="34" rx="3" ry="2" fill="#3a2208" opacity="0.7" />
      <line x1="18" y1="26" x2="20" y2="30" stroke="#3a5a80" strokeWidth="0.5" />
      <line x1="21" y1="24" x2="22" y2="29" stroke="#3a5a80" strokeWidth="0.5" />
      <line x1="35" y1="24" x2="34" y2="29" stroke="#3a5a80" strokeWidth="0.5" />
      <line x1="38" y1="26" x2="36" y2="30" stroke="#3a5a80" strokeWidth="0.5" />
      <path
        d="M16 40 Q28 44 40 40"
        fill="none"
        stroke={primary}
        strokeWidth="0.6"
        strokeDasharray="2 2"
        opacity="0.5"
      />
      <ellipse cx="22" cy="14" rx="4" ry="2" fill={primary} opacity="0.1" transform="rotate(-20 22 14)" />
      <circle cx="28" cy="34" r="1.8" fill={accent} opacity="0.9" />
    </svg>
  );
}
