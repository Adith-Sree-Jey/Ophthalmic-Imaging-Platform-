/*
Sidebar map before this icon extraction:
- Sidebar component file: `src/components/Navbar.jsx`.
- Existing module icons were rendered with Lucide React (`Eye`, `CircleDot`, `ScanLine`) via the `icon` field in `buildModules(...)`.
- Active module state is derived from the `selectedModule` prop and compared with each module label in `SidebarRow`.
- Active nav colors use `bg-[rgba(255,255,255,0.08)]` on the row plus a left accent bar `bg-[#8AA4FF]`; inactive rows use muted white text classes.
- The sidebar's current module icon size was `18px` (`<Icon size={18} ... />`) in both expanded and collapsed modes.
*/
export default function GlaucomaIcon({ size = 28, active = false }) {
  const primary = active ? "#a8c4f0" : "#4a6fa5";
  const accent = active ? "#e8955b" : "#6a4a35";
  const blue = active ? "#5b8fe8" : "#3a5a80";
  const bg = "#0f1729";

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 56 56"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Glaucoma module"
    >
      <circle cx="28" cy="28" r="22" fill="#1a2a4a" stroke={primary} strokeWidth="0.8" />
      <circle
        cx="28"
        cy="28"
        r="22"
        fill="none"
        stroke={accent}
        strokeWidth="4"
        strokeDasharray="90 48"
        strokeDashoffset="0"
        strokeLinecap="round"
      />
      <circle cx="28" cy="28" r="13" fill={bg} stroke={blue} strokeWidth="0.8" />
      <line x1="28" y1="16" x2="28" y2="40" stroke={blue} strokeWidth="0.5" opacity="0.7" />
      <line x1="16" y1="28" x2="40" y2="28" stroke={blue} strokeWidth="0.5" opacity="0.7" />
      <circle cx="26" cy="27" r="2" fill={accent} />
      <circle cx="30" cy="29" r="1.2" fill={primary} />
      <circle
        cx="28"
        cy="28"
        r="26"
        fill="none"
        stroke={primary}
        strokeWidth="0.5"
        strokeDasharray="2 3"
        opacity="0.5"
      />
    </svg>
  );
}
