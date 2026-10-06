/*
Sidebar map before this icon extraction:
- Sidebar component file: `src/components/Navbar.jsx`.
- Existing module icons were rendered with Lucide React (`Eye`, `CircleDot`, `ScanLine`) via the `icon` field in `buildModules(...)`.
- Active module state is derived from the `selectedModule` prop and compared with each module label in `SidebarRow`.
- Active nav colors use `bg-[rgba(255,255,255,0.08)]` on the row plus a left accent bar `bg-[#8AA4FF]`; inactive rows use muted white text classes.
- The sidebar's current module icon size was `18px` (`<Icon size={18} ... />`) in both expanded and collapsed modes.
*/
export default function RetinaIcon({ size = 28, active = false }) {
  const artery = active ? "#e05050" : "#803030";
  const vein = active ? "#4060c0" : "#263060";
  const disc = active ? "#f0d890" : "#806840";
  const overlay = active ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.06)";

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 56 56"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Retina segmentation module"
    >
      <circle cx="28" cy="28" r="26" fill="#180e06" />
      <circle cx="28" cy="28" r="20" fill="none" stroke="#1e1208" strokeWidth="5" opacity="0.4" />
      <ellipse cx="16" cy="24" rx="6" ry="7" fill="#e8c870" opacity="0.9" />
      <ellipse cx="16" cy="24" rx="3.5" ry="4.5" fill={disc} opacity="0.8" />
      <ellipse cx="16" cy="24" rx="1.8" ry="2.5" fill="#e8a840" opacity="0.9" />
      <path d="M19 20 Q26 14 34 10 Q42 7 50 5" stroke={artery} strokeWidth="1.6" strokeLinecap="round" />
      <path d="M34 10 Q40 8 46 9 Q50 10 53 8" stroke={artery} strokeWidth="1.1" strokeLinecap="round" />
      <path d="M42 7 Q48 5 52 4" stroke={artery} strokeWidth="0.8" strokeLinecap="round" />
      <path d="M19 28 Q26 34 34 40 Q42 46 50 50" stroke={artery} strokeWidth="1.6" strokeLinecap="round" />
      <path d="M34 40 Q40 44 46 45 Q50 46 53 49" stroke={artery} strokeWidth="1.1" strokeLinecap="round" />
      <path d="M42 46 Q48 48 52 50" stroke={artery} strokeWidth="0.8" strokeLinecap="round" />
      <path d="M16 18 Q14 12 13 6 Q12 2 10 0" stroke={artery} strokeWidth="1.1" strokeLinecap="round" />
      <path d="M13 6 Q10 3 7 2" stroke={artery} strokeWidth="0.7" strokeLinecap="round" />
      <path d="M16 30 Q14 36 12 42 Q10 48 8 52" stroke={artery} strokeWidth="1.1" strokeLinecap="round" />
      <path d="M12 42 Q9 46 6 48" stroke={artery} strokeWidth="0.7" strokeLinecap="round" />
      <path d="M18 21 Q25 16 33 12 Q41 8 48 7" stroke={vein} strokeWidth="1.3" strokeLinecap="round" />
      <path d="M33 12 Q39 10 45 11 Q49 12 52 10" stroke={vein} strokeWidth="0.9" strokeLinecap="round" />
      <path d="M18 27 Q25 33 33 38 Q41 44 48 48" stroke={vein} strokeWidth="1.3" strokeLinecap="round" />
      <path d="M33 38 Q39 42 45 43 Q49 44 52 47" stroke={vein} strokeWidth="0.9" strokeLinecap="round" />
      <path d="M15 19 Q13 13 12 7 Q11 3 9 1" stroke={vein} strokeWidth="0.9" strokeLinecap="round" />
      <path d="M15 29 Q13 35 11 41 Q9 47 7 51" stroke={vein} strokeWidth="0.9" strokeLinecap="round" />
      <path d="M19 20 Q26 14 34 10 Q42 7 50 5" stroke={overlay} strokeWidth="4" strokeLinecap="round" />
      <path d="M19 28 Q26 34 34 40 Q42 46 50 50" stroke={overlay} strokeWidth="4" strokeLinecap="round" />
      <circle cx="32" cy="28" r="8" fill="none" stroke="#3a2a10" strokeWidth="3" opacity="0.35" />
      <circle cx="32" cy="28" r="2.5" fill="#100a04" />
      <circle cx="31" cy="27" r="1" fill="#a08040" opacity="0.4" />
      <circle cx="28" cy="28" r="26" fill="none" stroke="#0a1220" strokeWidth="2" />
    </svg>
  );
}
