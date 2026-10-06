/*
Sidebar map before this icon extraction:
- Sidebar component file: `src/components/Navbar.jsx`.
- Existing module icons were rendered with Lucide React (`Eye`, `CircleDot`, `ScanLine`) via the `icon` field in `buildModules(...)`.
- Active module state is derived from the `selectedModule` prop and compared with each module label in `SidebarRow`.
- Active nav colors use `bg-[rgba(255,255,255,0.08)]` on the row plus a left accent bar `bg-[#8AA4FF]`; inactive rows use muted white text classes.
- The sidebar's current module icon size was `18px` (`<Icon size={18} ... />`) in both expanded and collapsed modes.
*/
export { default as GlaucomaIcon } from "./GlaucomaIcon";
export { default as CataractIcon } from "./CataractIcon";
export { default as RetinaIcon } from "./RetinaIcon";
