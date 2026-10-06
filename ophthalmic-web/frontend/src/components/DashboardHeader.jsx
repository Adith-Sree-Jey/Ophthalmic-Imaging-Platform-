import { useEffect, useRef, useState } from "react";
import { ChevronDown, LogOut, Moon, Sun } from "lucide-react";

function getInitials(username) {
  if (!username) return "AE";
  const parts = username.split(/\s+/).filter(Boolean).slice(0, 2);
  if (!parts.length) return "AE";
  return parts.map((part) => part[0]?.toUpperCase() || "").join("");
}

export default function DashboardHeader({
  title,
  username,
  isDarkMode = false,
  onToggleTheme,
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const profileMenuRef = useRef(null);

  useEffect(() => {
    function handlePointerDown(event) {
      if (!profileMenuRef.current?.contains(event.target)) {
        setMenuOpen(false);
      }
    }

    function handleEscape(event) {
      if (event.key === "Escape") {
        setMenuOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleEscape);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleEscape);
    };
  }, []);

  function handleSignOut() {
    localStorage.removeItem("token");
    localStorage.removeItem("access_token");
    localStorage.removeItem("username");
    localStorage.removeItem("role");
    window.location.replace("/login");
  }

  return (
    <header className={`h-[60px] shrink-0 border-b px-6 ${isDarkMode ? "border-[#2A3140] bg-[#171C26]" : "border-[#E7ECF3] bg-white"}`}>
      <div className="flex h-full items-center gap-4">
        <div className="min-w-[160px]">
          <h1 className={`text-[24px] font-semibold tracking-[-0.02em] ${isDarkMode ? "text-white" : "text-[#172033]"}`}>{title}</h1>
        </div>

        <div className="ml-auto flex items-center gap-4">
          <button
            type="button"
            onClick={onToggleTheme}
            aria-label="Toggle theme"
            title={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
            className={`relative inline-flex h-[48px] w-[104px] items-center rounded-full border p-1 transition-colors ${isDarkMode ? "border-[#2C3443] bg-[#12161E]" : "border-[#E6E8EC] bg-[#F8F8F8]"}`}
          >
            <span
              className={`absolute top-1 h-[38px] w-[46px] rounded-full shadow-sm transition-all duration-300 ${isDarkMode ? "left-[54px] bg-[#3A3A3A]" : "left-1 bg-white"}`}
            />
            <span className="relative z-10 flex flex-1 items-center justify-center">
              <Sun size={20} className={isDarkMode ? "text-[#5B6472]" : "text-[#F5A623]"} />
            </span>
            <span className="relative z-10 flex flex-1 items-center justify-center">
              <Moon size={20} className={isDarkMode ? "text-[#F5A623]" : "text-[#9AA3B2]"} />
            </span>
          </button>

          <div className="relative" ref={profileMenuRef}>
            <button
              type="button"
              onClick={() => setMenuOpen((current) => !current)}
              className={`flex items-center gap-3 rounded-2xl border px-2.5 py-1.5 text-left transition-colors ${
                isDarkMode
                  ? "border-[#2C3443] bg-[#171C26] hover:bg-[#1D2430]"
                  : "border-[#E6E8EC] bg-white hover:bg-[#F7F9FC]"
              }`}
              aria-haspopup="menu"
              aria-expanded={menuOpen}
              title="Profile menu"
            >
              <div className={`flex h-10 w-10 items-center justify-center rounded-full text-sm font-semibold text-white ${isDarkMode ? "bg-[#243248]" : "bg-[#0A2342]"}`}>
                {getInitials(username)}
              </div>
              <div className="hidden text-right md:block">
                <p className={`text-sm font-semibold ${isDarkMode ? "text-white" : "text-[#172033]"}`}>{username || "Clinician"}</p>
                <p className={`text-xs ${isDarkMode ? "text-[#7F8A9E]" : "text-[#8B97AB]"}`}>Clinical AI Workbench</p>
              </div>
              <ChevronDown
                size={16}
                className={`transition-transform ${menuOpen ? "rotate-180" : ""} ${isDarkMode ? "text-[#7F8A9E]" : "text-[#8B97AB]"}`}
              />
            </button>

            {menuOpen && (
              <div
                className={`absolute right-0 top-[calc(100%+10px)] z-20 min-w-[180px] rounded-2xl border p-2 shadow-[0_18px_48px_rgba(15,23,42,0.22)] ${
                  isDarkMode ? "border-[#2C3443] bg-[#171C26]" : "border-[#E7ECF3] bg-white"
                }`}
                role="menu"
              >
                <button
                  type="button"
                  onClick={handleSignOut}
                  className={`flex w-full items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium transition-colors ${
                    isDarkMode
                      ? "text-[#F3F6FB] hover:bg-[rgba(239,68,68,0.14)] hover:text-[#FCA5A5]"
                      : "text-[#172033] hover:bg-[#FEF2F2] hover:text-[#B91C1C]"
                  }`}
                  role="menuitem"
                >
                  <LogOut size={16} />
                  <span>Sign Out</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
