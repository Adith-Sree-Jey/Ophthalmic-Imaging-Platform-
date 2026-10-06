import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../api";
import platformLogo from "../assets/platform-logo.svg";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (localStorage.getItem("token") || localStorage.getItem("access_token")) {
      navigate("/dashboard", { replace: true });
    }
  }, [navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const data = await login(username, password);
      const token = data.access_token || data.token;
      localStorage.setItem("token", token);
      localStorage.setItem("access_token", token);
      localStorage.setItem("username", data.username || username);
      localStorage.setItem("role", data.role || "doctor");
      navigate("/dashboard");
    } catch (err) {
      setError(err.message || "Authentication failed. Please check your credentials.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden font-['Inter',system-ui,sans-serif]">
      {/* Left panel — navy branding */}
      <div className="hidden md:flex w-[40%] bg-[#0A2342] flex-col items-center justify-center px-10 relative">
        <img
          src={platformLogo}
          alt="Ophthalmic Imaging logo"
          className="mb-10 h-[22rem] w-[22rem] rounded-full bg-white object-contain p-3 shadow-2xl"
        />

        <h1 className="text-white text-5xl font-bold text-center leading-tight">
          Ophthalmic Imaging
        </h1>
        <p className="text-white/80 text-xl text-center mt-4 max-w-md leading-snug">
          Clinical AI Platform
        </p>
      </div>

      {/* Right panel — login form */}
      <div className="flex-1 bg-white flex flex-col items-center justify-center px-8 sm:px-16 relative">
        <div className="w-full max-w-sm">
          <h2 className="text-[#1A1A2E] text-xl font-bold mb-8">
            Sign In to Ophthalmic Imaging
          </h2>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-xs font-semibold text-[#0A2342] uppercase tracking-wide mb-1.5">
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                className="w-full border border-[#D0D7E2] rounded px-3 py-2.5 text-sm text-[#1A1A2E] focus:outline-none focus:border-[#0A2342] transition-colors"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-[#0A2342] uppercase tracking-wide mb-1.5">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                className="w-full border border-[#D0D7E2] rounded px-3 py-2.5 text-sm text-[#1A1A2E] focus:outline-none focus:border-[#0A2342] transition-colors"
                required
              />
            </div>

            <button
              type="submit"
              disabled={loading || !username || !password}
              className="w-full bg-[#0A2342] text-white py-2.5 rounded text-sm font-semibold hover:bg-[#1A4A7A] disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
            >
              {loading && (
                <svg className="animate-spin w-4 h-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              )}
              {loading ? "Signing in\u2026" : "Sign In"}
            </button>

            {error && (
              <p className="text-red-600 text-sm text-center mt-2">{error}</p>
            )}
          </form>
        </div>
      </div>
    </div>
  );
}
