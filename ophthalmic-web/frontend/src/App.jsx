import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import PatientHistory from "./pages/PatientHistory";
import ReviewQueue from "./pages/ReviewQueue";
import CaseReview from "./pages/CaseReview";
import AuditLog from "./pages/AuditLog";
import ValidationDashboard from "./pages/ValidationDashboard";
import PerformanceDashboard from "./pages/PerformanceDashboard";
import CalibrationAdmin from "./pages/CalibrationAdmin";
import ThresholdConfig from "./pages/ThresholdConfig";

function hasToken() {
  return !!(localStorage.getItem("token") || localStorage.getItem("access_token"));
}

function RequireAuth({ children }) {
  return hasToken() ? children : <Navigate to="/login" replace />;
}

function GuestOnly({ children }) {
  return hasToken() ? <Navigate to="/dashboard" replace /> : children;
}

function RequireAdmin({ children }) {
  return localStorage.getItem("role") === "admin" ? children : <Navigate to="/dashboard" replace />;
}

function RootRedirect() {
  return <Navigate to={hasToken() ? "/dashboard" : "/login"} replace />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/login"
          element={(
            <GuestOnly>
              <Login />
            </GuestOnly>
          )}
        />
        <Route
          path="/dashboard"
          element={(
            <RequireAuth>
              <Dashboard />
            </RequireAuth>
          )}
        />
        <Route
          path="/history"
          element={(
            <RequireAuth>
              <PatientHistory />
            </RequireAuth>
          )}
        />
        <Route path="/review-queue" element={<RequireAuth><ReviewQueue /></RequireAuth>} />
        <Route path="/cases/:caseId/review" element={<RequireAuth><CaseReview /></RequireAuth>} />
        <Route path="/audit-log" element={<RequireAuth><RequireAdmin><AuditLog /></RequireAdmin></RequireAuth>} />
        <Route path="/validation-dashboard" element={<RequireAuth><RequireAdmin><ValidationDashboard /></RequireAdmin></RequireAuth>} />
        <Route path="/performance-dashboard" element={<RequireAuth><RequireAdmin><PerformanceDashboard /></RequireAdmin></RequireAuth>} />
        <Route path="/admin/calibration" element={<RequireAuth><RequireAdmin><CalibrationAdmin /></RequireAdmin></RequireAuth>} />
        <Route path="/admin/thresholds" element={<RequireAuth><RequireAdmin><ThresholdConfig /></RequireAdmin></RequireAuth>} />
        <Route path="/" element={<RootRedirect />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
