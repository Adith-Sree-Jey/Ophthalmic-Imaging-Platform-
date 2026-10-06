import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AuditLog from "./AuditLog";
import { exportAuditLog, getAuditLog } from "../api";

vi.mock("../components/Navbar", () => ({ default: () => <nav>Navigation</nav> }));
vi.mock("../api", () => ({
  exportAuditLog: vi.fn(),
  getAuditLog: vi.fn(),
}));

describe("AuditLog direct component handling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.setItem("role", "doctor");
    getAuditLog.mockResolvedValue({ items: [], total: 0 });
  });

  it("catches a 403 export rejection and renders the error without an unhandled rejection", async () => {
    const unhandled = vi.fn((event) => event.preventDefault());
    window.addEventListener("unhandledrejection", unhandled);
    exportAuditLog.mockRejectedValue(new Error("Admin access required."));

    render(<AuditLog />, { wrapper: MemoryRouter });
    expect(await screen.findByRole("heading", { name: "Audit Log" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Export CSV" }));

    expect(await screen.findByText("Admin access required.")).toBeInTheDocument();
    expect(unhandled).not.toHaveBeenCalled();
    window.removeEventListener("unhandledrejection", unhandled);
  });
});
