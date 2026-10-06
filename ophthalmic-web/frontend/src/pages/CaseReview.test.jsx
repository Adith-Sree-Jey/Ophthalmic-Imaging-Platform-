import { act, render, screen } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import CaseReview from "./CaseReview";
import { getCase } from "../api";

vi.mock("../components/Navbar", () => ({ default: () => <nav>Navigation</nav> }));
vi.mock("../api", () => ({
  approveReport: vi.fn(),
  approveResult: vi.fn(),
  downloadReport: vi.fn(),
  getCase: vi.fn(),
  overrideResult: vi.fn(),
  rejectReport: vi.fn(),
  rejectResult: vi.fn(),
}));

function casePayload(id, patientName, mriNumber) {
  return {
    case: { id, status: "pending_review", priority: "routine", report_status: "pending" },
    patient_summary: { patient_name: patientName, mri_number: mriNumber },
    module_results: [],
    audit_trail: [],
  };
}

describe("CaseReview navigation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("clears the previous patient's identity before the next case resolves", async () => {
    let resolveCaseB;
    getCase.mockImplementation((caseId) => {
      if (caseId === "A") return Promise.resolve(casePayload("A", "Patient A", "MRI-A"));
      return new Promise((resolve) => {
        resolveCaseB = resolve;
      });
    });

    const router = createMemoryRouter(
      [{ path: "/cases/:caseId/review", element: <CaseReview /> }],
      { initialEntries: ["/cases/A/review"] },
    );
    render(<RouterProvider router={router} />);

    expect(await screen.findByText("Patient A")).toBeInTheDocument();
    expect(screen.getByText("MRI-A")).toBeInTheDocument();

    await act(async () => {
      await router.navigate("/cases/B/review");
    });

    expect(screen.queryByText("Patient A")).not.toBeInTheDocument();
    expect(screen.queryByText("MRI-A")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Loading case");

    await act(async () => {
      resolveCaseB(casePayload("B", "Patient B", "MRI-B"));
    });

    expect(await screen.findByText("Patient B")).toBeInTheDocument();
    expect(screen.getByText("MRI-B")).toBeInTheDocument();
    expect(screen.queryByText("Patient A")).not.toBeInTheDocument();
  });
});
