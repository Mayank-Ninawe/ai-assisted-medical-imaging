import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import App from "./App";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

test("renders disclaimer and handles classification success", async () => {
  fetchMock.mockResolvedValue({
    ok: true,
    json: async () => ({
      model: "densenet121",
      model_artifact: "best_classifier.pt",
      threshold: 0.5,
      predictions: [{
        label: "Cardiomegaly",
        probability: 0.6,
        positive_at_threshold: true,
      }],
      limitation: "Educational model output only.",
    }),
  });

  render(<App />);
  expect(screen.getByText(/not for clinical use/i)).toBeInTheDocument();

  const input = screen.getByLabelText("Chest X-ray image");
  const file = new File(["image"], "xray.png", { type: "image/png" });
  fireEvent.change(input, { target: { files: [file] } });
  fireEvent.click(screen.getByRole("button", { name: "Run classification" }));

  await waitFor(() => expect(screen.getByText(/Classification completed/i)).toBeInTheDocument());
  expect(screen.getByText(/Cardiomegaly: 60.00%/i)).toBeInTheDocument();
});

test("shows backend errors", async () => {
  fetchMock.mockResolvedValue({
    ok: false,
    json: async () => ({ detail: { message: "Model unavailable." } }),
  });

  render(<App />);
  const input = screen.getByLabelText("Chest X-ray image");
  fireEvent.change(input, {
    target: { files: [new File(["image"], "xray.png", { type: "image/png" })] },
  });
  fireEvent.click(screen.getByRole("button", { name: "Run classification" }));

  await waitFor(() => expect(screen.getByText("Model unavailable.")).toBeInTheDocument());
});
