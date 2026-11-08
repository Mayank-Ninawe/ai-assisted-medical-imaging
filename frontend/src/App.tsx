import { useState } from "react";
import { classify, generateGradCam, submitReview, type ClassificationResponse, type GradCamResponse } from "./api";
import "./styles.css";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ClassificationResponse | null>(null);
  const [gradCam, setGradCam] = useState<GradCamResponse | null>(null);
  const [status, setStatus] = useState("Choose a chest X-ray image to begin.");
  const [busy, setBusy] = useState(false);

  async function handleUpload() {
    if (!file) {
      setStatus("Select an image first.");
      return;
    }
    setBusy(true);
    setGradCam(null);
    setStatus("Running the educational classifier...");
    try {
      setResult(await classify(file));
      setStatus("Classification completed.");
    } catch (error) {
      setResult(null);
      setStatus(error instanceof Error ? error.message : "The classification request failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleGradCam(label: string) {
    if (!file) return;
    setBusy(true);
    setStatus(`Generating Grad-CAM for ${label}...`);
    try {
      setGradCam(await generateGradCam(file, label));
      setStatus("Grad-CAM generated.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Grad-CAM generation failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleReview(action: "approve" | "reject") {
    if (!result) return;
    setBusy(true);
    try {
      await submitReview(file?.name ?? "uploaded-image", action, result.model_artifact);
      setStatus(`Review action "${action}" was saved.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "The review action failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <header>
        <h1>AI-Assisted Medical Imaging</h1>
        <p>Educational demonstration only — not for clinical use or diagnosis.</p>
      </header>

      <section aria-labelledby="upload-heading">
        <h2 id="upload-heading">Classification</h2>
        <label htmlFor="image-upload">Chest X-ray image</label>
        <input
          id="image-upload"
          type="file"
          accept="image/png,image/jpeg,image/bmp,image/tiff"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
        <button type="button" onClick={handleUpload} disabled={busy || !file}>
          {busy ? "Working..." : "Run classification"}
        </button>
        <p role="status">{status}</p>
      </section>

      {result && (
        <section aria-labelledby="results-heading">
          <h2 id="results-heading">Model output</h2>
          <p>Model: {result.model} ({result.model_artifact})</p>
          <ul>
            {result.predictions.map((prediction) => (
              <li key={prediction.label}>
                <span>{prediction.label}: {(prediction.probability * 100).toFixed(2)}%</span>
                <button type="button" onClick={() => handleGradCam(prediction.label)} disabled={busy}>
                  Explain
                </button>
              </li>
            ))}
          </ul>
          <p>{result.limitation}</p>
          <button type="button" onClick={() => handleReview("approve")} disabled={busy}>Approve review</button>
          <button type="button" onClick={() => handleReview("reject")} disabled={busy}>Reject review</button>
        </section>
      )}

      {gradCam && (
        <section aria-labelledby="gradcam-heading">
          <h2 id="gradcam-heading">Grad-CAM</h2>
          <img
            src={`data:image/png;base64,${gradCam.overlay_base64}`}
            alt={`Grad-CAM overlay for ${gradCam.target_label}`}
          />
          <p>{gradCam.limitation}</p>
        </section>
      )}
    </main>
  );
}
