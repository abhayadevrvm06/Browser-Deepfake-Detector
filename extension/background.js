function toggleMonitoring(apiUrl) {
  if (window.__deepfakeMonitor) {
    clearTimeout(window.__deepfakeMonitor);
    window.__deepfakeMonitor = null;
    document.getElementById("deepfake-monitor-banner")?.remove();
    return;
  }

  const banner = document.createElement("div");
  banner.id = "deepfake-monitor-banner";
  Object.assign(banner.style, {
    position: "fixed", top: "20px", right: "20px", zIndex: 999999,
    padding: "12px 20px", background: "#555", color: "white",
    fontFamily: "sans-serif", fontSize: "14px", borderRadius: "8px",
    boxShadow: "0 2px 10px rgba(0,0,0,0.3)",
  });
  banner.textContent = "Deepfake Monitor: starting...";
  document.body.appendChild(banner);

  const recentPredictions = []; // holds last 3 predictions for smoothing

  async function checkFrame() {
    const video = document.querySelector("video");
    if (!video || video.readyState < 2) {
      banner.textContent = "Deepfake Monitor: no video found";
      window.__deepfakeMonitor = setTimeout(checkFrame, 1500);
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) {
        window.__deepfakeMonitor = setTimeout(checkFrame, 1500);
        return;
      }
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");
      try {
        const response = await fetch(apiUrl, { method: "POST", body: formData });
        const result = await response.json();

        if (!result.face_detected) {
          banner.textContent = "Deepfake Monitor: no face detected";
          banner.style.background = "#555";
        } else {
          recentPredictions.push(result.prediction);
          if (recentPredictions.length > 3) recentPredictions.shift();

          const fakeCount = recentPredictions.filter(p => p === "fake").length;
          const smoothedVerdict = fakeCount >= 2 ? "fake" : "real";
          const pct = (result.confidence * 100).toFixed(1);

          banner.textContent = `Deepfake Monitor: ${smoothedVerdict.toUpperCase()} (latest: ${pct}%)`;
          banner.style.background = smoothedVerdict === "fake" ? "#d32f2f" : "#2e7d32";
        }
      } catch (err) {
        banner.textContent = "Deepfake Monitor: check failed";
      }
      window.__deepfakeMonitor = setTimeout(checkFrame, 1500);
    }, "image/jpeg", 0.9);
  }

  checkFrame();
}