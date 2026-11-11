"""
SAFE CALL: Interactive Web Browser Dashboard.

Runs a zero-dependency local web server using Python standard library (http.server).
Provides a modern visual interface for call analysis:
- Drag & Drop Audio Upload (.m4a, .wav, .mp3, .aac, .flac)
- Browser Microphone Voice Recording
- 1-Click Pre-Packaged Demo Scenarios
- Animated Authenticity & Scam Risk Meters
- Real-Time Transcription Display
- Red Flag Indicators & Cyber Helpline 1930 Guidance

Run:
    python web_app.py
Then open:
    http://localhost:8000
"""

import sys
import os
import json
import cgi
import tempfile
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import get_pipeline

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SAFE CALL | AI Voice Clone & Scam Detection</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(17, 24, 39, 0.85);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --critical: #dc2626;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background: radial-gradient(circle at top left, #1e1b4b 0%, #090d16 60%);
      color: var(--text);
      min-height: 100vh;
      padding: 2rem 1rem;
    }
    .container { max-width: 1100px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 2.5rem; }
    .badge {
      display: inline-block;
      padding: 0.35rem 0.85rem;
      background: rgba(59, 130, 246, 0.15);
      border: 1px solid rgba(59, 130, 246, 0.3);
      color: #60a5fa;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 0.75rem;
    }
    h1 { font-size: 2.5rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.5rem; }
    .gradient-text { background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    p.subtitle { color: var(--text-muted); font-size: 1.05rem; max-width: 650px; margin: 0 auto; }

    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 1rem;
      padding: 1.75rem;
      backdrop-filter: blur(12px);
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .card-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 1.25rem; display: flex; align-items: center; gap: 0.5rem; }

    /* Upload Area */
    .drop-zone {
      border: 2px dashed rgba(59, 130, 246, 0.4);
      border-radius: 0.75rem;
      padding: 2.5rem 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: rgba(15, 23, 42, 0.5);
    }
    .drop-zone:hover, .drop-zone.dragover { border-color: var(--primary); background: rgba(59, 130, 246, 0.08); }
    .drop-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
    .drop-text { font-size: 0.95rem; font-weight: 500; margin-bottom: 0.25rem; }
    .drop-sub { font-size: 0.8rem; color: var(--text-muted); }

    /* Buttons */
    .btn-group { display: flex; gap: 0.75rem; margin-top: 1.25rem; }
    .btn {
      flex: 1;
      padding: 0.75rem 1rem;
      border-radius: 0.5rem;
      border: none;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      transition: all 0.15s ease;
    }
    .btn-primary { background: var(--primary); color: white; }
    .btn-primary:hover { background: var(--primary-hover); transform: translateY(-1px); }
    .btn-danger { background: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.4); color: #fca5a5; }
    .btn-danger:hover { background: rgba(239, 68, 68, 0.3); }

    /* Preset Scenarios */
    .presets { display: flex; flex-direction: column; gap: 0.6rem; margin-top: 1rem; }
    .preset-btn {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 0.75rem 1rem;
      border-radius: 0.5rem;
      color: var(--text);
      text-align: left;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: all 0.15s ease;
      font-size: 0.88rem;
    }
    .preset-btn:hover { background: rgba(255, 255, 255, 0.08); border-color: rgba(255, 255, 255, 0.2); }
    .preset-tag { font-size: 0.75rem; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-weight: 600; }
    .tag-safe { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; }
    .tag-danger { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }

    /* Results */
    .results-card { grid-column: 1 / -1; }
    .banner {
      padding: 1rem 1.5rem;
      border-radius: 0.75rem;
      font-weight: 700;
      font-size: 1.1rem;
      margin-bottom: 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .banner-safe { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); color: #6ee7b7; }
    .banner-moderate { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); color: #fcd34d; }
    .banner-critical { background: rgba(220, 38, 38, 0.2); border: 1px solid rgba(220, 38, 38, 0.5); color: #fca5a5; }

    .meter-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
    .meter-box {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 0.75rem;
      padding: 1.25rem;
      text-align: center;
    }
    .meter-val { font-size: 2rem; font-weight: 800; margin: 0.35rem 0; font-family: 'JetBrains Mono', monospace; }
    .meter-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }

    .transcript-box {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 0.5rem;
      padding: 1rem;
      font-style: italic;
      color: #e5e7eb;
      margin-bottom: 1.5rem;
      line-height: 1.6;
    }

    .redflags-list { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.5rem; }
    .redflag-badge {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #fca5a5;
      padding: 0.35rem 0.75rem;
      border-radius: 0.35rem;
      font-size: 0.82rem;
      font-weight: 600;
    }

    .actions-list { list-style: none; display: flex; flex-direction: column; gap: 0.5rem; }
    .actions-list li {
      padding: 0.6rem 0.9rem;
      background: rgba(255, 255, 255, 0.03);
      border-left: 3px solid var(--primary);
      border-radius: 0 0.35rem 0.35rem 0;
      font-size: 0.9rem;
    }

    .loading { display: none; text-align: center; padding: 2rem; }
    .spinner {
      border: 3px solid rgba(255, 255, 255, 0.1);
      border-top: 3px solid var(--primary);
      border-radius: 50%;
      width: 40px; height: 40px;
      animation: spin 0.8s linear infinite;
      margin: 0 auto 1rem;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge">Semester 1 Defense & Live Evaluation</div>
      <h1><span class="gradient-text">SAFE CALL</span> Dashboard</h1>
      <p class="subtitle">Real-time Multi-Modal Voice Deepfake Detection, Whisper Transcription, and Intelligent Scam Protection.</p>
    </header>

    <div class="grid">
      <!-- Input Card -->
      <div class="card">
        <h2 class="card-title">🎙️ Audio Input</h2>
        <div class="drop-zone" id="dropZone">
          <div class="drop-icon">📁</div>
          <div class="drop-text">Click to choose or drag & drop audio</div>
          <div class="drop-sub">Supports .m4a, .wav, .mp3, .aac, .flac</div>
          <input type="file" id="fileInput" accept="audio/*" style="display: none;">
        </div>

        <div class="btn-group">
          <button class="btn btn-primary" id="recordBtn">🔴 Record Live Mic</button>
        </div>
      </div>

      <!-- Presets Card -->
      <div class="card">
        <h2 class="card-title">⚡ 1-Click Demonstration Scenarios</h2>
        <div class="presets">
          <button class="preset-btn" onclick="analyzePreset('user_sample_1.m4a')">
            <span>👤 User Real Voice (Customer Refund)</span>
            <span class="preset-tag tag-safe">HUMAN VOICE</span>
          </button>
          <button class="preset-btn" onclick="analyzePreset('user_sample_2.m4a')">
            <span>👤 User Real Voice (Service Checkup)</span>
            <span class="preset-tag tag-safe">HUMAN VOICE</span>
          </button>
          <button class="preset-btn" onclick="analyzePreset('2_bank_otp_scam.wav')">
            <span>🚨 AI Cloned Bank OTP Scam</span>
            <span class="preset-tag tag-danger">AI SCAM</span>
          </button>
          <button class="preset-btn" onclick="analyzePreset('3_police_arrest_threat.wav')">
            <span>🚨 AI Cloned Police Arrest Threat</span>
            <span class="preset-tag tag-danger">AI SCAM</span>
          </button>
        </div>
      </div>

      <!-- Loading Indicator -->
      <div class="card results-card loading" id="loadingBox">
        <div class="spinner"></div>
        <p style="font-weight: 600; color: #93c5fd;">Processing call through 6-Stage SAFE CALL Pipeline...</p>
        <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem;">Extracting 40-dim MFCCs • Whisper ASR • Scam Intent NLP • Fusion</p>
      </div>

      <!-- Results Card -->
      <div class="card results-card" id="resultsBox" style="display: none;">
        <div id="banner" class="banner"></div>

        <div class="meter-grid">
          <div class="meter-box">
            <div class="meter-label">Voice Authenticity</div>
            <div class="meter-val" id="voiceVal">-</div>
            <div id="voiceLabel" style="font-size: 0.85rem; font-weight: 600;">-</div>
          </div>
          <div class="meter-box">
            <div class="meter-label">Scam Risk Score</div>
            <div class="meter-val" id="riskVal">-</div>
            <div id="riskLabel" style="font-size: 0.85rem; font-weight: 600;">-</div>
          </div>
          <div class="meter-box">
            <div class="meter-label">Threat Classification</div>
            <div class="meter-val" id="threatVal" style="font-size: 1.25rem; padding-top: 0.5rem;">-</div>
          </div>
        </div>

        <h3 style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem; color: #93c5fd;">📝 Speech Transcription</h3>
        <div class="transcript-box" id="transcriptText"></div>

        <h3 style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem; color: #fca5a5;">🚩 Detected Scam Indicators</h3>
        <div class="redflags-list" id="redFlagsList"></div>

        <h3 style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem; color: #a7f3d0;">🛡️ Recommended Countermeasures</h3>
        <ul class="actions-list" id="actionsList"></ul>
      </div>
    </div>
  </div>

  <script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const loadingBox = document.getElementById('loadingBox');
    const resultsBox = document.getElementById('resultsBox');
    const recordBtn = document.getElementById('recordBtn');

    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;

    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) uploadAudio(e.target.files[0]);
    });

    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length) uploadAudio(e.dataTransfer.files[0]);
    });

    // Microphone Recording
    recordBtn.addEventListener('click', async () => {
      if (!isRecording) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          mediaRecorder = new MediaRecorder(stream);
          audioChunks = [];
          mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
          mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            uploadAudio(audioBlob, 'mic_recording.webm');
          };
          mediaRecorder.start();
          isRecording = true;
          recordBtn.innerText = '⏹️ Stop & Analyze';
          recordBtn.classList.add('btn-danger');
        } catch (err) {
          alert('Microphone access denied or not available: ' + err.message);
        }
      } else {
        mediaRecorder.stop();
        isRecording = false;
        recordBtn.innerText = '🔴 Record Live Mic';
        recordBtn.classList.remove('btn-danger');
      }
    });

    async function uploadAudio(file, filename = null) {
      loadingBox.style.display = 'block';
      resultsBox.style.display = 'none';

      const formData = new FormData();
      formData.append('audio', file, filename || file.name);

      try {
        const res = await fetch('/api/analyze', { method: 'POST', body: formData });
        const data = await res.json();
        renderResults(data);
      } catch (err) {
        alert('Analysis error: ' + err.message);
      } finally {
        loadingBox.style.display = 'none';
      }
    }

    async function analyzePreset(presetName) {
      loadingBox.style.display = 'block';
      resultsBox.style.display = 'none';

      try {
        const res = await fetch('/api/analyze_sample?name=' + encodeURIComponent(presetName));
        const data = await res.json();
        renderResults(data);
      } catch (err) {
        alert('Preset error: ' + err.message);
      } finally {
        loadingBox.style.display = 'none';
      }
    }

    function renderResults(data) {
      resultsBox.style.display = 'block';
      const guidance = data.guidance || {};
      const fusion = data.fusion || {};
      const voice = data.voice_detection || {};
      const trans = data.transcription || {};
      const scam = data.scam_detection || {};

      // Banner
      const banner = document.getElementById('banner');
      banner.innerText = guidance.title || 'ANALYSIS COMPLETE';
      banner.className = 'banner';
      if (fusion.risk_level === 'CRITICAL' || fusion.risk_level === 'HIGH') {
        banner.classList.add('banner-critical');
      } else if (fusion.risk_level === 'MODERATE') {
        banner.classList.add('banner-moderate');
      } else {
        banner.classList.add('banner-safe');
      }

      // Voice Authenticity
      const isSpoof = voice.raw_label === 1;
      const voiceVal = document.getElementById('voiceVal');
      const voiceLabel = document.getElementById('voiceLabel');
      if (isSpoof) {
        voiceVal.innerText = (voice.spoof_probability * 100).toFixed(1) + '%';
        voiceVal.style.color = '#ef4444';
        voiceLabel.innerText = '⚠️ AI-GENERATED / CLONE';
        voiceLabel.style.color = '#fca5a5';
      } else {
        voiceVal.innerText = (voice.bonafide_probability * 100).toFixed(1) + '%';
        voiceVal.style.color = '#10b981';
        voiceLabel.innerText = '✅ GENUINE HUMAN';
        voiceLabel.style.color = '#6ee7b7';
      }

      // Risk Score
      const riskVal = document.getElementById('riskVal');
      const riskLabel = document.getElementById('riskLabel');
      const score = (fusion.composite_risk_score * 100).toFixed(1);
      riskVal.innerText = score + '%';
      riskLabel.innerText = 'LEVEL: ' + (fusion.risk_level || 'LOW');
      if (fusion.risk_level === 'CRITICAL' || fusion.risk_level === 'HIGH') {
        riskVal.style.color = '#ef4444';
      } else if (fusion.risk_level === 'MODERATE') {
        riskVal.style.color = '#f59e0b';
      } else {
        riskVal.style.color = '#10b981';
      }

      // Threat
      document.getElementById('threatVal').innerText = (fusion.primary_threat || 'BENIGN').replace(/_/g, ' ');

      // Transcript
      document.getElementById('transcriptText').innerText = trans.text ? `"${trans.text}"` : "(No clear spoken speech detected)";

      // Red flags
      const flagsList = document.getElementById('redFlagsList');
      flagsList.innerHTML = '';
      const flags = scam.red_flags || [];
      if (flags.length === 0) {
        flagsList.innerHTML = '<span style="color: #9ca3af; font-size: 0.85rem;">✅ No social engineering or scam red flags detected.</span>';
      } else {
        flags.forEach(f => {
          const span = document.createElement('span');
          span.className = 'redflag-badge';
          span.innerText = `[${f.severity}] ${f.category.replace(/_/g, ' ').toUpperCase()}: "${f.matched_phrase}"`;
          flagsList.appendChild(span);
        });
      }

      // Actions
      const actionsList = document.getElementById('actionsList');
      actionsList.innerHTML = '';
      const actions = guidance.action_items || [];
      actions.forEach(act => {
        const li = document.createElement('li');
        li.innerText = act;
        actionsList.appendChild(li);
      });

      resultsBox.scrollIntoView({ behavior: 'smooth' });
    }
  </script>
</body>
</html>
"""


class SafeCallServer(BaseHTTPRequestHandler):
    pipeline = None

    @classmethod
    def get_pipeline(cls):
        if cls.pipeline is None:
            cls.pipeline = get_pipeline()
        return cls.pipeline

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

        elif parsed.path == "/api/analyze_sample":
            query = urllib.parse.parse_qs(parsed.query)
            sample_name = query.get("name", [""])[0]

            # Search in demo_samples or tests/temp_audio
            possible_paths = [
                PROJECT_ROOT / "data" / "demo_samples" / sample_name,
                PROJECT_ROOT / "tests" / "temp_audio" / sample_name,
            ]
            target_path = None
            for p in possible_paths:
                if p.is_file():
                    target_path = p
                    break

            if not target_path:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Sample not found: {sample_name}"}).encode("utf-8"))
                return

            pipe = self.get_pipeline()
            result = pipe.process_call(target_path)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/analyze":
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Expected multipart/form-data")
                return

            # Parse multipart upload using FieldStorage
            environ = {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            }
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ=environ,
            )

            fileitem = form["audio"]
            if not fileitem.file:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"No audio file received")
                return

            # Save uploaded bytes to a temp audio file
            ext = Path(fileitem.filename or "recording.m4a").suffix or ".m4a"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(fileitem.file.read())
                temp_path = Path(tmp.name)

            try:
                pipe = self.get_pipeline()
                result = pipe.process_call(temp_path)
                response_data = json.dumps(result).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response_data)
            except Exception as exc:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
            finally:
                if temp_path.is_file():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port: int = 8000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, SafeCallServer)
    print("=" * 70)
    print("   🛡️   SAFE CALL: WEB BROWSER DASHBOARD   🛡️")
    print(f"   Server active at: http://localhost:{port}")
    print("   Open this URL in Google Chrome, Microsoft Edge, or any browser.")
    print("   Press Ctrl+C in this terminal to stop the server.")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_server(port)
