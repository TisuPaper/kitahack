// Live detection screen — embeds the full live detection UI inline.
// This is a snapshot of app/static/live.html with API_BASE injected at runtime.
// Web-only: uses dart:html for Blob URL + iframe rendering.

import 'dart:convert';
import 'dart:html' as html;
import 'dart:ui_web' as ui_web;
import 'package:flutter/material.dart';
import '../services/api_service.dart';

const String _liveHtmlTemplate = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Live Deepfake Detection</title>
  <link rel="icon" id="pageFavicon" type="image/png" href="">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: transparent; /* Changed to transparent to let Flutter background show through */
      color: #1E293B; /* Changed to dark text for light theme */
      min-height: 100vh;
    }

    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 16px 24px;
      background: rgba(255, 255, 255, 0.6); /* Glassmorphism light */
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border-bottom: 1px solid rgba(0,0,0,0.05);
      border-radius: 16px 16px 0 0;
    }

    .header h1 {
      font-size: 20px;
      font-weight: 800;
      display: flex;
      align-items: center;
      gap: 10px;
      color: #1E293B;
      letter-spacing: -0.5px;
    }

    .header h1 .icon {
      width: 32px;
      height: 32px;
      background: rgba(0,0,0,0.05);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
    }

    .controls {
      display: flex;
      gap: 12px;
      align-items: center;
    }

    .btn {
      padding: 10px 20px;
      border: none;
      border-radius: 10px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .btn-start {
      background: #1E293B; /* Dark button like home screen */
      color: white;
    }

    .btn-start:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }

    .btn-stop {
      background: #EF4444;
      color: white;
    }

    .btn-stop:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(239,68,68,0.3); }

    .btn:disabled {
      opacity: 0.4;
      cursor: not-allowed;
      transform: none !important;
      box-shadow: none !important;
    }

    .btn-pip {
      background: rgba(0,0,0,0.05);
      border: 1px solid rgba(0,0,0,0.1);
      color: #1E293B;
    }
    .btn-pip:hover { background: rgba(0,0,0,0.08); transform: translateY(-1px); }
    .btn-pip.active {
      background: rgba(0,0,0,0.1);
      border-color: rgba(0,0,0,0.2);
      box-shadow: 0 0 12px rgba(0,0,0,0.05);
    }

    .main {
      display: flex;
      gap: 20px;
      padding: 20px;
      height: calc(100vh - 68px);
    }

    .video-panel {
      flex: 1;
      position: relative;
      background: rgba(255, 255, 255, 0.6); /* Glassmorphism light */
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.8);
      box-shadow: 0 8px 32px rgba(0,0,0,0.05);
    }

    #screenVideo {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #000; /* Keep video background black for contrast */
      border-radius: 16px;
    }

    #faceOverlay {
      position: absolute;
      top: 0; left: 0;
      width: 100%; height: 100%;
      pointer-events: none;
    }

    .placeholder {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      text-align: center;
      color: rgba(0,0,0,0.5);
    }

    .placeholder .big-icon { font-size: 64px; margin-bottom: 16px; }
    .placeholder p { font-size: 16px; font-weight: 600; color: #1E293B; }

    .verdict-overlay {
      position: absolute; top: 16px; right: 16px;
      padding: 14px 22px; border-radius: 14px;
      display: none; flex-direction: column; gap: 4px;
      backdrop-filter: blur(16px); z-index: 10;
      transition: all 0.3s ease; max-width: 340px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .verdict-overlay .vo-main { font-size: 20px; font-weight: 800; letter-spacing: 2px; }
    .verdict-overlay .vo-band { font-size: 12px; font-weight: 600; opacity: 0.8; }
    .verdict-overlay .vo-why  { font-size: 11px; opacity: 0.8; margin-top: 2px; }

    .verdict-overlay.real {
      display: flex; background: rgba(255,255,255,0.9);
      border: 1px solid rgba(34,197,94,0.4); color: #22C55E;
    }
    .verdict-overlay.fake {
      display: flex; background: rgba(255,255,255,0.9);
      border: 1px solid rgba(239,68,68,0.4); color: #EF4444;
      animation: pulse-red 1.5s infinite;
    }
    .verdict-overlay.uncertain {
      display: flex; background: rgba(255,255,255,0.9);
      border: 1px solid rgba(245,158,11,0.4); color: #F59E0B;
    }
    .verdict-overlay.analyzing {
      display: flex; background: rgba(255,255,255,0.9);
      border: 1px solid rgba(0,0,0,0.1); color: #1E293B;
    }

    @keyframes pulse-red {
      0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.3); }
      50% { box-shadow: 0 0 20px 4px rgba(239,68,68,0.2); }
    }

    .side-panel { width: 370px; display: flex; flex-direction: column; gap: 14px; overflow-y: auto; }
    .card {
      background: rgba(255, 255, 255, 0.6); /* Glassmorphism light */
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border-radius: 16px; padding: 20px;
      border: 1px solid rgba(255,255,255,0.8);
      box-shadow: 0 8px 32px rgba(0,0,0,0.05);
    }
    .card h3 {
      font-size: 13px; color: rgba(0,0,0,0.5); text-transform: uppercase;
      letter-spacing: 1px; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
      font-weight: 700;
    }

    .vc-icon     { font-size: 36px; line-height: 1; }
    .vc-verdict  { font-size: 28px; font-weight: 800; letter-spacing: 3px; }
    .vc-band     { font-size: 13px; font-weight: 600; opacity: 0.75; margin-top: 2px; }
    .vc-why      { font-size: 13px; color: rgba(0,0,0,0.6); margin-top: 10px; line-height: 1.4; }
    .vc-steps    { list-style: none; margin-top: 10px; padding: 0; }
    .vc-steps li {
      font-size: 12px; color: rgba(0,0,0,0.6); padding: 4px 0; padding-left: 18px;
      position: relative; line-height: 1.5;
    }
    .vc-steps li::before {
      content: ''; position: absolute; left: 0; top: 9px;
      width: 8px; height: 8px; border-radius: 50%; border: 1.5px solid rgba(0,0,0,0.25);
    }

    .audio-indicator {
      display: flex; align-items: center; gap: 8px; padding: 8px 12px;
      border-radius: 8px; background: rgba(0,0,0,0.05); font-size: 13px; color: #1E293B;
      font-weight: 600;
    }
    .audio-bars { display: flex; gap: 2px; align-items: flex-end; height: 16px; }
    .audio-bars span {
      width: 3px; background: #1E293B; border-radius: 2px;
      animation: audio-bar 0.8s ease-in-out infinite;
    }
    .audio-bars span:nth-child(1) { height: 6px; animation-delay: 0s; }
    .audio-bars span:nth-child(2) { height: 12px; animation-delay: 0.15s; }
    .audio-bars span:nth-child(3) { height: 8px; animation-delay: 0.3s; }
    .audio-bars span:nth-child(4) { height: 14px; animation-delay: 0.45s; }
    .audio-bars span:nth-child(5) { height: 10px; animation-delay: 0.6s; }
    @keyframes audio-bar { 0%, 100% { transform: scaleY(1); } 50% { transform: scaleY(0.4); } }
    #audioCanvas { width: 100%; height: 32px; border-radius: 8px; background: rgba(0,0,0,0.05); margin-top: 8px; }

    .details-toggle {
      display: flex; align-items: center; justify-content: space-between;
      cursor: pointer; user-select: none;
    }
    .details-toggle::after {
      content: '\25B6'; font-size: 10px; color: rgba(0,0,0,0.3);
      transition: transform 0.2s;
    }
    .details-toggle.open::after { transform: rotate(90deg); }
    .details-body { display: none; margin-top: 10px; }
    .details-body.open { display: block; }
    .detail-row {
      display: flex; justify-content: space-between; padding: 4px 0; font-size: 12px;
    }
    .detail-row .dl { color: rgba(0,0,0,0.5); }
    .detail-row .dv { font-weight: 600; color: #1E293B; }

    .log-entry {
      padding: 6px 10px; border-radius: 8px; margin-bottom: 4px;
      font-size: 11px; display: flex; align-items: center; gap: 6px;
      font-weight: 500;
    }
    .log-entry.real { background: rgba(34,197,94,0.1); color: #16A34A; }
    .log-entry.fake { background: rgba(239,68,68,0.1); color: #DC2626; }
    .log-entry.uncertain { background: rgba(245,158,11,0.1); color: #D97706; }
    .log-entry.analyzing { background: rgba(0,0,0,0.05); color: #1E293B; }
    .log-container { max-height: 260px; overflow-y: auto; }

    .interval-select {
      background: rgba(255,255,255,0.8); color: #1E293B;
      border: 1px solid rgba(0,0,0,0.1);
      border-radius: 8px; padding: 8px 12px; font-size: 13px;
      font-weight: 600;
    }

    /* Responsive Layout for Mobile/Smaller Screens */
    @media (max-width: 800px) {
      .header {
        padding: 16px;
        justify-content: center;
      }
      .header h1 {
        display: none; /* Hide title to save space, as requested */
      }
      .controls {
        flex-wrap: wrap;
        justify-content: center;
        width: 100%;
      }
      .main {
        flex-direction: column;
        height: auto;
        padding: 16px;
        gap: 16px;
      }
      .side-panel {
        width: 100%;
        overflow-y: visible;
      }
      .video-panel {
        min-height: 350px;
      }
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>
      <span class="icon">🛡️</span>
      Live Deepfake Detection
    </h1>
    <div class="controls">
      <label style="font-size:13px; color:rgba(0,0,0,0.5); font-weight: 600;">
        Scan every:
        <select id="intervalSelect" class="interval-select">
          <option value="2000">2s</option>
          <option value="3000" selected>3s</option>
          <option value="5000">5s</option>
          <option value="10000">10s</option>
        </select>
      </label>
      <button id="btnPip" class="btn btn-pip" onclick="toggleWidget()" title="Float results in a mini window">
        ⧉ Pop Out
      </button>
      <button id="btnStart" class="btn btn-start" onclick="startCapture()">
        ▶ Start Capture
      </button>
      <button id="btnStop" class="btn btn-stop" onclick="stopCapture()" disabled>
        ■ Stop
      </button>
    </div>
  </div>

  <div class="main">
    <div class="video-panel">
      <video id="screenVideo" autoplay muted></video>
      <canvas id="faceOverlay"></canvas>
      <div id="placeholder" class="placeholder">
        <div class="big-icon">🖥️</div>
        <p>Click "Start Capture" to share your screen</p>
        <p style="font-size:13px; margin-top:8px; opacity:0.5">
          Select a window, tab, or entire screen
        </p>
      </div>
      <div id="verdictOverlay" class="verdict-overlay"></div>
    </div>

    <div class="side-panel">
      <div class="card" id="videoVerdictCard">
        <h3>🖥 Video Analysis</h3>
        <div id="videoVerdictArea" style="text-align:center; padding:10px 0">
          <div style="font-size:13px; color:rgba(0,0,0,0.4); font-weight: 500;">
            Click "Start Capture" to begin
          </div>
        </div>
      </div>

      <div class="card" id="audioVerdictCard">
        <h3>🎙 Audio Analysis</h3>
        <div id="audioStatus">
          <div style="font-size:13px; color:rgba(0,0,0,0.4); font-weight: 500;">
            Audio capture starts with screen share
          </div>
        </div>
        <canvas id="audioCanvas"></canvas>
      </div>

      <div class="card">
        <h3 class="details-toggle" id="detailsToggle" onclick="toggleDetails()">
          🔬 Technical Details
        </h3>
        <div class="details-body" id="detailsBody">
          <div class="detail-row"><span class="dl">State</span><span class="dv" id="statState">Idle</span></div>
          <div class="detail-row"><span class="dl">Frames analyzed</span><span class="dv" id="statFrames">0</span></div>
          <div class="detail-row"><span class="dl">Fake detections</span><span class="dv" id="statFakes" style="color:#DC2626">0</span></div>
          <div class="detail-row"><span class="dl">Real detections</span><span class="dv" id="statReals" style="color:#16A34A">0</span></div>
          <div class="detail-row"><span class="dl">Avg confidence</span><span class="dv" id="statConfidence">-</span></div>
          <div id="detailsExtra" style="margin-top:6px; font-size:11px; color:rgba(0,0,0,0.4)"></div>
        </div>
      </div>

      <div class="card" style="flex:1;">
        <h3>📋 Detection Log</h3>
        <div id="logContainer" class="log-container">
          <div class="log-entry analyzing">Waiting to start...</div>
        </div>
      </div>
    </div>
  </div>

  <canvas id="frameCanvas" style="display:none;"></canvas>

  <script>
    const faceOverlay = document.getElementById('faceOverlay');
    const faceCtx = faceOverlay.getContext('2d');

    const VERDICT_COLOR = {
      real:      '#22C55E',
      fake:      '#EF4444',
      uncertain: '#F59E0B',
    };

    function drawFaceBox(bbox, prediction) {
      faceOverlay.width  = faceOverlay.clientWidth;
      faceOverlay.height = faceOverlay.clientHeight;
      faceCtx.clearRect(0, 0, faceOverlay.width, faceOverlay.height);

      if (!bbox) return;

      const videoEl  = document.getElementById('screenVideo');
      const vw       = videoEl.videoWidth;
      const vh       = videoEl.videoHeight;
      const cw       = faceOverlay.clientWidth;
      const ch       = faceOverlay.clientHeight;

      if (!vw || !vh) return;

      const scale    = Math.min(cw / vw, ch / vh);
      const offsetX  = (cw - vw * scale) / 2;
      const offsetY  = (ch - vh * scale) / 2;

      const rx = offsetX + bbox.x * vw * scale;
      const ry = offsetY + bbox.y * vh * scale;
      const rw = bbox.w * vw * scale;
      const rh = bbox.h * vh * scale;

      const color = VERDICT_COLOR[prediction.toLowerCase()] || '#818CF8';

      faceCtx.shadowColor  = color;
      faceCtx.shadowBlur   = 12;
      faceCtx.strokeStyle  = color;
      faceCtx.lineWidth    = 2.5;
      faceCtx.strokeRect(rx, ry, rw, rh);
      faceCtx.shadowBlur   = 0;

      const cs = Math.min(rw, rh) * 0.18;
      faceCtx.lineWidth = 3.5;
      faceCtx.strokeStyle = color;

      const corners = [
        [rx,      ry,      cs,  0,  0, cs],
        [rx + rw, ry,     -cs,  0,  0, cs],
        [rx,      ry + rh, cs,  0,  0,-cs],
        [rx + rw, ry + rh,-cs,  0,  0,-cs],
      ];
      for (const [x, y, dx1, dy1, dx2, dy2] of corners) {
        faceCtx.beginPath();
        faceCtx.moveTo(x + dx1, y + dy1);
        faceCtx.lineTo(x, y);
        faceCtx.lineTo(x + dx2, y + dy2);
        faceCtx.stroke();
      }

      const label  = prediction.toUpperCase();
      const font   = 'bold 13px Inter, -apple-system, sans-serif';
      faceCtx.font = font;
      const tw     = faceCtx.measureText(label).width;
      const ph     = 22;
      const pw     = tw + 18;
      const px     = rx + (rw - pw) / 2;
      const py     = ry - ph - 6;

      faceCtx.fillStyle = color + 'CC';
      roundRect(faceCtx, px, py, pw, ph, 6);
      faceCtx.fill();

      faceCtx.fillStyle = '#fff';
      faceCtx.font      = font;
      faceCtx.textAlign = 'center';
      faceCtx.fillText(label, px + pw / 2, py + 15);
      faceCtx.textAlign = 'left';
    }

    function roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y);
      ctx.arcTo(x + w, y, x + w, y + r, r);
      ctx.lineTo(x + w, y + h - r);
      ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
      ctx.lineTo(x + r, y + h);
      ctx.arcTo(x, y + h, x, y + h - r, r);
      ctx.lineTo(x, y + r);
      ctx.arcTo(x, y, x + r, y, r);
      ctx.closePath();
    }

    function clearFaceBox() {
      faceOverlay.width  = faceOverlay.clientWidth;
      faceOverlay.height = faceOverlay.clientHeight;
      faceCtx.clearRect(0, 0, faceOverlay.width, faceOverlay.height);
    }

    let stream = null;
    let audioStream = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let audioAnalysisBuffer = [];
    let audioHeaderChunk = null;
    const AUDIO_BUFFER_MAX = 5;
    let isAnalyzingAudio = false;
    let scanInterval = null;
    let frameCount = 0;
    let fakeCount = 0;
    let realCount = 0;
    let totalConfidence = 0;
    let isAnalyzing = false;
    let audioContext = null;
    let analyserNode = null;
    let animFrameId = null;
    let audioChunkWatchdog = null;

    const API_BASE = '__API_BASE__';

    const video = document.getElementById('screenVideo');
    const placeholder = document.getElementById('placeholder');
    const verdictOverlay = document.getElementById('verdictOverlay');
    const logContainer = document.getElementById('logContainer');
    const canvas = document.getElementById('frameCanvas');
    const ctx = canvas.getContext('2d');
    const audioCanvas = document.getElementById('audioCanvas');
    const audioCtx2d = audioCanvas.getContext('2d');

    async function startCapture() {
      try {
        stream = await navigator.mediaDevices.getDisplayMedia({
          video: { cursor: 'always' },
          audio: true
        });

        video.srcObject = stream;
        placeholder.style.display = 'none';

        document.getElementById('btnStart').disabled = true;
        document.getElementById('btnStop').disabled = false;
        document.getElementById('statState').textContent = '🟢 Capturing';

        let audioSourceStream = null;
        const screenAudioTracks = stream.getAudioTracks();
        const surface = stream.getVideoTracks()[0]?.getSettings?.().displaySurface || '';
        const isBrowserTabCapture = surface === 'browser';

        if (screenAudioTracks.length > 0 && isBrowserTabCapture) {
          audioSourceStream = stream;
          document.getElementById('audioStatus').innerHTML =
            '<div class="audio-indicator"><div class="audio-bars">' +
            '<span></span><span></span><span></span><span></span><span></span>' +
            '</div>Recording tab audio...</div>';
        } else {
          try {
            const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioSourceStream = micStream;
            audioStream = micStream;
            document.getElementById('audioStatus').innerHTML =
              '<div class="audio-indicator"><div class="audio-bars">' +
              '<span></span><span></span><span></span><span></span><span></span>' +
              '</div>🎤 Recording via microphone...' +
              (!isBrowserTabCapture && screenAudioTracks.length > 0
                ? ' (screen/window audio not supported reliably)'
                : '') +
              '</div>';
          } catch (micErr) {
            console.warn('Microphone access denied:', micErr);
            document.getElementById('audioStatus').innerHTML =
              '<div style="font-size:13px; color:#F59E0B">' +
              '⚠️ No audio source — share a browser tab (not screen/window) or allow mic access</div>';
          }
        }

        if (audioSourceStream) {
          startAudioVisualization(audioSourceStream);
          startAudioRecording(audioSourceStream);
        }

        const interval = parseInt(document.getElementById('intervalSelect').value);
        scanInterval = setInterval(() => {
          analyzeFrame();
          analyzeAudio();
        }, interval);

        stream.getVideoTracks()[0].onended = () => stopCapture();

        addLog('analyzing', '🟢 Capture started — scanning every ' +
          (interval / 1000) + 's');
      } catch (err) {
        console.error('Screen capture failed:', err);
        addLog('uncertain', '❌ Screen capture cancelled or denied');
      }
    }

    function stopCapture() {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
      }
      if (audioStream) {
        audioStream.getTracks().forEach(t => t.stop());
        audioStream = null;
      }

      if (scanInterval) {
        clearInterval(scanInterval);
        scanInterval = null;
      }

      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
      }
      if (audioChunkWatchdog) {
        clearTimeout(audioChunkWatchdog);
        audioChunkWatchdog = null;
      }

      if (audioContext) {
        audioContext.close();
        audioContext = null;
      }

      if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
      }

      video.srcObject = null;
      placeholder.style.display = '';
      verdictOverlay.className = 'verdict-overlay';
      verdictOverlay.style.display = 'none';
      clearFaceBox();

      document.getElementById('btnStart').disabled = false;
      document.getElementById('btnStop').disabled = true;
      document.getElementById('statState').textContent = '⏹ Stopped';

      audioAnalysisBuffer = [];
      audioHeaderChunk = null;
      isAnalyzingAudio = false;
      addLog('analyzing', '⏹ Capture stopped');
      resetTabTitle();

      if (audioChunks.length > 0) {
        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        addLog('analyzing',
          '🎙️ Audio recorded — <a href="' + url + '" download="captured_audio.webm" style="color:#818CF8">Download</a>');
        audioChunks = [];
      }
    }

    async function analyzeFrame() {
      if (isAnalyzing || !stream) return;
      isAnalyzing = true;

      try {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        const blob = await new Promise(resolve =>
          canvas.toBlob(resolve, 'image/jpeg', 0.85)
        );

        if (!blob) {
          isAnalyzing = false;
          return;
        }

        verdictOverlay.className = 'verdict-overlay analyzing';
        verdictOverlay.textContent = '🔍 Analyzing...';

        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        const response = await fetch(API_BASE + '/predict', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || 'API error');
        }

        const result = await response.json();
        frameCount++;

        const v     = result.verdict || 'UNCERTAIN';
        const vl    = v.toLowerCase();
        const pFake = result.final_p_fake ?? 0.5;
        const band  = result.confidence_band || '';
        const bandLabel = BAND_LABELS[band] || '';
        const why   = result.advice ? result.advice.why : '';

        verdictOverlay.className = 'verdict-overlay ' + vl;
        verdictOverlay.innerHTML =
          '<div class="vo-main">' + getEmoji(vl) + ' ' + v + '</div>' +
          '<div class="vo-band">' + bandLabel + '</div>' +
          (why ? '<div class="vo-why">' + why + '</div>' : '');

        const bbox = result.signals ? result.signals.face_bbox : null;
        const faceFound = result.signals ? result.signals.face_found : false;
        drawFaceBox(bbox, v);

        renderVerdictCard(document.getElementById('videoVerdictArea'), result, 'video');

        if (vl === 'fake') fakeCount++;
        else if (vl === 'real') realCount++;
        totalConfidence += Math.max(pFake, 1 - pFake) * 100;

        document.getElementById('statFrames').textContent = frameCount;
        document.getElementById('statFakes').textContent = fakeCount;
        document.getElementById('statReals').textContent = realCount;
        document.getElementById('statConfidence').textContent =
          (totalConfidence / frameCount).toFixed(1) + '%';

        const detExtra = document.getElementById('detailsExtra');
        if (detExtra && result.models) {
          detExtra.innerHTML = result.models.map(m =>
            '<div class="detail-row"><span class="dl">' + m.name + '</span><span class="dv">' + (m.p_fake*100).toFixed(1) + '%' + (m.used ? '' : ' (unused)') + '</span></div>'
          ).join('') +
          (result.timing_ms ? '<div class="detail-row"><span class="dl">Timing</span><span class="dv">' + result.timing_ms.total + 'ms</span></div>' : '');
        }

        const time = new Date().toLocaleTimeString();
        addLog(vl, getEmoji(vl) + ' ' + time + ' — ' + v + ' &middot; ' + bandLabel);

        updateTabTitle(v, bandLabel);
        updateFavicon(v);
        const thumbDataUrl = createFaceThumbnail(bbox, v);
        updateWidget(v, pFake, band, time, thumbDataUrl, why);

      } catch (err) {
        console.error('Analysis error:', err);
        addLog('uncertain', '⚠️ Error: ' + err.message);
      }

      isAnalyzing = false;
    }

    async function startAudioRecording(mediaStream) {
      try {
        const audioTracks = mediaStream.getAudioTracks();
        if (audioTracks.length === 0) {
          _showAudioError('No audio tracks found in stream');
          return;
        }

        // Route through AudioContext to create a fresh MediaStream.
        // Chrome blocks MediaRecorder on raw getDisplayMedia audio
        // tracks inside iframes; piping via AudioContext bypasses this.
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === 'suspended') await audioCtx.resume();
        const source = audioCtx.createMediaStreamSource(new MediaStream(audioTracks));
        const dest   = audioCtx.createMediaStreamDestination();
        source.connect(dest);
        const recordableStream = dest.stream;

        const mimeTypes = [
          'audio/webm;codecs=opus',
          'audio/webm',
          'audio/ogg;codecs=opus',
          '',
        ];
        let chosenMime = '';
        for (const mt of mimeTypes) {
          if (!mt || MediaRecorder.isTypeSupported(mt)) { chosenMime = mt; break; }
        }

        const opts = chosenMime ? { mimeType: chosenMime } : undefined;
        mediaRecorder = new MediaRecorder(recordableStream, opts);
        audioChunks = [];
        audioAnalysisBuffer = [];
        audioHeaderChunk = null;

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            audioChunks.push(e.data);
            if (!audioHeaderChunk) {
              audioHeaderChunk = e.data;
            }
            audioAnalysisBuffer.push(e.data);
            if (audioAnalysisBuffer.length > AUDIO_BUFFER_MAX) {
              audioAnalysisBuffer.shift();
            }
          }
        };

        mediaRecorder.onerror = (e) => {
          console.error('MediaRecorder error:', e);
          _showAudioError('Audio recorder error');
        };

        mediaRecorder.start(1000);
        console.log('MediaRecorder started with mimeType:', mediaRecorder.mimeType);

        // If recorder starts but emits no usable chunks, tab audio is likely unavailable.
        // Common on macOS when sharing non-browser apps; fallback to mic if possible.
        if (audioChunkWatchdog) clearTimeout(audioChunkWatchdog);
        audioChunkWatchdog = setTimeout(() => {
          if (audioAnalysisBuffer.length === 0) {
            try {
              if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
            } catch (_) {}
            trySwitchToMicAudio('No tab audio detected');
          }
        }, 4500);
      } catch (err) {
        console.error('Audio recording failed:', err);
        _showAudioError('Audio recording failed: ' + (err.message || err));
      }
    }

    async function trySwitchToMicAudio(reason) {
      try {
        const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioStream = micStream;
        startAudioVisualization(micStream);
        startAudioRecording(micStream);
        document.getElementById('audioStatus').innerHTML =
          '<div class="audio-indicator"><div class="audio-bars">' +
          '<span></span><span></span><span></span><span></span><span></span>' +
          '</div>🎤 Switched to microphone (' + reason + ')</div>';
      } catch (micErr) {
        _showAudioError(
          reason + '. On macOS, tab capture records browser-tab audio only; for desktop apps, allow microphone.'
        );
      }
    }

    function _showAudioError(msg) {
      const el = document.getElementById('audioStatus');
      if (el) el.innerHTML = '<div style="font-size:12px;color:#F59E0B">⚠️ ' + msg + '</div>';
    }

    async function analyzeAudio() {
      if (isAnalyzingAudio || audioAnalysisBuffer.length < 3 || !stream) return;
      isAnalyzingAudio = true;

      try {
        const parts = audioHeaderChunk && audioAnalysisBuffer[0] !== audioHeaderChunk
          ? [audioHeaderChunk, ...audioAnalysisBuffer]
          : [...audioAnalysisBuffer];
        const blob = new Blob(parts, { type: 'audio/webm' });
        if (blob.size < 2000) { isAnalyzingAudio = false; return; }

        const fd = new FormData();
        fd.append('file', blob, 'live_audio.webm');

        const resp = await fetch(API_BASE + '/predict-audio', { method: 'POST', body: fd });
        if (!resp.ok) {
          const errBody = await resp.json().catch(() => ({}));
          const detail  = errBody.detail || ('HTTP ' + resp.status);
          console.warn('Audio API error:', detail);
          _showAudioError(detail);
          isAnalyzingAudio = false;
          return;
        }

        const result = await resp.json();

        updateAudioDisplay(result);
        updateWidgetAudio(result);

        const v    = result.verdict || 'UNCERTAIN';
        const vl2  = v.toLowerCase();
        const bandLabel2 = BAND_LABELS[result.confidence_band] || '';
        const time = new Date().toLocaleTimeString();
        addLog(vl2, '🎙 ' + time + ' — Audio ' + v + ' &middot; ' + bandLabel2);

      } catch (err) {
        console.error('Audio analysis error:', err);
        _showAudioError(err.message || 'Unknown error');
      }

      isAnalyzingAudio = false;
    }

    function updateAudioDisplay(result) {
      const statusEl = document.getElementById('audioStatus');
      if (!statusEl) return;
      renderVerdictCard(statusEl, result, 'audio');
    }

    function startAudioVisualization(mediaStream) {
      try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        if (audioContext.state === 'suspended') {
          audioContext.resume().catch(() => {});
        }
        const source = audioContext.createMediaStreamSource(mediaStream);
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 256;
        source.connect(analyserNode);
        drawAudioWaveform();
      } catch (err) {
        console.error('Audio viz failed:', err);
      }
    }

    function drawAudioWaveform() {
      if (!analyserNode) return;
      const bufferLength = analyserNode.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      function draw() {
        animFrameId = requestAnimationFrame(draw);
        analyserNode.getByteFrequencyData(dataArray);

        const w = audioCanvas.width = audioCanvas.clientWidth;
        const h = audioCanvas.height = audioCanvas.clientHeight;
        audioCtx2d.clearRect(0, 0, w, h);

        const barWidth = (w / bufferLength) * 2;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const barHeight = (dataArray[i] / 255) * h;
          const hue = 240 + (dataArray[i] / 255) * 60;
          audioCtx2d.fillStyle = 'hsla(' + hue + ', 70%, 60%, 0.8)';
          audioCtx2d.fillRect(x, h - barHeight, barWidth - 1, barHeight);
          x += barWidth;
        }
      }
      draw();
    }

    const BAND_LABELS = { HIGH: 'High confidence', MEDIUM: 'Medium confidence', LOW: 'Low confidence' };

    function getEmoji(pred) {
      switch (pred) {
        case 'fake': return '🚨';
        case 'real': return '✅';
        default: return '❓';
      }
    }

    function addLog(type, message) {
      const entry = document.createElement('div');
      entry.className = 'log-entry ' + type;
      entry.innerHTML = message;
      logContainer.prepend(entry);
      while (logContainer.children.length > 50) {
        logContainer.removeChild(logContainer.lastChild);
      }
    }

    function toggleDetails() {
      const toggle = document.getElementById('detailsToggle');
      const body   = document.getElementById('detailsBody');
      toggle.classList.toggle('open');
      body.classList.toggle('open');
    }

    function renderVerdictCard(container, result, mediaLabel) {
      const v     = result.verdict || 'UNCERTAIN';
      const vl    = v.toLowerCase();
      const band  = BAND_LABELS[result.confidence_band] || 'Checking...';
      const why   = result.advice ? result.advice.why : '';
      const steps = result.advice ? result.advice.next_steps || [] : [];
      const emoji = getEmoji(vl);
      const colors = { real: '#22C55E', fake: '#EF4444', uncertain: '#F59E0B' };
      const color  = colors[vl] || '#818CF8';

      const showSteps = mediaLabel === 'video' && v === 'UNCERTAIN';
      const stepsHtml = showSteps ? steps.map(s => '<li>' + s + '</li>').join('') : '';

      container.innerHTML =
        '<div style="text-align:center; padding:8px 0">' +
          '<div class="vc-icon">' + emoji + '</div>' +
          '<div class="vc-verdict" style="color:' + color + '">' + v + '</div>' +
          '<div class="vc-band" style="color:' + color + '">' + band + '</div>' +
        '</div>' +
        '<div class="vc-why">' + why + '</div>' +
        (stepsHtml ? '<ul class="vc-steps">' + stepsHtml + '</ul>' : '');
    }

    let widgetWindow = null;

    const WIDGET_CSS =
      '* { margin:0; padding:0; box-sizing:border-box; }' +
      'body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0F172A; color: #fff; height: 100vh; display: flex; flex-direction: column; overflow: hidden; user-select: none; }' +
      '.w-header { background: #1E293B; padding: 7px 12px; font-size: 10px; color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 1.2px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); flex-shrink: 0; }' +
      '.w-header .dot { width: 7px; height: 7px; border-radius: 50%; background: #818CF8; flex-shrink:0; }' +
      '.w-thumb-wrap { flex-shrink: 0; overflow: hidden; background: #000; border-bottom: 1px solid rgba(255,255,255,0.05); display: none; }' +
      '.w-thumb-wrap img { width: 100%; height: auto; display: block; }' +
      '.w-main { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 12px 16px; gap: 4px; transition: background 0.4s; text-align: center; }' +
      '.w-verdict { font-size: 26px; font-weight: 800; letter-spacing: 3px; transition: color 0.3s; }' +
      '.w-band { font-size: 11px; font-weight: 600; opacity: 0.75; }' +
      '.w-why { font-size: 10px; color: rgba(255,255,255,0.45); margin-top: 4px; line-height: 1.3; }' +
      '.w-divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 6px 0; width: 100%; display: none; }' +
      '.w-audio-row { display: none; align-items: center; gap: 6px; font-size: 11px; }' +
      '.w-audio-row .wa-icon { font-size: 14px; }' +
      '.w-audio-row .wa-verdict { font-weight: 700; letter-spacing: 1px; }' +
      '.w-audio-row .wa-band { font-size: 10px; opacity: 0.6; }';

    const WIDGET_HTML =
      '<div class="w-header"><span>🛡 Deepfake Detector</span><div class="dot" id="wDot"></div></div>' +
      '<div class="w-thumb-wrap" id="wThumbWrap"><img id="wThumb" alt=""></div>' +
      '<div class="w-main" id="wMain">' +
        '<div class="w-verdict" id="wVerdict" style="color:#818CF8">⏳ Waiting</div>' +
        '<div class="w-band" id="wBand" style="color:#818CF8">—</div>' +
        '<div class="w-why" id="wWhy"></div>' +
        '<hr class="w-divider" id="wAudioDivider">' +
        '<div class="w-audio-row" id="wAudioRow"><span class="wa-icon">🎙</span><span class="wa-verdict" id="wAudioVerdict">—</span><span class="wa-band" id="wAudioBand"></span></div>' +
      '</div>';

    function _setupWidgetDoc(doc) {
      const style = doc.createElement('style');
      style.textContent = WIDGET_CSS;
      doc.head.appendChild(style);
      doc.body.innerHTML = WIDGET_HTML;
    }

    async function toggleWidget() {
      if (widgetWindow && !widgetWindow.closed) {
        widgetWindow.close();
        widgetWindow = null;
        _refreshPipBtn();
        return;
      }

      if ('documentPictureInPicture' in window) {
        try {
          const pip = await window.documentPictureInPicture.requestWindow({
            width: 290, height: 340,
          });
          _setupWidgetDoc(pip.document);
          widgetWindow = pip;
          pip.addEventListener('pagehide', () => { widgetWindow = null; _refreshPipBtn(); });
          _refreshPipBtn();
          return;
        } catch (e) {}
      }

      const popup = window.open(
        '', 'deepfake-widget',
        'width=290,height=340,toolbar=no,menubar=no,location=no,status=no,resizable=yes'
      );
      if (!popup) {
        addLog('uncertain', '⚠️ Popup blocked — allow popups for this site to use the widget.');
        return;
      }
      popup.document.open();
      popup.document.write('<!DOCTYPE html><html><head><meta charset="UTF-8">' +
        '<title>Deepfake Widget</title>' +
        '<style>' + WIDGET_CSS + '</style></head><body>' + WIDGET_HTML + '</body></html>');
      popup.document.close();
      widgetWindow = popup;
      popup.addEventListener('beforeunload', () => { widgetWindow = null; _refreshPipBtn(); });
      _refreshPipBtn();
    }

    function _refreshPipBtn() {
      const btn = document.getElementById('btnPip');
      if (!btn) return;
      const open = widgetWindow && !widgetWindow.closed;
      btn.textContent = open ? '✕ Close Widget' : '⧉ Pop Out';
      btn.classList.toggle('active', open);
    }

    function updateWidget(verdict, pFake, band, time, thumbDataUrl, why) {
      if (!widgetWindow || widgetWindow.closed) return;
      const doc = widgetWindow.document;
      const colors = { REAL: '#22C55E', FAKE: '#EF4444', UNCERTAIN: '#F59E0B' };
      const bgTints = { REAL: 'rgba(34,197,94,0.07)', FAKE: 'rgba(239,68,68,0.10)', UNCERTAIN: 'rgba(245,158,11,0.08)' };
      const emojis = { REAL: '✅', FAKE: '🚨', UNCERTAIN: '❓' };
      const color  = colors[verdict]  || '#818CF8';
      const bgTint = bgTints[verdict] || 'transparent';
      const emoji  = emojis[verdict]  || '⏳';
      const bandLabel = BAND_LABELS[band] || '';

      const $ = id => doc.getElementById(id);
      if (!$('wVerdict')) return;

      const thumbWrap = $('wThumbWrap');
      const thumbImg  = $('wThumb');
      if (thumbWrap && thumbImg) {
        if (thumbDataUrl) {
          thumbImg.src            = thumbDataUrl;
          thumbWrap.style.display = 'block';
        } else {
          thumbWrap.style.display = 'none';
        }
      }

      $('wMain').style.background = bgTint;
      $('wVerdict').textContent   = emoji + ' ' + verdict;
      $('wVerdict').style.color   = color;
      $('wBand').textContent      = bandLabel;
      $('wBand').style.color      = color;
      const whyEl = $('wWhy');
      if (whyEl) whyEl.textContent = why || '';

      const dot = $('wDot');
      if (dot) {
        dot.style.background = color;
        dot.style.boxShadow  = '0 0 6px ' + color;
        setTimeout(() => { if (dot) dot.style.boxShadow = 'none'; }, 600);
      }
    }

    function updateWidgetAudio(result) {
      if (!widgetWindow || widgetWindow.closed) return;
      const doc = widgetWindow.document;
      const $ = id => doc.getElementById(id);
      if (!$('wAudioVerdict')) return;

      const verdict = result.verdict || 'UNCERTAIN';
      const band    = result.confidence_band || '';
      const colors  = { REAL: '#22C55E', FAKE: '#EF4444', UNCERTAIN: '#F59E0B' };
      const emojis  = { REAL: '✅', FAKE: '🚨', UNCERTAIN: '❓' };
      const color   = colors[verdict] || '#818CF8';
      const bandLabel = BAND_LABELS[band] || '';

      const divider = $('wAudioDivider');
      const row     = $('wAudioRow');
      if (divider) divider.style.display = 'block';
      if (row)     row.style.display     = 'flex';

      $('wAudioVerdict').textContent = emojis[verdict] + ' ' + verdict;
      $('wAudioVerdict').style.color = color;
      $('wAudioBand').textContent    = bandLabel;
      $('wAudioBand').style.color    = color;
    }

    function createFaceThumbnail(faceBox, verdict) {
      if (!canvas.width || !canvas.height) return null;

      const tw = 290, th = Math.round(290 * canvas.height / canvas.width);
      const tc = document.createElement('canvas');
      tc.width = tw; tc.height = th;
      const tx = tc.getContext('2d');

      tx.drawImage(canvas, 0, 0, tw, th);

      if (!faceBox) return tc.toDataURL('image/jpeg', 0.8);

      const colors = { REAL: '#22C55E', FAKE: '#EF4444', UNCERTAIN: '#F59E0B' };
      const color  = colors[verdict] || '#818CF8';

      const rx = faceBox.x * tw;
      const ry = faceBox.y * th;
      const rw = faceBox.w * tw;
      const rh = faceBox.h * th;

      tx.shadowColor  = color;
      tx.shadowBlur   = 10;
      tx.strokeStyle  = color;
      tx.lineWidth    = 2;
      tx.strokeRect(rx, ry, rw, rh);
      tx.shadowBlur   = 0;

      const cs = Math.min(rw, rh) * 0.20;
      tx.lineWidth   = 3;
      const corners  = [
        [rx,      ry,      cs,  0,  0, cs],
        [rx + rw, ry,     -cs,  0,  0, cs],
        [rx,      ry + rh, cs,  0,  0,-cs],
        [rx + rw, ry + rh,-cs,  0,  0,-cs],
      ];
      for (const [x, y, dx1, dy1, dx2, dy2] of corners) {
        tx.beginPath();
        tx.moveTo(x + dx1, y + dy1);
        tx.lineTo(x, y);
        tx.lineTo(x + dx2, y + dy2);
        tx.stroke();
      }

      return tc.toDataURL('image/jpeg', 0.80);
    }

    const _faviconCanvas = document.createElement('canvas');
    _faviconCanvas.width = _faviconCanvas.height = 32;

    function updateTabTitle(verdict, bandLabel) {
      const e = { REAL: '✅', FAKE: '🚨', UNCERTAIN: '❓' }[verdict] || '🔍';
      document.title = e + ' ' + verdict + ' — ' + (bandLabel || 'Checking') + ' — Deepfake Detector';
    }

    function updateFavicon(verdict) {
      const colors = { REAL: '#22C55E', FAKE: '#EF4444', UNCERTAIN: '#F59E0B' };
      const marks  = { REAL: '✓',       FAKE: '!',        UNCERTAIN: '?' };
      const color  = colors[verdict] || '#818CF8';
      const mark   = marks[verdict]  || '…';

      const c = _faviconCanvas;
      const x = c.getContext('2d');
      x.clearRect(0, 0, 32, 32);

      x.fillStyle = color;
      x.beginPath(); x.arc(16, 16, 15, 0, Math.PI * 2); x.fill();

      x.strokeStyle = 'rgba(255,255,255,0.25)';
      x.lineWidth = 1.5;
      x.beginPath(); x.arc(16, 16, 13, 0, Math.PI * 2); x.stroke();

      x.fillStyle = '#fff';
      x.font = 'bold 16px sans-serif';
      x.textAlign = 'center';
      x.textBaseline = 'middle';
      x.fillText(mark, 16, 17);

      const link = document.getElementById('pageFavicon');
      if (link) link.href = c.toDataURL('image/png');
    }

    function resetTabTitle() {
      document.title = 'Live Deepfake Detection';
      const link = document.getElementById('pageFavicon');
      if (link) link.href = '';
    }
  </script>
</body>
</html>''';

class LiveScreen extends StatefulWidget {
  const LiveScreen({super.key});

  @override
  State<LiveScreen> createState() => _LiveScreenState();
}

class _LiveScreenState extends State<LiveScreen> {
  static const _viewType = 'live-detection-blob';
  static bool _registered = false;

  @override
  void initState() {
    super.initState();
    if (!_registered) {
      final htmlContent = _liveHtmlTemplate.replaceAll(
        '__API_BASE__',
        ApiService.baseUrl,
      );

      ui_web.platformViewRegistry.registerViewFactory(_viewType, (int viewId) {
        return html.IFrameElement()
          ..srcdoc = htmlContent
          ..style.border = 'none'
          ..style.width = '100%'
          ..style.height = '100%'
          ..allow = 'camera *; microphone *; display-capture *; picture-in-picture *; autoplay *'
          ..setAttribute('allowfullscreen', 'true');
      });
      _registered = true;
    }
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        return Scaffold(
          backgroundColor: Colors.transparent, // Let the main layout background show through
          body: Padding(
            padding: EdgeInsets.only(
              left: constraints.maxWidth > 600 ? 40 : 24, 
              right: constraints.maxWidth > 600 ? 40 : 24,
              top: 80, // Match home_screen top padding
              bottom: 40,
            ),
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.4), // Glassmorphism base
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: Colors.white.withValues(alpha: 0.8), width: 1.5),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.05),
                    blurRadius: 24,
                    offset: const Offset(0, 8),
                  ),
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(24),
                child: const HtmlElementView(viewType: _viewType),
              ),
            ),
          ),
        );
      },
    );
  }
}
