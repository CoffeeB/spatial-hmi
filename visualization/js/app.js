/**
 * Main WebGL Spatial Visualizer Client.
 * Connects to Python HMI WebSocket server and translates spatial commands into 3D interactions.
 */

class SpatialHMIApp {
  constructor() {
    this.container = document.getElementById("canvas-container");
    this.wsStatusDot = document.getElementById("ws-status-dot");
    this.wsStatusText = document.getElementById("ws-status-text");
    this.controlModeBadge = document.getElementById("control-mode-badge");
    this.stateBadge = document.getElementById("state-badge");
    this.actionLabel = document.getElementById("action-label");
    this.gesturePrompt = document.getElementById("gesture-prompt");

    // Full-Screen AR Camera Feed Element
    this.arCameraFeed = document.getElementById("ar-camera-feed");

    // Cluster Card Elements
    this.clusterCard = document.getElementById("cluster-card");
    this.clusterCloseBtn = document.getElementById("cluster-close-btn");

    // Recording Elements
    this.recordBtn = document.getElementById("record-btn");
    this.recordBtnText = document.getElementById("record-btn-text");
    this.exportBtn = document.getElementById("export-btn");
    this.recordingStatsText = document.getElementById("recording-stats-text");
    this.telemetryRecFrames = document.getElementById("telemetry-rec-frames");
    this.telemetryRecSize = document.getElementById("telemetry-rec-size");

    // Recording State
    this.isRecording = false;
    this.recordedPackets = [];
    this.recordStartTime = null;
    this.recordingTimerId = null;

    this._initThree();
    this._initGlobeAndNodes();
    this._initSubsystems();
    this._initClusterHandlers();
    this._initRecordingHandlers();
    this._initKeyboardShortcuts();
    this._initWebSocket();
    this._initFallbackMouseControls();

    window.addEventListener("resize", () => this._onWindowResize());
    this.clock = new THREE.Clock();
    this._animate();
  }

  _initThree() {
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(
      45,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    this.targetCameraDistance = 6.8;
    this.currentCameraDistance = 6.8;
    this.camera.position.set(0, 0, this.currentCameraDistance);

    // Transparent WebGL Renderer for AR Hologram overlay
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setClearColor(0x000000, 0.0);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.container.appendChild(this.renderer.domElement);

    // Hologram Lighting
    const ambientLight = new THREE.AmbientLight(0x00f0ff, 0.6);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x00f0ff, 2.0);
    dirLight1.position.set(5, 4, 6);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 1.2);
    dirLight2.position.set(-6, -3, 4);
    this.scene.add(dirLight2);
  }

  _initGlobeAndNodes() {
    this.globe = new InteractiveGlobe(this.scene, 2.3);
    this.nodeManager = new SpatialNodeManager(this.globe);
  }

  _initSubsystems() {
    this.cursor = new SpatialCursor();
    this.debugOverlay = new DebugOverlay();
    this.activeManipulatedNode = null;
    this.hasActiveVisionHand = false;
  }

  _initClusterHandlers() {
    if (this.clusterCloseBtn) {
      this.clusterCloseBtn.addEventListener("click", () => {
        this.nodeManager.collapseCluster();
      });
    }
  }

  _initKeyboardShortcuts() {
    window.addEventListener("keydown", (e) => {
      const key = e.key.toLowerCase();
      if (key === "d") {
        this.debugOverlay.toggle();
      } else if (key === "r") {
        this._toggleRecording();
      } else if (e.key === "Escape") {
        this.nodeManager.collapseCluster();
      }
    });
  }

  _initRecordingHandlers() {
    if (this.recordBtn) {
      this.recordBtn.addEventListener("click", () => this._toggleRecording());
    }
    if (this.exportBtn) {
      this.exportBtn.addEventListener("click", () => this._exportDataset());
    }
  }

  _toggleRecording() {
    if (!this.isRecording) {
      // Start Recording
      this.isRecording = true;
      this.recordedPackets = [];
      this.recordStartTime = Date.now();
      this.recordBtn.classList.add("recording");
      this.recordBtnText.textContent = "STOP RECORDING";
      if (this.recordingBanner) this.recordingBanner.classList.remove("hidden");
      this.exportBtn.classList.add("hidden");

      this.recordingTimerId = setInterval(() => {
        const elapsedSec = Math.floor((Date.now() - this.recordStartTime) / 1000);
        const mins = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
        const secs = String(elapsedSec % 60).padStart(2, "0");
        const frameCount = this.recordedPackets.length;
        if (this.recordingStatsText) {
          this.recordingStatsText.textContent = `REC: ${mins}:${secs} | ${frameCount} frames`;
        }
        if (this.telemetryRecFrames) {
          this.telemetryRecFrames.textContent = frameCount;
          const kbSize = (JSON.stringify(this.recordedPackets).length / 1024).toFixed(1);
          this.telemetryRecSize.textContent = `${kbSize} KB`;
        }
      }, 250);
    } else {
      // Stop Recording
      this.isRecording = false;
      clearInterval(this.recordingTimerId);
      this.recordBtn.classList.remove("recording");
      this.recordBtnText.textContent = "START RECORDING";
      if (this.recordingBanner) this.recordingBanner.classList.add("hidden");

      if (this.recordedPackets.length > 0) {
        this.exportBtn.classList.remove("hidden");
        this.actionLabel.textContent = `Recording Complete (${this.recordedPackets.length} frames saved)`;
      }
    }
  }

  _exportDataset() {
    if (this.recordedPackets.length === 0) return;

    const dataset = {
      metadata: {
        session_id: `hmi_session_${Date.now()}`,
        recorded_at: new Date().toISOString(),
        total_frames: this.recordedPackets.length,
        system: "Spatial HMI Perception & Intent Engine",
        landmarks_format: "21 3D Normalized Coordinates (x, y, z)",
      },
      frames: this.recordedPackets,
    };

    const blob = new Blob([JSON.stringify(dataset, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `spatial_hmi_dataset_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  _initWebSocket(host = "127.0.0.1", port = 8765) {
    const wsUrl = `ws://${host}:${port}`;
    console.log(`Connecting to HMI Server at ${wsUrl}...`);

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.wsStatusDot.className = "pill-dot connected";
      this.wsStatusText.textContent = "CONNECTED (60Hz)";
      if (this.controlModeBadge) {
        this.controlModeBadge.textContent = "VISION GESTURE ENGINE";
        this.controlModeBadge.className = "pill control-mode-badge mode-vision";
      }
      console.log("Connected to Spatial HMI WebSocket Server.");
    };

    this.ws.onmessage = (event) => {
      try {
        const packet = JSON.parse(event.data);
        this._processHMIPacket(packet);
      } catch (err) {
        console.error("Error parsing HMI packet:", err);
      }
    };

    this.ws.onclose = () => {
      this.wsStatusDot.className = "pill-dot disconnected";
      this.wsStatusText.textContent = "DISCONNECTED";
      if (this.controlModeBadge) {
        this.controlModeBadge.textContent = "MOUSE FALLBACK";
        this.controlModeBadge.className = "pill control-mode-badge";
      }
      // Auto reconnect after 2 seconds
      setTimeout(() => this._initWebSocket(host, port), 2000);
    };

    this.ws.onerror = (err) => {
      console.warn("WebSocket error occurred. Retrying...");
      this.ws.close();
    };
  }

  _processHMIPacket(packet) {
    // Session Recording Collection (omits base64 image for clean lightweight telemetry dataset)
    if (this.isRecording) {
      const recFrame = {
        timestamp: packet.timestamp,
        intent_state: packet.intent_state,
        active_gesture: packet.active_gesture,
        intent_confidence: packet.intent_confidence,
        command: packet.command,
        hands: packet.hands,
        fps: packet.fps,
        latency_ms: packet.latency_ms,
      };
      this.recordedPackets.push(recFrame);
    }

    const cmd = packet.command;
    const state = packet.intent_state;
    const numHands = packet.hands ? packet.hands.length : 0;
    this.hasActiveVisionHand = numHands > 0;
    const [ndcX, ndcY] = cmd.cursor_ndc || [0, 0];

    // 1. Update Full-Screen AR Camera Perception Live Feed
    if (packet.video_frame_b64 && this.arCameraFeed) {
      const srcUrl = "data:image/jpeg;base64," + packet.video_frame_b64;
      if (this.arCameraFeed.src !== srcUrl) {
        this.arCameraFeed.src = srcUrl;
      }
    }

    // 2. Update Debug Overlay
    this.debugOverlay.update(packet);

    // 3. Update UI State Badge
    this._updateStateBadge(state, packet.active_gesture);

    // 4. Update Spatial Cursor & Pinch Meter
    const pinchConf = packet.hands && packet.hands[0] ? packet.hands[0].pinch_confidence : 0;
    this.cursor.updateFromNDC(ndcX, ndcY, state, cmd.is_pinch_active, pinchConf);

    // 5. Spatial Ray-Casting for Node Hover & Pointing
    const ndcVector = new THREE.Vector2(ndcX, ndcY);
    const hoveredNode = this.nodeManager.testRaycast(this.camera, ndcVector);

    // If actively pointing at a node, auto-select & expand its sub-node cluster
    if (packet.active_gesture === "POINT" && hoveredNode) {
      if (hoveredNode.isSubnode) {
        this.nodeManager.selectSubnode(hoveredNode);
        this.activeManipulatedNode = hoveredNode;
        this.actionLabel.textContent = `Pointing at Sub-Node: ${hoveredNode.label}`;
      } else if (this.nodeManager.expandedCluster !== hoveredNode) {
        this.nodeManager.selectNode(hoveredNode);
        this.activeManipulatedNode = hoveredNode;
        this.actionLabel.textContent = `Pointing: Expanded ${hoveredNode.label}`;
        this.gesturePrompt.textContent = "Cluster opened! Pinch to manipulate sub-nodes, or spread hands to zoom.";
      }
    }

    // 6. Execute High-Level Spatial Commands
    switch (cmd.command_type) {
      case "HOVER":
        if (hoveredNode) {
          this.actionLabel.textContent = hoveredNode.isSubnode
            ? `Hovering Sub-Node: ${hoveredNode.label} (${hoveredNode.metric})`
            : `Hovering Cluster: ${hoveredNode.label}`;
          this.gesturePrompt.textContent = "POINT at node to zoom into cluster, or PINCH to manipulate.";
        } else {
          this.actionLabel.textContent = "Exploring Hologram Space";
          this.gesturePrompt.textContent = "Point at glowing nodes to zoom into clusters, or make a fist to rotate.";
        }
        break;

      case "SELECT":
        if (hoveredNode) {
          if (hoveredNode.isSubnode) {
            this.nodeManager.selectSubnode(hoveredNode);
          } else {
            this.nodeManager.selectNode(hoveredNode);
          }
          this.activeManipulatedNode = hoveredNode;
          this.actionLabel.textContent = `Selected: ${hoveredNode.label}`;
          this.gesturePrompt.textContent = "Node locked. PINCH and drag to reposition in 3D orbit.";
        }
        break;

      case "ROTATE_OBJECT":
        if (cmd.delta_rotation) {
          const [deltaYaw, deltaPitch] = cmd.delta_rotation;
          this.globe.applyRotationDelta(deltaYaw, deltaPitch);
          this.actionLabel.textContent = "Rotating Hologram";
          this.gesturePrompt.textContent = "Move closed fist across screen to rotate. Open hand to release.";
        }
        break;

      case "SCALE_OBJECT":
        if (cmd.delta_scale) {
          this.targetCameraDistance = Math.max(3.6, Math.min(11.0, this.targetCameraDistance / cmd.delta_scale));
          this.actionLabel.textContent = "Spatial Zooming";
          this.gesturePrompt.textContent = "Spread two hands apart to zoom in, bring closer to zoom out.";
        }
        break;

      case "TRANSLATE_NODE":
        if (this.activeManipulatedNode && cmd.delta_translation) {
          const [dx, dy] = cmd.delta_translation;
          this.nodeManager.translateNode(this.activeManipulatedNode, dx, dy);
          this.actionLabel.textContent = `Manipulating: ${this.activeManipulatedNode.label}`;
          this.gesturePrompt.textContent = "Holding node. Release pinch to anchor new spatial position.";
        } else if (hoveredNode) {
          this.activeManipulatedNode = hoveredNode;
          if (hoveredNode.isSubnode) {
            this.nodeManager.selectSubnode(hoveredNode);
          } else {
            this.nodeManager.selectNode(hoveredNode);
          }
        }
        break;

      case "RELEASE_OBJECT":
        this.activeManipulatedNode = null;
        this.actionLabel.textContent = "Interaction Released";
        this.gesturePrompt.textContent = "Hand opened. Hologram steady.";
        break;

      case "IDLE":
      default:
        this.actionLabel.textContent = "Awaiting Hand Gesture...";
        this.gesturePrompt.textContent = "Show your hand to the camera to interact with holographic nodes";
        break;
    }
  }

  _updateStateBadge(state, activeGesture) {
    this.stateBadge.textContent = state;
    this.stateBadge.className = `pill state-badge state-${state.toLowerCase()}`;
  }

  _initFallbackMouseControls() {
    let isMouseDown = false;
    let prevMouse = { x: 0, y: 0 };

    window.addEventListener("mousedown", (e) => {
      // Don't trigger 3D mouse drag if clicking UI buttons
      if (e.target.closest(".top-bar") || e.target.closest(".camera-container") || e.target.closest(".telemetry-panel")) {
        return;
      }
      isMouseDown = true;
      prevMouse = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener("mousemove", (e) => {
      // If hands are detected by computer vision, vision has primary control
      if (this.hasActiveVisionHand) return;

      const ndcX = (e.clientX / window.innerWidth) * 2 - 1;
      const ndcY = -(e.clientY / window.innerHeight) * 2 + 1;

      this.cursor.updateFromNDC(ndcX, ndcY, "OBSERVING", false, 0);
      this.nodeManager.testRaycast(this.camera, new THREE.Vector2(ndcX, ndcY));

      if (isMouseDown) {
        const dx = (e.clientX - prevMouse.x) * 0.005;
        const dy = (e.clientY - prevMouse.y) * 0.005;
        this.globe.applyRotationDelta(dx, dy);
        prevMouse = { x: e.clientX, y: e.clientY };
      }
    });

    window.addEventListener("mouseup", () => {
      isMouseDown = false;
    });

    window.addEventListener("wheel", (e) => {
      if (!this.hasActiveVisionHand) {
        this.targetCameraDistance += e.deltaY * 0.005;
        this.targetCameraDistance = Math.max(4.0, Math.min(12.0, this.targetCameraDistance));
      }
    });
  }

  _onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    const deltaTime = this.clock.getDelta();

    // Smooth camera distance interpolation
    this.currentCameraDistance += (this.targetCameraDistance - this.currentCameraDistance) * 0.1;
    this.camera.position.z = this.currentCameraDistance;

    this.globe.update(deltaTime);
    this.nodeManager.update(deltaTime);

    this.renderer.render(this.scene, this.camera);
  }
}

// Bootstrap application on DOM load
window.addEventListener("DOMContentLoaded", () => {
  window.hmiApp = new SpatialHMIApp();
});

