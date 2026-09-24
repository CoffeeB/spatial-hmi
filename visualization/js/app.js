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

    // Full-Screen Camera Screen Feed Element
    this.cameraScreenFeed = document.getElementById("camera-screen-feed");
    this.cameraContainer = document.getElementById("camera-container");
    this.cameraFeed = document.getElementById("camera-feed");
    this.cameraPlaceholder = document.getElementById("camera-placeholder");
    this.cameraFeedOverlay = document.getElementById("camera-feed-overlay");
    this.cameraHandCount = document.getElementById("camera-hand-count");
    this.camGestureBadge = document.getElementById("cam-gesture-badge");
    this.camPinchBar = document.getElementById("cam-pinch-bar");
    this.cameraExpandBtn = document.getElementById("camera-expand-btn");

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
    this._initCameraControls();
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
    this.cameraFocusTarget = new THREE.Vector3(0, 0, 0);
    this.focusTarget = null;

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
    this.heldNode = null;
    this.isZoomedOut = false;
    this.hasActiveVisionHand = false;
  }

  _initCameraControls() {
    if (this.cameraExpandBtn && this.cameraContainer) {
      this.cameraExpandBtn.addEventListener("click", () => {
        this.cameraContainer.classList.toggle("expanded");
      });
    }
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

    // 1. Update Full-Screen Camera Live Screen Feed
    if (packet.video_frame_b64) {
      const srcUrl = "data:image/jpeg;base64," + packet.video_frame_b64;
      if (this.cameraScreenFeed) {
        if (this.cameraScreenFeed.src !== srcUrl) {
          this.cameraScreenFeed.src = srcUrl;
        }
        this.cameraScreenFeed.style.opacity = "1";
      }
      if (this.cameraFeed && this.cameraFeed.src !== srcUrl) {
        this.cameraFeed.src = srcUrl;
      }
      if (this.cameraPlaceholder) this.cameraPlaceholder.classList.add("hidden");
      if (this.cameraFeedOverlay) this.cameraFeedOverlay.classList.remove("hidden");
    }

    if (this.cameraHandCount) {
      this.cameraHandCount.textContent = numHands === 1 ? "1 HAND" : `${numHands} HANDS`;
    }

    if (this.camGestureBadge) {
      this.camGestureBadge.textContent = packet.active_gesture || "NONE";
    }

    // 2. Update Debug Overlay
    this.debugOverlay.update(packet);

    // 3. Update UI State Badge
    this._updateStateBadge(state, packet.active_gesture);

    // 4. Update Spatial Cursor & Pinch Meter
    const pinchConf = packet.hands && packet.hands[0] ? packet.hands[0].pinch_confidence : 0;
    if (this.camPinchBar) {
      this.camPinchBar.style.width = `${Math.round(pinchConf * 100)}%`;
    }
    this.cursor.updateFromNDC(ndcX, ndcY, state, cmd.is_pinch_active, pinchConf);

    // 5. Spatial Ray-Casting for Node Hover & Pointing
    const ndcVector = new THREE.Vector2(ndcX, ndcY);
    const hoveredNode = this.nodeManager.testRaycast(this.camera, ndcVector);

    // 6. Node Holding: One hand can hold ANY node (parent or sub-node) with Pinch or Grab
    // 6. Node Holding & Pinching: Pinch or Grab holds any node
    const isPinchOrGrab = numHands === 1 && (
      packet.active_gesture === "PINCH" ||
      packet.active_gesture === "GRAB" ||
      cmd.is_pinch_active ||
      cmd.command_type === "FOCUS_NODE"
    );

    if (isPinchOrGrab) {
      if (!this.heldNode) {
        // Lock onto hovered node or the nearest visible front-facing node
        const target = hoveredNode || this.nodeManager.getNearestFrontFacingNode(this.camera, ndcVector);
        if (target) {
          this.heldNode = target;
          this.nodeManager.setNodeHolding(this.heldNode, true);
          if (!this.heldNode.isSubnode && !this.nodeManager.expandedCluster) {
            this.nodeManager.selectNode(this.heldNode);
          }
          const worldPos = this.nodeManager.getNodeWorldPosition(this.heldNode);
          if (worldPos) {
            this.focusTarget = worldPos.clone().multiplyScalar(0.35);
          }
          this.targetCameraDistance = 4.4;
        }
      }

      if (this.heldNode) {
        // Node is held: follow hand movement
        const [dx, dy] = cmd.delta_translation || [0, 0];
        if (Math.abs(dx) > 0.0001 || Math.abs(dy) > 0.0001) {
          this.nodeManager.translateNode(this.heldNode, dx, dy);
        }
        this.actionLabel.textContent = `Holding: ${this.heldNode.label}`;
        this.gesturePrompt.textContent = "Node is held. Move hand to reposition, Open Palm to anchor.";
        return; // Prevent world spin/zoom while holding a node
      }
    } else {
      if (this.heldNode) {
        this.nodeManager.setNodeHolding(this.heldNode, false);
        this.heldNode = null;
        this.actionLabel.textContent = "Node Anchored";
      }
    }

    // 7. Two-Hand World Manipulation (Rotating, Expanding, Shrinking)
    if (numHands >= 2 || cmd.command_type === "BIMANUAL_NAV") {
      // Rotation: bounded strictly to hands edge-to-edge / top-to-bottom (NOT 360 spin)
      if (cmd.delta_rotation) {
        const [deltaYaw, deltaPitch, deltaRoll] = cmd.delta_rotation;
        this.globe.applyBoundedHandRotation(deltaYaw, deltaPitch, deltaRoll || 0);
      }

      // Translation
      if (cmd.delta_translation) {
        const [dx, dy] = cmd.delta_translation;
        if (Math.abs(dx) > 0.001 || Math.abs(dy) > 0.001) {
          this.globe.applyTranslationDelta(dx * 1.5, dy * 1.5);
        }
      }

      // Expanding & Shrinking
      const scale = cmd.delta_scale;
      if (scale && Math.abs(scale - 1.0) > 0.001) {
        this.targetCameraDistance = Math.max(3.8, Math.min(11.5, this.targetCameraDistance / scale));
        if (scale > 1.002) {
          this.actionLabel.textContent = "Two Hands Apart → Expanding Globe";
        } else if (scale < 0.998) {
          this.actionLabel.textContent = "Two Hands Together → Shrinking Globe";
        }
      } else {
        this.actionLabel.textContent = "Two-Hand Bounded Manipulation (Edge-to-Edge)";
      }
      this.gesturePrompt.textContent = "Hands rotate edge-to-edge & top-to-bottom. Move apart to expand, together to shrink.";
      return;
    }

    // 8. One-Hand Gestures: Swiping (360 spin), Grabbing (Zoom out), Hand Release (Expand)
    const isSwipeGesture = [
      "SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN"
    ].includes(packet.active_gesture);

    if (isSwipeGesture || cmd.command_type === "ROTATE_OBJECT") {
      if (state !== "RELEASE" && state !== "RELEASING" && state !== "IDLE") {
        if (cmd.delta_rotation) {
          const [deltaYaw, deltaPitch] = cmd.delta_rotation;
          // Swiping triggers gradual 360-degree rotation vertical / horizontal
          this.globe.applySwipeSpin(deltaYaw, deltaPitch);
          const dir = packet.active_gesture ? packet.active_gesture.replace("SWIPE_", "") : "360";
          this.actionLabel.textContent = `👋 360° Swipe Spin: ${dir}`;
          this.gesturePrompt.textContent = "Swipe intensity determines 360° spin velocity.";
          return;
        }
      }
    }

    // Hand Release / Open / Spread when zoomed out: expands globe back!
    const isReleaseOrOpen = (
      packet.active_gesture === "OPEN_PALM" ||
      packet.active_gesture === "SPREAD_FINGERS" ||
      state === "RELEASE" ||
      cmd.command_type === "RELEASE_OBJECT" ||
      cmd.command_type === "WORLD_ZOOM_MIN"
    );

    if (this.isZoomedOut && isReleaseOrOpen) {
      this.targetCameraDistance = 5.2; // Smooth gradual expanded view
      this.isZoomedOut = false;
      this.actionLabel.textContent = "Hand Released → Expanding Globe";
      this.gesturePrompt.textContent = "Globe expanded back into active view.";
      return;
    }

    // Grab (closed fist in space) → Zoom out overview
    if (packet.active_gesture === "GRAB" || cmd.command_type === "WORLD_ZOOM_MAX") {
      this.focusTarget = null;
      this.targetCameraDistance = 11.2;
      this.isZoomedOut = true;
      this.actionLabel.textContent = "Closed Fist Grab → Zoom Out Overview";
      this.gesturePrompt.textContent = "Globe zoomed out. Release hand to expand back in.";
      return;
    }

    // Spread Hand in normal mode → Close view
    if (packet.active_gesture === "SPREAD_FINGERS" || cmd.command_type === "WORLD_ZOOM_MIN") {
      this.targetCameraDistance = 3.8;
      this.actionLabel.textContent = "Spread Hand → Close View";
      this.gesturePrompt.textContent = "Camera easing to closest allowable distance.";
      return;
    }

    // Hover / Pointing
    if (hoveredNode) {
      this.actionLabel.textContent = hoveredNode.isSubnode
        ? `Hover Sub-Node: ${hoveredNode.label} (${hoveredNode.metric})`
        : `Hover Node: ${hoveredNode.label}`;
      this.gesturePrompt.textContent = "Pinch or Grab to hold this node. Point highlights.";
      return;
    }

    // Release / Steady
    if (cmd.command_type === "RELEASE_OBJECT" || state === "RELEASE" || state === "IDLE") {
      this.globe.freeze();
      if (this.focusTarget && packet.active_gesture === "OPEN_PALM") {
        this.focusTarget = null;
        this.targetCameraDistance = 6.8;
        this.nodeManager.collapseCluster();
      }
      this.actionLabel.textContent = this.nodeManager.expandedCluster
        ? `Observing Cluster: ${this.nodeManager.expandedCluster.label}`
        : "Observing Hologram";
      this.gesturePrompt.textContent = "Open palm release. Hologram steady.";
      return;
    }

    // Default Idle
    this.globe.freeze();
    this.actionLabel.textContent = "Observing Hologram";
    this.gesturePrompt.textContent = "Pinch/Grab node to hold, Swipe for 360° spin, 2 hands to rotate & zoom.";
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

    // Smooth camera distance interpolation (easing)
    this.currentCameraDistance += (this.targetCameraDistance - this.currentCameraDistance) * 0.045;
    this.camera.position.z = this.currentCameraDistance;

    // Smooth focus mode camera orientation
    if (this.focusTarget) {
      this.cameraFocusTarget.lerp(this.focusTarget, 0.08);
      this.camera.lookAt(this.cameraFocusTarget);
    } else {
      this.cameraFocusTarget.lerp(new THREE.Vector3(0, 0, 0), 0.08);
      this.camera.lookAt(this.cameraFocusTarget);
    }

    this.globe.update(deltaTime);
    this.nodeManager.update(deltaTime);

    // ── Compute globe's projected screen-space circle every frame ──────────
    // Project world origin (globe center) → screen pixel
    const globeCenter3D = new THREE.Vector3(0, 0, 0);
    const projected = globeCenter3D.clone().project(this.camera);
    const screenCX = (projected.x * 0.5 + 0.5) * window.innerWidth;
    const screenCY = (1 - (projected.y * 0.5 + 0.5)) * window.innerHeight;

    // Project a point on the globe equator to get radius in pixels.
    // Use globe.radius (2.3). Apply a slight margin (0.92) so cursor
    // stays comfortably inside the wireframe boundary.
    const equatorPoint = new THREE.Vector3(this.globe.radius * 0.92, 0, 0);
    const projectedEdge = equatorPoint.clone().project(this.camera);
    const edgeScreenX = (projectedEdge.x * 0.5 + 0.5) * window.innerWidth;
    const screenR = Math.abs(edgeScreenX - screenCX);

    this.cursor.setGlobeBounds({ cx: screenCX, cy: screenCY, r: screenR });
    // ────────────────────────────────────────────────────────────────────────

    this.renderer.render(this.scene, this.camera);
  }

}

// Bootstrap application on DOM load
window.addEventListener("DOMContentLoaded", () => {
  window.hmiApp = new SpatialHMIApp();
});

