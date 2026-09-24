/**
 * Main WebGL Spatial Visualizer Client.
 * Connects to Python HMI WebSocket server and translates spatial commands into 3D interactions.
 */

class SpatialHMIApp {
  constructor() {
    this.container = document.getElementById("canvas-container");
    this.wsStatusDot = document.getElementById("ws-status-dot");
    this.wsStatusText = document.getElementById("ws-status-text");
    this.stateBadge = document.getElementById("state-badge");
    this.actionLabel = document.getElementById("action-label");

    this._initThree();
    this._initGlobeAndNodes();
    this._initSubsystems();
    this._initWebSocket();
    this._initFallbackMouseControls();

    window.addEventListener("resize", () => this._onWindowResize());
    this.clock = new THREE.Clock();
    this._animate();
  }

  _initThree() {
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x07090e, 0.035);

    this.camera = new THREE.PerspectiveCamera(
      45,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    this.targetCameraDistance = 7.0;
    this.currentCameraDistance = 7.0;
    this.camera.position.set(0, 0, this.currentCameraDistance);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.container.appendChild(this.renderer.domElement);

    // Lighting setup
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x00f0ff, 1.6);
    dirLight1.position.set(5, 4, 6);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x3b82f6, 1.0);
    dirLight2.position.set(-6, -3, 4);
    this.scene.add(dirLight2);
  }

  _initGlobeAndNodes() {
    this.globe = new InteractiveGlobe(this.scene, 2.4);
    this.nodeManager = new SpatialNodeManager(this.globe);
  }

  _initSubsystems() {
    this.cursor = new SpatialCursor();
    this.debugOverlay = new DebugOverlay();
    this.activeManipulatedNode = null;
  }

  _initWebSocket(host = "127.0.0.1", port = 8765) {
    const wsUrl = `ws://${host}:${port}`;
    console.log(`Connecting to HMI Server at ${wsUrl}...`);

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.wsStatusDot.className = "pill-dot connected";
      this.wsStatusText.textContent = "CONNECTED (60Hz)";
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
      // Auto reconnect after 2 seconds
      setTimeout(() => this._initWebSocket(host, port), 2000);
    };

    this.ws.onerror = (err) => {
      console.warn("WebSocket error occurred. Retrying...");
      this.ws.close();
    };
  }

  _processHMIPacket(packet) {
    const cmd = packet.command;
    const state = packet.intent_state;
    const [ndcX, ndcY] = cmd.cursor_ndc || [0, 0];

    // 1. Update Debug Overlay
    this.debugOverlay.update(packet);

    // 2. Update UI State Badge
    this._updateStateBadge(state, packet.active_gesture);

    // 3. Update Spatial Cursor
    const pinchConf = packet.hands && packet.hands[0] ? packet.hands[0].pinch_confidence : 0;
    this.cursor.updateFromNDC(ndcX, ndcY, state, cmd.is_pinch_active, pinchConf);

    // 4. Spatial Ray-Casting for Node Hover
    const ndcVector = new THREE.Vector2(ndcX, ndcY);
    const hoveredNode = this.nodeManager.testRaycast(this.camera, ndcVector);

    // 5. Execute High-Level Spatial Commands
    switch (cmd.command_type) {
      case "HOVER":
        this.actionLabel.textContent = hoveredNode ? `Hovering: ${hoveredNode.label}` : "Exploring Space";
        break;

      case "SELECT":
        if (hoveredNode) {
          this.nodeManager.selectNode(hoveredNode);
          this.activeManipulatedNode = hoveredNode;
          this.actionLabel.textContent = `Selected: ${hoveredNode.label}`;
        }
        break;

      case "ROTATE_OBJECT":
        if (cmd.delta_rotation) {
          const [deltaYaw, deltaPitch] = cmd.delta_rotation;
          this.globe.applyRotationDelta(deltaYaw, deltaPitch);
          this.actionLabel.textContent = "Manipulating Globe Rotation";
        }
        break;

      case "SCALE_OBJECT":
        if (cmd.delta_scale) {
          this.targetCameraDistance = Math.max(4.0, Math.min(12.0, this.targetCameraDistance / cmd.delta_scale));
          this.actionLabel.textContent = "Bimanual Zooming";
        }
        break;

      case "TRANSLATE_NODE":
        if (this.activeManipulatedNode && cmd.delta_translation) {
          const [dx, dy] = cmd.delta_translation;
          this.nodeManager.translateNode(this.activeManipulatedNode, dx, dy);
          this.actionLabel.textContent = `Translating: ${this.activeManipulatedNode.label}`;
        } else if (hoveredNode) {
          this.activeManipulatedNode = hoveredNode;
          this.nodeManager.selectNode(hoveredNode);
        }
        break;

      case "RELEASE_OBJECT":
        this.activeManipulatedNode = null;
        this.actionLabel.textContent = "Interaction Released";
        break;

      case "IDLE":
      default:
        this.actionLabel.textContent = "Awaiting Hand Gesture...";
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
      isMouseDown = true;
      prevMouse = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener("mousemove", (e) => {
      // Simulate NDC cursor
      const ndcX = (e.clientX / window.innerWidth) * 2 - 1;
      const ndcY = -(e.clientY / window.innerHeight) * 2 + 1;

      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        this.cursor.updateFromNDC(ndcX, ndcY, "OBSERVING", false, 0);
        this.nodeManager.testRaycast(this.camera, new THREE.Vector2(ndcX, ndcY));

        if (isMouseDown) {
          const dx = (e.clientX - prevMouse.x) * 0.005;
          const dy = (e.clientY - prevMouse.y) * 0.005;
          this.globe.applyRotationDelta(dx, dy);
          prevMouse = { x: e.clientX, y: e.clientY };
        }
      }
    });

    window.addEventListener("mouseup", () => {
      isMouseDown = false;
    });

    window.addEventListener("wheel", (e) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
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
