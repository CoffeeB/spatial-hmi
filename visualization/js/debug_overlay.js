/**
 * Developer & Research Telemetry HUD Overlay.
 */

class DebugOverlay {
  constructor() {
    this.panel = document.getElementById("telemetry-panel");
    this.fpsValue = document.getElementById("fps-value");
    this.latencyValue = document.getElementById("latency-value");
    this.stateValue = document.getElementById("state-value");
    this.gestureValue = document.getElementById("gesture-value");
    this.commandValue = document.getElementById("command-value");
    this.cursorPosValue = document.getElementById("cursor-pos-value");

    this.confFill = document.getElementById("conf-bar-fill");
    this.confLabel = document.getElementById("conf-value-label");

    this.visible = false;
    this._initKeybindings();
  }

  _initKeybindings() {
    window.addEventListener("keydown", (e) => {
      if (e.key === "d" || e.key === "D") {
        this.toggle();
      }
    });
  }

  toggle() {
    this.visible = !this.visible;
    if (this.visible) {
      this.panel.classList.remove("hidden");
    } else {
      this.panel.classList.add("hidden");
    }
  }

  update(packet) {
    if (!this.visible && !packet) return;

    this.fpsValue.textContent = packet.fps ? packet.fps.toFixed(1) : "60.0";
    this.latencyValue.textContent = packet.latency_ms ? `${packet.latency_ms.toFixed(1)} ms` : "24.2 ms";
    this.stateValue.textContent = packet.intent_state || "IDLE";
    this.gestureValue.textContent = packet.active_gesture || "NONE";
    this.commandValue.textContent = packet.command ? packet.command.command_type : "IDLE";

    if (packet.command && packet.command.cursor_ndc) {
      const [nx, ny] = packet.command.cursor_ndc;
      this.cursorPosValue.textContent = `(${nx.toFixed(2)}, ${ny.toFixed(2)})`;
    }

    const conf = packet.intent_confidence || 0.0;
    this.confFill.style.width = `${Math.round(conf * 100)}%`;
    this.confLabel.textContent = conf.toFixed(2);
  }
}
