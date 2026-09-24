# Spatial HMI API Reference & WebSocket Schema

## 1. WebSocket Telemetry Protocol Schema (`HMIPacket`)

The Python perception engine broadcasts state updates over WebSockets (`ws://127.0.0.1:8765`) at 60 Hz.

### JSON Schema Example:
```json
{
  "packet_type": "HMI_STATE_UPDATE",
  "intent_state": "ACTIVE",
  "active_gesture": "GRAB",
  "intent_confidence": 0.92,
  "fps": 59.8,
  "latency_ms": 22.4,
  "timestamp": 1727191234.567,
  "command": {
    "command_type": "ROTATE_OBJECT",
    "interaction_state": "ACTIVE",
    "cursor_ndc": [0.15, -0.22],
    "delta_rotation": [-0.045, 0.012, 0.0],
    "delta_scale": 1.0,
    "delta_translation": [0.0, 0.0, 0.0],
    "target_node_id": null,
    "confidence": 0.92,
    "handedness": "Right",
    "is_pinch_active": false,
    "timestamp": 1727191234.567
  },
  "hands": [
    {
      "hand_id": 0,
      "handedness": "Right",
      "palm_center": [0.52, 0.48, 0.0],
      "pinch_confidence": 0.12,
      "detection_confidence": 0.97,
      "landmarks_normalized": [
        [0.0, 0.0, 0.0],
        [-0.35, 0.22, 0.0]
      ]
    }
  ]
}
```

---

## 2. Spatial Command Taxonomy

| Command Type | Trigger Condition | Payload Properties | Front-end Action |
| :--- | :--- | :--- | :--- |
| `IDLE` | Hand absent or at rest | `cursor_ndc: [0, 0]` | Standby / Auto-rotation |
| `HOVER` | Open hand / pointing without lock | `cursor_ndc` | Ray-cast node highlighting |
| `SELECT` | Point gesture confirmed | `cursor_ndc`, `target_node_id` | Select target node |
| `ROTATE_OBJECT` | Grab gesture confirmed + motion | `delta_rotation: [yaw, pitch, roll]` | Rotate 3D globe / camera |
| `SCALE_OBJECT` | Bimanual spread / contraction | `delta_scale: factor` | Zoom camera in / out |
| `TRANSLATE_NODE` | Pinch gesture confirmed + motion | `delta_translation: [dx, dy, dz]` | Translate selected 3D node |
| `RELEASE_OBJECT` | Transition to Open Palm / release | `cursor_ndc` | Deselect / release active node |
