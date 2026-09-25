"""
Main entry point for running the Spatial HMI perception pipeline and WebSocket server.
"""

import argparse
import http.server
import os
from pathlib import Path
import socketserver
import sys
import threading
import time
import cv2
import numpy as np

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera.frame_capture import CameraStream
from src.camera.zoom_controller import CameraZoomController
from src.communication.protocol import HandTelemetry, HMIPacket
from src.communication.websocket_server import HMIWebSocketServer
from src.interaction.engine import InteractionEngine
from src.perception.hand_detector import HandDetector
from src.utils.config_loader import load_config
from src.utils.logging_config import setup_logger

logger = setup_logger("hmi_main")


def start_http_server(directory: Path, port: int = 8080):
    """Starts a background HTTP server serving the WebGL visualizer."""
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format, *args):
            pass  # Silence noisy static asset HTTP requests

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), Handler) as httpd:
        logger.info(f"Visualizer Web App running at: http://localhost:{port}/index.html")
        httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Spatial HMI Perception & WebSocket Server")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--dev", action="store_true", help="Open local OpenCV developer debug window")
    parser.add_argument("--http-port", type=int, default=8080, help="Port for static WebGL visualizer")
    parser.add_argument("--synthetic", action="store_true", help="Run in headless synthetic evaluation mode")
    parser.add_argument("--record", type=str, default=None, help="Path to output JSONL file for recording session telemetry")
    parser.add_argument("--no-auto-zoom", action="store_true", help="Disable dynamic auto-zoom & auto-focus hand tracking")
    parser.add_argument("--max-zoom", type=float, default=None, help="Maximum digital zoom level for distance hand tracking (default: 3.5)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Prepare recording file if requested
    record_file = None
    if args.record:
        record_path = Path(args.record)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_file = open(record_path, "w", encoding="utf-8")
        logger.info(f"Server-side telemetry recording active: {record_path}")

    # 1. Start Static HTTP Server for Three.js WebGL Visualization
    vis_dir = Path(__file__).parent.parent / "visualization"
    http_thread = threading.Thread(
        target=start_http_server, args=(vis_dir, args.http_port), daemon=True, name="HttpServerThread"
    )
    http_thread.start()

    # 2. Start WebSocket Server
    ws_server = HMIWebSocketServer(host=config.communication.host, port=config.communication.port)
    ws_server.start()

    # 3. Initialize Perception & Interaction Subsystems
    detector = HandDetector(
        max_num_hands=config.perception.max_num_hands,
        min_detection_confidence=config.perception.min_detection_confidence,
        min_tracking_confidence=config.perception.min_tracking_confidence,
        model_complexity=config.perception.model_complexity,
    )
    engine = InteractionEngine(config)

    # 4. Start Camera Stream
    camera = CameraStream(
        device_index=config.camera.device_index,
        width=config.camera.width,
        height=config.camera.height,
        target_fps=config.camera.target_fps,
        auto_reconnect=config.camera.auto_reconnect,
    )

    if not args.synthetic:
        camera.start()

    # 5. Initialize Dynamic Auto-Zoom & Auto-Focus Controller
    auto_zoom_enabled = getattr(config.camera, "auto_zoom", True) and not args.no_auto_zoom
    max_zoom_val = args.max_zoom if args.max_zoom is not None else getattr(config.camera, "max_zoom", 3.5)
    zoom_controller = CameraZoomController(
        enabled=auto_zoom_enabled,
        min_zoom=getattr(config.camera, "min_zoom", 1.0),
        max_zoom=max_zoom_val,
        target_hand_scale=getattr(config.camera, "target_hand_scale", 0.32),
        zoom_speed=getattr(config.camera, "zoom_speed", 0.10),
        pan_speed=getattr(config.camera, "pan_speed", 0.12),
        hold_frames=getattr(config.camera, "hold_frames", 8),
    )
    if auto_zoom_enabled:
        logger.info(f"Dynamic Hand Auto-Zoom & Auto-Focus active (max zoom: {max_zoom_val:.1f}x).")

    logger.info("Spatial HMI System running. Press Ctrl+C in terminal or 'q' in debug window to exit.")
    logger.info(f"Open your browser to http://localhost:{args.http_port}/index.html for the Gestura Level 0 Finger State Laboratory.")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            loop_start = time.time()

            if args.synthetic:
                # Synthetic blank frame for headless testing
                frame = np.zeros((config.camera.height, config.camera.width, 3), dtype=np.uint8)
                ret = True
                capture_time = loop_start
                zoomed_frame = frame
            else:
                ret, frame, capture_time = camera.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Crop and dynamic auto-zoom/focus
                zoomed_frame, _ = zoom_controller.crop_and_resize(
                    frame,
                    out_w=min(config.camera.width, 1280),
                    out_h=min(config.camera.height, 720),
                )

            # Process Perception on zoomed/focused frame
            hand_states, annotated_frame = detector.process_frame(zoomed_frame, timestamp=capture_time)

            # High-speed fallback check: if no hands found in zoomed crop, but currently zoomed in > 1.15x
            if not args.synthetic and len(hand_states) == 0 and zoom_controller.current_zoom > 1.15:
                wide_states, _ = detector.process_frame(frame, timestamp=capture_time)
                if len(wide_states) > 0:
                    zoom_controller.notify_full_frame_detection(wide_states)
                    zoomed_frame, _ = zoom_controller.crop_and_resize(
                        frame,
                        out_w=min(config.camera.width, 1280),
                        out_h=min(config.camera.height, 720),
                    )
                    hand_states, annotated_frame = detector.process_frame(zoomed_frame, timestamp=capture_time)

            # Update zoom controller with observed hand states for continuous smooth auto-framing
            zoom_controller.update(hand_states, timestamp=capture_time)

            # Process Interaction Engine (FSM + Smoothing + Command Mapping)
            command, intent_ctx, primary_hand = engine.process_hands(hand_states, timestamp=capture_time)

            # Build Telemetry Packet
            now = time.time()
            latency_ms = (now - capture_time) * 1000.0

            telemetry_hands = []
            for h in hand_states:
                norm_pts = [[float(p[0]), float(p[1]), float(p[2])] for p in h.normalized_landmarks_array]
                telemetry_hands.append(
                    HandTelemetry(
                        hand_id=h.hand_id,
                        handedness=h.handedness,
                        palm_center=h.palm_center,
                        palm_velocity=h.palm_velocity,
                        pinch_confidence=h.pinch_confidence,
                        detection_confidence=h.detection_confidence,
                        landmarks_normalized=norm_pts,
                        finger_states=h.finger_states.as_dict() if getattr(h, "finger_states", None) else {},
                        finger_details=h.finger_states.as_details_dict() if getattr(h, "finger_states", None) else {},
                        orientation_angles=h.orientation_angles,
                        hand_scale_ref=float(h.hand_scale_ref),
                        hand_pose_id=h.derived_pose.pose_id.value if getattr(h, "derived_pose", None) else "UNKNOWN",
                        hand_pose_name=h.derived_pose.canonical_name if getattr(h, "derived_pose", None) else "NONE",
                        pose_predicates=h.derived_pose.satisfied_predicates if getattr(h, "derived_pose", None) else [],
                        finger_config_summary=h.derived_pose.configuration.summary() if getattr(h, "derived_pose", None) else "",
                        palm_facing=getattr(h, "palm_facing", "PALM"),
                    )
                )

            # Compress annotated frame to base64 JPEG for browser PiP feed
            video_b64 = None
            if annotated_frame is not None and not args.synthetic:
                feed_h, feed_w = annotated_frame.shape[:2]
                out_feed_w = 640
                out_feed_h = int(round(640 * (feed_h / feed_w)))
                small_frame = cv2.resize(annotated_frame, (out_feed_w, out_feed_h), interpolation=cv2.INTER_AREA)
                ret_enc, buf = cv2.imencode(".jpg", small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                if ret_enc:
                    import base64
                    video_b64 = base64.b64encode(buf).decode("ascii")

            packet = HMIPacket(
                command=command,
                intent_state=intent_ctx.state.value,
                active_gesture=intent_ctx.active_gesture.value,
                candidate_gesture=intent_ctx.candidate_gesture.value if hasattr(intent_ctx, "candidate_gesture") else "NONE",
                dominant_hand=command.dominant_hand,
                modifier_hand=command.modifier_hand,
                intent_confidence=intent_ctx.intent_confidence,
                hands=telemetry_hands,
                fps=float(fps),
                latency_ms=float(latency_ms),
                video_frame_b64=video_b64,
                camera_zoom=float(zoom_controller.current_zoom),
                camera_zoom_tracking=bool(zoom_controller.is_tracking),
                timestamp=now,
            )

            # Broadcast via WebSocket
            ws_server.broadcast_packet(packet)

            # Record to disk if server-side recording enabled
            if record_file is not None:
                record_file.write(packet.model_dump_json() + "\n")
                record_file.flush()

            # Developer GUI Mode
            if args.dev and annotated_frame is not None:
                # Add HUD text
                cv2.putText(
                    annotated_frame,
                    f"State: {intent_ctx.state.value} | Intent Conf: {intent_ctx.intent_confidence:.2f}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 120),
                    2,
                )
                cv2.putText(
                    annotated_frame,
                    f"Command: {command.command_type.value} | FPS: {fps:.1f}",
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 255),
                    2,
                )
                cv2.imshow("Spatial HMI - Perception Debug", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Calculate Pipeline FPS
            elapsed = time.time() - prev_time
            if elapsed >= 0.5:
                fps = 1.0 / max(time.time() - loop_start, 1e-4)
                prev_time = time.time()

            # Target 60 Hz loop rate
            duration = time.time() - loop_start
            target_delay = 1.0 / 60.0
            if duration < target_delay:
                time.sleep(target_delay - duration)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    finally:
        if record_file is not None:
            record_file.close()
            logger.info("Recording file cleanly closed.")
        if not args.synthetic:
            camera.stop()
        detector.close()
        ws_server.stop()
        if args.dev:
            cv2.destroyAllWindows()
        logger.info("Spatial HMI cleanly shut down.")


if __name__ == "__main__":
    main()
