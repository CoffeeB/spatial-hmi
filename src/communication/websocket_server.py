"""
Asynchronous WebSocket server for real-time spatial HMI communication.
"""

import asyncio
import json
import threading
from typing import Optional, Set
import websockets
from websockets.server import WebSocketServerProtocol

from src.communication.protocol import HMIPacket
from src.utils.logging_config import setup_logger

logger = setup_logger("websocket_server")


class HMIWebSocketServer:
    """
    Asynchronous WebSocket server running on a dedicated asyncio event loop in a background thread.
    Broadcasts HMIPackets to all connected front-end visualizer clients.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.server = None
        self.thread: Optional[threading.Thread] = None
        self.running: bool = False

    def start(self):
        """Starts the server in a background daemon thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="WebSocketServerThread")
        self.thread.start()
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

    def _run_loop(self):
        """Event loop runner."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start_server())
        self.loop.run_forever()

    async def _start_server(self):
        """Initializes the websockets server."""
        self.server = await websockets.serve(self._handler, self.host, self.port)

    async def _handler(self, websocket: WebSocketServerProtocol, path: str):
        """Handles client connection lifecycle."""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"Visualizer client connected from {client_addr}. Total clients: {len(self.clients)}")
        try:
            async for message in websocket:
                # Handle client incoming messages (e.g., ping or mode switch)
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"Visualizer client disconnected from {client_addr}. Total clients: {len(self.clients)}")

    def broadcast_packet(self, packet: HMIPacket):
        """
        Thread-safe method called from perception thread to broadcast telemetry packet to all connected clients.
        """
        if not self.running or not self.loop or not self.clients:
            return

        payload = packet.model_dump_json()
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.loop)

    async def _broadcast(self, payload: str):
        """Broadcasts payload to all active clients."""
        if not self.clients:
            return
        # Broadcast concurrently across active connections
        tasks = [client.send(payload) for client in self.clients]
        await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self):
        """Stops the WebSocket server and cleans up asyncio loop."""
        self.running = False
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        logger.info("WebSocket server stopped.")
