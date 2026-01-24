"""
WebSocket server for real-time trade visualization.

Handles client connections, playback control, and data streaming.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Set, Optional

import aiohttp
from aiohttp import web

from .config import VisualizerConfig, DEFAULT_PORT
from .simulator import TradeSimulator
from .model_wrapper import load_model, get_device


class VisualizationServer:
    """
    WebSocket server for trade visualization.

    Handles:
    - HTML page serving
    - WebSocket connections for real-time updates
    - Playback control (play, pause, step, speed)
    """

    def __init__(
        self,
        simulator: TradeSimulator,
        config: VisualizerConfig,
        html_template: str
    ):
        """
        Initialize the server.

        Args:
            simulator: TradeSimulator instance
            config: Visualizer configuration
            html_template: HTML template string
        """
        self.simulator = simulator
        self.config = config
        self.html_template = html_template

        # Server state
        self.playing = False
        self.speed = 10
        self.clients: Set[web.WebSocketResponse] = set()

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.clients.add(ws)

        try:
            # Send initial state immediately on connect
            initial_update = self.simulator.step()
            if initial_update:
                await ws.send_str(json.dumps(initial_update))

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(ws, msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f'WebSocket error: {ws.exception()}')
        finally:
            self.clients.discard(ws)

        return ws

    async def _handle_message(self, ws: web.WebSocketResponse, data: str):
        """Handle incoming WebSocket message."""
        try:
            message = json.loads(data)
            cmd = message.get('command')

            if cmd == 'play':
                self.playing = True
            elif cmd == 'pause':
                self.playing = False
            elif cmd == 'step':
                update = self.simulator.step()
                if update:
                    await ws.send_str(json.dumps(update))
            elif cmd == 'next_trade':
                update = self.simulator.step_to_next_trade()
                if update:
                    await ws.send_str(json.dumps(update))
            elif cmd == 'reset':
                self.simulator.reset()
                self.playing = False
                # Send initial state after reset
                update = self.simulator.step()
                if update:
                    await ws.send_str(json.dumps(update))
            elif cmd == 'speed':
                self.speed = message.get('value', 10)
        except json.JSONDecodeError:
            print(f"Invalid JSON message: {data}")

    async def index_handler(self, request: web.Request) -> web.Response:
        """Serve the HTML page."""
        return web.Response(text=self.html_template, content_type='text/html')

    async def playback_loop(self):
        """Background loop for auto-playback."""
        while True:
            if self.playing and self.clients:
                update = self.simulator.step()
                if update:
                    msg = json.dumps(update)
                    # Send to all connected clients
                    for client in list(self.clients):
                        try:
                            await client.send_str(msg)
                        except Exception:
                            self.clients.discard(client)
                else:
                    # End of data
                    self.playing = False

            # Sleep based on speed
            await asyncio.sleep(1.0 / max(self.speed, 1))

    async def static_handler(self, request: web.Request) -> web.Response:
        """Serve static files (CSS, JS)."""
        path = request.match_info.get('path', '')
        static_dir = Path(__file__).parent.parent / 'static'
        file_path = static_dir / path

        if not file_path.exists() or not file_path.is_file():
            return web.Response(status=404, text="Not found")

        # Determine content type
        suffix = file_path.suffix.lower()
        content_types = {
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.html': 'text/html',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.svg': 'image/svg+xml',
        }
        content_type = content_types.get(suffix, 'text/plain')

        return web.Response(
            body=file_path.read_bytes(),
            content_type=content_type
        )

    async def run(self):
        """Run the server."""
        app = web.Application()
        app.router.add_get('/', self.index_handler)
        app.router.add_get('/ws', self.websocket_handler)
        app.router.add_get('/static/{path:.*}', self.static_handler)

        # Start playback loop
        asyncio.create_task(self.playback_loop())

        # Run server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.config.port)
        await site.start()

        print(f"\n{'='*60}")
        print(f" RL Trade Visualizer Running")
        print(f"{'='*60}")
        print(f" Strategy: {self.config.strategy_name}")
        print(f" Open: http://localhost:{self.config.port}")
        print(f" Press Ctrl+C to stop")
        print(f"{'='*60}\n")

        # Keep running
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass


async def run_server(config: VisualizerConfig, html_template: str):
    """
    Main entry point to run the visualization server.

    Args:
        config: Visualizer configuration
        html_template: HTML template string
    """
    import pandas as pd

    # Determine device
    device = get_device()
    print(f"Using {device.upper()} acceleration")

    # Load model
    print(f"Loading model from {config.model_path}...")
    strategy_dir = config.model_path.parent.parent
    model = load_model(config.model_path, strategy_dir, device)

    # Load data
    print(f"Loading price data from {config.price_data_path}...")
    price_data = pd.read_csv(config.price_data_path, index_col=0, parse_dates=True)

    print(f"Loading trades data from {config.trades_data_path}...")
    trades_data = pd.read_csv(config.trades_data_path)

    # Filter to OOS period
    price_data = price_data[
        (price_data.index >= config.oos_start) &
        (price_data.index <= config.oos_end)
    ]
    print(f"Loaded {len(price_data):,} bars, {len(trades_data):,} trades")

    # Create simulator
    simulator = TradeSimulator(price_data, trades_data, model, config, device)
    print(f"Trade entries mapped: {len(simulator.trade_entries):,}")

    # Create and run server
    server = VisualizationServer(simulator, config, html_template)
    await server.run()
