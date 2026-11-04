#!/usr/bin/env python3
"""
Simple HTTP Server to preview WeatherForecast-SEA Docsify documentation locally.
Usage:
    python3 docs/serve.py [port]
"""

import http.server
import socketserver
import os
import sys

DEFAULT_PORT = 3000

def run_server(port=DEFAULT_PORT):
    # Set directory to docs/
    docs_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(docs_dir)

    Handler = http.server.SimpleHTTPRequestHandler
    Handler.extensions_map.update({
        '.md': 'text/plain; charset=utf-8',
        '.html': 'text/html; charset=utf-8',
        '.js': 'application/javascript; charset=utf-8',
        '.css': 'text/css; charset=utf-8',
        '.json': 'application/json; charset=utf-8'
    })

    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True

    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"🌦️ WeatherForecast-SEA Docsify Documentation Server")
            print(f"📂 Serving directory: {docs_dir}")
            print(f"🚀 Local preview URL: http://localhost:{port}")
            print(f"🛑 Press Ctrl+C to stop the server.")
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            alt_port = port + 1
            print(f"⚠️ Port {port} is in use, trying port {alt_port}...")
            run_server(alt_port)
        else:
            raise e

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    run_server(port)
