"""Local-only access boundary and bounded HTTP bodies (including chunked uploads)."""
from __future__ import annotations

import ipaddress
import json
import logging
import time
from urllib.parse import urlsplit

from starlette.responses import JSONResponse

MAX_BODY_BYTES = 4 * 1024 * 1024


class LocalOnlyMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])}
        host = headers.get("host", "")
        try:
            hostname = urlsplit("http://" + host).hostname
        except ValueError:
            hostname = None
        peer = (scope.get("client") or ("",))[0]
        local_peer = peer == "testclient"
        try:
            local_peer |= ipaddress.ip_address(peer).is_loopback
        except ValueError:
            pass
        allowed_host = hostname in {"localhost", "127.0.0.1", "::1"} or (peer == "testclient" and hostname == "testserver")
        error = None
        if not local_peer or not allowed_host:
            error = (403, "ForestMind is a local single-user service. Remote and untrusted Host requests are disabled.")
        if scope["method"] not in {"GET", "HEAD", "OPTIONS"}:
            origin = headers.get("origin")
            if origin and origin.rstrip("/") != f"{scope.get('scheme','http')}://{host}":
                error = (403, "Cross-origin changes to this local workspace are blocked")
            if headers.get("sec-fetch-site") == "cross-site":
                error = (403, "Cross-site changes to this local workspace are blocked")
        try:
            if int(headers.get("content-length", "0")) > MAX_BODY_BYTES:
                error = (413, "Request exceeds the 4 MiB upload limit")
        except ValueError:
            error = (400, "Invalid Content-Length")
        if error:
            return await JSONResponse({"detail": error[1]}, status_code=error[0])(scope, receive, send)
        # Buffer only mutating requests, with an independent cap for streaming bodies.
        if scope["method"] not in {"GET", "HEAD", "OPTIONS"}:
            messages, size = [], 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                size += len(message.get("body", b""))
                if size > MAX_BODY_BYTES:
                    return await JSONResponse({"detail": "Request exceeds the 4 MiB upload limit"}, status_code=413)(scope, receive, send)
                messages.append(message)
                if not message.get("more_body", False):
                    break
            async def bounded_receive():
                return messages.pop(0) if messages else await receive()
        else:
            bounded_receive = receive
        start = time.monotonic()
        code = 500
        async def observed_send(message):
            nonlocal code
            if message["type"] == "http.response.start":
                code = message["status"]
            await send(message)
        try:
            await self.app(scope, bounded_receive, observed_send)
        finally:
            logging.getLogger("forestmind.http").info(json.dumps({"event": "http_request", "method": scope["method"],
                "path": scope["path"][:200], "status": code, "duration_ms": round((time.monotonic() - start) * 1000)}))
