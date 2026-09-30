"""Optional macOS NIC binding in pool-owned Colab CLI and keep-alive processes.

Do not install globally or change system network settings.
"""

import os
import socket
import sys

interface = os.environ.get("COLAB_POOL_INTERFACE")
if interface:
    try:
        if sys.platform != "darwin":
            raise RuntimeError("Colab pool interface binding requires macOS")
        index = socket.if_nametoindex(interface)
        # Bind at the socket layer to cover urllib auth, requests transfers,
        # and Jupyter websockets, without changing their library internals.
        # macOS netinet/in.h defines IP_BOUND_IF=25.
        original_getaddrinfo = socket.getaddrinfo
        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex

        def ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

        def bound_connect(self, address):
            if self.family == socket.AF_INET:
                self.setsockopt(socket.IPPROTO_IP, 25, index)
            return original_connect(self, address)

        def bound_connect_ex(self, address):
            if self.family == socket.AF_INET:
                self.setsockopt(socket.IPPROTO_IP, 25, index)
            return original_connect_ex(self, address)

        socket.getaddrinfo = ipv4_getaddrinfo
        socket.socket.connect = bound_connect
        socket.socket.connect_ex = bound_connect_ex
    except Exception as error:  # noqa: BLE001 - Python otherwise ignores sitecustomize failures
        print(f"Colab pool network setup failed: {error}", file=sys.stderr, flush=True)
        os._exit(78)
