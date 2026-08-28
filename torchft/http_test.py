# Copyright (c) Panocular AI.
#
# Tests for torchft/http.py's address-family selection.
#
# Regression: _IPv6HTTPServer hardcoded AF_INET6, so torchft.Manager died with
# `OSError: [Errno 97] Address family not supported by protocol` on every HPC compute
# node that ships with IPv6 disabled (TU Darmstadt lcluster13: login nodes have IPv6,
# GPU nodes do not). Three servers share the class, so all three crashed.
#
# The module is loaded straight from its file rather than imported as `torchft.http`,
# so these tests need neither the package's compiled extension nor a built venv.
import errno
import importlib.util
import pathlib
import socket
import sys
import unittest
from unittest import mock

_HTTP_PY = pathlib.Path(__file__).with_name("http.py")

# This package contains a module named `http.py`, so with the package directory on
# sys.path (which is what happens when this file is run directly) `import http.server`
# inside it would resolve to that sibling instead of the STDLIB http package. Drop the
# package dir so the module under test can import its own dependency.
_HERE = str(pathlib.Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p and p not in (".", _HERE)]


def _load_http_module():
    spec = importlib.util.spec_from_file_location("_torchft_http_under_test", _HTTP_PY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _host_has_ipv6() -> bool:
    try:
        socket.socket(socket.AF_INET6, socket.SOCK_STREAM).close()
        return True
    except OSError:
        return False


class AddressFamilyTest(unittest.TestCase):
    def test_falls_back_to_ipv4_when_af_inet6_is_unsupported(self) -> None:
        """THE regression: an IPv6-less node must still get a working server."""
        real_socket = socket.socket

        def no_ipv6(family, *args, **kwargs):
            if family == socket.AF_INET6:
                raise OSError(errno.EAFNOSUPPORT,
                              "Address family not supported by protocol")
            return real_socket(family, *args, **kwargs)

        with mock.patch("socket.socket", side_effect=no_ipv6):
            module = _load_http_module()
            self.assertFalse(module._ipv6_supported())
            self.assertEqual(module._IPv6HTTPServer.address_family, socket.AF_INET)

    def test_ignores_has_ipv6_when_the_socket_still_fails(self) -> None:
        """`socket.has_ipv6` is True even where the kernel refuses AF_INET6 (observed
        on lcluster13's compute nodes), so the flag alone must not decide."""
        real_socket = socket.socket

        def no_ipv6(family, *args, **kwargs):
            if family == socket.AF_INET6:
                raise OSError(errno.EAFNOSUPPORT, "nope")
            return real_socket(family, *args, **kwargs)

        with mock.patch("socket.has_ipv6", True), \
             mock.patch("socket.socket", side_effect=no_ipv6):
            self.assertFalse(_load_http_module()._ipv6_supported())

    def test_skips_the_probe_when_python_lacks_ipv6(self) -> None:
        with mock.patch("socket.has_ipv6", False):
            self.assertFalse(_load_http_module()._ipv6_supported())

    @unittest.skipUnless(_host_has_ipv6(), "host has no IPv6 to prefer")
    def test_prefers_ipv6_when_available(self) -> None:
        """Unchanged behavior on a dual-stack host: still AF_INET6, which also serves
        v4-mapped clients."""
        module = _load_http_module()
        self.assertTrue(module._ipv6_supported())
        self.assertEqual(module._IPv6HTTPServer.address_family, socket.AF_INET6)

    def test_server_actually_binds_under_the_selected_family(self) -> None:
        """End to end: construct the real server the way torchft does and confirm it
        binds. This is what threw [Errno 97] before the fix."""
        from http.server import BaseHTTPRequestHandler

        module = _load_http_module()
        server = module._IPv6HTTPServer(("", 0), BaseHTTPRequestHandler)
        try:
            self.assertGreater(server.socket.getsockname()[1], 0)
        finally:
            server.server_close()


class ManagerBindAddressTest(unittest.TestCase):
    """`Manager` hands its bind string to the RUST ManagerServer, so an IPv6 wildcard
    on an IPv6-less host fails as `RuntimeError: ... (os error 97)`. torchft/manager.py
    imports torch, which is not always available where these unit tests run, so pin the
    exact expression from the source rather than importing the module.
    """

    EXPECTED = 'bind = f"[::]:{port}" if _ipv6_supported() else f"0.0.0.0:{port}"'

    def test_manager_chooses_the_wildcard_by_ipv6_support(self) -> None:
        manager_py = pathlib.Path(__file__).with_name("manager.py").read_text()
        self.assertIn(self.EXPECTED, manager_py,
                      "manager.py must pick its bind wildcard via _ipv6_supported(); a "
                      "hardcoded [::] breaks every host with IPv6 disabled")

    def test_both_wildcards_are_bindable_shapes(self) -> None:
        """Sanity-check the two strings the line above can produce: the one matching
        this host must actually bind."""
        module = _load_http_module()
        family = module._IPv6HTTPServer.address_family
        wildcard = "" if family == socket.AF_INET6 else "0.0.0.0"
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.bind((wildcard, 0))
            self.assertGreater(sock.getsockname()[1], 0)
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()
