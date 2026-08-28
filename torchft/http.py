# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import socket
from http.server import ThreadingHTTPServer


def _ipv6_supported() -> bool:
    """Whether this host can actually open an IPv6 socket.

    `socket.has_ipv6` only reports that PYTHON was built with IPv6 support, which is
    true even on hosts where the kernel has IPv6 disabled -- so the flag alone is not
    enough and we open a throwaway socket to find out for real.
    """
    if not socket.has_ipv6:
        return False
    try:
        socket.socket(socket.AF_INET6, socket.SOCK_STREAM).close()
    except OSError:
        return False
    return True


class _IPv6HTTPServer(ThreadingHTTPServer):
    # Panocular: prefer IPv6 -- on a dual-stack host it also serves v4-mapped clients,
    # which is why upstream hardcodes it -- but FALL BACK to IPv4 on a host with no
    # IPv6 at all. Hardcoding AF_INET6 killed every training run on TU Darmstadt
    # lcluster13's GPU nodes with `OSError: [Errno 97] Address family not supported by
    # protocol`, thrown the moment torchft.Manager built its checkpoint transport --
    # HPC compute nodes commonly ship with IPv6 disabled while the LOGIN nodes have it
    # enabled, so nothing upstream of the job sees a problem (debugged live
    # 2026-08-28). Three servers share this class (checkpoint transport, parameter
    # server, async_diloco relay), so all three needed the one fix.
    #
    # Probed once at import: a host does not gain or lose IPv6 mid-process, and
    # `address()` reports a hostname rather than a bracketed literal, so the family
    # switch needs no change in the addresses we advertise.
    # pyrefly: ignore [bad-override-mutable-attribute]
    address_family: socket.AddressFamily = (
        socket.AF_INET6 if _ipv6_supported() else socket.AF_INET
    )
    request_queue_size: int = 1024
