# FORK-DELTA — PanocularAI/torchft

What this fork adds on top of upstream `meta-pytorch/torchft`, why it cannot all live
outside the fork, and which patches are candidates to send upstream.

Measured vs upstream base `3a8d174` (2026-09-29, just past v0.2.0): **13 files, +2502 / −16.**

> The async DiLoCo / HeLoCo algorithms **used to live here** (+2701 lines plus ~2200 of
> tests). They moved to `panoengine.decentralized` in panofabric-engine, because they were
> leaf modules: no torchtitan references, subclassing nothing from torchft, importing only
> two small private helpers, and nothing here imported them back. What remains is what
> genuinely cannot leave — see "Why these cannot leave" below.

## The cost model

A rebase conflicts only on files **both sides touched**. Added files never conflict. So the
carrying cost of this fork is not the 2,502 added lines — it is the **16 deleted lines
across 6 modified files**. Shrinking that set is the only thing that makes upstream syncs
cheap; the new modules are free to keep in-tree forever.

### Modified — the entire conflict surface

| File | Δ | What we changed |
|---|---|---|
| `torchft/manager.py` | +478 / −9 | rank0 synchronization: only rank 0 of a replica participates in the pseudo-gradient exchange. This is what makes islands of *different sizes* train together. Also binds the ManagerServer to `0.0.0.0` on hosts without IPv6. |
| `torchft/local_sgd.py` | +74 / −2 | rank0 synchronization's side of fragment sync: rank-0 state-dict gather/scatter and `scatter_grads()` after each sync. |
| `torchft/parameter_server.py` | +32 / −4 | `bind_host` / `advertise_host` (`$TORCHFT_PS_ADVERTISE_HOST`) so peers behind NAT can reach it. The engine's async DiLoCo imports `_resolve_advertise_host`. |
| `torchft/http.py` | +32 / −1 | `_ipv6_supported()`: the checkpoint HTTP server falls back to IPv4 on hosts without IPv6. |
| `src/lighthouse.rs` | +32 / −0 | `/status.json`, a machine-readable twin of `/status` (quorum id, participants, max step). |
| `torchft/local_sgd_test.py` | +4 / −0 | The DiLoCo tests' autospec'd `Manager` needs the `_rank0_synchronization_only` attribute `local_sgd.py` reads. |

### Added — never conflicts, stays here

| File | Lines | What it is |
|---|---|---|
| `torchft/semi_async_diloco.py` | 331 | Semi-synchronous DiLoCo. |
| `torchft/semi_async_heloco.py` | 217 | Semi-synchronous HeLoCo. |
| `semi_async_*_test.py` (2 files) | 780 | Tests for the above. |
| `train_semi_async_*.py` (2 files) | 391 | Standalone drivers. |
| `torchft/http_test.py` | 131 | Tests for the IPv4 fallback in `http.py` and `manager.py`. |

## Why these cannot leave the fork

Two reasons, both about the language rather than about tidiness:

1. **You cannot add methods to a class from outside it.** The rank0-synchronization work is
   +478 lines of NEW methods on `Manager` (`_allreduce_rank0`, `_async_quorum_rank0`,
   `scatter_grads`, …). Subclassing would not do: upstream constructs `Manager` itself.
2. **You cannot safely subclass a private class out-of-tree.**
   `_SemiAsyncDiLoCoFragment` extends `local_sgd._StreamingDiLoCoFragment`, so every upstream
   refactor of that private class would break it with no way to fix both sides atomically.

This is also the test for anything added here in future: if it only *composes* public
torchft classes, it belongs in panofabric-engine, not in this fork.

## Upstreaming candidates

Each merged PR permanently deletes fork surface. Ordered by value:

| Patch | Size | Note |
|---|---|---|
| IPv4 fallback in `http.py` + `manager.py` | small | Hosts without IPv6 (common on HPC compute nodes) crash in `Manager()` before step 1. Small, obviously correct, independent of everything else here, and upstream just took the matching store-address fix (#335) — the easiest win. |
| `parameter_server.py` `bind_host` / `advertise_host` | small | Needed on any NAT'd multi-host deployment; nothing Panocular-specific in it. |
| rank0 synchronization in `manager.py` | +475 | Long shot, highest value. It is a heterogeneity feature Meta plausibly wants, and it is the difference between "`pip install torchft` works" and "fork required forever". Propose it even if it stalls. |

**Target: 6 modified files → 1.** Then the next upstream sync is minutes, not a branch.

Nothing in this repo blocks on a PR landing. The fork already carries every patch and keeps
carrying it; each merge is simply a free deletion whenever it happens.

## Consumers

This fork is pinned by SHA (never by branch) from
[panofabric-engine](https://github.com/PanocularAI/panofabric-engine)'s `[train]` extra.
A moving ref in a published dist is not reproducible and breaks outright when the branch is
deleted — which is exactly what happened to the old `@async-diloco` ref.

**Build note for consumers:** this package builds via maturin, so installing it from source
requires a Rust toolchain, `protoc` 32.0, and CPython ≤ 3.13 (pyo3 0.24's ceiling). Publishing
per-(CPython, platform) wheels on tag is the single highest-value thing we could do for
outside users, and it would remove Rust from every consumer including the cloud-node bootstrap.
