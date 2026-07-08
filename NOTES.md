# Issue #3506 — `OSError: libtvm.so: cannot open shared object file`

## 1. Root cause

`libtvm.so` is **not** part of the `mlc-llm` wheel. It is shipped by the separate
`mlc-ai` / TVM package. This is deliberate: `ci/task/build_lib.sh` runs
`auditwheel repair` with `--exclude libtvm --exclude libtvm_runtime --exclude
libtvm_ffi ...`, so the mlc-llm wheel only bundles `libmlc_llm.so` /
`libmlc_llm_module.so`, both of which link against `libtvm.so` at load time.

At import, `python/mlc_llm/base.py` loads `libmlc_llm.so` via `ctypes.CDLL`. When the
matching `mlc-ai` package is missing, incomplete, or its CUDA variant does not match
(the exact situation reported for the CUDA 13.0 nightly wheels), the dynamic linker
cannot resolve the `libtvm.so` dependency and `ctypes.CDLL` raises a bare
`OSError: libtvm.so: cannot open shared object file: No such file or directory`.

The underlying missing-file problem lives in the `mlc-ai`/TVM wheel packaging, which is
built and published **outside this repository** — there is no `libtvm.so`-producing
build in mlc-llm to fix here. What *is* in this repo's control is the loader's behavior:
it turns a resolvable-diagnosis situation ("your `mlc-ai` install is missing/mismatched")
into an opaque `OSError` that gives the user no path forward. That unhelpful failure is
the fixable defect.

## 2. The fix and why

Added `load_lib(path)` to `python/mlc_llm/libinfo.py` and routed `base.py`'s load
through it. It wraps `ctypes.CDLL` and, on `OSError`, re-raises a `RuntimeError` that:

- preserves the original loader error (kept as `__cause__` and in the message) for
  debugging, and
- explains the actual cause and fix: a matching `mlc-ai` package must be installed, and
  for pip wheels the CUDA variant must match (`mlc-ai-nightly-cuXYZ` alongside
  `mlc-llm-nightly-cuXYZ`), with a link to the install docs.

This mirrors the existing convention in the same file: `find_lib_path(..., optional=False)`
already raises a descriptive `RuntimeError` (with candidate paths) when the mlc-llm
library itself is absent. The change extends the same "clear, actionable error" treatment
to the missing-**dependency** case.

The helper was placed in `libinfo.py` (not `base.py`) on purpose: `libinfo.py` is
standalone — it imports only `os`/`sys`/`ctypes`, has no `tvm` dependency, and is even
`exec`-ed directly by `setup.py`. That keeps the new logic unit-testable without a fully
built `mlc_llm`/`tvm` install. `base.py` no longer references `ctypes` directly, so that
now-unused import was removed.

## 3. Files changed

- `python/mlc_llm/libinfo.py` — add `import ctypes`; add `load_lib(path)` helper.
- `python/mlc_llm/base.py` — call `libinfo.load_lib(...)` instead of `ctypes.CDLL(...)`;
  drop the now-unused `import ctypes`.
- `tests/python/test_libinfo.py` — new focused unit test.

## 4. Risk / uncertainty

- **Low behavioral risk.** On the success path `load_lib` returns exactly what
  `ctypes.CDLL` returned; only the *error* path changes (a clearer `RuntimeError` in place
  of the raw `OSError`). Callers of `_load_mlc_llm_lib` did not catch `OSError`
  specifically, so no error-handling contract is broken.
- **Scope caveat (honest):** this does not make CUDA 13.0 wheels ship `libtvm.so` — that
  requires a change in the `mlc-ai`/TVM wheel build, which is not in this repository. This
  fix converts the confusing symptom into a self-service diagnostic that points users to
  the real remedy; it is a robustness/UX fix, not a repackaging of the upstream wheel.
- I could not exercise the real dlopen failure locally because `tvm`/`mlc_llm` are not
  installed in this environment; the test simulates the `OSError` from `ctypes.CDLL`
  instead (see below).

## 5. How I verified

- `python3 -m pytest tests/python/test_libinfo.py -v` → 2 passed. Covers both the
  missing-dependency error path (asserts the message retains `libtvm.so`, mentions
  `mlc-ai`, and chains the original `OSError` as `__cause__`) and the success path.
- Confirmed `libinfo.py` still loads two ways: via `importlib` and via the exact
  `exec(compile(...))` pattern `setup.py` uses (so packaging is unaffected).
- `python3 -m py_compile` on all three files, and `ruff check` on them → all checks passed.
