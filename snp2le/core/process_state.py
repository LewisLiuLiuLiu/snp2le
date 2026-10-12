# SPDX-FileCopyrightText: 2026 Simon Dorrer
# SPDX-License-Identifier: Apache-2.0
"""process_state.py - process-wide settings that overlapping conversions share.

A conversion changes two things that belong to the whole process rather than to its thread:
the standard streams, silenced while scikit-rf fits, and the BLAS thread count, held at one.
A plain context manager saves the value it finds and puts it back on exit.  When two
conversions overlap in threads, the second one saves the first one's setting as the
original and restores it last, so the process keeps it for good: sys.stdout stays a StringIO
and BLAS stays on one thread.  A ProcessSetting is applied by the first caller in and undone
by the last one out, so overlapping callers share one setting and the process gets its own
value back.
"""
from __future__ import annotations
import contextlib
import threading


class ProcessSetting:
    """A process-wide change held while at least one caller is inside `held()`.

    `apply()` makes the change and returns a callable that undoes it, or None when there is
    nothing to undo.  The first caller in decides, and the decision holds for every caller
    until the last one has left."""

    def __init__(self, apply):
        self._apply = apply
        self._lock = threading.Lock()
        self._users = 0
        self._undo = None

    @contextlib.contextmanager
    def held(self):
        with self._lock:
            if self._users == 0:
                self._undo = self._apply()
            self._users += 1
        try:
            yield
        finally:
            with self._lock:
                self._users -= 1
                if self._users == 0:
                    undo, self._undo = self._undo, None
                    if undo is not None:
                        undo()
