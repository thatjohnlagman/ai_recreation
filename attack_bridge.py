"""
attack_bridge.py
────────────────
Shared file-bus between the Adversarial Probe Console (attacker_app.py)
and the IDS Security Posture Console (app.py).

Protocol
--------
• Attacker calls  write_attack(probe_dict)   →  appends one entry to QUEUE_FILE
• Defender calls  drain_attacks()            →  reads all pending entries, clears the file
                                                 returns list[dict] (empty if nothing pending)

Concurrency safety: We use an atomic rename-swap pattern so neither app
can corrupt the file.  No external dependencies required.
"""

import json
import os
import time

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
QUEUE_FILE  = os.path.join(_HERE, "shared_attack_queue.json")
_LOCK_FILE  = QUEUE_FILE + ".lock"
_TMP_FILE   = QUEUE_FILE + ".tmp"

# Maximum entries kept in the queue to avoid unbounded growth
_MAX_QUEUE  = 200

# How long (seconds) to spin-wait for the lock before giving up
_LOCK_TIMEOUT = 0.25


# ── Internal lock helpers ─────────────────────────────────────────────────────

def _acquire_lock(timeout: float = _LOCK_TIMEOUT) -> bool:
    """Try to acquire a filesystem lock by exclusive creation of a lock file."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            # O_CREAT | O_EXCL is atomic on NTFS and POSIX
            fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.01)
    return False


def _release_lock():
    try:
        os.remove(_LOCK_FILE)
    except FileNotFoundError:
        pass


# ── Public API ────────────────────────────────────────────────────────────────

def write_attack(probe: dict) -> bool:
    """
    Append one probe entry to the shared attack queue.
    Called by attacker_app.py on each tick that produces a probe.

    Parameters
    ----------
    probe : dict
        Must contain at minimum:
            probe_id     (str)
            timestamp    (float, epoch seconds)
            method       (str)  — e.g. "silent_probing"
            bypassed     (bool)
            attack_score (float)
            source_ip    (str)
            full_vec     (dict[str, float])  — feature vector

    Returns True on success, False if the lock could not be acquired.
    """
    if not _acquire_lock():
        return False
    try:
        # Read existing queue
        entries: list = []
        if os.path.exists(QUEUE_FILE):
            try:
                with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                if not isinstance(entries, list):
                    entries = []
            except (json.JSONDecodeError, OSError):
                entries = []

        entries.append(probe)

        # Trim to cap
        if len(entries) > _MAX_QUEUE:
            entries = entries[-_MAX_QUEUE:]

        # Atomic write via temp-file + rename
        with open(_TMP_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f)
        os.replace(_TMP_FILE, QUEUE_FILE)
        return True
    except OSError:
        return False
    finally:
        _release_lock()


def drain_attacks() -> list:
    """
    Read and clear all pending attack entries from the shared queue.
    Called by app.py on each simulation tick.

    Returns a (possibly empty) list of probe dicts.
    """
    if not os.path.exists(QUEUE_FILE):
        return []

    if not _acquire_lock():
        return []

    try:
        entries: list = []
        try:
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                entries = json.load(f)
            if not isinstance(entries, list):
                entries = []
        except (json.JSONDecodeError, OSError):
            entries = []

        # Clear the file atomically
        with open(_TMP_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        os.replace(_TMP_FILE, QUEUE_FILE)

        return entries
    except OSError:
        return []
    finally:
        _release_lock()


def clear_queue():
    """Utility: wipe the queue file entirely (useful for tests / resets)."""
    if not _acquire_lock():
        return
    try:
        with open(_TMP_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        os.replace(_TMP_FILE, QUEUE_FILE)
    except OSError:
        pass
    finally:
        _release_lock()
