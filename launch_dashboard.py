#!/usr/bin/env python3
"""
Local Dashboard Launcher:
Launches the authorized SOC Web Dashboard in the system default browser
using a private one-time launch ticket, establishing an HttpOnly operator session.
Does not require manual token entry or login dialogs in the dashboard.
"""

import os
import sys
import json
import urllib.request
import urllib.error
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PRIVATE_TOKEN_FILE = REPO_ROOT / ".operator_token"
ALT_TOKEN_FILE = REPO_ROOT / "operator_token.txt"
BASE_URL = os.environ.get("IDS_BASE_URL", "http://127.0.0.1:8000")


def get_token() -> str:
    env_tok = os.environ.get("IDS_OPERATOR_TOKEN")
    if env_tok and env_tok.strip():
        return env_tok.strip()
    if PRIVATE_TOKEN_FILE.exists():
        try:
            val = PRIVATE_TOKEN_FILE.read_text(encoding="utf-8").strip()
            if val:
                return val
        except Exception:
            pass
    if ALT_TOKEN_FILE.exists():
        try:
            val = ALT_TOKEN_FILE.read_text(encoding="utf-8").strip()
            if val:
                return val
        except Exception:
            pass
    return ""


def main():
    print("=" * 65)
    print("  IDS + RECALL-AWARE DEFENSE — LOCAL DASHBOARD LAUNCHER")
    print("=" * 65)

    token = get_token()

    # 1. Check if server is running
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/dashboard/stats")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status != 200:
                print(f"[Launcher] Server returned unexpected status {resp.status}.")
                sys.exit(1)
    except Exception as e:
        print(f"[Launcher] Error connecting to server at {BASE_URL}: {e}")
        print("\n[Launcher] Please start the server first in another terminal:")
        print("    $env:IDS_DATA_PROFILE = 'expanded'")
        print("    .\\.venv\\Scripts\\python.exe server.py\n")
        sys.exit(1)

    # 2. Request one-time launch ticket from server
    ticket_url = f"{BASE_URL}/api/operator/issue-ticket"
    req_body = json.dumps({}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Operator-Token"] = token

    try:
        req = urllib.request.Request(ticket_url, data=req_body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            launch_url = data.get("launch_url")
            if not launch_url:
                raise ValueError("Response missing launch_url")
    except Exception as e:
        print(f"[Launcher] Failed to issue launch ticket: {e}")
        print("[Launcher] Ensure operator token is configured or server is accessible.")
        sys.exit(1)

    # 3. Open launch URL in browser (consumes ticket and sets HttpOnly cookie)
    print(f"[Launcher] Server confirmed. Opening authorized dashboard in browser...")
    webbrowser.open(launch_url)
    print("[Launcher] Authorized operator session established via HttpOnly cookie.")
    print("=" * 65)


if __name__ == "__main__":
    main()
