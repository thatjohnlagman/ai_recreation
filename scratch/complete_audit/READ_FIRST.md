# READ FIRST

**Snapshot**: HEAD `62736afb746c2ee57a0c92a36ac5e5401f3ea5c7`, Research HEAD `77cd29e04de6e0371845d28628dad3d8e0dc5053`
**Architecture**: Python backend (server.py) with mock attacker traffic stream (attacker_sim.py), exposing live metrics and configs to a vanilla JS dashboard (frontend/).
**Startup**: `python server.py --open`, `python attacker_sim.py --mode stream`
**Limitations**: The UI blends fully pre-computed mathematical aggregates from the research repo's JSON files with mock live simulated traffic metrics generated locally without an operational SIEM oracle.
