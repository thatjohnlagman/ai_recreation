# READ FIRST

**Snapshot**: HEAD `c012cf393a17d41e884b450a29a08ebf2eda6cd6`
**Architecture**: Python backend, Mock stream attacker, Vanilla JS dashboard.
**Startup**: `python server.py --open`
**Package Layout**: `src/` contains exact copies of the working tree, including all Python modules, frontend, and `.research_repo` configs/csvs.
**Limitations**: The UI blends mock generated metrics with scraped historical values that fail strict statistical pairing validations against canonical Phase 11 research tables.
