# CONFIG COMPARISON
- **Research Configs (.research_repo)**: Extensive YAML definitions for C1-C7. AFP minimums and growths vary strictly per config.
- **Tool Settings (server.py)**: Hardcoded. `get_base_intensity()` returns static floats. Controller uses generic exponential scaling without respecting the exact strict thresholds laid out in the research repo.
