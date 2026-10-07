# CURRENT STATE HANDOFF

1. **Branch & State**: Branch `feature/ids-pages-correction`. HEAD is at `e4f23b8d`. Working tree is clean except for runtime SQLite db. Changes are committed and pushed.
2. **Current AFP Settings (C1)**:
   - Initial (Base) Epsilon: 0.0003
   - Minimum Epsilon: 0.00003
   - Maximum Epsilon: 0.0003
   - Red (Fast Decay) Multiplier: 0.40
   - Yellow (Slow Decay) Multiplier: 0.90
   - Green (Growth) Multiplier: 1.30 (Modified from 1.05)
   - Recall Thresholds: Rcritical = 0.85, Rmin = 0.95
   - Window Size: 5
   - Cadence: Atomic batch updates
3. **Edits Since Pre-Supplementary (30ab73a)**:
   - `defenses.yaml` & `expanded_data/configs/defenses.yaml`: Lowered `intensity_min` from 0.0001 (or 0.0003) to 0.00003.
   - `controllers.yaml`: Increased C1 `growth_factor` from 1.05 to 1.30 to allow faster epsilon recovery.
   - `attacker_sim.py`: Reduced stream interactive menu delay from 1.0s to 0.1s.
   - `frontend/*`: Removed Data Expanded badge, removed Export Metrics button from Dashboard, moved Export buttons in Explorer and Sessions to filters and headers respectively.
   - `server.py`: Fixed a float-casting bug where `rolling_recall` being None caused a server 500 error.
4. **Effect on RS/FS**: No effect on RS/FS bounds. RS/FS would use C1's new 1.30 growth factor if assigned to C1, but their min/max intensities are unchanged.
5. **Reset & Metrics**: Baseline resets clear the session rolling recall and history arrays. Rolling recall is updated atomically per batch. Dashboard strictly displays session recall (total TP / (total TP + total FN)) and rolling controller recall.
6. **Supplementary Attacker Effort Code**: Removed entirely. The repo was hard-reset back to the pre-supplementary snapshot `30ab73a`.
7. **Known Issues & Tests**: Interactive stream tested successfully. Bouncing epsilon and correct UI placement verified. No crash observed on empty recall.
8. **Deviations from Phase 1-11 Research**:
   - `intensity_min` lowered to 0.00003 (originally 0.0001).
   - `growth_factor` increased to 1.30 (originally 1.05).
   - These modifications invalidate direct comparison to the frozen experiment paper results, as the controller now recovers significantly faster.
