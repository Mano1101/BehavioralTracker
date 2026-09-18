# BehavioralTracker Best V1

Combined foundation using the uploaded **AnimalBehaviourTracker** tracking/analysis engine plus the earlier BehavioralTracker workflow concept.

## What is included
- Dependency-aware workflow with **Edit Selected Step** and downstream stale-state invalidation.
- Uploaded project's perspective correction, polygon ROI/object architecture, transition calculations, EPM functions, interaction bouts, two-mouse tracking, merged-blob splitting and Hungarian identity matching.
- 1 to 3 animal configuration.
- Grayscale/RGB/HSV/LAB configuration layer.
- Preview/QC stage before full tracking.
- Assay choices: Open Field, EPM, Light/Dark, NOR, Y-Maze, T-Maze, Social Interaction, Custom.
- BORIS-style event coding data layer placeholder.
- CSV/Excel/plot output engine retained from the uploaded project.
- Save/open `.btproj` project state.
- Background-safe PyInstaller entry point.

## Important scientific note
No computer-vision tracker can guarantee zero errors for every video. Identical mice can swap identities during close contact, and grooming/rearing require pose-level validation. This version is a strong integrated foundation, not a validated replacement for manual QC.

## Install
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```

## Packaging
```bash
pip install pyinstaller
pyinstaller --noconfirm --windowed --name BehavioralTracker main.py
```

## Correction workflow
If you discover a mistake while working at step 9, select step 3 and click **Edit Selected Step**. Step 3 becomes pending and steps 4 onward become stale. Correct step 3, then run forward. Earlier valid work is preserved.
