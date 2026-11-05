# ChemiLab Studio

An enhanced educational cheminformatics workspace built with Streamlit and RDKit.

## Features
- SMILES parsing and canonical SMILES export
- 2D molecular structure rendering (PNG)
- Interactive 3D conformer viewer using RDKit + py3Dmol
- Molecular descriptors: formula, molecular weight, exact mass, LogP, TPSA,
  H-bond donors/acceptors, rotatable bonds, rings, heavy atoms, fraction Csp3,
  and molar refractivity
- Batch descriptor analysis with CSV export
- PDF molecular reports
- SDF and canonical SMILES downloads

## Run locally
Python 3.11 or 3.12 recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a new GitHub repository.
2. Upload the extracted project files (not just the ZIP).
3. In Streamlit Community Cloud, create an app from that repository.
4. Select `app.py` as the main file.
5. Deploy. If dependency resolution fails, check the build log and pin compatible package versions.

## Notes
- `packages.txt` is intentionally minimal; no extra apt packages are currently specified.
- 3D conformers are computationally generated and are not experimentally determined geometries.
- Descriptor values are educational computational estimates, not experimental data.

## Project background
The learning objectives are based on the course repository:
https://github.com/subramar4/Cheminformatics-Representation-of-2D-and-3D-chemical-structure
