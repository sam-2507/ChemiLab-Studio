import streamlit as st
import pandas as pd
from io import BytesIO
from pathlib import Path
import tempfile

from rdkit import Chem
from rdkit.Chem import (
    AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors,
    Draw, rdDepictor
)
from rdkit.Chem.Draw import rdMolDraw2D
import py3Dmol
import streamlit.components.v1 as components
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

st.set_page_config(
    page_title="ChemiLab Studio | Molecular Workspace",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Theme ---------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#17233b; --muted:#68758d; --teal:#087e8b; --line:#e5eaf1; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f6f9fc 0%,#ffffff 42%); color:var(--ink); }
.block-container { padding-top:1.7rem; max-width:1440px; }
.hero { padding:1.55rem 1.8rem; border-radius:22px; color:white;
 background:linear-gradient(115deg,#10233f 0%,#145b70 58%,#15969a 100%);
 box-shadow:0 14px 35px rgba(16,35,63,.14); margin-bottom:1.2rem; }
.hero h1 {font-family:'Space Grotesk',sans-serif; font-size:2.25rem; margin:0; color:white;}
.hero p {margin:.45rem 0 0; color:#d8edf0; font-size:1rem;}
.eyebrow {text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; color:#a9e2df;}
.panel {background:white; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; }
.metric-card {background:white; border:1px solid var(--line); border-radius:14px; padding: .9rem 1rem;}
.metric-label {font-size:.78rem;color:#748198;font-weight:600;}
.metric-value {font-family:'Space Grotesk',sans-serif;font-size:1.55rem;font-weight:700;color:#152b45;}
.small-muted {color:#718096;font-size:.85rem;}
.section-head {font-family:'Space Grotesk',sans-serif;font-size:1.22rem;font-weight:700;color:#18304d;margin:.35rem 0 .7rem;}
div[data-testid="stMetric"] {background:white;border:1px solid #e5eaf1;padding:12px 14px;border-radius:14px;}
.stButton>button, .stDownloadButton>button {border-radius:10px;font-weight:650;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

EXAMPLES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "Lactic acid": "CC(O)C(=O)O",
    "Lidocaine": "CCN(CC)C(=O)C1=NC(C)=CC=C1",
}

st.markdown("""
<div class="hero">
  <div class="eyebrow">CHEMINFORMATICS • LEARN / VISUALISE / ANALYSE</div>
  <h1>ChemiLab Studio</h1>
  <p>Your interactive workspace for molecular structures, descriptors and downloadable reports.</p>
</div>
""", unsafe_allow_html=True)

def parse_smiles(smiles: str):
    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            Chem.rdchem.Mol.__init__ if False else None
        return mol
    except Exception:
        return None

def get_properties(mol):
    return {
        "Molecular formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular weight (g/mol)": round(Descriptors.MolWt(mol), 3),
        "Exact molecular weight": round(Descriptors.ExactMolWt(mol), 4),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "H-bond donors": Lipinski.NumHDonors(mol),
        "H-bond acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable bonds": Lipinski.NumRotatableBonds(mol),
        "Total rings": Lipinski.RingCount(mol),
        "Aromatic rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Aliphatic rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "Heavy atoms": mol.GetNumHeavyAtoms(),
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "Molar refractivity": round(Crippen.MolMR(mol), 3),
    }

def make_2d_png(mol, size=(900, 620), atom_indices=None):
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.bondLineWidth = 2.2
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=atom_indices or [])
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

def make_3d_block(mol):
    work = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    status = AllChem.EmbedMolecule(work, params)
    if status != 0:
        return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(work):
            AllChem.MMFFOptimizeMolecule(work, maxIters=250)
        else:
            AllChem.UFFOptimizeMolecule(work, maxIters=250)
    except Exception:
        pass
    return Chem.MolToMolBlock(work)

def build_pdf(smiles, mol, props, png_bytes):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=17*mm,
                            leftMargin=17*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("LabTitle", parent=styles["Title"], textColor=colors.HexColor("#12344a"),
                           fontSize=20, leading=24, spaceAfter=5)
    sub = ParagraphStyle("Sub", parent=styles["Normal"], textColor=colors.HexColor("#63758a"),
                         fontSize=9, leading=13)
    story = [Paragraph("ChemiLab Studio — Molecular Report", title),
             Paragraph("Generated molecular structure analysis • Educational use", sub),
             Spacer(1, 9*mm),
             Paragraph("<b>Structure input</b>", styles["Heading2"]),
             Paragraph(f"Input SMILES: {smiles}", styles["BodyText"]),
             Paragraph(f"Canonical SMILES: {Chem.MolToSmiles(mol)}", styles["BodyText"]),
             Paragraph(f"Formula: <b>{props['Molecular formula']}</b>", styles["BodyText"]),
             Spacer(1, 4*mm)]
    try:
        img = RLImage(BytesIO(png_bytes), width=145*mm, height=100*mm, kind="proportional")
        story.extend([img, Spacer(1, 5*mm)])
    except Exception:
        pass
    story.append(Paragraph("Molecular descriptors", styles["Heading2"]))
    rows = [["Descriptor", "Calculated value"]] + [[str(k), str(v)] for k, v in props.items()]
    table = Table(rows, colWidths=[92*mm, 75*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#123f55")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f1f6f8")]),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#d8e1e8")),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.extend([table, Spacer(1, 5*mm),
                  Paragraph("Note: Descriptors are computed from the submitted molecular graph using RDKit. "
                            "They are computational estimates and should not be treated as experimental measurements.",
                            sub)])
    doc.build(story)
    return buffer.getvalue()

# ----------------------------- Sidebar -------------------------------------
with st.sidebar:
    st.markdown("## 🧪 Workspace")
    st.caption("Molecular analysis toolkit")
    page = st.radio("Navigate", ["Molecule workspace", "Batch analysis", "About & guide"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Quick examples**")
    chosen = st.selectbox("Load a molecule", list(EXAMPLES.keys()), index=0)
    if st.button("Load example", use_container_width=True):
        st.session_state["smiles_input"] = EXAMPLES[chosen]
        st.rerun()
    st.divider()
    st.caption("Powered by RDKit · Streamlit · py3Dmol")

if page == "About & guide":
    st.markdown('<div class="section-head">About this virtual laboratory</div>', unsafe_allow_html=True)
    st.write("ChemiLab Studio is an educational cheminformatics workspace for translating molecular line notation into visual structures and calculated descriptors.")
    c1, c2, c3 = st.columns(3)
    c1.metric("2D", "Structure drawing")
    c2.metric("3D", "Conformer viewer")
    c3.metric("PDF", "Exportable report")
    st.markdown("### How to use")
    st.markdown("""
    1. Choose a sample molecule or paste a valid **SMILES** string.
    2. Select **Analyse molecule** to validate the structure and calculate descriptors.
    3. Explore the 2D depiction, interactive 3D model and descriptor table.
    4. Download the molecular image, SDF structure or PDF report.
    5. Use Batch analysis to calculate descriptors for several SMILES at once.
    """)
    st.info("SMILES is a text notation for describing molecular structures. Invalid or unsupported strings will be rejected with a clear message.")
    st.markdown("### Scientific notes")
    st.write("LogP, TPSA and other descriptors are model-derived values. They do not replace laboratory measurements, validated pharmacokinetic models or expert chemical review.")
    st.markdown("### Credits")
    st.write("This enhanced educational project is based on the cheminformatics learning objectives and workflow in the original course repository.")
    st.markdown("[View original course repository](https://github.com/subramar4/Cheminformatics-Representation-of-2D-and-3D-chemical-structure)")
    st.stop()

if page == "Batch analysis":
    st.markdown('<div class="section-head">Batch molecular descriptor analysis</div>', unsafe_allow_html=True)
    st.write("Enter one molecule per line using the format `Name,SMILES` or just `SMILES`.")
    batch_text = st.text_area("Molecule list", value="Ethanol,CCO\nBenzene,c1ccccc1\nAspirin,CC(=O)Oc1ccccc1C(=O)O", height=180)
    if st.button("Analyse batch", type="primary"):
        rows, errors = [], []
        for line_no, raw in enumerate(batch_text.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                name, smi = line.split(",", 1)
                name, smi = name.strip(), smi.strip()
            else:
                name, smi = f"Molecule {line_no}", line
            mol = parse_smiles(smi)
            if mol is None:
                errors.append(f"Line {line_no}: could not parse `{smi}`")
                continue
            row = {"Name": name, "SMILES": smi}
            row.update(get_properties(mol))
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            st.success(f"Successfully analysed {len(rows)} molecule(s).")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                               "chemiLab_batch_descriptors.csv", "text/csv")
        for err in errors:
            st.warning(err)
    st.stop()

# ----------------------------- Main workspace ------------------------------
if "smiles_input" not in st.session_state:
    st.session_state["smiles_input"] = EXAMPLES["Ethanol"]

left, right = st.columns([1.05, 1.45], gap="large")
with left:
    st.markdown('<div class="section-head">01 / Molecular input</div>', unsafe_allow_html=True)
    smiles = st.text_area("SMILES notation", key="smiles_input", height=105,
                          help="Paste a SMILES string, for example CCO for ethanol.")
    analyse = st.button("⚡ Analyse molecule", type="primary", use_container_width=True)
    st.caption("Tip: use the example selector in the sidebar to load a known structure.")
    if analyse:
        st.session_state["analysed_smiles"] = smiles.strip()

active_smiles = st.session_state.get("analysed_smiles", st.session_state["smiles_input"]).strip()
mol = parse_smiles(active_smiles)

with right:
    st.markdown('<div class="section-head">02 / Structure preview</div>', unsafe_allow_html=True)
    if mol is None:
        st.markdown('<div class="panel">Enter a valid SMILES string and click <b>Analyse molecule</b> to generate a molecular workspace.</div>', unsafe_allow_html=True)
    else:
        png_bytes = make_2d_png(mol)
        st.image(png_bytes, caption="2D molecular structure", use_container_width=True)
        st.download_button("Download 2D structure (PNG)", png_bytes,
                           "molecule_2d.png", "image/png")

if mol is None:
    if active_smiles:
        st.error("This SMILES string could not be interpreted. Check the notation and try again.")
    st.stop()

props = get_properties(mol)
st.divider()
st.markdown('<div class="section-head">03 / Molecular profile</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Molecular formula", props["Molecular formula"])
m2.metric("Molecular weight", f'{props["Molecular weight (g/mol)"]:.3f} g/mol')
m3.metric("LogP", props["LogP"])
m4.metric("TPSA", f'{props["TPSA (Å²)"]:.1f} Å²')

tab2d, tab3d, tabdata, tabexport = st.tabs(["🖼️ 2D structure", "🧊 Interactive 3D", "📊 Descriptors", "⬇️ Export"])

with tab2d:
    st.write("A 2D depiction of the molecular graph. Atom labels and bond connectivity are generated by RDKit.")
    st.image(png_bytes, use_container_width=True)
    st.download_button("Save high-resolution PNG", png_bytes, "molecular_structure.png", "image/png", key="png_tab")

with tab3d:
    style = st.selectbox("Rendering style", ["Stick", "Ball and Stick", "Sphere"])
    show_h = st.checkbox("Include hydrogens", value=True)
    if st.button("Generate 3D conformer", type="primary"):
        with st.spinner("Generating and optimising a 3D conformer…"):
            block = make_3d_block(mol)
            st.session_state["mol3d_block"] = block
    block = st.session_state.get("mol3d_block")
    if block:
        view = py3Dmol.view(width=900, height=540)
        view.addModel(block, "mol")
        if style == "Stick":
            view.setStyle({"stick": {"radius": 0.16}})
        elif style == "Ball and Stick":
            view.setStyle({"stick": {"radius": 0.13}, "sphere": {"scale": 0.27}})
        else:
            view.setStyle({"sphere": {"scale": 0.75}})
        if not show_h:
            view.removeAllModels()
            heavy = Chem.RemoveHs(Chem.MolFromMolBlock(block, removeHs=False))
            view.addModel(Chem.MolToMolBlock(heavy), "mol")
            if style == "Stick":
                view.setStyle({"stick": {"radius": 0.16}})
            elif style == "Ball and Stick":
                view.setStyle({"stick": {"radius": 0.13}, "sphere": {"scale": 0.27}})
            else:
                view.setStyle({"sphere": {"scale": 0.75}})
        view.setBackgroundColor("#f7fafc")
        view.zoomTo()
        components.html(view._make_html(), height=570, scrolling=False)
        st.download_button("Download 3D structure (SDF)",
                           Chem.MolToMolBlock(Chem.MolFromMolBlock(block, removeHs=False)).encode("utf-8"),
                           "molecule_3d.sdf", "chemical/x-mdl-sdfile")
    else:
        st.info("Generate a conformer to open the interactive 3D viewer. Some large or unusual molecules may not embed successfully.")

with tabdata:
    st.write("Calculated molecular descriptors")
    df = pd.DataFrame([{"Descriptor": k, "Value": v} for k, v in props.items()])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Values are calculated from the molecular graph. They are not experimental measurements.")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download descriptors (CSV)", csv, "molecular_descriptors.csv", "text/csv")

with tabexport:
    st.markdown("#### Molecular report")
    st.write("Create a printable PDF containing the input notation, 2D depiction and calculated descriptors.")
    pdf = build_pdf(active_smiles, mol, props, png_bytes)
    st.download_button("📄 Download PDF report", pdf, "ChemiLab_molecular_report.pdf",
                       "application/pdf", type="primary")
    st.download_button("Download canonical SMILES", Chem.MolToSmiles(mol).encode("utf-8"),
                       "canonical_smiles.txt", "text/plain")
    st.download_button("Download structure (SDF)", Chem.MolToMolBlock(mol).encode("utf-8"),
                       "molecule_2d.sdf", "chemical/x-mdl-sdfile")

st.divider()
st.markdown('<p class="small-muted">ChemiLab Studio · Educational cheminformatics workspace · Built with RDKit</p>', unsafe_allow_html=True)
