import streamlit as st
import pandas as pd
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import py3Dmol
import streamlit.components.v1 as components
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Image as RLImage,
)

st.set_page_config(
    page_title="Virtual Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# macOS-style theme
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0c0d12;
  --panel: rgba(255,255,255,.045);
  --panel-hover: rgba(255,255,255,.075);
  --line: rgba(255,255,255,.09);
  --text: #f5f5f7;
  --muted: #9a9aa3;
  --accent: #0a84ff;
  --accent-soft: rgba(10,132,255,.16);
  --good: #30d158;
  --radius: 14px;
  --ease: cubic-bezier(.22,.8,.3,1);
}

html, body, [class*="css"], .stApp, button, input, textarea {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", "Helvetica Neue", sans-serif !important;
  -webkit-font-smoothing: antialiased;
}

/* ---------- Canvas ---------- */
.stApp {
  background:
    radial-gradient(900px 520px at 85% -10%, rgba(10,132,255,.16), transparent 60%),
    radial-gradient(700px 480px at -5% 105%, rgba(191,90,242,.12), transparent 60%),
    var(--bg);
  color: var(--text);
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { max-width: 1180px; padding-top: 1.2rem; padding-bottom: 4rem; }
h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.02em; }
p, li, label, .stMarkdown { color: var(--text); }
::selection { background: var(--accent-soft); }

/* ---------- Sidebar (frosted glass) ---------- */
section[data-testid="stSidebar"] {
  background: rgba(30,31,40,.72);
  backdrop-filter: blur(28px) saturate(180%);
  -webkit-backdrop-filter: blur(28px) saturate(180%);
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
.brand { display:flex; align-items:center; gap:.6rem; font-size:1.45rem; font-weight:700;
         letter-spacing:-.02em; margin: .3rem 0 1rem; }
.brand .logo { width:34px; height:34px; border-radius:9px; display:grid; place-items:center;
         background: linear-gradient(145deg,#0a84ff,#5e5ce6); box-shadow: 0 6px 16px rgba(10,132,255,.35); font-size:1.05rem; }
.side-title { font-size:.95rem; font-weight:600; color:var(--muted); margin:1.2rem 0 .4rem; }

/* Radio -> macOS source list */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
  padding: .5rem .75rem; border-radius: 9px; width: 100%;
  transition: background .18s var(--ease), transform .18s var(--ease);
  cursor: pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
  background: var(--panel-hover); transform: translateX(3px);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
  background: var(--accent); box-shadow: 0 4px 14px rgba(10,132,255,.35);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color:#fff; font-weight:600; }

/* ---------- Window title bar ---------- */
.titlebar { display:flex; align-items:center; gap:.5rem; padding:.7rem 1rem; margin-bottom:1.6rem;
  border:1px solid var(--line); border-radius: var(--radius); background: var(--panel);
  backdrop-filter: blur(16px); }
.dot { width:12px; height:12px; border-radius:50%; display:inline-block; transition: filter .15s; }
.titlebar:hover .dot { filter: brightness(1.15); }
.dot.r { background:#ff5f57; } .dot.y { background:#febc2e; } .dot.g { background:#28c840; }
.titlebar .ttl { flex:1; text-align:center; color:var(--muted); font-size:.85rem; font-weight:500; margin-right:46px; }

/* ---------- Headings ---------- */
.page-title { font-size: 2.6rem; font-weight: 700; letter-spacing:-.035em; line-height:1.1; margin: 0 0 .5rem; }
.page-sub { color: var(--muted); font-size: 1.05rem; margin-bottom: 1.8rem; max-width: 62ch; }

/* ---------- Cards ---------- */
.card { background: var(--panel); border:1px solid var(--line); border-radius: var(--radius);
  padding: 1.15rem 1.25rem; height: 100%;
  transition: transform .25s var(--ease), background .25s var(--ease), border-color .25s var(--ease), box-shadow .25s var(--ease); }
.card:hover { transform: translateY(-4px); background: var(--panel-hover);
  border-color: rgba(10,132,255,.45); box-shadow: 0 18px 40px rgba(0,0,0,.35), 0 0 0 4px rgba(10,132,255,.07); }
.card .ic { font-size: 1.6rem; margin-bottom:.5rem; }
.card h4 { margin:0 0 .3rem; font-size:1.05rem; font-weight:600; }
.card p { margin:0; color: var(--muted); font-size:.92rem; line-height:1.45; }

/* ---------- Inputs ---------- */
.stTextInput input, .stTextArea textarea, div[data-baseweb="select"] > div, .stNumberInput input {
  background: rgba(255,255,255,.055) !important; border: 1px solid var(--line) !important;
  border-radius: 11px !important; color: var(--text) !important;
  transition: border-color .18s var(--ease), box-shadow .18s var(--ease), background .18s var(--ease);
}
.stTextInput input:hover, .stTextArea textarea:hover, div[data-baseweb="select"] > div:hover {
  background: rgba(255,255,255,.085) !important; border-color: rgba(255,255,255,.2) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--accent) !important; box-shadow: 0 0 0 4px var(--accent-soft) !important;
}
.stTextArea textarea { font-family: "SF Mono", ui-monospace, Menlo, monospace !important; font-size: .95rem; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
  border-radius: 11px; font-weight: 600; padding: .6rem 1.1rem;
  background: rgba(255,255,255,.07); color: var(--text); border: 1px solid var(--line);
  transition: transform .18s var(--ease), box-shadow .18s var(--ease), background .18s var(--ease), border-color .18s var(--ease);
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-2px) scale(1.015); background: rgba(255,255,255,.12);
  border-color: rgba(255,255,255,.25); box-shadow: 0 10px 24px rgba(0,0,0,.35); color: #fff;
}
.stButton > button:active, .stDownloadButton > button:active { transform: scale(.98); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: linear-gradient(180deg,#2b95ff,#0a84ff); border-color: transparent; color:#fff;
  box-shadow: 0 8px 20px rgba(10,132,255,.35);
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
  box-shadow: 0 14px 30px rgba(10,132,255,.5);
}

/* ---------- Tabs, metrics, tables, alerts ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--panel); padding: 4px; border-radius: 12px; border:1px solid var(--line); width: fit-content; }
.stTabs [data-baseweb="tab"] { border-radius: 9px; padding: .4rem 1rem; height: auto; transition: background .18s; }
.stTabs [data-baseweb="tab"]:hover { background: var(--panel-hover); }
.stTabs [aria-selected="true"] { background: rgba(255,255,255,.14) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

div[data-testid="stMetric"] { background: var(--panel); border:1px solid var(--line); border-radius: var(--radius);
  padding: 14px 16px; transition: transform .22s var(--ease), border-color .22s, background .22s; }
div[data-testid="stMetric"]:hover { transform: translateY(-3px); background: var(--panel-hover); border-color: rgba(10,132,255,.4); }
div[data-testid="stMetricValue"] { font-weight: 700; letter-spacing:-.02em; }
div[data-testid="stMetricLabel"] p { color: var(--muted); }

div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius: var(--radius); overflow:hidden;
  transition: box-shadow .25s var(--ease), border-color .25s; }
div[data-testid="stDataFrame"]:hover { border-color: rgba(255,255,255,.2); box-shadow: 0 14px 34px rgba(0,0,0,.35); }

div[data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--line); }
div[data-testid="stExpander"] { border:1px solid var(--line); border-radius: var(--radius); background: var(--panel);
  transition: background .2s, border-color .2s; }
div[data-testid="stExpander"]:hover { background: var(--panel-hover); border-color: rgba(255,255,255,.18); }
div[data-testid="stImage"] img { border-radius: var(--radius); transition: transform .3s var(--ease), box-shadow .3s var(--ease); }
div[data-testid="stImage"] img:hover { transform: scale(1.015); box-shadow: 0 20px 46px rgba(0,0,0,.45); }
hr { border-color: var(--line); }

/* Scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.15); border-radius: 10px; border: 2px solid transparent; background-clip: padding-box; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,.28); background-clip: padding-box; border:2px solid transparent; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Data + chemistry helpers
# =============================================================================
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

PAGES = [
    "🏠 Home",
    "📖 Theory",
    "🧬 Molecular Visualization",
    "📊 Molecular Properties Report",
    "📈 Structure–Property Analysis",
    "📝 Assessment",
    "📄 Final Project Report",
]


def parse_smiles(smiles: str):
    if not smiles or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles.strip())
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


def lipinski_violations(p):
    return sum([
        p["Molecular weight (g/mol)"] > 500,
        p["LogP"] > 5,
        p["H-bond donors"] > 5,
        p["H-bond acceptors"] > 10,
    ])


def make_2d_png(mol, size=(900, 620)):
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().bondLineWidth = 2.2
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def make_3d_block(mol):
    work = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    if AllChem.EmbedMolecule(work, params) != 0:
        return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(work):
            AllChem.MMFFOptimizeMolecule(work, maxIters=250)
        else:
            AllChem.UFFOptimizeMolecule(work, maxIters=250)
    except Exception:
        pass
    return Chem.MolToMolBlock(work)


def render_3d(block, style, show_h):
    m = Chem.MolFromMolBlock(block, removeHs=False)
    if not show_h:
        m = Chem.RemoveHs(m)
    view = py3Dmol.view(width=900, height=520)
    view.addModel(Chem.MolToMolBlock(m), "mol")
    if style == "Stick":
        view.setStyle({"stick": {"radius": 0.16}})
    elif style == "Ball and Stick":
        view.setStyle({"stick": {"radius": 0.13}, "sphere": {"scale": 0.27}})
    else:
        view.setStyle({"sphere": {"scale": 0.75}})
    view.setBackgroundColor("#14151c")
    view.zoomTo()
    components.html(view._make_html(), height=540, scrolling=False)


def molecules_to_df(text):
    rows, errors = [], []
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            name, smi = [x.strip() for x in line.split(",", 1)]
        else:
            name, smi = f"Molecule {n}", line
        mol = parse_smiles(smi)
        if mol is None:
            errors.append(f"Line {n}: couldn't read `{smi}`")
            continue
        row = {"Name": name, "SMILES": smi}
        row.update(get_properties(mol))
        rows.append(row)
    return rows, errors


# =============================================================================
# PDF (single report for all molecules)
# =============================================================================
def build_report_pdf(student, reg_no, smiles_list):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=17*mm, leftMargin=17*mm,
                            topMargin=16*mm, bottomMargin=16*mm)
    ss = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=ss["Title"], textColor=colors.HexColor("#12344a"), fontSize=20, leading=24)
    sub = ParagraphStyle("S", parent=ss["Normal"], textColor=colors.HexColor("#63758a"), fontSize=9, leading=13)

    story = [
        Paragraph("Final Student Molecular Analysis Report", title),
        Paragraph(f"Student: <b>{student or '—'}</b> &nbsp;&nbsp; Registration no.: <b>{reg_no or '—'}</b>", sub),
        Paragraph(f"Molecules analysed: {len(smiles_list)}", sub),
        Spacer(1, 8*mm),
    ]
    for i, smi in enumerate(smiles_list, start=1):
        mol = parse_smiles(smi)
        props = get_properties(mol)
        story += [
            Paragraph(f"Molecule {i}", ss["Heading2"]),
            Paragraph(f"Input SMILES: {smi}", ss["BodyText"]),
            Paragraph(f"Canonical SMILES: {Chem.MolToSmiles(mol)}", ss["BodyText"]),
            Spacer(1, 3*mm),
        ]
        try:
            story += [RLImage(BytesIO(make_2d_png(mol, (700, 460))), width=110*mm, height=72*mm, kind="proportional"),
                      Spacer(1, 3*mm)]
        except Exception:
            pass
        rows = [["Descriptor", "Value"]] + [[k, str(v)] for k, v in props.items()]
        t = Table(rows, colWidths=[92*mm, 75*mm], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123f55")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f6f8")]),
            ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#d8e1e8")),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        if i < len(smiles_list):
            story.append(PageBreak())
    story += [Spacer(1, 6*mm),
              Paragraph("Descriptors are computed from the molecular graph with RDKit. They are estimates, "
                        "not experimental measurements.", sub)]
    doc.build(story)
    return buf.getvalue()


# =============================================================================
# UI helpers
# =============================================================================
def titlebar(text):
    st.markdown(
        f'<div class="titlebar"><span class="dot r"></span><span class="dot y"></span>'
        f'<span class="dot g"></span><span class="ttl">{text}</span></div>',
        unsafe_allow_html=True)


def header(title, sub=""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="page-sub">{sub}</div>', unsafe_allow_html=True)


def card(icon, title, body):
    return f'<div class="card"><div class="ic">{icon}</div><h4>{title}</h4><p>{body}</p></div>'


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">🧪</div>Virtual Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()
    st.markdown("### 🧑‍🎓 Student Information")
    st.text_input("Student Name", key="student_name")
    st.text_input("Registration Number", key="reg_number")
    st.markdown("### 🎓 Learning Objectives")
    st.markdown("Students can learn:\n- SMILES notation\n- Molecular representation\n- 2D and 3D structures\n- Molecular descriptors")

titlebar(f"Virtual Lab — {page.split(' ', 1)[1]}")

# =============================================================================
# Pages
# =============================================================================
if page == PAGES[0]:  # Home
    header("Virtual Lab", "Turn a line of text into a molecule. Draw it, spin it, measure it, and export a report.")
    c = st.columns(3, gap="medium")
    c[0].markdown(card("📖", "Theory", "SMILES, descriptors and Lipinski's rule in plain language."), unsafe_allow_html=True)
    c[1].markdown(card("🧬", "Visualize", "See any molecule in 2D, then rotate it in 3D."), unsafe_allow_html=True)
    c[2].markdown(card("📊", "Measure", "Weight, LogP, TPSA and more for one molecule or many."), unsafe_allow_html=True)
    st.write("")
    c = st.columns(3, gap="medium")
    c[0].markdown(card("📈", "Compare", "Chart a property across molecules and check Lipinski."), unsafe_allow_html=True)
    c[1].markdown(card("📝", "Assess", "A short quiz to check what you've picked up."), unsafe_allow_html=True)
    c[2].markdown(card("📄", "Report", "One PDF with every molecule, your name and number."), unsafe_allow_html=True)
    st.write("")
    st.info("Start with **Molecular Visualization**, or jump to **Final Project Report** when you're ready to export.")

elif page == PAGES[1]:  # Theory
    header("Theory", "The three ideas behind everything in this lab.")
    with st.expander("What is SMILES?", expanded=True):
        st.write("SMILES is a text notation for molecules. Atoms are letters, bonds are symbols, and "
                 "branches go in parentheses.")
        st.code("CCO            # ethanol\nc1ccccc1       # benzene (lowercase = aromatic)\nCC(=O)O        # acetic acid", language="text")
    with st.expander("What are molecular descriptors?"):
        st.write("Numbers computed from a molecule's structure. **Molecular weight** is its mass, **LogP** estimates "
                 "how fat-loving it is, and **TPSA** is the polar surface area, which relates to how well it crosses membranes.")
    with st.expander("Lipinski's rule of five"):
        st.write("A drug-likeness rule of thumb. Oral drugs tend to have molecular weight ≤ 500, LogP ≤ 5, "
                 "H-bond donors ≤ 5 and H-bond acceptors ≤ 10.")
    with st.expander("2D vs 3D structure"):
        st.write("2D drawings show connectivity. 3D conformers show how atoms sit in space, which affects how "
                 "molecules interact.")

elif page == PAGES[2]:  # Visualization
    header("Molecular Visualization", "Paste a SMILES string or pick an example.")
    if "viz_smiles" not in st.session_state:
        st.session_state["viz_smiles"] = "CCO"
    a, b = st.columns([3, 2], gap="large")
    with b:
        pick = st.selectbox("Example molecule", list(EXAMPLES.keys()))
        if st.button("Load example", use_container_width=True):
            st.session_state["viz_smiles"] = EXAMPLES[pick]
            st.session_state.pop("mol3d_block", None)
            st.rerun()
    with a:
        smi = st.text_area("SMILES", key="viz_smiles", height=100)
    mol = parse_smiles(smi)
    if mol is None:
        st.error("That SMILES couldn't be read. Check the notation and try again.")
    else:
        props = get_properties(mol)
        m = st.columns(4)
        m[0].metric("Formula", props["Molecular formula"])
        m[1].metric("Weight", f'{props["Molecular weight (g/mol)"]:.2f} g/mol')
        m[2].metric("LogP", props["LogP"])
        m[3].metric("TPSA", f'{props["TPSA (Å²)"]:.1f} Å²')
        st.write("")
        t2d, t3d = st.tabs(["🖼️ 2D structure", "🧊 3D viewer"])
        with t2d:
            png = make_2d_png(mol)
            st.image(png, use_container_width=True)
            st.download_button("Download PNG", png, "molecule_2d.png", "image/png")
        with t3d:
            c1, c2 = st.columns([2, 1])
            style = c1.selectbox("Style", ["Stick", "Ball and Stick", "Sphere"])
            show_h = c2.checkbox("Show hydrogens", value=True)
            if st.button("Generate 3D conformer", type="primary"):
                with st.spinner("Building a 3D conformer…"):
                    st.session_state["mol3d_block"] = make_3d_block(mol)
            block = st.session_state.get("mol3d_block")
            if block:
                render_3d(block, style, show_h)
                st.download_button("Download SDF", block.encode(), "molecule_3d.sdf", "chemical/x-mdl-sdfile")
            else:
                st.info("Generate a conformer to open the viewer. Very large molecules may not embed.")

elif page == PAGES[3]:  # Properties report
    header("Molecular Properties Report", "One molecule per line, as `Name,SMILES` or just `SMILES`.")
    text = st.text_area("Molecule list", height=170,
                        value="Ethanol,CCO\nBenzene,c1ccccc1\nAspirin,CC(=O)Oc1ccccc1C(=O)O")
    rows, errors = molecules_to_df(text)
    for e in errors:
        st.warning(e)
    if rows:
        df = pd.DataFrame(rows)
        st.success(f"{len(rows)} molecule(s) analysed.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode(), "descriptors.csv", "text/csv")

elif page == PAGES[4]:  # Structure-property
    header("Structure–Property Analysis", "Compare a property across molecules and check drug-likeness.")
    default = "\n".join(f"{k},{v}" for k, v in list(EXAMPLES.items())[:6])
    text = st.text_area("Molecules (Name,SMILES per line)", value=default, height=150)
    rows, errors = molecules_to_df(text)
    for e in errors:
        st.warning(e)
    if rows:
        df = pd.DataFrame(rows)
        numeric = [c for c in df.columns if c not in ("Name", "SMILES", "Molecular formula")]
        prop = st.selectbox("Property to compare", numeric, index=numeric.index("Molecular weight (g/mol)"))
        st.bar_chart(df.set_index("Name")[prop], color="#0a84ff")
        st.markdown("#### Lipinski check")
        lip = pd.DataFrame({
            "Name": df["Name"],
            "MW ≤ 500": df["Molecular weight (g/mol)"] <= 500,
            "LogP ≤ 5": df["LogP"] <= 5,
            "HBD ≤ 5": df["H-bond donors"] <= 5,
            "HBA ≤ 10": df["H-bond acceptors"] <= 10,
        })
        lip["Violations"] = (~lip.iloc[:, 1:5]).sum(axis=1)
        st.dataframe(lip, use_container_width=True, hide_index=True)

elif page == PAGES[5]:  # Assessment
    header("Assessment", "Four quick questions.")
    QUIZ = [
        ("What does SMILES stand for?",
         ["Simplified Molecular Input Line Entry System", "Standard Molecular Index of Linked Elements", "Structured Model of Ionic Labelled Species"], 0),
        ("Which SMILES is ethanol?", ["CCO", "CC(=O)O", "c1ccccc1"], 0),
        ("Which descriptor estimates how fat-loving a molecule is?", ["TPSA", "LogP", "Heavy atoms"], 1),
        ("Lipinski's rule says oral drugs tend to have a molecular weight of at most…", ["200", "500", "900"], 1),
    ]
    with st.form("quiz"):
        answers = [st.radio(q, opts, index=None, key=f"q{i}") for i, (q, opts, _) in enumerate(QUIZ)]
        done = st.form_submit_button("Check answers", type="primary")
    if done:
        score = sum(a == QUIZ[i][1][QUIZ[i][2]] for i, a in enumerate(answers))
        (st.success if score == len(QUIZ) else st.info)(f"You scored {score} out of {len(QUIZ)}.")
        for i, (q, opts, ans) in enumerate(QUIZ):
            if answers[i] != opts[ans]:
                st.write(f"• **{q}** → {opts[ans]}")

else:  # Final report
    header("📄 Final Student Molecular Analysis Report",
           "Enter one SMILES string per line. The report will include every valid molecule in a single downloadable PDF.")
    name_in = st.text_input("Student Name for Report")
    reg_in = st.text_input("Registration Number for Report")
    smiles_text = st.text_area("Enter SMILES (one molecule per line)", value="CCO", height=170)

    student = name_in or st.session_state.get("student_name", "")
    reg_no = reg_in or st.session_state.get("reg_number", "")

    valid, table_rows, bad = [], [], []
    for n, raw in enumerate(smiles_text.splitlines(), start=1):
        s = raw.strip()
        if not s:
            continue
        m = parse_smiles(s)
        if m is None:
            bad.append(f"Line {n}: couldn't read `{s}`")
            continue
        valid.append(s)
        table_rows.append({
            "No.": len(valid),
            "Input SMILES": s,
            "Molecular Formula": rdMolDescriptors.CalcMolFormula(m),
            "Molecular Weight (g/mol)": round(Descriptors.MolWt(m), 3),
        })
    for e in bad:
        st.warning(e)

    if valid:
        st.success(f"{len(valid)} valid molecule(s) ready for the report.")
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        if st.button("📄 Generate Single PDF for All Molecules"):
            with st.spinner("Building your report…"):
                st.session_state["final_pdf"] = build_report_pdf(student, reg_no, valid)
        if st.session_state.get("final_pdf"):
            st.download_button("⬇️ Download PDF report", st.session_state["final_pdf"],
                               "Final_Molecular_Report.pdf", "application/pdf", type="primary")
    else:
        st.info("Add at least one valid SMILES to build a report.")
