"""SensoryRec chat PROTOTYPE (S3, first version). Run with `make app`.

A request in everyday words -> top 5 wines from the chosen model -> refine
with one click. No LLM yet: the explanation box shows the words each tasting
note shares with the request. Everything is ranked by the same models as in
`make eval` (fitted with the test wines held out).
"""
import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))   # app/style.py
import style  # noqa: E402

from sensoryrec import images, prototype
from sensoryrec.config import load_config, repo_path
from sensoryrec.recommenders import build_models
from sensoryrec.refinement import Refiner, load_lexicons

st.set_page_config(page_title="SensoryRec · prototype", page_icon="🍷", layout="wide")
st.markdown(style.CSS, unsafe_allow_html=True)
cfg = load_config()
K = cfg["k"]


@st.cache_resource(show_spinner="Loading the wines…")
def catalogue():
    processed = repo_path(cfg["paths"]["processed_dir"])
    wines = pd.read_parquet(processed / "wines.parquet")
    test_ids = pd.read_csv(processed / "test_ids.csv")["wine_id"]
    return wines, test_ids


@st.cache_resource
def bottle_info():
    """Shapes by grape, and the label photos that exist (data/images/sources.csv)."""
    shapes = images.load_shapes(repo_path(cfg["images"]["shapes"]))
    sources = repo_path(cfg["images"]["sources"])
    labels = dict(zip(*pd.read_csv(sources)[["wine_id", "file"]].T.values)) if sources.exists() else {}
    return shapes, labels


def picture(row, colour):
    """The wine's label photo if we have one (CC0, X-Wines), else its drawn bottle."""
    shapes, labels = bottle_info()
    if row.wine_id in labels:
        import base64
        data = base64.b64encode(repo_path(labels[row.wine_id]).read_bytes()).decode()
        return f'<img src="data:image/jpeg;base64,{data}" alt="label" style="height:140px">'
    return images.bottle_svg(images.bottle_shape(row.variety, colour, shapes), colour, 140)


@st.cache_resource
def variety_colour():
    table = pd.read_csv(repo_path(cfg["attribute"]["colour_table"]))
    return dict(zip(table["variety"], table["colour"]))


@st.cache_resource(show_spinner="Preparing the model (first time only)…")
def fitted_model(name):
    wines, test_ids = catalogue()
    model = next(m for m in build_models(cfg) if m.name == name)
    return model.fit(wines, holdout_ids=test_ids)


@st.cache_resource(show_spinner="Preparing refinement…")
def refiner(name):
    model = fitted_model(name)
    _, test_ids = catalogue()
    lexicons = load_lexicons(repo_path(cfg["refinement"]["lexicons"]))
    return Refiner(model.embeddings, model.catalogue, lexicons).fit(test_ids)


def reset(query=""):
    st.session_state.query = query
    st.session_state.steps = []          # taste refinements, in order
    st.session_state.max_price = None    # set by "cheaper"


if "steps" not in st.session_state:
    reset()

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("### Settings")
# The app shows the semantic engine only (student's decision, 16 Nov 2026);
# Gemini embeddings will be added here as the one alternative. The baselines
# stay in the evaluation (`make eval`), not in the app.
APP_MODELS = {"semantic": "Sentence-BERT", "semantic_gemini": "Gemini embeddings"}
names = [m.name for m in build_models(cfg) if m.name in APP_MODELS]   # Gemini appears once enabled
model_name = names[0] if len(names) == 1 else st.sidebar.selectbox(
    "Engine", names, format_func=APP_MODELS.get)
st.sidebar.caption(f"Engine: {APP_MODELS[model_name]}." + ("" if len(names) > 1 else " Gemini embeddings come next."))
wines, _ = catalogue()
country = st.sidebar.selectbox("Country (optional)", ["any"] + sorted(wines["country"].dropna().unique()))
st.sidebar.caption("PROTOTYPE: no LLM explanation yet (S3). Bottles are drawn by shape and colour; label photos (X-Wines, CC0) where a wine matches.")

# ---------------------------------------------------------------- request
st.markdown('<div class="maison-nav"><span class="maison-brand">SensoryRec</span>'
            '<span class="maison-proto">Research prototype · demonstration only</span></div>', unsafe_allow_html=True)
st.markdown('<div class="maison-tag">Find your bottle</div>'
            '<div class="maison-hero">Discover your next favourite bottle.</div>'
            '<div class="maison-rule"></div>'
            '<div class="maison-lead">Describe the wine you would like, in your own words, as you would to a '
            'sommelier. We compare your words with 120,000 professional tasting notes and show you the closest.</div>',
            unsafe_allow_html=True)
with st.form("ask"):
    text = st.text_input("What would you like?", value=st.session_state.query,
                         placeholder="Something red that isn't too heavy, a bit fruity and easy to drink")
    if st.form_submit_button("Find wines") and text.strip():
        reset(text.strip())

query = st.session_state.query
if not query:
    st.info("Try: *A fresh white that tastes like lemon and green apple, not sweet*")
    st.stop()

model = fitted_model(model_name)
can_refine = hasattr(model, "embeddings")   # taste directions live in embedding space
lexicons = load_lexicons(repo_path(cfg["refinement"]["lexicons"]))
alpha, alpha_from_dev = prototype.chosen_alpha(repo_path(cfg["refinement"]["out_dir"]) / "refinement_alpha_dev.csv",
                                               default=min(cfg["refinement"]["alphas"]))

filters = {}
if country != "any":
    filters["country"] = country
if st.session_state.max_price is not None:
    filters["max_price"] = st.session_state.max_price

steps = st.session_state.steps if can_refine else []
top = prototype.recommend(model, query, K, filters, steps,
                          refiner(model_name) if steps else None, alpha)

# ---------------------------------------------------------------- refinement
labels = [lex["more"] if s > 0 else lex["less"] for t, s in steps for lex in [lexicons[t]]]
if st.session_state.max_price is not None:
    labels.append(f"at most ${st.session_state.max_price:.0f}")
st.markdown(f"### Our selection for you")
st.markdown(f"*“{query}”*" + (f"  \n**Refined:** {', '.join(labels)}" if labels else ""))

st.markdown("**Refine the selection**")
buttons = prototype.refinement_buttons(lexicons) if can_refine else []
cols = st.columns(len(buttons) + 2)
for col, (label, trait, sign) in zip(cols, buttons):
    if col.button(label):
        st.session_state.steps = steps + [(trait, sign)]
        st.rerun()
if cols[-2].button("cheaper"):
    limit = prototype.cheaper_limit(top)
    if limit is not None:
        st.session_state.max_price = limit
        st.rerun()
if cols[-1].button("start over"):
    reset(query)
    st.rerun()
if not can_refine:
    st.caption(f"Taste refinement works in embedding space: choose a semantic model. Only 'cheaper' works for {model_name}.")
elif not alpha_from_dev:
    st.caption(f"Step size {alpha} is a placeholder until `make refine` has chosen it on the dev queries.")

# ---------------------------------------------------------------- results
if top.empty:
    st.warning("No wine passes these filters. Press 'start over'.")
colours = variety_colour()
for row in top.itertuples():
    price = f"${row.price:.0f}" if pd.notna(row.price) else "price unknown"
    words = prototype.shared_words(query, row.description)
    why = ", ".join(words) if words else "no shared words; matched on meaning"
    colour = colours.get(row.variety)
    st.markdown(
        f'<div class="maison-card" style="display:grid;grid-template-columns:70px 1fr;gap:20px">'
        f'<div style="display:flex;align-items:flex-end;justify-content:center;background:#EFE6D4">{picture(row, colour)}</div><div>'
        f'<div><span class="rank">{row.rank}</span>'
        f'<span class="name">{html.escape(row.title)}</span>{style.badge(colour)}</div>'
        f'<div class="meta">{html.escape(str(row.variety))} · {html.escape(str(row.country))} · {price} · '
        f'{row.points} points (Wine Enthusiast)</div>'
        f'<div class="note">{html.escape(row.description)}</div>'
        f'<div class="why"><b>Why this wine</b> (prototype: words the tasting note shares with your request): '
        f'{html.escape(why)}. <i>A written explanation, grounded in this note, comes next (S3).</i></div></div></div>',
        unsafe_allow_html=True)
