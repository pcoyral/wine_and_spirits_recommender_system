"""SensoryRec chat PROTOTYPE (S3). Run with `make app`.

A request in everyday words -> top 5 wines from the engine (rag.engine in
config.yaml) -> refine with one click -> a short explanation per wine,
written by Gemini from the retrieved tasting notes only (rag.py). The engine
ranks; the LLM only explains. Without a Gemini key, cached explanations are
still shown; otherwise the box falls back to the words the note shares with
the request, labelled as such.
"""
import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))   # app/style.py
import style  # noqa: E402

from sensoryrec import images, prototype, rag
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
    labels = {}
    if sources.exists():
        t = pd.read_csv(sources)
        # a photo checked by the student (Open Food Facts) comes before an X-Wines label photo
        t = t.assign(order=(~t["source"].astype(str).str.startswith("Open Food Facts")).astype(int))
        t = t.sort_values("order").drop_duplicates("wine_id")
        labels = dict(zip(t["wine_id"], t["file"]))
    return shapes, labels


def picture(row, colour):
    """A real photo if we have one (approved Open Food Facts photo, or X-Wines label), else the drawn bottle."""
    shapes, labels = bottle_info()
    if row.wine_id in labels:
        import base64
        data = base64.b64encode(repo_path(labels[row.wine_id]).read_bytes()).decode()
        return f'<img src="data:image/jpeg;base64,{data}" alt="label" style="height:140px">'
    return images.bottle_svg(images.bottle_shape(row.variety, colour, shapes), colour, 140)


def show_featured(n=8):
    """A row of bottles we have real photos of (X-Wines, CC0), before the first request."""
    sources = repo_path(cfg["images"]["sources"])
    if not sources.exists():
        return
    wines, _ = catalogue()
    by_id = wines.set_index("wine_id")
    ids = images.featured(pd.read_csv(sources), by_id, variety_colour(), n, cfg["seed"])
    if not ids:
        return
    st.markdown("### Featured bottles")
    st.caption("Wines from the catalogue that we have a real bottle photo of (X-Wines, CC0). "
               "Scores and notes: Wine Enthusiast.")
    shapes, labels = bottle_info()
    cols = st.columns(4)
    for i, wine_id in enumerate(ids):
        w = by_id.loc[wine_id]
        price = f"${w['price']:.0f}" if pd.notna(w["price"]) else "price unknown"
        with cols[i % 4]:
            st.image(str(repo_path(labels[wine_id])), use_container_width=True)
            st.markdown(f'<div class="maison-serif" style="font-size:1.15rem;line-height:1.2">'
                        f'{html.escape(w["title"])}</div><div class="meta" style="font-size:.72rem;'
                        f'letter-spacing:.06em;text-transform:uppercase;color:#6b665e">'
                        f'{html.escape(str(w["variety"]))} · {html.escape(str(w["country"]))} · {price} · '
                        f'{w["points"]} pts</div>', unsafe_allow_html=True)


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


@st.cache_resource
def explainer():
    return rag.from_config(cfg)


@st.cache_data(show_spinner="Writing the explanation…")
def explanation(request, wine_ids, change):
    """Cached per request, list and change, so a rerun does not ask again."""
    wines, _ = catalogue()
    top = wines.set_index("wine_id").loc[list(wine_ids)].reset_index()
    return rag.explain(explainer(), cfg["rag"]["prompt"], request, top, change)


def reset(query=""):
    st.session_state.change = None       # (label, wine_ids before) of the last refinement
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
APP_MODELS = {"semantic": "Sentence-BERT", "semantic_bge": "bge-small", "semantic_gemini": "Gemini embeddings"}
# rag.engine chooses the engine (student's decision D2, 19 Nov 2026); Gemini
# appears next to it once its embeddings are enabled.
names = [m.name for m in build_models(cfg) if m.name == cfg["rag"]["engine"] or m.name == "semantic_gemini"]
model_name = names[0] if len(names) == 1 else st.sidebar.selectbox(
    "Engine", names, format_func=APP_MODELS.get)
st.sidebar.caption(f"Engine: {APP_MODELS[model_name]}." + ("" if len(names) > 1 else " Gemini embeddings come next."))
wines, _ = catalogue()
country = st.sidebar.selectbox("Country (optional)", ["any"] + sorted(wines["country"].dropna().unique()))
st.sidebar.caption("PROTOTYPE. Explanations: Gemini, from the tasting notes only; the engine alone ranks. "
                   "Bottles are drawn by shape and colour; label photos (X-Wines, CC0) where a wine matches.")

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
    show_featured()
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
        st.session_state.change = (label, top["wine_id"].tolist())
        st.session_state.steps = steps + [(trait, sign)]
        st.rerun()
if cols[-2].button("cheaper"):
    limit = prototype.cheaper_limit(top)
    if limit is not None:
        st.session_state.change = ("cheaper", top["wine_id"].tolist())
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
change = ""
if st.session_state.change:
    change = rag.change_text(st.session_state.change[0], st.session_state.change[1], top["wine_id"].tolist())
out = explanation(query, tuple(top["wine_id"]), change) if not top.empty else {"intro": "", "whys": [], "source": "llm"}
if out["intro"]:
    st.markdown(f"*{out['intro']}*")
why_label = ("Why this wine" if out["source"] == "llm"
             else "Why this wine (no written explanation available: words the note shares with your request)")
for row, why in zip(top.itertuples(), out["whys"]):
    price = f"${row.price:.0f}" if pd.notna(row.price) else "price unknown"
    colour = colours.get(row.variety)
    st.markdown(
        f'<div class="maison-card" style="display:grid;grid-template-columns:70px 1fr;gap:20px">'
        f'<div style="display:flex;align-items:flex-end;justify-content:center;background:#EFE6D4">{picture(row, colour)}</div><div>'
        f'<div><span class="rank">{row.rank}</span>'
        f'<span class="name">{html.escape(row.title)}</span>{style.badge(colour)}</div>'
        f'<div class="meta">{html.escape(str(row.variety))} · {html.escape(str(row.country))} · {price} · '
        f'{row.points} points (Wine Enthusiast)</div>'
        f'<div class="note">{html.escape(row.description)}</div>'
        f'<div class="why"><b>{why_label}</b>: {html.escape(why)}</div></div></div>',
        unsafe_allow_html=True)
