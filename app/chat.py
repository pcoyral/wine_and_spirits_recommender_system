"""SensoryRec chat PROTOTYPE (S3, first version). Run with `make app`.

A request in everyday words -> top 5 wines from the chosen model -> refine
with one click. No LLM yet: the explanation box shows the words each tasting
note shares with the request. Everything is ranked by the same models as in
`make eval` (fitted with the test wines held out).
"""
import pandas as pd
import streamlit as st

from sensoryrec import prototype
from sensoryrec.config import load_config, repo_path
from sensoryrec.recommenders import build_models
from sensoryrec.refinement import Refiner, load_lexicons

st.set_page_config(page_title="SensoryRec prototype", page_icon="🍷", layout="wide")
cfg = load_config()
K = cfg["k"]


@st.cache_resource(show_spinner="Loading the wines…")
def catalogue():
    processed = repo_path(cfg["paths"]["processed_dir"])
    wines = pd.read_parquet(processed / "wines.parquet")
    test_ids = pd.read_csv(processed / "test_ids.csv")["wine_id"]
    return wines, test_ids


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
names = [m.name for m in build_models(cfg)]
model_name = st.sidebar.selectbox("Model", names, index=names.index("semantic") if "semantic" in names else 0)
wines, _ = catalogue()
country = st.sidebar.selectbox("Country (optional)", ["any"] + sorted(wines["country"].dropna().unique()))
st.sidebar.caption("PROTOTYPE: no LLM explanation and no bottle images yet (S3, increment A).")

# ---------------------------------------------------------------- request
st.title("🍷 SensoryRec")
st.caption("Describe a wine you would like, in your own words. **Prototype** for the supervisor meeting of 16 Nov 2026.")
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
st.markdown(f"**You asked:** {query}" + (f"  \n**Refined:** {', '.join(labels)}" if labels else ""))

st.markdown("**Refine:**")
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
for row in top.itertuples():
    with st.container(border=True):
        price = f"${row.price:.0f}" if pd.notna(row.price) else "price unknown"
        st.markdown(f"**{row.rank}. {row.title}**  \n{row.variety} · {row.country} · {price} · {row.points} points")
        st.write(row.description)
        words = prototype.shared_words(query, row.description)
        st.caption("Why this wine (prototype: words the note shares with your request): "
                   + (", ".join(words) if words else "no shared words, matched on meaning")
                   + ". *The LLM explanation grounded in this note comes in S3.*")
