"""The "French Maison" look of the prototype (student's design brief,
16 Nov 2026): palette, typography and the pastel colour of each wine
category. Visual only: nothing here changes what is recommended."""

BURGUNDY, CREAM, GOLD, IVORY, CHARCOAL = "#49272D", "#F8F4EB", "#B59A68", "#FFFFFF", "#292923"

# pastel identity of each wine category (text stays charcoal for contrast)
CATEGORY_COLOURS = {"red": "#B77B82", "white": "#E9DDBB", "rosé": "#E9B8C0", "sparkling": "#D6C9A7"}
CATEGORY_LABELS = {"red": "Red", "white": "White", "rosé": "Rosé", "sparkling": "Sparkling"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Inter:wght@300;400;500&display=swap');
html, body, [class*="css"], .stMarkdown, p, label, button, input {{ font-family: 'Inter', sans-serif; color: {CHARCOAL}; }}
h1, h2, h3, .maison-serif {{ font-family: 'Cormorant Garamond', Georgia, serif !important; color: {BURGUNDY}; font-weight: 500; letter-spacing: .01em; }}
[data-testid="stSidebar"] {{ background: {IVORY}; border-right: 1px solid rgba(181,154,104,.35); }}
.stButton button, .stFormSubmitButton button {{ border-radius: 2px; border: 1px solid {BURGUNDY}; color: {BURGUNDY}; background: transparent; font-weight: 500; letter-spacing: .04em; }}
.stButton button:hover, .stFormSubmitButton button:hover {{ background: {BURGUNDY}; color: {IVORY}; border-color: {BURGUNDY}; }}
.stFormSubmitButton button, .stFormSubmitButton button p {{ background: {BURGUNDY}; color: {IVORY} !important; }}
.stFormSubmitButton button:hover {{ background: #5c3239; }}
.stTextInput input {{ border-radius: 2px; background: {IVORY}; border: 1px solid rgba(181,154,104,.6); }}
[data-testid="stForm"] {{ border: 1px solid rgba(181,154,104,.45); border-radius: 2px; background: {IVORY}; }}
.maison-nav {{ display:flex; justify-content:space-between; align-items:baseline; border-bottom:1px solid rgba(181,154,104,.5); padding-bottom:.6rem; margin-bottom:2.2rem; }}
.maison-brand {{ font-family:'Cormorant Garamond', Georgia, serif; font-size:1.7rem; color:{BURGUNDY}; letter-spacing:.12em; text-transform:uppercase; }}
.maison-tag {{ font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; color:{GOLD}; }}
.maison-hero {{ font-family:'Cormorant Garamond', Georgia, serif; font-size:3.1rem; line-height:1.05; color:{BURGUNDY}; margin:0 0 .5rem 0; }}
.maison-lead {{ font-size:1rem; color:#5a5650; max-width:46rem; margin-bottom:1.6rem; }}
.maison-rule {{ width:56px; height:1px; background:{GOLD}; margin:.8rem 0 1.4rem 0; }}
.maison-card {{ background:{IVORY}; border:1px solid rgba(181,154,104,.35); border-radius:2px; padding:1.4rem 1.6rem; margin-bottom:1rem; }}
.maison-card .rank {{ font-family:'Cormorant Garamond', Georgia, serif; color:{GOLD}; font-size:1.5rem; margin-right:.5rem; }}
.maison-card .name {{ font-family:'Cormorant Garamond', Georgia, serif; color:{BURGUNDY}; font-size:1.55rem; line-height:1.2; }}
.maison-card .meta {{ font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; color:#6b665e; margin:.35rem 0 .8rem 0; }}
.maison-card .note {{ font-size:.97rem; line-height:1.6; }}
.maison-card .why {{ font-size:.8rem; color:#6b665e; border-top:1px solid rgba(181,154,104,.3); margin-top:.9rem; padding-top:.6rem; }}
.maison-badge {{ display:inline-block; font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; padding:.18rem .6rem; border-radius:1px; color:{CHARCOAL}; margin-left:.6rem; vertical-align:middle; }}
.maison-proto {{ font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; color:{BURGUNDY}; border:1px solid {GOLD}; padding:.15rem .5rem; }}
</style>
"""


def badge(colour):
    """Small pastel badge for a wine category, or nothing if the colour is unknown."""
    if colour not in CATEGORY_COLOURS:
        return ""
    return (f'<span class="maison-badge" style="background:{CATEGORY_COLOURS[colour]}">'
            f'{CATEGORY_LABELS[colour]}</span>')
