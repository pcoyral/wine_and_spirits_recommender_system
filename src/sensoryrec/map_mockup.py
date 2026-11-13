"""`make map-mockup`: one map MOCK-UP for the Future Work chapter.

Maps and availability are outside the scope of the thesis (CLAUDE.md). This
script only shows what a map view could look like, in one HTML page:

1. Where the recommended wines come from (REAL): the countries of origin of
   the top k wines that one model returns for one demo query.
2. Where to buy them (MOCK): pins for shops that DO NOT EXIST. Their names
   and positions are invented in this file; no shop or price data is used.

Output: docs/map_mockup.html (open it in a browser; it needs an internet
connection for the map background). The library choice is explained in
docs/map_libraries.md.
"""
import argparse
import logging

import pandas as pd
import plotly.graph_objects as go

from sensoryrec.config import load_config, repo_path
from sensoryrec.demo import read_queries
from sensoryrec.recommenders import build_models

log = logging.getLogger(__name__)

# Invented shops around the centre of Madrid. NOT real places.
MOCK_SHOPS = pd.DataFrame({
    "name": ["MOCK shop A", "MOCK shop B", "MOCK shop C", "MOCK shop D"],
    "lat": [40.4215, 40.4130, 40.4262, 40.4098],
    "lon": [-3.7005, -3.7110, -3.6890, -3.6935],
})
BANNER = ("MOCK-UP for Future Work. Map 1 uses real recommendations. "
          "Map 2 shows INVENTED shops: no real shop, stock or price data.")


def origin_counts(top, iso):
    """One row per country of origin of the recommended wines: its ISO code,
    how many of the wines come from it, and their titles.
    top: the recommended wines (columns country, title); iso: country -> ISO-3."""
    known = top[top["country"].isin(iso.index)]
    if len(known) < len(top):
        log.warning("%d wines without a mappable country are left off the map", len(top) - len(known))
    rows = known.groupby("country", sort=False).agg(n_wines=("title", "size"),
                                                    wines=("title", "<br>".join)).reset_index()
    rows["iso3"] = rows["country"].map(iso)
    return rows


def origin_figure(origins, query, model_name):
    """World map: one bubble per country, larger when more of the wines come from it."""
    fig = go.Figure(go.Scattergeo(
        locations=origins["iso3"], locationmode="ISO-3",
        text=origins["country"] + ": " + origins["n_wines"].astype(str) + " wine(s)<br>" + origins["wines"],
        hoverinfo="text",
        marker={"size": 14 + 8 * origins["n_wines"], "color": "#7b1e3a", "line": {"width": 1, "color": "white"}}))
    fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="#bbbbbb",
                    showland=True, landcolor="#f3efe6")
    fig.update_layout(title=f"1. Where the recommended wines come from (real)<br>"
                            f"<sup>Query: \"{query}\" | model: {model_name}</sup>",
                      margin={"l": 0, "r": 0, "t": 70, "b": 0}, height=460)
    return fig


def shops_figure(shops):
    """City map with the invented shops."""
    fig = go.Figure(go.Scattermap(
        lat=shops["lat"], lon=shops["lon"], mode="markers+text",
        text=shops["name"], textposition="top center",
        marker={"size": 16, "color": "#d9480f"}, hoverinfo="text"))
    fig.update_layout(title="2. Where to buy (MOCK: invented shops, not real data)",
                      map={"style": "open-street-map", "zoom": 13,
                           "center": {"lat": shops["lat"].mean(), "lon": shops["lon"].mean()}},
                      margin={"l": 0, "r": 0, "t": 50, "b": 0}, height=460)
    return fig


def page(figures):
    """One HTML page: the banner, then the figures (plotly.js loaded once)."""
    parts = [fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
             for i, fig in enumerate(figures)]
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>SensoryRec map mock-up</title></head>\n"
            "<body style=\"font-family: sans-serif; max-width: 960px; margin: auto\">\n"
            f"<p style=\"background: #fff3cd; border: 2px solid #d9480f; padding: 10px\"><b>{BANNER}</b></p>\n"
            + "\n".join(parts) + "\n</body></html>\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Build the map mock-up")
    parser.add_argument("--model", default="semantic", help="model name as in the result tables")
    parser.add_argument("--query-no", type=int, default=1, help="which line of the demo queries")
    args = parser.parse_args()

    cfg = load_config()
    query = read_queries(repo_path(cfg["demo"]["queries"]))[args.query_no - 1]
    processed = repo_path(cfg["paths"]["processed_dir"])
    wines = pd.read_parquet(processed / "wines.parquet")
    test_ids = pd.read_csv(processed / "test_ids.csv")["wine_id"]
    iso = pd.read_csv(repo_path(cfg["map_mockup"]["country_codes"])).set_index("country")["iso3"]

    model = next(m for m in build_models(cfg) if m.name == args.model)
    top = model.fit(wines, holdout_ids=test_ids).recommend(query, k=cfg["k"])

    out = repo_path(cfg["map_mockup"]["out"])
    out.write_text(page([origin_figure(origin_counts(top, iso), query, model.name),
                         shops_figure(MOCK_SHOPS)]), encoding="utf-8")
    log.info("saved %s (query %d, model %s)", out, args.query_no, model.name)


if __name__ == "__main__":
    main()
