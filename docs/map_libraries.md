# Python map libraries: comparison note

Written after the supervisor meeting of Nov 2026 (DECISIONS.md, option E1).
Maps and "where to buy" are **Future Work** in this thesis, not part of the
recommender. This note compares three Python libraries and explains which one
the single mock-up (`docs/map_mockup.html`, built by `make map-mockup`) uses.

## What our data allows

The Wine Enthusiast dataset gives each wine a **country, province and region
name**, and no coordinates. That decides most of the choice:

- A map **by country** needs no coordinates if the library can place a
  country from its ISO code (`data/country_iso3.csv`).
- A map **by province or region** needs coordinates for each name. Getting
  them means geocoding with an external service, which has usage limits and
  terms to respect. That is Future Work.
- A map of **shops** needs shop and stock data that the thesis does not have
  and will not scrape. The mock-up therefore uses invented pins, labelled MOCK.

## The three libraries

| | folium | plotly | pydeck |
|---|---|---|---|
| Built on | Leaflet (JavaScript) | plotly.js, with MapLibre for street maps | deck.gl (WebGL) |
| Street-level background | OpenStreetMap by default, no key | OpenStreetMap or Carto styles, no key | Carto styles by default, no key; Mapbox styles need a token |
| Map by country without coordinates | Only with a borders file (GeoJSON) that you supply | Yes: accepts ISO country codes directly | No: needs coordinates or a borders file |
| Pins with labels and pop-ups | Yes, the simplest of the three | Yes | Yes, as a "layer" with a tooltip |
| Large data (100,000+ points), 3D | Slow with many markers | Fine for thousands | Its main strength |
| Output | HTML page | HTML page; static image with an extra package | HTML page |
| Streamlit | Needs the extra package `streamlit-folium` | Built in (`st.plotly_chart`) | Built in (`st.pydeck_chart`) |
| Gradio | Through an HTML block | Built in (`gr.Plot`) | Through an HTML block |
| Learning curve | Low | Low to medium | Medium to high |
| Also draws ordinary charts | No | Yes | No |

All three load their JavaScript and the map background from the internet when
the page is opened, so a map page needs a connection.

Other tools seen but not compared in detail: `st.map` in Streamlit (points
only, very little control) and GeoPandas with matplotlib (static figures for
a printed thesis, needs border files).

## The same four pins in each library

`shops` is a table with the columns `name`, `lat`, `lon`. Each snippet below
was run with the versions listed at the end of this note.

**folium**

```python
import folium

m = folium.Map(location=[shops["lat"].mean(), shops["lon"].mean()], zoom_start=13)
for shop in shops.itertuples():
    folium.Marker([shop.lat, shop.lon], popup=shop.name).add_to(m)
m.save("map.html")
```

**plotly**

```python
import plotly.graph_objects as go

fig = go.Figure(go.Scattermap(lat=shops["lat"], lon=shops["lon"], text=shops["name"], mode="markers+text"))
fig.update_layout(map={"style": "open-street-map", "zoom": 13,
                       "center": {"lat": shops["lat"].mean(), "lon": shops["lon"].mean()}})
fig.write_html("map.html", include_plotlyjs="cdn")
```

**pydeck**

```python
import pydeck as pdk

layer = pdk.Layer("ScatterplotLayer", data=shops, get_position="[lon, lat]", get_radius=60,
                  get_fill_color=[217, 72, 15], pickable=True)
view = pdk.ViewState(latitude=shops["lat"].mean(), longitude=shops["lon"].mean(), zoom=13)
pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{name}"}).to_html("map.html")
```

## Choice for the mock-up: plotly

1. **It maps countries from ISO codes.** The "where the recommended wines
   come from" map works with the data we have, with no coordinates and no
   borders file.
2. **It fits the chat app either way.** Streamlit and Gradio both display
   plotly figures directly, so the choice of app framework (S3) stays open.
3. **One library for both maps** in the mock-up (world view and city view).

folium would be the choice for a map that is only pins on streets (it is the
simplest). pydeck would be the choice if the map had to show very many
points or 3D layers, which a top-5 list never needs.

## The mock-up

`make map-mockup` writes `docs/map_mockup.html` (open it in a browser):

1. **Where the recommended wines come from (real).** The countries of origin
   of the top k wines for one demo query and one model; both are named in the
   title. Country level only.
2. **Where to buy (MOCK).** Four invented shops in the centre of Madrid. The
   names and positions are made up in `src/sensoryrec/map_mockup.py`; every
   pin, the title and a banner on the page say so. No shop, stock or price
   data is used.

The copy in the repository was built with `MODEL=tfidf`, because the Semantic
model cannot run on the machine where it was written. The Demo workflow
rebuilds it with the Semantic model.

## What a real version would need (Future Work)

- **Shop and stock data** from a source that allows it (a partner retailer or
  an official API). No scraping of sites whose terms forbid it.
- **Geocoding** of provinces, regions and shop addresses, with a service
  whose terms allow it, and the results cached.
- **The user's location**, which is personal data: consent, and no storage
  beyond the session.
- **Freshness**: stock changes daily, the review dataset does not.

## Versions used for this note

folium 0.20.0, plotly 7.1.0, pydeck 0.9.3 (13 Nov 2026). Only plotly is in
`requirements.txt`; the other two were installed for the comparison only.
