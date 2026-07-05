"""Render an IMDF archive to a self-contained Leaflet map (open in a browser).

    uv run python view_imdf.py --archive <dir>   ->  <dir>/view.html

The GeoJSON is inlined into the HTML, so the only network use is the OpenStreetMap basemap
tiles + Leaflet from CDN (loaded by your browser). Opens at the venue's real-world location.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STYLE = {
    "walkway": {"color": "#8a6d3b", "fillColor": "#deb887", "fillOpacity": 0.75, "weight": 1},
    "vegetation": {"color": "#3d7a2f", "fillColor": "#78be64", "fillOpacity": 0.6, "weight": 1},
    "structure": {"color": "#7a4a2f", "fillColor": "#c8824a", "fillOpacity": 0.7, "weight": 1},
    "stairs": {"color": "#8a5a8a", "fillColor": "#d2a0d2", "fillOpacity": 0.75, "weight": 1},
}
VENUE_STYLE = {"color": "#111", "weight": 3, "fill": False}

HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}}
.legend{{background:#fff;padding:8px 10px;border-radius:6px;font:13px sans-serif;line-height:1.6;box-shadow:0 1px 4px rgba(0,0,0,.3)}}
.sw{{display:inline-block;width:12px;height:12px;margin-right:6px;border:1px solid #0003;vertical-align:middle}}</style>
</head><body><div id="map"></div><script>
const UNIT={units}, VENUE={venue}, LEVEL={level}, STYLE={style};
const map=L.map('map');
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom:20,attribution:'&copy; OpenStreetMap'}}).addTo(map);
const lvl=L.geoJSON(LEVEL,{{style:{{color:'#888',weight:1,fillColor:'#eee',fillOpacity:0.15}}}}).addTo(map);
const units=L.geoJSON(UNIT,{{
  style:f=>STYLE[f.properties.category]||{{color:'#888',fillColor:'#bbb',fillOpacity:.5,weight:1}},
  onEachFeature:(f,l)=>l.bindPopup(`<b>${{f.feature_type}}</b><br>category: ${{f.properties.category}}<br>id: ${{f.id}}`)
}}).addTo(map);
const ven=L.geoJSON(VENUE,{{style:{venue_style}}}).addTo(map);
map.fitBounds(ven.getBounds().pad(0.1));
const lg=L.control({{position:'bottomright'}});
lg.onAdd=()=>{{const d=L.DomUtil.create('div','legend');d.innerHTML='<b>{title}</b><br>'+
  Object.entries(STYLE).map(([k,v])=>`<span class=sw style="background:${{v.fillColor}}"></span>${{k}}`).join('<br>')+
  '<br><span class=sw style="background:none;border:2px solid #111"></span>venue';return d;}};
lg.addTo(map);
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="IMDF archive dir (has *.geojson)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = Path(args.archive)

    venue = json.load(open(d / "venue.geojson"))
    title = venue["features"][0]["properties"]["name"].get("en", "IMDF Venue")
    html = HTML.format(
        title=title,
        units=json.dumps(json.load(open(d / "unit.geojson"))),
        venue=json.dumps(venue),
        level=json.dumps(json.load(open(d / "level.geojson"))),
        style=json.dumps(STYLE),
        venue_style=json.dumps(VENUE_STYLE),
    )
    out = Path(args.out) if args.out else d / "view.html"
    out.write_text(html)
    print(f"wrote {out}  (open in a browser)")


if __name__ == "__main__":
    main()
