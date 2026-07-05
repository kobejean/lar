"""Vectorise the semantic BEV into a schema-valid IMDF archive (Apple Maps).

    Level rasters (semantic/coverage) + georef  ->  IMDF GeoJSON (venue/level/unit + address)

Pipeline:
  1. Georeference: fit world(X,Z) metres -> WGS84 from the camera trajectory vs GPS
     (GPS-accuracy-limited absolute placement ~18 m; relative shape is metric-accurate).
  2. Polygonise: per IMDF unit category (walkway/vegetation/stairs), extract the semantic
     mask, trace contours *with holes*, clean/simplify in metres (shapely), drop slivers.
  3. Assemble: venue + level(outdoor, ordinal 0) + units + address, with UUID ids, foreign
     keys, display points, and a manifest.
  4. Self-validate: enum membership, FK integrity, geometry validity, WGS84 bounds.

No open-source IMDF CLI validator exists; run the produced archive through Safe Software's
free online FME IMDF validator for the authoritative check.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, mapping
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from colmap_io import qvec2rotmat, read_model
from taxonomy import CLASSES, Klass

# --- allowed category enums (OGC IMDF 1.0.0 / docs.ogc.org/cs/20-094) ---------------
VENUE_CATEGORIES = {"airport", "aquarium", "businesscampus", "casino", "communitycenter",
                    "conventioncenter", "governmentfacility", "healthcarefacility", "hotel",
                    "museum", "parkingfacility", "resort", "retailstore", "shoppingcenter",
                    "stadium", "stripmall", "theater", "themepark", "trainstation",
                    "transitstation", "university"}
LEVEL_CATEGORIES = {"arrivals", "departures", "parking", "transit", "unspecified"}
UNIT_CATEGORIES = {"walkway", "vegetation", "stairs", "structure", "road", "terrace",
                   "unenclosedarea", "unspecified", "room", "steps", "ramp", "parking"}


# ---- georeferencing -----------------------------------------------------------------

def _umeyama(A: np.ndarray, B: np.ndarray):
    """Best similarity mapping A->B (B ~= s*R@A + t), reflection allowed (2D)."""
    mA, mB = A.mean(0), B.mean(0)
    A0, B0 = A - mA, B - mB
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T                       # optimal orthogonal; may be a reflection (det -1)
    s = S.sum() / (A0 ** 2).sum()
    t = mB - s * R @ mA
    return s, R, t


def fit_world_to_wgs84(recon, frames_path: str | Path, gps_path: str | Path):
    """Return (to_wgs84(world_xz)->lonlat, info) fitting the camera track to GPS East-North."""
    frames = {f["id"]: f for f in json.load(open(frames_path))}
    gps = json.load(open(gps_path))
    G = np.array([[g["global"][0], g["global"][1]] for g in gps])  # lat, lon
    gtime = np.array([g["timestamp"] for g in gps])
    lat0, lon0 = G.mean(0)
    Re = 6378137.0
    coslat = np.cos(np.radians(lat0))
    E = np.radians(G[:, 1] - lon0) * Re * coslat
    N = np.radians(G[:, 0] - lat0) * Re

    ft = {fid: f["timestamp"] for fid, f in frames.items()}
    xz, T = [], []
    for im in recon.images.values():
        fid = int(im.name.split("_")[0])
        if fid in ft:
            c = -qvec2rotmat(im.qvec).T @ im.tvec
            xz.append([c[0], c[2]])
            T.append(ft[fid])
    xz = np.array(xz)
    idx = np.clip(np.searchsorted(gtime, np.array(T)), 0, len(gtime) - 1)
    EN = np.c_[E[idx], N[idx]]
    s, R, t = _umeyama(xz, EN)
    res = float(np.sqrt((((xz @ (s * R).T) + t - EN) ** 2).sum(1)).mean())

    def to_wgs84(world_xz: np.ndarray) -> np.ndarray:
        en = world_xz @ (s * R).T + t
        lat = lat0 + np.degrees(en[:, 1] / Re)
        lon = lon0 + np.degrees(en[:, 0] / (Re * coslat))
        return np.c_[lon, lat]

    return to_wgs84, {"residual_m": res, "scale": float(s), "ref_latlon": [float(lat0), float(lon0)]}


# ---- polygonisation -----------------------------------------------------------------

def _pix_to_world(pts_xy: np.ndarray, meta: dict) -> np.ndarray:
    """cv2 contour pixels (col=x, row=y) -> world (X,Z) metres."""
    world = np.empty_like(pts_xy, dtype=np.float64)
    world[:, 0] = meta["origin_u"] + pts_xy[:, 0] * meta["cell_size"]  # X
    world[:, 1] = meta["origin_v"] + pts_xy[:, 1] * meta["cell_size"]  # Z
    return world


def mask_to_polygons(binary: np.ndarray, meta: dict, simplify_m: float = 0.5,
                     min_area_m2: float = 2.0) -> list[Polygon]:
    """Binary raster -> clean shapely polygons (with holes) in world metres."""
    b = binary.astype(np.uint8)
    contours, hierarchy = cv2.findContours(b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    polys: list[Polygon] = []
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] != -1 or len(cnt) < 3:      # skip holes (handled) / degenerate
            continue
        shell = _pix_to_world(cnt.reshape(-1, 2), meta)
        holes = [_pix_to_world(contours[j].reshape(-1, 2), meta)
                 for j in range(len(contours)) if hierarchy[j][3] == i and len(contours[j]) >= 3]
        try:
            poly = Polygon(shell, holes).buffer(0)      # buffer(0) repairs self-intersections
        except Exception:
            continue
        for p in (poly.geoms if poly.geom_type == "MultiPolygon" else [poly]):
            p = p.simplify(simplify_m, preserve_topology=True)
            if p.is_valid and not p.is_empty and p.area >= min_area_m2:
                polys.append(orient(p, sign=1.0))       # exterior CCW (GeoJSON/IMDF)
    return polys


def smooth_polygon(poly: Polygon, radius: float, simplify_m: float,
                   min_hole_m2: float = 2.0) -> list[Polygon]:
    """Cartographic smoothing: rounded morphological open+close (removes pixel-staircase and
    thin protrusions/nicks), then simplify vertices and drop tiny holes. Returns 0+ polygons."""
    p = (poly.buffer(-radius, join_style=1)      # open: erode+dilate -> drop protrusions
         .buffer(2 * radius, join_style=1)        # ...then close: dilate+erode -> fill nicks
         .buffer(-radius, join_style=1))          # (round joins => smooth curves)
    if p.is_empty:
        return []
    out = []
    for q in (p.geoms if p.geom_type == "MultiPolygon" else [p]):
        q = q.simplify(simplify_m, preserve_topology=True)
        if not q.is_valid:
            q = q.buffer(0)
        for r in (q.geoms if q.geom_type == "MultiPolygon" else [q]):
            if r.geom_type != "Polygon" or r.is_empty:
                continue
            holes = [h for h in r.interiors if Polygon(h).area >= min_hole_m2]
            r = Polygon(r.exterior, holes)
            if r.is_valid and r.area > 0:
                out.append(orient(r, sign=1.0))
    return out


# ---- IMDF assembly ------------------------------------------------------------------

def _feature(feature_type: str, geometry, props: dict) -> dict:
    return {"id": str(uuid.uuid4()), "type": "Feature", "feature_type": feature_type,
            "geometry": geometry, "properties": props}


def _labels(name: str) -> dict:
    return {"en": name}


def _display_point(poly: Polygon, to_wgs84) -> dict:
    p = poly.representative_point()
    lon, lat = to_wgs84(np.array([[p.x, p.y]]))[0]
    return {"type": "Point", "coordinates": [float(lon), float(lat)]}


def _geom_wgs84(poly: Polygon, to_wgs84) -> dict:
    """shapely polygon in world metres -> GeoJSON geometry in WGS84."""
    g = mapping(poly)
    g["coordinates"] = [[[float(x), float(y)] for x, y in to_wgs84(np.array(ring)).tolist()]
                        for ring in [np.array(r) for r in g["coordinates"]]]
    return g


def build_imdf(level_dir: str | Path, model_dir: str | Path, frames_path: str | Path,
               gps_path: str | Path, venue_name: str = "Maguro Park",
               venue_category: str = "themepark", locality: str = "Tokyo",
               country: str = "JP", smooth: bool = True, log=print) -> dict:
    d = Path(level_dir)
    meta = json.load(open(d / "level0.meta.json"))
    npz = np.load(d / "level0.npz")
    semantic, coverage = npz["semantic"], npz["coverage"]

    log("georeferencing (camera trajectory vs GPS)...")
    recon = read_model(model_dir)
    to_wgs84, geo = fit_world_to_wgs84(recon, frames_path, gps_path)
    log(f"  fit residual {geo['residual_m']:.1f} m (GPS-limited), scale {geo['scale']:.3f}")

    # unit category -> the internal classes that map to it (via taxonomy)
    cat_classes: dict[str, list[int]] = {}
    for c in CLASSES:
        if c.imdf and c.imdf[0] == "unit":
            cat_classes.setdefault(c.imdf[1], []).append(int(c.klass))

    address = _feature("address", None,
                       {"address": venue_name, "unit": None, "locality": locality,
                        "province": locality, "country": country, "postal_code": None,
                        "postal_code_ext": None, "postal_code_vanity": None})

    cs = meta["cell_size"]
    smooth_r, simp_m = 1.4 * cs, 0.8 * cs             # smoothing radius / simplify tol in metres

    # venue / level boundary = outer boundary of observed coverage (gaps closed)
    cov = cv2.morphologyEx(coverage.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    boundary_polys = mask_to_polygons(cov, meta, simplify_m=1.0, min_area_m2=20.0)
    if not boundary_polys:
        raise ValueError("no coverage to form a venue boundary")
    venue_poly = Polygon(max(boundary_polys, key=lambda p: p.area).exterior)  # no holes
    if smooth:
        parts = smooth_polygon(venue_poly, smooth_r * 2, simp_m * 2)
        if parts:
            venue_poly = Polygon(max(parts, key=lambda p: p.area).exterior)

    venue = _feature("venue", _geom_wgs84(venue_poly, to_wgs84),
                     {"category": venue_category, "restriction": None, "name": _labels(venue_name),
                      "alt_name": None, "hours": None, "phone": None, "website": None,
                      "display_point": _display_point(venue_poly, to_wgs84),
                      "address_id": address["id"]})

    level = _feature("level", _geom_wgs84(venue_poly, to_wgs84),
                     {"category": "unspecified", "restriction": None, "outdoor": True,
                      "ordinal": 0, "name": _labels("Ground"), "short_name": _labels("G"),
                      "display_point": _display_point(venue_poly, to_wgs84),
                      "address_id": address["id"], "building_ids": None})

    units = []
    for category, klasses in cat_classes.items():
        if category not in UNIT_CATEGORIES:
            log(f"  skipping unit category '{category}' (not in IMDF enum)")
            continue
        mask = (np.isin(semantic, klasses) & coverage).astype(np.uint8)
        # consolidate before tracing: drop specks (open) then fill pinholes (close).
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        polys = mask_to_polygons(mask, meta, simplify_m=0.6, min_area_m2=5.0)

        # merge touching pieces, then cartographically smooth each region
        merged = unary_union(polys) if polys else None
        parts = ([] if merged is None or merged.is_empty
                 else list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged])
        cat_polys = []
        for part in parts:
            cat_polys.extend(smooth_polygon(part, smooth_r, simp_m) if smooth else [part])
        cat_polys = [p for p in cat_polys if p.area >= 5.0]

        for poly in cat_polys:
            units.append(_feature("unit", _geom_wgs84(poly, to_wgs84),
                                  {"category": category, "restriction": None, "accessibility": None,
                                   "name": None, "alt_name": None, "display_point": None,
                                   "level_id": level["id"]}))
        log(f"  {category}: {len(cat_polys)} unit polygons")

    return {
        "manifest": {"version": "1.0.0", "created": datetime.now(timezone.utc).isoformat(),
                     "generated_by": "lar semantic_bev", "language": "en", "extensions": []},
        "address": [address], "venue": [venue], "level": [level], "unit": units,
        "_geo": geo,
    }


def write_archive(imdf: dict, out_dir: str | Path, log=print) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(imdf["manifest"], indent=2))
    for ftype in ("address", "venue", "level", "unit"):
        fc = {"type": "FeatureCollection", "name": ftype, "features": imdf[ftype]}
        (out / f"{ftype}.geojson").write_text(json.dumps(fc))
    log(f"wrote IMDF archive -> {out} "
        f"({len(imdf['unit'])} units, {len(imdf['level'])} level, {len(imdf['venue'])} venue)")


# ---- self-validation ----------------------------------------------------------------

def validate(imdf: dict, log=print) -> bool:
    errs: list[str] = []
    ids = set()
    for ftype in ("address", "venue", "level", "unit"):
        for f in imdf[ftype]:
            if f["id"] in ids:
                errs.append(f"duplicate id {f['id']}")
            ids.add(f["id"])
            if f["feature_type"] != ftype:
                errs.append(f"{ftype}: wrong feature_type {f['feature_type']}")
            if ftype != "address" and (f["geometry"] is None or f["geometry"]["type"] != "Polygon"):
                errs.append(f"{ftype}: geometry must be Polygon")
            # WGS84 bounds
            if f["geometry"]:
                for ring in f["geometry"]["coordinates"]:
                    for lon, lat in ring:
                        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                            errs.append(f"{ftype}: coord out of WGS84 bounds ({lon},{lat})")
                            break
    cat_ok = {"venue": VENUE_CATEGORIES, "level": LEVEL_CATEGORIES, "unit": UNIT_CATEGORIES}
    for ftype, allowed in cat_ok.items():
        for f in imdf[ftype]:
            if f["properties"].get("category") not in allowed:
                errs.append(f"{ftype}: invalid category {f['properties'].get('category')!r}")
    # foreign keys
    level_ids = {f["id"] for f in imdf["level"]}
    addr_ids = {f["id"] for f in imdf["address"]}
    for u in imdf["unit"]:
        if u["properties"]["level_id"] not in level_ids:
            errs.append("unit: dangling level_id")
    for f in imdf["venue"] + imdf["level"]:
        if f["properties"].get("address_id") not in addr_ids:
            errs.append(f"{f['feature_type']}: dangling address_id")
    # geometry validity via shapely
    for ftype in ("venue", "level", "unit"):
        for f in imdf[ftype]:
            rings = f["geometry"]["coordinates"]
            if not Polygon(rings[0], rings[1:]).is_valid:
                errs.append(f"{ftype}: invalid polygon geometry")

    if errs:
        for e in errs[:20]:
            log(f"  INVALID: {e}")
        log(f"self-validation FAILED ({len(errs)} issues)")
        return False
    log("self-validation PASSED (enums, FKs, geometry, WGS84 bounds)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Vectorise semantic BEV -> IMDF archive")
    ap.add_argument("--level", required=True, help="dir with level0.npz / .meta.json")
    ap.add_argument("--model", required=True, help="COLMAP poses_txt dir")
    ap.add_argument("--frames", required=True, help="frames.json (ARKit)")
    ap.add_argument("--gps", required=True, help="gps.json")
    ap.add_argument("--out", default=None, help="archive dir (default <level>/imdf)")
    ap.add_argument("--venue-name", default="Maguro Park")
    ap.add_argument("--venue-category", default="themepark", choices=sorted(VENUE_CATEGORIES))
    ap.add_argument("--no-smooth", action="store_true", help="disable cartographic smoothing")
    args = ap.parse_args()

    imdf = build_imdf(args.level, args.model, args.frames, args.gps,
                      venue_name=args.venue_name, venue_category=args.venue_category,
                      smooth=not args.no_smooth)
    ok = validate(imdf)
    write_archive(imdf, args.out or (Path(args.level) / "imdf"))
    if not ok:
        raise SystemExit("IMDF self-validation failed")


if __name__ == "__main__":
    main()
