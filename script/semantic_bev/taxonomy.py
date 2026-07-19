"""The semantic contract for the BEV pipeline.

This is the one place that reconciles three vocabularies:

  1. **Open-vocab prompts** -- free text fed to the segmenter (many phrasings per class,
     because open-vocab models respond to wording).
  2. **Internal classes** -- the fixed enum stored in ``Level.semantic`` rasters and used
     for ground/occupancy logic.
  3. **IMDF categories** -- Apple's *closed* enums (``unit.category`` / ``amenity.category``)
     that the validator accepts. Open-vocab freedom must collapse onto these.

Keeping all three mappings here means the rest of the pipeline is vocabulary-agnostic:
segmenter emits internal class ids, ground/occupancy read ``role``, IMDF export reads
``imdf``. Add a new class in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Role(IntEnum):
    """How a class participates in the 2.5D model."""

    IGNORE = 0     # dynamic / unreliable -> excluded from geometry (people, vehicles, sky)
    GROUND = 1     # defines the walkable surface height field
    OBSTACLE = 2   # static, above-ground -> occupancy + IMDF fixtures/amenities


class Klass(IntEnum):
    """Internal fixed enum. Stored as uint8 in semantic rasters."""

    UNKNOWN = 0
    PATH = 1        # paved walkway / trail
    GRASS = 2       # lawn
    TERRAIN = 3     # dirt / sand / gravel / bare ground
    PAVEMENT = 4    # plaza / hardscape (non-path paved ground)
    WATER = 5
    STAIRS = 6
    TREE = 7
    BUILDING = 8
    WALL = 9        # wall / fence / railing
    FURNITURE = 10  # bench, table, bin, sign, light pole
    PERSON = 11
    VEHICLE = 12
    SKY = 13        # bright sky through the canopy -> ignore (never ground/obstacle)


@dataclass(frozen=True)
class ClassDef:
    klass: Klass
    role: Role
    prompts: list[str]                # open-vocab phrasings -> this class
    imdf: tuple[str, str] | None = None  # (feature_type, category) for strict IMDF, or None
    color: tuple[int, int, int] = (0, 0, 0)  # RGB for visualisation


# Prompts tuned for a wooded Tokyo park (Rinshi-no-mori): asphalt paths + dirt forest
# trails + leaf litter + low undergrowth + tall trees, plus joggers/cyclists.
# Order matters only for stable prompt indexing; lookups are by Klass.
CLASSES: list[ClassDef] = [
    ClassDef(Klass.UNKNOWN, Role.IGNORE, [], None, (0, 0, 0)),
    ClassDef(Klass.PATH, Role.GROUND,
             ["an asphalt walking path", "paved asphalt road", "asphalt pavement",
              "a paved footpath"],
             ("unit", "walkway"), (222, 184, 135)),
    ClassDef(Klass.GRASS, Role.GROUND,
             ["grass lawn", "green grass", "low green ground cover plants"],
             ("unit", "vegetation"), (120, 190, 100)),
    ClassDef(Klass.TERRAIN, Role.GROUND,
             ["bare dirt ground", "forest floor with fallen leaves", "a dirt trail",
              "gravel ground"],
             ("unit", "vegetation"), (160, 130, 90)),
    ClassDef(Klass.PAVEMENT, Role.GROUND,
             ["stone plaza pavement", "brick paving", "a concrete plaza"],
             ("unit", "walkway"), (190, 190, 195)),
    ClassDef(Klass.WATER, Role.OBSTACLE,
             ["pond water", "a water surface", "a lake"],
             ("unit", "structure"), (70, 130, 200)),
    ClassDef(Klass.STAIRS, Role.GROUND,
             ["outdoor stairs", "stone steps", "a staircase"],
             ("unit", "stairs"), (210, 160, 210)),
    ClassDef(Klass.TREE, Role.OBSTACLE,
             ["a tree", "a tree trunk", "green tree canopy", "a bush", "a shrub", "a hedge"],
             ("amenity", "landmark"), (30, 110, 40)),
    ClassDef(Klass.BUILDING, Role.OBSTACLE,
             ["a building", "a building wall", "a small park structure", "a restroom building"],
             ("unit", "structure"), (150, 110, 90)),
    ClassDef(Klass.WALL, Role.OBSTACLE,
             ["a fence", "a railing", "a low stone wall", "a wooden fence"],
             ("fixture", "wall"), (130, 130, 130)),
    ClassDef(Klass.FURNITURE, Role.OBSTACLE,
             ["a park bench", "a trash can", "a signboard", "a lamp post", "a wooden bench"],
             ("amenity", "furniture"), (230, 200, 60)),
    ClassDef(Klass.PERSON, Role.IGNORE,
             ["a person", "a pedestrian", "a jogger"],
             None, (255, 0, 0)),
    ClassDef(Klass.VEHICLE, Role.IGNORE,
             ["a bicycle", "a car", "a baby stroller"],
             None, (255, 120, 0)),
    ClassDef(Klass.SKY, Role.IGNORE,
             ["the bright sky", "sky through the trees"],
             None, (150, 200, 235)),
]

_BY_KLASS: dict[Klass, ClassDef] = {c.klass: c for c in CLASSES}


def by_klass(k: Klass | int) -> ClassDef:
    return _BY_KLASS[Klass(k)]


def role_of(k: Klass | int) -> Role:
    return _BY_KLASS[Klass(k)].role


def color_lut() -> "list[tuple[int, int, int]]":
    """RGB per Klass id, index-aligned (for palette PNG output)."""
    lut = [(0, 0, 0)] * (max(Klass) + 1)
    for c in CLASSES:
        lut[int(c.klass)] = c.color
    return lut


# Keyword -> Klass, for mapping *closed-set* model labels (e.g. ADE20K's 150 names) onto our
# taxonomy. Checked in order; first substring hit wins, so put specific before generic.
_KEYWORD_KLASS: list[tuple[str, Klass]] = [
    # --- explicit disambiguations, must precede the generic keys below ---
    # Word boundaries fix the accidental hits, but a few labels genuinely contain a keyword
    # while meaning something else, and a few lost their (accidentally correct) mapping once
    # the loose match was removed. Both are named here rather than left to chance.
    ("pool table", Klass.UNKNOWN), ("billiard table", Klass.UNKNOWN),  # indoor, not WATER
    ("skyscraper", Klass.BUILDING),                                    # a building, not SKY
    ("playing field", Klass.GRASS), ("soccer field", Klass.GRASS),
    ("seat", Klass.FURNITURE),
    ("streetlight", Klass.FURNITURE), ("street lamp", Klass.FURNITURE),
    ("vending machine", Klass.FURNITURE), ("kiosk", Klass.BUILDING),
    ("bike rack", Klass.FURNITURE), ("picnic table", Klass.FURNITURE),
    ("planter", Klass.FURNITURE), ("bollard", Klass.FURNITURE),
    ("statue", Klass.FURNITURE), ("monument", Klass.FURNITURE),
    # --- compounds that the OLD loose match happened to get right ---
    # Anchoring alone is a net regression: "stairway"/"escalator" used to hit via "stair" in
    # "staircase", "waterfall" via "water", "minibike" via "bike", "traffic light" via "sign"
    # in "signal". Substring matching was accidentally correct about as often as it was wrong,
    # so each compound has to be named rather than left to a lucky collision.
    ("stairway", Klass.STAIRS), ("staircase", Klass.STAIRS), ("escalator", Klass.STAIRS),
    ("waterfall", Klass.WATER),
    ("minibike", Klass.VEHICLE), ("motorbike", Klass.VEHICLE), ("motorcycle", Klass.VEHICLE),
    ("boat", Klass.VEHICLE), ("ship", Klass.VEHICLE), ("bus", Klass.VEHICLE),
    ("airplane", Klass.VEHICLE), ("aeroplane", Klass.VEHICLE),
    ("tower", Klass.BUILDING), ("grandstand", Klass.BUILDING),
    ("traffic light", Klass.FURNITURE), ("traffic signal", Klass.FURNITURE),
    ("stoplight", Klass.FURNITURE), ("light source", Klass.FURNITURE),
    ("column", Klass.FURNITURE), ("pillar", Klass.FURNITURE), ("pedestal", Klass.FURNITURE),
    ("flowerpot", Klass.FURNITURE), ("vase", Klass.FURNITURE),
    ("bulletin board", Klass.FURNITURE), ("notice board", Klass.FURNITURE),
    ("sidewalk", Klass.PATH), ("pavement", Klass.PAVEMENT), ("runway", Klass.PATH),
    ("road", Klass.PATH), ("path", Klass.PATH),
    ("stair", Klass.STAIRS), ("step", Klass.STAIRS),
    ("grass", Klass.GRASS), ("lawn", Klass.GRASS), ("flower", Klass.GRASS),
    ("field", Klass.GRASS),          # ADE "field" is a grass field, not bare ground
    ("palm", Klass.TREE), ("tree", Klass.TREE), ("plant", Klass.TREE),
    ("bush", Klass.TREE), ("shrub", Klass.TREE), ("hedge", Klass.TREE),
    ("earth", Klass.TERRAIN), ("ground", Klass.TERRAIN), ("soil", Klass.TERRAIN),
    ("dirt", Klass.TERRAIN), ("land", Klass.TERRAIN),
    ("sand", Klass.TERRAIN), ("gravel", Klass.TERRAIN), ("hill", Klass.TERRAIN),
    ("mountain", Klass.TERRAIN), ("rock", Klass.TERRAIN),
    ("water", Klass.WATER), ("sea", Klass.WATER), ("river", Klass.WATER),
    ("lake", Klass.WATER), ("pool", Klass.WATER), ("fountain", Klass.WATER),
    ("fence", Klass.WALL), ("railing", Klass.WALL), ("bannister", Klass.WALL),
    ("wall", Klass.WALL),
    ("building", Klass.BUILDING), ("house", Klass.BUILDING), ("hut", Klass.BUILDING),
    ("hovel", Klass.BUILDING), ("booth", Klass.BUILDING),
    ("bench", Klass.FURNITURE), ("pole", Klass.FURNITURE), ("signboard", Klass.FURNITURE),
    ("sign", Klass.FURNITURE), ("streetlight", Klass.FURNITURE), ("lamp", Klass.FURNITURE),
    ("sculpture", Klass.FURNITURE), ("ashcan", Klass.FURNITURE), ("trash", Klass.FURNITURE),
    ("person", Klass.PERSON), ("bicycle", Klass.VEHICLE), ("bike", Klass.VEHICLE),
    ("car", Klass.VEHICLE), ("van", Klass.VEHICLE), ("truck", Klass.VEHICLE),
    ("sky", Klass.SKY),
]


_KW_RE: "list[tuple[object, Klass]]" = []


def keyword_klass(name: str) -> Klass:
    """Map an arbitrary closed-set label name to our taxonomy by keyword; UNKNOWN if no hit.

    Matching is on WORD BOUNDARIES, not raw substrings. Plain ``kw in name`` silently produced
    real mislabels, because our keywords are short and English is full of them:

        "seat"           contained "sea"    -> WATER
        "streetlight"    contained "tree"   -> TREE     (s-TREE-t)
        "rug, carpet"    contained "car"    -> VEHICLE
        "kitchen island" contained "land"   -> TERRAIN
        "skyscraper"     started  "sky"     -> SKY

    Ordering cannot fix this -- ``streetlight`` was already listed as FURNITURE, it just lost
    the race to ``tree``. Only anchoring the match does. These were not hypothetical: the park
    BEV carried 9 WATER cells (no water in the park) and misfiled street lamps as trees.

    A name may be a comma-separated synonym list ("rug, carpet, carpeting"); boundary matching
    spans it naturally, so callers should pass the whole string rather than splitting and
    combining, which had no principled way to break ties.
    """
    global _KW_RE
    if not _KW_RE:
        import re
        # ``s?`` because the keys are singular but label sets are not ("stair" must still
        # catch "stairs, steps"). Anchoring without this silently drops every plural -- it sent
        # STAIRS to UNKNOWN on the first attempt.
        _KW_RE = [(re.compile(r"\b" + re.escape(kw) + r"s?\b"), k) for kw, k in _KEYWORD_KLASS]
    n = name.lower()
    for rx, k in _KW_RE:
        if rx.search(n):
            return k
    return Klass.UNKNOWN


@dataclass
class PromptTable:
    """Flattened prompt list + mapping back to Klass, for open-vocab segmenters."""

    prompts: list[str] = field(default_factory=list)
    prompt_klass: list[Klass] = field(default_factory=list)

    @classmethod
    def build(cls) -> "PromptTable":
        t = cls()
        for c in CLASSES:
            for p in c.prompts:
                t.prompts.append(p)
                t.prompt_klass.append(c.klass)
        return t
