"""Pre-checks before screening colour corrections for the regressor's crop.

Report Section 12, step 3 plans to extend the gray-world test of Section 7.7 to
two more corrections: take the brightest pixels as white, or take the sclera as
white. Each has a known weakness, and both are checked here before any
haemoglobin model is fitted:

  1. **Clipping census.** Glare saturates at 255. A clipped pixel no longer
     records the colour of the light, and if every channel reaches 255 somewhere
     in the photograph, "scale each channel so its maximum is white" applies no
     correction at all. Counted at the original resolution and in the 512-pixel
     frame where `prepare()` computes the correction, and located: on the
     conjunctiva, below the lower lid, or elsewhere.
  2. **Sclera colour against age and haemoglobin.** The sclera yellows with age
     and with jaundice. If its measured colour tracks age or haemoglobin, dividing
     the conjunctiva by it adds or removes something other than the light.

The white plastic of the capture device is visible below the lower lid in almost
every photograph. It is a fixed white object, so its colour records only the
light and the phone's own processing. It serves two purposes: as the reference
against which each estimate of the light is compared (angular difference), and
to remove the light from the sclera's colour (sclera b* minus plastic b*), so
that what remains is the sclera itself.

Nothing is trained and no haemoglobin model is fitted. The conjunctiva mask comes
from the deployed U-Net, given the input it was trained on.

The sclera and plastic detectors were fixed by looking at overlays only, before
any colour was compared with age or haemoglobin. Every sclera overlay was then
graded by eye from index-labelled sheets in shuffled order, so that site, age and
colour were hidden (`sclera_grading.json`).

    python experiments/26_correction_prechecks/run.py              # reuse measurements.csv
    python experiments/26_correction_prechecks/run.py --remeasure  # run the extractor again
    python experiments/26_correction_prechecks/run.py --sheets     # write the grading sheets

Abbreviations
-------------
Hb    haemoglobin (g/dL)
b*    CIELAB yellow-blue axis; positive is yellow, negative is blue
a*    CIELAB red-green axis; positive is red
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "datasets/eyes_defy/dataset anemia"
MEASURED = HERE / "measurements.csv"
GRADING = HERE / "sclera_grading.json"

WORK = 512            # the frame prepare() balances (PreprocessConfig.work_size)
NEAR_CLIP = 250       # a pixel with any channel at or above this is not trusted
TOP_BRIGHT = 0.01     # brightest share of unclipped pixels used as "white"
SCLERA_TOP = 0.30     # sclera = brightest 30% of the band above the conjunctiva
SCLERA_A_MAX = 20.0   # drop clearly red pixels (vessels, tissue); not a cut on b*
BAND, MARGIN = 220, 20
PLASTIC_MIN = 3000    # unclipped plastic pixels needed to use it as a reference
BOOTSTRAP, SEED = 2000, 0
ESTIMATES = ("gray_world", "max_rgb", "bright_frame", "bright_eye", "sclera")
LABEL = {"gray_world": "Gray-world (frame mean)", "max_rgb": "Max-RGB (512 frame)",
         "bright_frame": "Brightest 1% unclipped, whole frame",
         "bright_eye": "Brightest 1% unclipped, above the lower lid",
         "sclera": "Sclera mean"}


# ---------------------------------------------------------------- detectors
def lab(img: np.ndarray) -> np.ndarray:
    """CIELAB (D65) of an 8-bit sRGB image: L* 0-100, a* and b* signed."""

    return cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


def near_clipped(img: np.ndarray) -> np.ndarray:
    """Pixels at or near saturation, grown by 2 px to catch their blurred rims."""

    near = (img >= NEAR_CLIP).any(-1).astype(np.uint8)
    return cv2.dilate(near, np.ones((5, 5), np.uint8)) > 0


def _edge(conj: np.ndarray, top: bool) -> np.ndarray:
    """Per-column top (or bottom) row of the conjunctiva, ends extended flat."""

    h, w = conj.shape
    cols = np.flatnonzero(conj.any(0))
    rows = np.argmax(conj[:, cols], axis=0) if top else h - 1 - np.argmax(conj[::-1][:, cols], axis=0)
    return np.interp(np.arange(w), cols, rows)


def eye_band(conj: np.ndarray) -> np.ndarray:
    """The band just above the everted lower lid, where the sclera lies.

    Every photograph has the lid at the bottom (the conjunctiva's centroid is in
    the lower half in all 217), so "above" is well defined. The band stops 220 px
    above the lid, which keeps the upper eyelid's skin out.
    """

    rows = np.arange(conj.shape[0])[:, None]
    edge = _edge(conj, top=True)[None, :]
    return (rows < edge - MARGIN) & (rows >= edge - BAND)


def below_lid(conj: np.ndarray, margin: int = 10) -> np.ndarray:
    rows = np.arange(conj.shape[0])[:, None]
    return rows > _edge(conj, top=False)[None, :] + margin


def sclera_mask(img: np.ndarray, conj: np.ndarray) -> np.ndarray:
    """Brightest 30% of the band, minus clipped and clearly red pixels, largest piece.

    Selection is on lightness, not on chroma, so a yellow sclera is not dropped
    for being yellow; only pixels redder than a* 20 (vessels, tissue) are removed.
    """

    L = lab(img)
    valid = eye_band(conj) & ~near_clipped(img)
    if valid.sum() < 100:
        return np.zeros_like(conj)
    thr = np.quantile(L[..., 0][valid], 1 - SCLERA_TOP)
    cand = valid & (L[..., 0] >= thr) & (L[..., 1] <= SCLERA_A_MAX)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(cand, 8)
    if n <= 1:
        return np.zeros_like(conj)
    return cc == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])


def plastic_mask(img: np.ndarray, conj: np.ndarray) -> np.ndarray:
    """The device's white plastic: bottom 15% of the frame, well below the lid.

    A loose chroma limit (C* <= 30) keeps out pale skin; it trims a strongly
    tinted plastic, which can only shrink, not create, a link with the sclera.
    """

    L = lab(img)
    rows = np.arange(conj.shape[0])[:, None]
    region = (rows > _edge(conj, top=False)[None, :] + 30) & (rows >= int(0.85 * conj.shape[0]))
    chroma = np.hypot(L[..., 1], L[..., 2])
    return region & ~near_clipped(img) & (L[..., 0] >= 75) & (chroma <= 30)


# ---------------------------------------------------------------- measuring
def mean_rgb(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return img[mask].reshape(-1, 3).astype(np.float64).mean(0) if mask.any() else np.full(3, np.nan)


def share(weights: np.ndarray, region: np.ndarray) -> float:
    total = weights.sum()
    return float((weights * region).sum() / total) if total > 0 else float("nan")


def measure_one(full: np.ndarray, seg) -> tuple[dict, dict]:
    from anemia.imaging import gray_world_white_balance, resize

    row: dict = {}
    clip_full = (full == 255).any(-1)
    row["full_clip_any"] = float(clip_full.mean())
    row["full_clip_all"] = float((full == 255).all(-1).mean())
    row["full_max_is_white"] = bool((full.reshape(-1, 3).max(0) == 255).all())

    small = resize(full, (WORK, WORK))
    seg_input = gray_world_white_balance(small) if seg.wants_gray_world else small
    conj = seg(seg_input)[0] > 0                      # the extractor's own training condition
    row["small_clip_any"] = float((small == 255).any(-1).mean())
    row["small_max_is_white"] = bool((small.reshape(-1, 3).max(0) == 255).all())

    # Where the original-resolution clipped pixels fall, in 512-frame regions.
    weight = cv2.resize(clip_full.astype(np.float32), (WORK, WORK), interpolation=cv2.INTER_AREA)
    below = below_lid(conj)
    row["clip_on_conjunctiva"] = share(weight, conj)
    row["clip_below_lid"] = share(weight, below & ~conj)
    row["clip_elsewhere"] = share(weight, ~below & ~conj)
    row["conj_px"] = int(conj.sum())
    row["conj_clipped_frac"] = float((weight[conj] > 0).mean())

    # The brightest unclipped pixels: where they are, and the white they imply.
    unclipped = ~near_clipped(small)
    total = small.astype(np.int32).sum(-1)
    sel = unclipped & (total >= np.quantile(total[unclipped], 1 - TOP_BRIGHT))
    sclera = sclera_mask(small, conj)
    plastic = plastic_mask(small, conj)
    for name, region in (("below_lid", below & ~conj), ("sclera", sclera), ("conjunctiva", conj)):
        row[f"bright_on_{name}"] = float((sel & region).sum() / sel.sum())
    eye = unclipped & ~below
    sel_eye = eye & (total >= np.quantile(total[eye], 1 - TOP_BRIGHT))

    est = {"gray_world": small.reshape(-1, 3).astype(np.float64).mean(0),
           "max_rgb": small.reshape(-1, 3).max(0).astype(np.float64),
           "bright_frame": mean_rgb(small, sel), "bright_eye": mean_rgb(small, sel_eye),
           "sclera": mean_rgb(small, sclera), "plastic": mean_rgb(small, plastic),
           "conjunctiva": mean_rgb(small, conj)}
    for name, v in est.items():
        row.update({f"{name}_{c}": float(x) for c, x in zip("RGB", v)})

    L = lab(small)
    for name, m in (("sclera", sclera), ("plastic", plastic)):
        row[f"{name}_px"] = int(m.sum())
        for k, ch in enumerate(("L", "a", "b")):
            row[f"{name}_{ch}"] = float(np.median(L[..., k][m])) if m.any() else float("nan")
    return row, {"small": small, "conj": conj, "sclera": sclera, "plastic": plastic}


def measure(remeasure: bool) -> pd.DataFrame:
    if MEASURED.exists() and not remeasure:
        return pd.read_csv(MEASURED)
    from anemia.data import load_dataset
    from anemia.imaging import read_rgb
    from anemia.segment import get_extractor

    seg = get_extractor("unet")
    rows = []
    for s in load_dataset(DATA):
        row, _ = measure_one(read_rgb(s.image_path), seg)
        rows.append({"patient": s.patient_id, "site": s.source.split("/")[-1], "hb": s.hb,
                     "age": s.age_years, "sex": s.sex, "file": Path(s.image_path).name, **row})
        print(f"  {len(rows):3d} {s.patient_id}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(MEASURED, index=False)
    return df


def write_sheets() -> None:
    """Index-labelled sclera overlays in shuffled order, 20 per sheet, for grading."""

    from anemia.data import load_dataset
    from anemia.imaging import read_rgb
    from anemia.segment import get_extractor

    seg = get_extractor("unet")
    samples = load_dataset(DATA)
    order = np.random.default_rng(123).permutation(len(samples))
    out = HERE / "grading"
    out.mkdir(exist_ok=True)
    tiles, key = [], {}
    for n, i in enumerate(order):
        _, m = measure_one(read_rgb(samples[i].image_path), seg)
        ov = m["small"].copy()
        ov[m["sclera"]] = (0.45 * ov[m["sclera"]] + 0.55 * np.array([0, 120, 255])).astype(np.uint8)
        cv2.putText(ov, str(i), (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 0), 3)
        tiles.append(cv2.resize(ov, (230, 230)))
        key[int(i)] = samples[i].patient_id
        if len(tiles) == 20 or n == len(order) - 1:
            while len(tiles) % 5:
                tiles.append(np.full((230, 230, 3), 255, np.uint8))
            sheet = np.vstack([np.hstack(tiles[k:k + 5]) for k in range(0, len(tiles), 5)])
            cv2.imwrite(str(out / f"sheet_{n // 20:02d}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
            tiles = []
    (out / "index_to_patient.json").write_text(json.dumps(key, indent=1))
    print(f"wrote {out}")


# ---------------------------------------------------------------- analysis
def angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle in degrees between colour vectors, row by row (the standard error
    measure for an estimate of the light: it ignores brightness)."""

    cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def rgb(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[[f"{name}_{c}" for c in "RGB"]].to_numpy(float)


def spearman(x, y) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    r, p = stats.spearmanr(x[ok], y[ok])
    return {"rho": float(r), "p": float(p), "n": int(ok.sum())}


def ols_boot(df: pd.DataFrame, target: str, seed: int = SEED) -> dict:
    """target ~ age (per decade) + Hb + female, with a patient bootstrap CI."""

    d = df.dropna(subset=[target])
    X = np.column_stack([np.ones(len(d)), d["age"] / 10, d["hb"], (d["sex"] == "F").astype(float)])
    y = d[target].to_numpy(float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, len(y), len(y))
        boots.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    names = ["intercept", "per_decade", "per_g_dl_hb", "female"]
    return {"n": len(y), **{n: {"coef": float(c), "ci": [float(a), float(b)]}
                            for n, c, a, b in zip(names, coef, lo, hi)},
            "sd_target": float(y.std(ddof=1))}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj.tolist()


def leakage(df: pd.DataFrame) -> dict:
    """What dividing the conjunctiva by each estimate of the light would do to it.

    A per-channel correction by an estimate E changes the conjunctiva's
    log(R/G) by exactly -log(E_R/E_G). So if log(E_R/E_G) rises with Hb, the
    correction removes that much of the conjunctiva's own Hb slope, and if it
    moves with age, the correction adds age to the feature. Both are fitted
    together (feature ~ Hb + age per decade + female), within each site. The
    plastic is the control: a fixed white object should carry neither.
    Reported: share of the conjunctiva's Hb slope removed, and the age shift
    expressed as apparent g/dL per decade (age slope / conjunctiva Hb slope).
    """

    out = {}
    for site, pick in GROUPS.items():
        d = df[pick(df) & df.sclera_ok & (df.plastic_px >= PLASTIC_MIN)]
        X = np.column_stack([np.ones(len(d)), d["hb"], d["age"] / 10, (d["sex"] == "F").astype(float)])
        conj = np.log(d["conjunctiva_R"] / d["conjunctiva_G"]).to_numpy()
        feats = {name: np.log(d[f"{name}_R"] / d[f"{name}_G"]).to_numpy()
                 for name in (*ESTIMATES, "plastic")}
        rng = np.random.default_rng(SEED)
        idx = [np.arange(len(d))] + [rng.integers(0, len(d), len(d)) for _ in range(BOOTSTRAP)]
        fit = lambda y, i: np.linalg.lstsq(X[i], y[i], rcond=None)[0]
        conj_fits = np.array([fit(conj, i) for i in idx])
        row = {"n": len(d), "conj_hb_slope": float(conj_fits[0, 1]),
               "conj_hb_slope_ci": np.percentile(conj_fits[1:, 1], [2.5, 97.5]).tolist(),
               "conj_age_slope": float(conj_fits[0, 2])}
        for name, y in feats.items():
            f = np.array([fit(y, i) for i in idx])
            removed = f[:, 1] / conj_fits[:, 1]
            age_equiv = f[:, 2] / conj_fits[:, 1]
            row[name] = {"hb_slope": float(f[0, 1]), "age_slope": float(f[0, 2]),
                         "share_removed": float(removed[0]),
                         "share_removed_ci": np.percentile(removed[1:], [2.5, 97.5]).tolist(),
                         "age_g_dl_per_decade": float(age_equiv[0]),
                         "age_g_dl_per_decade_ci": np.percentile(age_equiv[1:], [2.5, 97.5]).tolist()}
        out[site] = row
    return out


def add_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Italy holds two capture batches, told apart by the file name.

    Patients 1-31 have files named "1.jpg" .. "31.jpg" and were photographed on
    other days (late May to mid July 2019, from EXIF) than the rest, whose files
    are named "T_<n>_<date>_<time>.jpg" and were taken mostly in one week of June.
    Same phone model and manual white balance in both. India is one batch.
    """

    dated = df.file.str.contains(r"20\d{6}", regex=True)
    df["batch"] = np.where(df.site == "india", "india", np.where(dated, "italy_dated", "italy_numbered"))
    return df


GROUPS = {"india": lambda d: d.site == "india", "italy": lambda d: d.site == "italy",
          "italy_dated": lambda d: d.batch == "italy_dated"}


def batches(df: pd.DataFrame) -> dict:
    d = df[df.site == "italy"].copy()
    d["plastic_rg"] = np.where(d.plastic_px >= PLASTIC_MIN, np.log(d.plastic_R / d.plastic_G), np.nan)
    d["conj_rg"] = np.log(d.conjunctiva_R / d.conjunctiva_G)
    d["r_chroma"] = d.conjunctiva_R / d[["conjunctiva_R", "conjunctiva_G", "conjunctiva_B"]].sum(1)
    out = {}
    for b, g in d.groupby("batch"):
        out[b] = {"n": len(g), "hb_mean": float(g.hb.mean()), "age_mean": float(g.age.mean()),
                  "plastic_log_rg_mean": float(g.plastic_rg.mean()),
                  "plastic_log_rg_sd": float(g.plastic_rg.std()),
                  "conj_log_rg_mean": float(g.conj_rg.mean()),
                  "r_vs_hb_pearson": float(np.corrcoef(g.r_chroma, g.hb)[0, 1]),
                  "plastic_rg_vs_hb": spearman(g.plastic_rg.values, g.hb.values)}
    out["pooled_r_vs_hb_pearson"] = float(np.corrcoef(d.r_chroma, d.hb)[0, 1])
    # Conjunctiva log(R/G) slope on Hb (age and sex held), pooled over Italy: as is,
    # with a batch term, and with the light removed (minus plastic log(R/G)).
    e = d[d.sclera_ok & (d.plastic_px >= PLASTIC_MIN)]
    base = [np.ones(len(e)), e.hb, e.age / 10, (e.sex == "F").astype(float)]
    hb_slope = lambda y, extra=(): float(np.linalg.lstsq(np.column_stack(base + list(extra)), y,
                                                          rcond=None)[0][1])
    numbered = (e.batch == "italy_numbered").astype(float)
    out["conj_hb_slope"] = {"n": len(e), "pooled": hb_slope(e.conj_rg),
                            "with_batch_term": hb_slope(e.conj_rg, [numbered]),
                            "minus_plastic": hb_slope(e.conj_rg - e.plastic_rg)}
    for b, g in e.groupby("batch"):
        gb = [np.ones(len(g)), g.hb, g.age / 10, (g.sex == "F").astype(float)]
        out["conj_hb_slope"][b] = float(np.linalg.lstsq(np.column_stack(gb), g.conj_rg, rcond=None)[0][1])
    a, b = d[d.batch == "italy_numbered"], d[d.batch == "italy_dated"]
    out["mann_whitney_p"] = {c: float(stats.mannwhitneyu(a[c].dropna(), b[c].dropna()).pvalue)
                             for c in ("hb", "age", "plastic_rg")}
    return out


def clipping(df: pd.DataFrame) -> dict:
    out = {}
    for site, d in [("india", df[df.site == "india"]), ("italy", df[df.site == "italy"]), ("all", df)]:
        out[site] = {
            "n": len(d),
            "full_max_is_white": float(d.full_max_is_white.mean()),
            "small_max_is_white": float(d.small_max_is_white.mean()),
            "full_clip_any_median": float(d.full_clip_any.median()),
            "full_clip_any_p90": float(d.full_clip_any.quantile(0.9)),
            "full_clip_any_max": float(d.full_clip_any.max()),
            "clip_location_mean": {k: float(d[f"clip_{k}"].mean())
                                   for k in ("on_conjunctiva", "below_lid", "elsewhere")},
            "photos_with_glare_on_conjunctiva": float((d.conj_clipped_frac > 0).mean()),
            "conj_clipped_frac_p90": float(d.conj_clipped_frac.quantile(0.9)),
            "bright_location_mean": {k: float(d[f"bright_on_{k}"].mean())
                                     for k in ("below_lid", "sclera", "conjunctiva")},
        }
    return out


def estimates(df: pd.DataFrame) -> dict:
    out = {}
    has_plastic = df.plastic_px >= PLASTIC_MIN
    for site in ("india", "italy"):
        d = df[(df.site == site) & has_plastic & df.sclera_ok]
        ref = rgb(d, "plastic")
        neutral = np.ones_like(ref)
        row = {"n": len(d), "plastic_from_neutral": float(np.median(angle(ref, neutral)))}
        for name in ESTIMATES:
            a = angle(rgb(d, name), ref)
            row[name] = {"median": float(np.median(a)), "iqr": [float(np.percentile(a, 25)),
                                                                 float(np.percentile(a, 75))],
                         "from_neutral": float(np.median(angle(rgb(d, name), neutral)))}
        row["no_correction"] = {"median": row["plastic_from_neutral"]}
        out[site] = row
    return out


def sclera(df: pd.DataFrame) -> dict:
    d0 = df[df.sclera_ok].copy()
    d0["sclera_b_rel"] = np.where(d0.plastic_px >= PLASTIC_MIN, d0.sclera_b - d0.plastic_b, np.nan)
    d0["sclera_a_rel"] = np.where(d0.plastic_px >= PLASTIC_MIN, d0.sclera_a - d0.plastic_a, np.nan)
    out = {"per_site": {}}
    primary = []
    for site, pick in GROUPS.items():
        d = d0[pick(d0)]
        res = {"n": len(d),
               "age_range": [float(d.age.min()), float(d.age.max())],
               "b_vs_age": spearman(d.sclera_b.values, d.age.values),
               "b_vs_hb": spearman(d.sclera_b.values, d.hb.values),
               "a_vs_age": spearman(d.sclera_a.values, d.age.values),
               "a_vs_hb": spearman(d.sclera_a.values, d.hb.values),
               "age_vs_hb": spearman(d.age.values, d.hb.values),
               "b_rel_vs_age": spearman(d.sclera_b_rel.values, d.age.values),
               "b_rel_vs_hb": spearman(d.sclera_b_rel.values, d.hb.values),
               "a_rel_vs_hb": spearman(d.sclera_a_rel.values, d.hb.values),
               "a_rel_vs_age": spearman(d.sclera_a_rel.values, d.age.values),
               "sclera_b_vs_plastic_b": spearman(d.sclera_b.values, d.plastic_b.where(d.plastic_px >= PLASTIC_MIN).values),
               "ols_b": ols_boot(d, "sclera_b"),
               "ols_b_rel": ols_boot(d, "sclera_b_rel"),
               "sclera_b_median": float(d.sclera_b.median()),
               "plastic_b_median": float(d.plastic_b[d.plastic_px >= PLASTIC_MIN].median())}
        out["per_site"][site] = res
        if site != "italy_dated":            # primary test: one per site
            primary.append(res["b_vs_age"]["p"])
    for site, p in zip(("india", "italy"), holm(primary)):
        out["per_site"][site]["b_vs_age"]["p_holm"] = p
    return out


def report(clip: dict, est: dict, scl: dict, leak: dict, bat: dict, excluded: list) -> None:
    print("\n1. Clipping census")
    for site, c in clip.items():
        print(f"  {site:5s} n={c['n']:3d}  max is (255,255,255): full {c['full_max_is_white']:.0%}, "
              f"512 {c['small_max_is_white']:.0%};  clipped px median {c['full_clip_any_median']:.3%}, "
              f"90th {c['full_clip_any_p90']:.2%}, max {c['full_clip_any_max']:.1%}")
        loc = c["clip_location_mean"]
        print(f"        clipped px on conjunctiva {loc['on_conjunctiva']:.0%}, below lid {loc['below_lid']:.0%}, "
              f"elsewhere {loc['elsewhere']:.0%};  photos with glare on conjunctiva "
              f"{c['photos_with_glare_on_conjunctiva']:.0%} (90th pct of mask {c['conj_clipped_frac_p90']:.2%})")
        b = c["bright_location_mean"]
        print(f"        brightest 1% unclipped: below lid {b['below_lid']:.0%}, sclera {b['sclera']:.0%}, "
              f"conjunctiva {b['conjunctiva']:.0%}")
    print("\n   Angle to the plastic's colour (degrees, median [IQR]); 'from neutral' = size of correction")
    for site, e in est.items():
        print(f"  {site} (n={e['n']}): no correction {e['no_correction']['median']:.2f}")
        for name in ESTIMATES:
            r = e[name]
            print(f"     {LABEL[name]:46s} {r['median']:5.2f} [{r['iqr'][0]:.2f}, {r['iqr'][1]:.2f}]"
                  f"   from neutral {r['from_neutral']:.2f}")
    print(f"\n2. Sclera colour (excluded by grading: {excluded})")
    for site, s in scl["per_site"].items():
        print(f"  {site} n={s['n']} age {s['age_range'][0]:.0f}-{s['age_range'][1]:.0f}; "
              f"sclera b* median {s['sclera_b_median']:.1f}, plastic b* {s['plastic_b_median']:.1f}")
        for k in ("b_vs_age", "b_vs_hb", "a_vs_age", "a_vs_hb", "age_vs_hb", "b_rel_vs_age",
                  "b_rel_vs_hb", "a_rel_vs_hb", "a_rel_vs_age", "sclera_b_vs_plastic_b"):
            r = s[k]
            extra = f"  Holm {r['p_holm']:.3f}" if "p_holm" in r else ""
            print(f"     {k:22s} rho {r['rho']:+.2f}  p {r['p']:.3g}  n {r['n']}{extra}")
        for k in ("ols_b", "ols_b_rel"):
            o = s[k]
            print(f"     {k:8s} per decade {o['per_decade']['coef']:+.2f} {np.round(o['per_decade']['ci'], 2)}"
                  f"  per g/dL {o['per_g_dl_hb']['coef']:+.2f} {np.round(o['per_g_dl_hb']['ci'], 2)}"
                  f"  SD {o['sd_target']:.2f}  n {o['n']}")
    print("\n4. Italian capture batches")
    for b in ("italy_numbered", "italy_dated"):
        r = bat[b]
        print(f"  {b:15s} n {r['n']:3d}  Hb {r['hb_mean']:.1f}  age {r['age_mean']:.0f}  plastic log(R/G) "
              f"{r['plastic_log_rg_mean']:+.3f} (SD {r['plastic_log_rg_sd']:.3f})  conjunctiva log(R/G) "
              f"{r['conj_log_rg_mean']:.3f}  r~Hb {r['r_vs_hb_pearson']:+.2f}  plastic~Hb rho "
              f"{r['plastic_rg_vs_hb']['rho']:+.2f}")
    print(f"  pooled r~Hb {bat['pooled_r_vs_hb_pearson']:+.2f}; Mann-Whitney p {bat['mann_whitney_p']}")
    print(f"  conjunctiva log(R/G) Hb slope: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in bat['conj_hb_slope'].items()})}")
    print("\n3. What each correction would do to the conjunctiva's log(R/G)")
    for site, l in leak.items():
        print(f"  {site} n={l['n']}: conjunctiva Hb slope {l['conj_hb_slope']:+.4f} per g/dL "
              f"{np.round(l['conj_hb_slope_ci'], 4)}")
        for name in (*ESTIMATES, "plastic"):
            r = l[name]
            print(f"     {name:13s} share of Hb slope removed {r['share_removed']:+.2f} "
                  f"{np.round(r['share_removed_ci'], 2)}   age shift {r['age_g_dl_per_decade']:+.2f} g/dL "
                  f"per decade {np.round(r['age_g_dl_per_decade_ci'], 2)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--remeasure", action="store_true")
    ap.add_argument("--sheets", action="store_true")
    args = ap.parse_args()
    if args.sheets:
        write_sheets()
        return
    df = measure(args.remeasure)
    grading = json.loads(GRADING.read_text())
    df["sclera_ok"] = ~df.patient.isin(grading["fail"]) & (df.sclera_px > 0)
    df = add_batch(df)
    clip, est, scl, leak, bat = clipping(df), estimates(df), sclera(df), leakage(df), batches(df)
    report(clip, est, scl, leak, bat, grading["fail"])
    (HERE / "results.json").write_text(json.dumps(
        {"constants": {"work": WORK, "near_clip": NEAR_CLIP, "top_bright": TOP_BRIGHT,
                       "sclera_top": SCLERA_TOP, "sclera_a_max": SCLERA_A_MAX, "band": BAND,
                       "margin": MARGIN, "plastic_min": PLASTIC_MIN, "bootstrap": BOOTSTRAP},
         "n_photos": len(df), "n_plastic": int((df.plastic_px >= PLASTIC_MIN).sum()),
         "clipping": clip, "estimates_vs_plastic": est, "sclera": scl, "leakage": leak,
         "italy_batches": bat}, indent=1))
    print(f"\nwrote {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
