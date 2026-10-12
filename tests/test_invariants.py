"""
Tests for the things that fail silently.

This project's three invariants -- crop symmetry, site grouping, feature cache
keying -- have each been violated at least once, and none of them raised. They
just made every downstream number quietly wrong. Those are exactly the failures
worth a test, because no amount of looking at output reveals them.

Deliberately no model weights. Both crop_sky and check_assessable accept a
precomputed segmentation, so the whole invariant surface can be exercised with
synthetic masks in seconds. A test suite that needs a 1.7 GB download is a test
suite nobody runs.
"""

import csv
import time

import numpy as np
import pytest
from PIL import Image

import dataset as D
import model_utils as mu
import probe as P

SKY = mu.BACKGROUND_CLASS_INDICES[0]      # ADE20K "sky"
ROCK = 34                                  # ADE20K "rock"
FLOOR = 3                                  # ADE20K "floor"
PERSON = mu.PERSON_CLASS_INDEX


def img(w, h, colour=(120, 110, 90)):
    return Image.new("RGB", (w, h), colour)


def noisy(w, h, seed=0):
    """An image with real variance, so the uniformity check does not fire."""
    rng = np.random.default_rng(seed)
    return Image.fromarray((rng.random((h, w, 3)) * 160 + 40).astype("uint8"))


# ====================================================== invariant 1: crop_sky

class TestCropSky:
    def test_leaves_image_alone_when_there_is_no_sky(self):
        """Below SKY_PIXEL_MIN_FRACTION there is no band worth trimming."""
        pred = np.full((100, 100), ROCK)
        out = mu.crop_sky(img(100, 100), pred)
        assert out.size == (100, 100)

    def test_trims_a_sky_band_off_the_top(self):
        pred = np.full((100, 100), ROCK)
        pred[:40, :] = SKY
        out = mu.crop_sky(img(100, 100), pred)
        assert out.height < 100, "a 40% sky band should have been trimmed"

    def test_never_removes_more_than_the_cap_from_an_edge(self):
        """
        A segmentation mistake must not be able to delete most of the photo.
        This is the guard that stops a bad mask eating the subject.
        """
        pred = np.full((100, 100), SKY)
        pred[95:, :] = ROCK                      # claims almost everything is sky
        out = mu.crop_sky(img(100, 100), pred)
        assert out.height >= 100 * (1 - mu.SKY_CROP_MAX_FRACTION) - 1

    def test_refuses_a_degenerate_crop(self):
        """Below the minimum edge there is nothing left to classify."""
        pred = np.full((40, 40), SKY)
        pred[38:, :] = ROCK
        out = mu.crop_sky(img(40, 40), pred)
        assert out.size == (40, 40), "should fall back to the original"

    def test_scales_the_box_when_the_mask_is_smaller_than_the_image(self):
        """
        The mask is capped at SEGMENT_MAX_EDGE, so on a large photograph it is
        smaller than the image and the box must be scaled back. Getting this
        wrong crops a tiny corner of a big photo and nothing errors.
        """
        pred = np.full((100, 100), ROCK)
        pred[:40, :] = SKY
        big = img(1000, 1000)
        out = mu.crop_sky(big, pred)
        assert out.width == 1000, "width was untouched, so it must survive scaling"
        assert out.height > 400, f"box was not scaled to image coordinates: {out.size}"


# ================================================ invariant 3: cache identity

class TestCacheKey:
    def test_changes_when_a_file_is_edited_in_place(self, tmp_path):
        """
        crop_dataset_sky.py rewrites images under their existing names, so the
        path list is unchanged. A key over paths alone would survive a re-crop
        and serve features computed on the pre-crop pixels.
        """
        f = tmp_path / "a.jpg"
        img(32, 32).save(f)
        before = D.index_key([f])
        time.sleep(1.1)                       # mtime has 1s resolution
        img(64, 64).save(f)
        assert D.index_key([f]) != before

    def test_differs_for_different_file_sets(self, tmp_path):
        a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
        img(32, 32).save(a)
        img(32, 32, (10, 20, 30)).save(b)
        assert D.index_key([a]) != D.index_key([b])

    def test_is_stable_for_the_same_input(self, tmp_path):
        f = tmp_path / "a.jpg"
        img(32, 32).save(f)
        assert D.index_key([f]) == D.index_key([f])


# ============================================== invariant 2: the labels join

class TestLoadIndex:
    def test_raises_when_a_csv_row_has_no_file(self, tmp_path, monkeypatch):
        """
        Silently skipping means the evaluation quietly runs on fewer images
        than the dataset claims, and the reported n is a lie.
        """
        ds = tmp_path / "slope_dataset" / "train" / "stable"
        ds.mkdir(parents=True)
        img(32, 32).save(ds / "present.jpg")
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "split", "label", "primary_indicator",
                        "confidence", "source", "site_id", "notes"])
            w.writerow(["present.jpg", "train", "stable", "bedrock", "high", "x", "", ""])
        monkeypatch.setattr(D, "DATASET_DIR", tmp_path / "slope_dataset")
        monkeypatch.setattr(D, "LABELS_CSV", csv_path)

        paths, labels, groups = D.load_index()
        assert len(paths) == 1

        # Now add a file with no row: the loader must refuse, not skip.
        img(32, 32).save(ds / "orphan.jpg")
        with pytest.raises(SystemExit):
            D.load_index()

    def test_blank_site_id_becomes_its_own_group(self, tmp_path, monkeypatch):
        """
        Blank ids must not collapse into one giant false group, which would put
        unrelated photographs on the same side of every split.
        """
        ds = tmp_path / "slope_dataset" / "train" / "stable"
        ds.mkdir(parents=True)
        for n in ("a.jpg", "b.jpg"):
            img(32, 32).save(ds / n)
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "split", "label", "primary_indicator",
                        "confidence", "source", "site_id", "notes"])
            for n in ("a.jpg", "b.jpg"):
                w.writerow([n, "train", "stable", "bedrock", "high", "x", "", ""])
        monkeypatch.setattr(D, "DATASET_DIR", tmp_path / "slope_dataset")
        monkeypatch.setattr(D, "LABELS_CSV", csv_path)

        _, _, groups = D.load_index()
        assert len(set(groups)) == 2, "blank site_ids were merged into one group"


# ======================================================== the abstention gate

class TestGate:
    """The cheap checks return before segmentation, so none of these need a model."""

    def test_refuses_an_image_that_is_too_small(self):
        r = mu.check_assessable(img(40, 40))
        assert not r["ok"] and r["reason"] == "too_small"

    def test_refuses_an_extreme_aspect_ratio(self):
        r = mu.check_assessable(noisy(3000, 200))
        assert not r["ok"] and r["reason"] == "extreme_aspect"

    def test_refuses_a_dark_frame(self):
        r = mu.check_assessable(img(400, 300, (3, 3, 4)))
        assert not r["ok"] and r["reason"] == "too_dark"

    def test_refuses_a_uniform_frame(self):
        r = mu.check_assessable(img(400, 300, (128, 128, 128)))
        assert not r["ok"] and r["reason"] == "no_detail"

    def test_refuses_an_all_sky_frame(self):
        pred = np.full((300, 400), SKY)
        r = mu.check_assessable(noisy(400, 300), pred)
        assert not r["ok"] and r["reason"] == "no_ground"

    def test_refuses_an_interior(self):
        pred = np.full((300, 400), FLOOR)
        r = mu.check_assessable(noisy(400, 300), pred)
        assert not r["ok"] and r["reason"] == "indoor_scene"

    def test_refuses_a_portrait(self):
        pred = np.full((300, 400), PERSON)
        r = mu.check_assessable(noisy(400, 300), pred)
        assert not r["ok"] and r["reason"] == "subject_is_person"

    def test_accepts_an_ordinary_slope(self):
        pred = np.full((300, 400), ROCK)
        pred[:60, :] = SKY
        r = mu.check_assessable(noisy(400, 300), pred)
        assert r["ok"], r.get("reason")

    def test_accepts_a_bare_engineered_face(self):
        """
        A close-up of shotcrete segments almost entirely as "wall": no terrain,
        no furniture. Refusing it would reject exactly the engineered slopes
        this project cares most about.
        """
        pred = np.full((300, 400), 0)          # ADE20K "wall"
        r = mu.check_assessable(noisy(400, 300), pred)
        assert r["ok"], r.get("reason")

    def test_stats_are_returned_on_success_too(self):
        pred = np.full((300, 400), ROCK)
        r = mu.check_assessable(noisy(400, 300), pred)
        assert "terrain_fraction" in r["stats"] and "mean_luminance" in r["stats"]


# ========================================================= the probe contract

class TestProbe:
    def test_calibration_folds_never_span_a_group(self):
        y = np.array([0, 1] * 20)
        g = np.repeat(np.arange(20), 2)
        for tr, va in P.inner_splits(y, g):
            assert not (set(g[tr]) & set(g[va])), "site leaked across a calibration fold"

    def test_score_is_perfect_on_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.01, 0.02, 0.98, 0.99])
        s = P.score(y, p, flag=0.35, high=0.65)
        assert s["acc"] == 100.0
        assert s["unstable_recall"] == 100.0
        assert s["flag_unstable_recall"] == 100.0

    def test_flag_threshold_catches_more_than_argmax(self):
        """The whole point of flagging below 0.5: more unstable caught."""
        y = np.array([1, 1, 0, 0])
        p = np.array([0.40, 0.80, 0.10, 0.20])   # 0.40 is missed at 0.5, caught at 0.35
        s = P.score(y, p, flag=0.35, high=0.65)
        assert s["unstable_recall"] == 50.0
        assert s["flag_unstable_recall"] == 100.0

    def test_fit_probe_produces_calibrated_probabilities(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, (30, 8)), rng.normal(3, 1, (30, 8))])
        y = np.array([0] * 30 + [1] * 30)
        g = np.arange(60)
        m = P.fit_probe(X, y, g)
        p = m.predict_proba(X)[:, 1]
        assert p.min() >= 0.0 and p.max() <= 1.0
        assert (p[y == 1].mean() > p[y == 0].mean()), "probe learned nothing separable"


# ================================================= deployed config consistency

def test_probe_records_the_config_it_was_fitted_with():
    """
    model_utils reads clip_id and tta out of the artifact rather than keeping
    its own copy. If the fitted probe stops recording them, inference silently
    falls back to a different backbone or augmentation than it was trained on.
    """
    import joblib
    from pathlib import Path
    f = Path("slope_probe.joblib")
    if not f.exists():
        pytest.skip("no fitted probe in the working tree")
    probe = joblib.load(f)
    assert "clip_id" in probe, "probe does not record its backbone"
    assert "tta" in probe, "probe does not record its augmentation"
    assert probe["clip_id"] == D.CLIP_ID, "artifact and dataset.py disagree on the backbone"
    assert probe["tta"] == D.CLIP_TTA, "artifact and dataset.py disagree on TTA"
