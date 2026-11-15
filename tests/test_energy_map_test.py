"""Final Energy Map test: paired bootstrap over strikes, per-strike C parts, model parsing (docs/preregistration_energy_map.md)."""
import numpy as np
import pandas as pd
import pytest

from deficit.evaluation import energy_map_test as emt
from deficit.evaluation import region_restoration as rr


def _ev(n_strikes=12, per=4, seed=0):
    """Episodes of strikes; `good` ranks them perfectly, `flat` is constant, `noisy` random, `twin` is a copy of `good`."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_strikes):
        h = rng.permutation(np.arange(1, per + 1)) * 10.0 + s
        for k in range(per):
            rows.append({"idx": s, "hours": h[k], "observed": bool(rng.random() < 0.8), "good": h[k], "flat": 50.0, "noisy": float(rng.normal())})
    df = pd.DataFrame(rows)
    df["twin"] = df["good"]
    return df


def test_c_parts_match_harrell_c_within():
    ev = _ev()
    for m in ("good", "flat", "noisy"):
        parts = emt.c_parts(ev[m].to_numpy(float), ev["hours"].to_numpy(float), ev["observed"].to_numpy(bool), ev["idx"].to_numpy())
        want = rr.harrell_c_within(ev[m].to_numpy(float), ev["hours"].to_numpy(float), ev["observed"].to_numpy(bool), ev["idx"].to_numpy())
        assert parts["num"].sum() / parts["den"].sum() == pytest.approx(want)


def test_paired_bootstrap_perfect_vs_constant():
    ev = _ev()
    r = emt.paired_c_bootstrap(ev, "good", "flat", n_boot=300, seed=42)
    assert r["C_model"] == pytest.approx(1.0) and r["C_base"] == pytest.approx(0.5) and r["diff"] == pytest.approx(0.5)
    # a constant forecast scores 0.5 on EVERY resample and a perfect one scores 1: the paired difference is exactly 0.5, no spread
    assert r["diff_lo"] == pytest.approx(0.5) and r["diff_hi"] == pytest.approx(0.5)
    assert r["p_diff_le_0"] == pytest.approx(1 / 301) and r["p_C_model_gt_half"] == pytest.approx(1 / 301)
    assert r["n_strikes"] == 12 and r["n_boot"] == 300


def test_paired_bootstrap_uses_the_same_resample_for_both_models():
    ev = _ev(seed=1)
    same = emt.paired_c_bootstrap(ev, "good", "twin", n_boot=300, seed=42)
    assert same["diff"] == 0 and same["diff_lo"] == 0 and same["diff_hi"] == 0 and same["p_diff_le_0"] == 1.0
    # independent resamples would give a nonzero spread even for identical forecasts; the marginal CI of C itself is not degenerate
    r = emt.paired_c_bootstrap(ev, "good", "noisy", n_boot=500, seed=42)
    assert r["C_model"] > r["C_base"] and r["diff_lo"] < r["diff"] < r["diff_hi"]
    assert r["diff"] == pytest.approx(r["C_model"] - r["C_base"])
    # a hand-made resample: strikes drawn once, both C computed on it
    S = 12
    rng = np.random.default_rng(42)
    draw = np.bincount(rng.integers(0, S, S), minlength=S).astype(float)
    a = emt.c_parts(ev["good"].to_numpy(float), ev["hours"].to_numpy(float), ev["observed"].to_numpy(bool), ev["idx"].to_numpy())
    b = emt.c_parts(ev["noisy"].to_numpy(float), ev["hours"].to_numpy(float), ev["observed"].to_numpy(bool), ev["idx"].to_numpy())
    first = (draw @ a["num"].to_numpy() - draw @ b["num"].to_numpy()) / (draw @ a["den"].to_numpy())
    assert np.isfinite(first)
    assert emt.paired_c_bootstrap(ev, "good", "noisy", n_boot=500, seed=42) == r      # deterministic for a fixed seed


def test_mae_vs_global_and_model_name_parsing(tmp_path):
    ev = _ev()
    ev["observed"] = True
    r = emt.mae_vs_global(ev.assign(oblast_median=ev["good"] + 1.0, global_median=ev["good"] + 5.0), "good")
    assert r["MAE_good"] == pytest.approx(0.0) and r["MAE_global_median"] == pytest.approx(5.0) and r["share_better_strikes"] == 1.0
    assert r["p_wilcoxon"] < 0.01
    f = tmp_path / "chosen.md"
    f.write_text("- **Model chosen for the test: `ridge_strike_oblast_a100`**, C = 0.7", encoding="utf-8")
    assert emt.chosen_model(f) == "ridge_strike_oblast_a100"
    with pytest.raises(ValueError):
        emt.forecast_chosen(pd.DataFrame(), pd.DataFrame(), "xgbaft_full_d2n60", 3)
