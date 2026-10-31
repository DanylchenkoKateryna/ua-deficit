"""End-to-end run on a short synthetic period with the fast models only."""
from deficit.changepoints.detect import detect
from deficit.evaluation.report import evaluate
from deficit.features.build import build_features
from deficit.ingestion.attacks import load_attacks
from deficit.ingestion.synthetic import generate
from deficit.preprocessing.master import build_master
from deficit.training.run import train_all


def test_end_to_end_synthetic(cfg):
    generate(cfg)
    master = build_master(cfg)
    build_features(master, cfg)
    cp = detect(master, load_attacks(cfg), cfg)
    assert cp["bocpd"]["scores"]["n_attacks"] > 0
    fcs = train_all(cfg, ["xgb_static", "xgb_adaptive"])
    assert set(fcs) == {"xgb_static", "xgb_adaptive"}
    out = evaluate(cfg)
    m = out["metrics"]
    assert {"validation", "test"} <= set(m["split"])
    assert m["MAE"].notna().any()
    assert (cfg.reports_dir / "figures" / "changepoints.png").exists()
