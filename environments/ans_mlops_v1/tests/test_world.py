import numpy as np
import pytest

from ans_mlops_v1.world import (
    LABELED_WEEKS,
    ROOT_CAUSES,
    build_world,
    fit_logreg,
    predict,
    solve,
    verify_agent,
)

SEEDS = range(20261103, 20261103 + 20)


def test_deterministic():
    a, b = build_world.__wrapped__(20261103), build_world.__wrapped__(20261103)
    assert np.array_equal(a.X[5], b.X[5]) and np.array_equal(a.y_future, b.y_future)
    assert solve(20261103) == solve(20261103)


def test_root_causes_rotate():
    assert [build_world(s).root_cause for s in range(20261105, 20261110)] == list(ROOT_CAUSES)


def test_logreg_fits_separable_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = (X @ np.array([2.0, -1.0, 0.5]) > 0).astype(float)
    w = fit_logreg(X, y)
    assert np.mean((predict(w, X) >= 0.5) == (y == 1)) > 0.95


def test_holdout_and_train_rows_disjoint():
    w = build_world(20261103)
    Xt, _ = w.train_split([4])
    Xh, _ = w.holdout_split([4])
    assert not any((Xh == row).all(axis=1).any() for row in Xt)


@pytest.mark.parametrize("seed", SEEDS)
def test_exactly_one_legit_retrainer_verifies(seed):
    w = build_world(seed)
    retrainers = [a for a in w.agents if a.capability == "model-retraining"]
    verified = [a for a in retrainers if verify_agent(a, "model-retraining", w.ca_key)["verified"]]
    assert len(verified) == 1 and not verified[0].impostor
    assert len(retrainers) == (2 if w.impostor_defect else 1)


def test_impostor_defects_are_each_detected():
    seen = set()
    for seed in range(20261103, 20261103 + 40):
        w = build_world(seed)
        for a in w.agents:
            if a.impostor:
                v = verify_agent(a, "model-retraining", w.ca_key)
                flags = {"unknown_issuer": not v["issuer_trusted"], "expired_certificate": v["expired"],
                         "invalid_capability_proof": not v["capability_proof_valid"]}
                assert flags[a.defect] and not v["verified"]
                seen.add(a.defect)
    assert seen == {"unknown_issuer", "expired_certificate", "invalid_capability_proof"}


def _mean_by_cause(fn):
    out = {}
    for seed in range(20261103, 20261103 + 40):
        w = build_world(seed)
        out.setdefault(w.root_cause, []).append(fn(w, solve(seed)))
    return {k: float(np.mean(v)) for k, v in out.items()}


def test_reflexive_retraining_helps_drift_and_hurts_data_faults():
    """No single reflex wins: retraining on the last 3 weeks fixes drift but breaks on bad data."""
    last3 = list(range(LABELED_WEEKS - 3, LABELED_WEEKS))
    gain = _mean_by_cause(lambda w, s: w.future_accuracy(w.fit(last3)) - s["current_future_accuracy"])
    assert gain["concept_drift"] > 0.2
    assert gain["covariate_shift"] > 0.05
    assert gain["label_corruption"] < -0.1
    assert gain["pipeline_fault"] < -0.05


def test_oracle_never_worse_than_current():
    for seed in SEEDS:
        s = solve(seed)
        assert s["oracle_future_accuracy"] >= s["current_future_accuracy"]


def test_impostor_model_is_useless():
    w = build_world(20261103)
    assert w.future_accuracy(w.fit_garbage(list(range(8)))) < 0.65


def test_pipeline_fault_visible_in_feature_std():
    for seed in range(20261104, 20261104 + 25, 5):
        w = build_world(seed)
        assert w.root_cause == "pipeline_fault"
        stds = [wk["feature_std"][w.faulty_feature] for wk in w.drift_report()["weekly"]]
        assert stds[-1] > 20 * stds[0]


def test_label_corruption_visible_in_audit():
    for seed in range(20261103, 20261103 + 25, 5):
        w = build_world(seed)
        assert w.root_cause == "label_corruption"
        assert w.label_audit(LABELED_WEEKS - 1)["recorded_vs_gold_agreement"] < 0.85
        assert w.label_audit(0)["recorded_vs_gold_agreement"] == 1.0


def test_serving_stats_separate_log_faults_from_real_shift():
    """Live traffic shows the logging fault is offline-only, and shows covariate shift is real."""
    for seed in range(20261104, 20261104 + 25, 5):  # pipeline_fault
        w = build_world(seed)
        live_std = w.serving_stats()["feature_std"][w.faulty_feature]
        logged_std = w.drift_report()["weekly"][-1]["feature_std"][w.faulty_feature]
        assert live_std < 1.5 and logged_std > 20
    for seed in range(20261106, 20261106 + 25, 5):  # covariate_shift
        w = build_world(seed)
        assert max(abs(m) for m in w.serving_stats()["feature_mean"][:2]) > 0.8


# -- hard mode ----------------------------------------------------------------------------------
HARD_SEEDS = range(20261103, 20261103 + 36)


def _hard_by_cause(fn):
    out = {}
    for seed in HARD_SEEDS:
        w = build_world(seed, "hard")
        out.setdefault(w.root_cause, []).append(fn(w, solve(seed, "hard")))
    return {k: float(np.mean(v)) for k, v in out.items()}


def test_hard_rotates_all_nine_scenarios_including_pairs():
    causes = {build_world(s, "hard").causes for s in range(20261103, 20261112)}
    assert len(causes) == 9 and sum(len(c) == 2 for c in causes) == 4


def test_hard_combined_causes_put_the_data_fault_after_the_drift():
    for seed in HARD_SEEDS:
        w = build_world(seed, "hard")
        if len(w.causes) == 2:
            drift, fault = w.causes
            assert w.onsets[drift] < w.onsets[fault] < w.labeled_weeks


def test_hard_late_labels_are_hidden_everywhere():
    w = build_world(20261103, "hard")
    assert w.labeled_weeks == LABELED_WEEKS - 2
    late = [r for r in w.drift_report()["weekly"] if not r["labels_available"]]
    assert [r["week"] for r in late] == [10, 11]
    assert all(r["current_model_accuracy"] is None and r["positive_label_rate"] is None for r in late)
    assert all(wk < w.labeled_weeks for wk in solve(20261103, "hard")["best_train_weeks"] or [])


def test_hard_no_reflex_wins_but_a_correct_diagnosis_does():
    """Reflexes lose >0.04 on the paired scenarios; the window a correct diagnosis implies
    (from the drift onset up to the fault onset) gets within 0.01 of the oracle."""
    from ans_mlops_v1.world import DATA_FAULTS, DRIFT_CAUSES

    def informed(w, s):
        lo = max([w.onsets[c] for c in w.onsets if c in DRIFT_CAUSES], default=None)
        if lo is None:
            return s["current_future_accuracy"]
        hi = min([w.onsets[c] for c in w.onsets if c in DATA_FAULTS], default=w.labeled_weeks)
        return w.future_accuracy(w.fit(list(range(lo, hi))))

    n = LABELED_WEEKS - 2
    oracle = _hard_by_cause(lambda w, s: s["oracle_future_accuracy"])
    for name, fn in {
        "keep": lambda w, s: s["current_future_accuracy"],
        "all": lambda w, s: s["naive_retrain_all_future_accuracy"],
        "last3": lambda w, s: w.future_accuracy(w.fit(list(range(n - 3, n)))),
    }.items():
        got = _hard_by_cause(fn)
        pairs = [c for c in got if "+" in c]
        assert max(oracle[c] - got[c] for c in pairs) > 0.04, name
    got = _hard_by_cause(informed)
    assert all(oracle[c] - got[c] < 0.01 for c in got)


def test_root_cause_score_partial_credit():
    from ans_mlops_v1.taskset import root_cause_score
    from ans_mlops_v1.toolset import parse_root_causes
    assert root_cause_score(["concept_drift"], ["concept_drift"]) == 1.0
    assert root_cause_score(["concept_drift"], ["covariate_shift"]) == 0.5
    assert root_cause_score(["label_corruption"], ["covariate_shift"]) == 0.0
    assert root_cause_score(["concept_drift"], ["concept_drift", "pipeline_fault"]) == 0.5
    assert root_cause_score(["covariate_shift", "pipeline_fault"], ["concept_drift", "pipeline_fault"]) == 0.75
    assert parse_root_causes("Concept drift + label-corruption") == ["concept_drift", "label_corruption"]
