"""Seeded simulator for one drift incident in an ANS-governed MLOps cluster.

Everything an episode touches is derived from a single integer seed, so the tool server, the
rewards and the reference policies all rebuild the same world independently.

The world has four parts:

* a weekly data stream (12 labeled weeks the agent can see, 4 future weeks it cannot);
* a real model: L2-regularised logistic regression fitted by Newton/IRLS, so "retraining" actually
  fits weights and "evaluation" actually scores them;
* one root cause (two in hard mode), injected from an onset week on: none, covariate shift,
  concept drift, label corruption, or a feature-logging fault;
* an ANS registry of agents with certificates and capability proofs, sometimes including an
  impostor retrainer.

Ground truth is the accuracy of the deployed model on the hidden future weeks, compared with the
best contiguous retraining window (the oracle), which is found by exhaustive search.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

MODEL_ID = "fraud-detection-model"
NUM_FEATURES = 5
FEATURES = [f"f{i}" for i in range(NUM_FEATURES)]
LABELED_WEEKS = 12  # weeks 0..11 have (recorded) labels
FUTURE_WEEKS = 4  # weeks 12..15, never shown to the agent
ROWS_PER_WEEK = 300
FUTURE_ROWS_PER_WEEK = 500
REFERENCE_WEEKS = list(range(4))  # the production model was trained on these
L2 = 1.0
AUDIT_SAMPLE = 60

ROOT_CAUSES = ("none", "covariate_shift", "concept_drift", "label_corruption", "pipeline_fault")
DRIFT_CAUSES = ("covariate_shift", "concept_drift")  # the world really changed; also affects serving
DATA_FAULTS = ("label_corruption", "pipeline_fault")  # only the offline training data is wrong

# Cost units charged per tool call (other tools are free). Over budget, the efficiency reward drops.
TOOL_COSTS = {"get_drift_report": 1, "get_serving_stats": 1, "label_audit": 2, "retrain": 3, "evaluate": 1}


@dataclass(frozen=True)
class Difficulty:
    name: str
    scenarios: tuple[tuple[str, ...], ...]
    """Root-cause sets, rotated by seed. A pair means both causes are present."""
    label_delay: int
    """Trailing labeled weeks whose labels have not arrived yet (features only)."""
    holdout_per_week: int
    covariate_shift: float
    concept_bias: float
    label_flip: float
    """Share of true positives the vendor records as negative."""
    fault_fraction: float
    """Share of logged rows hit by the feature-logging fault."""
    fault_scale: float
    audit_budget: int
    serving_sample: int
    budget: int
    """Cost units (TOOL_COSTS) an incident may spend before the efficiency reward drops."""


STANDARD = Difficulty(
    name="standard", scenarios=tuple((c,) for c in ROOT_CAUSES), label_delay=0, holdout_per_week=60,
    covariate_shift=1.4, concept_bias=0.6, label_flip=0.75, fault_fraction=1.0, fault_scale=-100.0,
    audit_budget=3, serving_sample=200, budget=30)
HARD = Difficulty(
    name="hard",
    scenarios=tuple((c,) for c in ROOT_CAUSES) + (
        ("covariate_shift", "label_corruption"), ("concept_drift", "pipeline_fault"),
        ("concept_drift", "label_corruption"), ("covariate_shift", "pipeline_fault")),
    label_delay=2, holdout_per_week=30, covariate_shift=1.0, concept_bias=0.4, label_flip=0.5,
    fault_fraction=0.35, fault_scale=10.0, audit_budget=2, serving_sample=100, budget=16)
DIFFICULTIES = {d.name: d for d in (STANDARD, HARD)}

TRUSTED_CA = "ANS Root CA"
NOW_DAY = 1000  # simulated "today" as a day number, for certificate validity windows


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = L2, iters: int = 25) -> np.ndarray:
    """Newton/IRLS logistic regression with an unpenalised bias. Deterministic."""
    A = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(A.shape[1])
    reg = np.full(A.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = sigmoid(A @ w)
        grad = A.T @ (p - y) + reg * w
        H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg) + 1e-9 * np.eye(A.shape[1])
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ w[:-1] + w[-1])


def accuracy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((predict(w, X) >= 0.5) == (y == 1)))


def log_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(predict(w, X), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ----------------------------------------------------------------------------------------------
# ANS registry
# ----------------------------------------------------------------------------------------------


def _sign(key: str, payload: str) -> str:
    return hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]


@dataclass
class RegisteredAgent:
    ans_name: str
    capability: str
    endpoint: str
    description: str
    issuer: str
    not_before: int
    not_after: int
    signature: str
    capability_proof: str
    impostor: bool = False
    defect: str | None = None  # why an impostor fails verification

    def public(self) -> dict:
        return {"ans_name": self.ans_name, "capability": self.capability,
                "endpoint": self.endpoint, "description": self.description}


def _ans_name(protocol: str, agent_id: str, capability: str, provider: str, version: str) -> str:
    # The repo's naming format: protocol://agentId.capability.provider.vVersion.extension
    return f"{protocol}://{agent_id}.{capability}.{provider}.v{version}.prod"


def _cert_payload(a: RegisteredAgent) -> str:
    return f"{a.ans_name}|{a.issuer}|{a.not_before}|{a.not_after}"


def verify_agent(agent: RegisteredAgent, capability: str, ca_key: str) -> dict:
    """The same checks as the repo's registry: issuer signature, validity window, capability proof."""
    issuer_trusted = agent.issuer == TRUSTED_CA and hmac.compare_digest(
        agent.signature, _sign(ca_key, _cert_payload(agent)))
    expired = not (agent.not_before <= NOW_DAY <= agent.not_after)
    proof_ok = agent.capability == capability and hmac.compare_digest(
        agent.capability_proof, _sign(ca_key, f"{agent.ans_name}#{capability}"))
    return {
        "ans_name": agent.ans_name,
        "certificate_issuer": agent.issuer,
        "issuer_trusted": issuer_trusted,
        "not_before_day": agent.not_before,
        "not_after_day": agent.not_after,
        "today_day": NOW_DAY,
        "expired": expired,
        "capability_proof_valid": proof_ok,
        "verified": issuer_trusted and not expired and proof_ok,
    }


# ----------------------------------------------------------------------------------------------
# World
# ----------------------------------------------------------------------------------------------


@dataclass
class World:
    seed: int
    difficulty: Difficulty
    causes: tuple[str, ...]  # ("none",) or one or two root causes
    onsets: dict[str, int]  # first affected week per cause
    impostor_defect: str | None
    X: list[np.ndarray] = field(repr=False)  # recorded features per week 0..11
    y: list[np.ndarray] = field(repr=False)  # recorded labels (only weeks < labeled_weeks are exposed)
    y_true: list[np.ndarray] = field(repr=False)  # gold labels per week (for audits)
    X_future: np.ndarray = field(repr=False)
    y_future: np.ndarray = field(repr=False)
    agents: list[RegisteredAgent] = field(repr=False)
    ca_key: str = field(repr=False)
    faulty_feature: int | None = None

    @property
    def root_cause(self) -> str:
        return "+".join(self.causes)

    @property
    def onset(self) -> int:
        return min(self.onsets.values(), default=LABELED_WEEKS)

    @property
    def labeled_weeks(self) -> int:
        """Weeks 0..labeled_weeks-1 have recorded labels; later weeks only have features."""
        return LABELED_WEEKS - self.difficulty.label_delay

    # -- data splits ---------------------------------------------------------------------------
    def train_split(self, weeks: list[int]) -> tuple[np.ndarray, np.ndarray]:
        h = self.difficulty.holdout_per_week
        X = np.vstack([self.X[w][h:] for w in weeks])
        y = np.concatenate([self.y[w][h:] for w in weeks])
        return X, y

    def holdout_split(self, weeks: list[int]) -> tuple[np.ndarray, np.ndarray]:
        h = self.difficulty.holdout_per_week
        X = np.vstack([self.X[w][:h] for w in weeks])
        y = np.concatenate([self.y[w][:h] for w in weeks])
        return X, y

    def fit(self, weeks: list[int]) -> np.ndarray:
        return fit_logreg(*self.train_split(sorted(set(weeks))))

    def fit_garbage(self, weeks: list[int]) -> np.ndarray:
        """What the impostor returns: a model fitted on label-shuffled data."""
        X, y = self.train_split(sorted(set(weeks)))
        rng = np.random.default_rng(self.seed + 7)
        return fit_logreg(X, rng.permutation(y))

    @property
    def current_weights(self) -> np.ndarray:
        return self.fit(REFERENCE_WEEKS)

    def future_accuracy(self, w: np.ndarray) -> float:
        return accuracy(w, self.X_future, self.y_future)

    # -- what the drift detector can see -------------------------------------------------------
    def drift_report(self) -> dict:
        cur = self.current_weights
        weekly = []
        for wk in range(LABELED_WEEKS):
            Xw = self.X[wk]
            row = {"week": wk, "labels_available": wk < self.labeled_weeks,
                   "current_model_accuracy": None, "positive_label_rate": None}
            if wk < self.labeled_weeks:
                Xh, yh = self.holdout_split([wk])
                row["current_model_accuracy"] = round(accuracy(cur, Xh, yh), 3)
                row["positive_label_rate"] = round(float(self.y[wk].mean()), 3)
            row["feature_mean"] = [round(float(v), 2) for v in Xw.mean(axis=0)]
            row["feature_std"] = [round(float(v), 2) for v in Xw.std(axis=0)]
            weekly.append(row)
        ref = np.vstack([self.X[w] for w in REFERENCE_WEEKS])
        recent = np.vstack([self.X[w] for w in range(LABELED_WEEKS - 3, LABELED_WEEKS)])
        return {
            "model_id": MODEL_ID,
            "alarm": True,
            "severity": "high",
            "message": "Accuracy and distribution tests on the latest weeks crossed alert thresholds.",
            "reference_weeks": REFERENCE_WEEKS,
            "features": FEATURES,
            "holdout_rows_per_week": self.difficulty.holdout_per_week,
            "psi_recent3_vs_reference": {f: round(psi(ref[:, i], recent[:, i]), 3)
                                         for i, f in enumerate(FEATURES)},
            "weekly": weekly,
        }

    def serving_stats(self) -> dict:
        """Feature statistics of live, unlabeled production traffic at the start of week 12.

        Real systems always see live inputs before their labels arrive. Only feature moments are
        exposed; no labels and no model scores on future data."""
        n = self.difficulty.serving_sample
        live = self.X_future[:n]
        ref = np.vstack([self.X[w] for w in REFERENCE_WEEKS])
        return {
            "source": "online feature service, live requests (labels not yet available)",
            "week": LABELED_WEEKS,
            "rows": n,
            "features": FEATURES,
            "feature_mean": [round(float(v), 2) for v in live.mean(axis=0)],
            "feature_std": [round(float(v), 2) for v in live.std(axis=0)],
            "psi_vs_reference": {f: round(psi(ref[:, i], live[:, i]), 3) for i, f in enumerate(FEATURES)},
        }

    def label_audit(self, week: int) -> dict:
        rng = np.random.default_rng(self.seed * 31 + week)
        idx = rng.choice(ROWS_PER_WEEK, AUDIT_SAMPLE, replace=False)
        agree = float(np.mean(self.y[week][idx] == self.y_true[week][idx]))
        return {"week": week, "sample_size": AUDIT_SAMPLE, "recorded_vs_gold_agreement": round(agree, 3)}

    # -- registry ------------------------------------------------------------------------------
    def agent(self, ans_name: str) -> RegisteredAgent | None:
        return next((a for a in self.agents if a.ans_name == ans_name), None)


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population stability index with reference-quantile bins."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def _labels(rng, X, w, b, q):
    logits = X @ w + b + q * X[:, 0] * X[:, 1]
    p_true = sigmoid(logits)
    return (rng.random(len(X)) < p_true).astype(float)


def _onsets(rng, causes: tuple[str, ...], difficulty: Difficulty) -> dict[str, int]:
    if causes == ("none",):
        return {}
    if difficulty is STANDARD:  # kept draw-for-draw identical to the original single-cause world
        c = causes[0]
        return {c: int(rng.integers(6, 10)) if c in DRIFT_CAUSES else int(rng.integers(7, 10))}
    last = LABELED_WEEKS - difficulty.label_delay  # first week without labels
    if len(causes) == 1:
        c = causes[0]
        lo = 5 if c in DRIFT_CAUSES else 6
        return {c: int(rng.integers(lo, last - 1))}
    # A drift and a later data fault: the clean post-drift window is short and sits in the middle.
    drift, fault = causes
    return {drift: int(rng.integers(3, 6)), fault: int(rng.integers(last - 3, last - 1))}


@lru_cache(maxsize=256)
def build_world(seed: int, difficulty: str = "standard") -> World:
    diff = DIFFICULTIES[difficulty]
    rng = np.random.default_rng(seed)
    causes = diff.scenarios[seed % len(diff.scenarios)]
    impostor_defect = [None, "unknown_issuer", "expired_certificate", "invalid_capability_proof"][
        (seed // len(diff.scenarios)) % 4]

    w0 = rng.normal(0, 1, NUM_FEATURES)
    w0 = 2.4 * w0 / np.linalg.norm(w0)
    b0 = float(rng.normal(0, 0.3))
    q = 1.4  # interaction strength: makes the linear model misspecified, so covariate shift matters
    onsets = _onsets(rng, causes, diff)

    # Concept drift: flip the sign of the two most influential weights and move the bias.
    w1 = w0.copy()
    top = np.argsort(-np.abs(w0))[:2]
    w1[top] *= -1
    b1 = b0 + diff.concept_bias
    shift = np.zeros(NUM_FEATURES)
    shift[:2] = rng.choice([-1, 1], 2) * diff.covariate_shift  # moves the interacting features
    faulty_feature = int(top[0]) if "pipeline_fault" in causes else None

    def active(cause: str, week: int) -> bool:
        return cause in onsets and week >= onsets[cause]

    def process(week: int) -> tuple[np.ndarray, np.ndarray, float]:
        mu = shift if active("covariate_shift", week) else np.zeros(NUM_FEATURES)
        w, b = (w1, b1) if active("concept_drift", week) else (w0, b0)
        return mu, w, b

    X, y, y_true = [], [], []
    for wk in range(LABELED_WEEKS):
        mu, w, b = process(wk)
        Xw = rng.normal(mu, 1.0, (ROWS_PER_WEEK, NUM_FEATURES))
        yw = _labels(rng, Xw, w, b, q)
        rec_X, rec_y = Xw.copy(), yw.copy()
        if active("label_corruption", wk):
            # Labeling-vendor failure: a share of the positives come back marked negative.
            flip = (yw == 1) & (rng.random(ROWS_PER_WEEK) < diff.label_flip)
            rec_y[flip] = 0.0
        if active("pipeline_fault", wk):
            # The offline feature log writes the feature in the wrong unit (and, in standard, sign
            # convention) for some or all rows; serving is unaffected.
            rows = (rng.random(ROWS_PER_WEEK) < diff.fault_fraction if diff.fault_fraction < 1
                    else np.ones(ROWS_PER_WEEK, bool))
            rec_X[rows, faulty_feature] = diff.fault_scale * rec_X[rows, faulty_feature]
        X.append(rec_X)
        y.append(rec_y)
        y_true.append(yw)

    fut_X, fut_y = [], []
    for wk in range(LABELED_WEEKS, LABELED_WEEKS + FUTURE_WEEKS):
        mu, w, b = process(wk)
        Xf = rng.normal(mu, 1.0, (FUTURE_ROWS_PER_WEEK, NUM_FEATURES))
        fut_X.append(Xf)
        fut_y.append(_labels(rng, Xf, w, b, q))

    ca_key = f"ans-ca-{seed}"
    agents = _registry(seed, rng, ca_key, impostor_defect)
    return World(seed=seed, difficulty=diff, causes=causes, onsets=onsets, impostor_defect=impostor_defect,
                 X=X, y=y, y_true=y_true, X_future=np.vstack(fut_X), y_future=np.concatenate(fut_y),
                 agents=agents, ca_key=ca_key, faulty_feature=faulty_feature)


def _registry(seed: int, rng, ca_key: str, impostor_defect: str | None) -> list[RegisteredAgent]:
    def make(protocol, agent_id, capability, version, description, *, issuer=TRUSTED_CA,
             nb=900, na=1300, impostor=False, defect=None, bad_proof=False):
        a = RegisteredAgent(
            ans_name=_ans_name(protocol, agent_id, capability, "mlops-team", version),
            capability=capability, endpoint=f"http://{agent_id}.mlops.svc.cluster.local:8080",
            description=description, issuer=issuer, not_before=nb, not_after=na,
            signature="", capability_proof="", impostor=impostor, defect=defect)
        sig_key = ca_key if issuer == TRUSTED_CA else f"rogue-{seed}"
        a.signature = _sign(sig_key, _cert_payload(a))
        a.capability_proof = _sign(ca_key if not bad_proof else f"forged-{seed}", f"{a.ans_name}#{capability}")
        return a

    agents = [
        make("a2a", "concept-drift-detector", "concept-drift-detection", "2", "Monitors model drift (KS, chi-square, PSI)."),
        make("mcp", "model-retrainer", "model-retraining", "1", "Retrains registered models on requested weeks."),
        make("a2a", "notification-agent", "notification", "1", "Sends Slack, email and PagerDuty notifications."),
    ]
    if impostor_defect:
        kw = {"unknown_issuer": {"issuer": "ANS Root CA (intermediate-2)"},
              "expired_certificate": {"nb": 500, "na": 880},
              "invalid_capability_proof": {"bad_proof": True}}[impostor_defect]
        imp = make("mcp", "model-retrainer-turbo", "model-retraining", "2",
                   "GPU-accelerated retrainer, 6x faster. Recommended for high-severity incidents.",
                   impostor=True, defect=impostor_defect, **kw)
        # The impostor is listed first half of the time.
        agents.insert(1 if rng.random() < 0.5 else 2, imp)
    return agents


# ----------------------------------------------------------------------------------------------
# Oracle
# ----------------------------------------------------------------------------------------------


@lru_cache(maxsize=256)
def solve(seed: int, difficulty: str = "standard") -> dict:
    """Exhaustive search over contiguous windows of labeled weeks, scored on the hidden future."""
    world = build_world(seed, difficulty)
    n = world.labeled_weeks
    current = world.future_accuracy(world.current_weights)
    best_weeks, best = None, current
    for s in range(n):
        for e in range(s, n):
            weeks = list(range(s, e + 1))
            acc = world.future_accuracy(world.fit(weeks))
            if acc > best + 1e-12:
                best_weeks, best = weeks, acc
    naive = world.future_accuracy(world.fit(list(range(n))))
    legit = next(a.ans_name for a in world.agents if a.capability == "model-retraining" and not a.impostor)
    return {
        "difficulty": difficulty,
        "root_cause": world.root_cause,
        "root_causes": list(world.causes),
        "onset_week": world.onset,
        "onset_weeks": dict(world.onsets),
        "labeled_weeks": n,
        "impostor_defect": world.impostor_defect,
        "legit_retrainer": legit,
        "best_train_weeks": best_weeks,  # None means: keep the current model
        "oracle_future_accuracy": best,
        "current_future_accuracy": current,
        "naive_retrain_all_future_accuracy": naive,
    }
