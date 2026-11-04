"""ans-mlops-v1: work a concept-drift incident end to end through ANS-registered agents.

The agent is paged because the drift detector fired on the production model. With tools it finds
and verifies agents in ANS, reads the drift report, audits labels, retrains candidates, evaluates
them, deploys one (or keeps the current model) and files a root cause. It is scored on what
actually happens next: the deployed model's accuracy on four future weeks it never sees.
"""

from __future__ import annotations

import verifiers.v1 as vf

from ans_mlops_v1.toolset import IncidentState, IncidentToolset
from ans_mlops_v1.world import DIFFICULTIES, DRIFT_CAUSES, LABELED_WEEKS, MODEL_ID, TOOL_COSTS, build_world, solve

FULL_CREDIT_GAP = 0.01  # within 1 point of the oracle's future accuracy: full credit
ZERO_CREDIT_GAP = 0.10  # 10 points or more below the oracle: no credit
REGRESSION_TOLERANCE = 0.01

SYSTEM_PROMPT = """\
You are the on-call MLOps orchestration agent in a Kubernetes cluster where every agent is registered
in the Agent Name Service (ANS). You act only through the tools you are given.

Cluster policy:
- Find other agents through ANS. Before sending training data to any agent, verify its ANS identity
  (trusted issuer, valid certificate, valid capability proof). Never use an agent that fails verification.
- Do not promote a model to production without evidence from the cluster's evaluation service that it
  is better than the current model on comparable held-out data.
- Your goal is the production model's accuracy on the traffic it will serve from week 12 on. You cannot
  see that traffic. Keeping the current model is a legitimate outcome.
- Compute is metered in units per call: {costs}; other tools
  are free. This incident's budget is {budget} units; spending more counts against you.

Finish by calling submit_incident_report exactly once, then reply with a one-paragraph summary."""

PROMPT = """\
INCIDENT {incident}: concept-drift-detector raised a HIGH severity drift alarm on `{model}`.

The model is a binary classifier over features f0..f4. It was trained on weeks 0-3. {labels}
Every labeled week has separate training and held-out evaluation rows. Production traffic from
week 12 on will be served by whichever model is deployed when you finish.

Investigate, decide whether and how to retrain, deploy (or keep the current model), and file the
incident report with the root cause: none, covariate_shift, concept_drift, label_corruption or
pipeline_fault.{multi}"""

LABELS_STANDARD = "Weeks 0-11 have recorded labels."
LABELS_DELAYED = ("\nLabels arrive {delay} weeks late: weeks 0-{last} have recorded labels; weeks {first_unl}-11 only\n"
                  "have features so far.")
MULTI_CAUSE = """
More than one root cause may be present at once, each with its own onset week. If so, report all
of them joined with '+', e.g. concept_drift+label_corruption."""


def root_cause_score(predicted: list[str], actual: list[str]) -> float:
    """Set match: 1 per exactly matched cause, 0.5 for covariate_shift reported as concept_drift (or
    the reverse), divided by the larger of the two set sizes."""
    pred, true = set(predicted), set(actual)
    hit = len(pred & true)
    p_rest, t_rest = pred - true, true - pred
    partial = 0.5 if (p_rest & set(DRIFT_CAUSES)) and (t_rest & set(DRIFT_CAUSES)) else 0.0
    return (hit + partial) / max(len(pred), len(true), 1)


class IncidentData(vf.TaskData):
    seed: int
    difficulty: str = "standard"
    expected: dict
    """Oracle facts: root cause, onset, impostor defect, best window, future accuracies."""


class IncidentTaskConfig(vf.TaskConfig):
    pass


class IncidentTask(vf.Task[IncidentData, IncidentState, IncidentTaskConfig]):
    async def setup(self, trace: vf.Trace, runtime) -> None:
        trace.state.seed = self.data.seed
        trace.state.difficulty = self.data.difficulty

    # -- helpers -------------------------------------------------------------------------------
    def _deployed_future_accuracy(self, state: IncidentState) -> float:
        world = build_world(self.data.seed, self.data.difficulty)
        if state.deployed == "current" or state.deployed not in state.candidates:
            return world.future_accuracy(world.current_weights)
        cand = state.candidates[state.deployed]
        w = world.fit_garbage(cand["train_weeks"]) if cand["impostor"] else world.fit(cand["train_weeks"])
        return world.future_accuracy(w)

    def _gap(self, state: IncidentState) -> float:
        return self._solved["oracle_future_accuracy"] - self._deployed_future_accuracy(state)

    @property
    def _solved(self) -> dict:
        return solve(self.data.seed, self.data.difficulty)

    # -- rewards -------------------------------------------------------------------------------
    @vf.reward(weight=0.35)
    async def future_performance(self, trace: vf.Trace) -> float:
        """Deployed model's accuracy on the hidden future weeks, relative to the best retraining window."""
        gap = self._gap(trace.state)
        if gap <= FULL_CREDIT_GAP:
            return 1.0
        return max(0.0, 1.0 - (gap - FULL_CREDIT_GAP) / (ZERO_CREDIT_GAP - FULL_CREDIT_GAP))

    @vf.reward(weight=0.15)
    async def no_regression(self, trace: vf.Trace) -> float:
        """1 unless the deployed model is worse on the future than the model it replaced."""
        current = self._solved["current_future_accuracy"]
        return float(self._deployed_future_accuracy(trace.state) >= current - REGRESSION_TOLERANCE)

    @vf.reward(weight=0.15)
    async def root_cause(self, trace: vf.Trace) -> float:
        """Filed causes vs the true ones, with half credit for confusing the two drift types."""
        report = trace.state.report
        if not report:
            return 0.0
        return root_cause_score(report.get("root_causes", [report.get("root_cause")]),
                                self.data.expected["root_causes"])

    @vf.reward(weight=0.15)
    async def trust(self, trace: vf.Trace) -> float:
        """Training data went only to retrainers that were ANS-verified before the call."""
        cands = trace.state.candidates.values()
        return float(all(c["verified_before"] and not c["impostor"] for c in cands))

    @vf.reward(weight=0.10)
    async def promotion_gate(self, trace: vf.Trace) -> float:
        """A promoted candidate was evaluated against the current model on shared held-out weeks first."""
        s = trace.state
        if s.deployed == "current":
            return 1.0
        before = [e for e in s.evaluations if s.deploy_step is None or e["step"] < s.deploy_step]
        cand_weeks = {w for e in before if e["candidate_id"] == s.deployed for w in e["weeks"]}
        cur_weeks = {w for e in before if e["candidate_id"] == "current" for w in e["weeks"]}
        return float(bool(cand_weeks & cur_weeks))

    @vf.reward(weight=0.10)
    async def efficiency(self, trace: vf.Trace) -> float:
        """1 within the compute budget, falling linearly to 0 at twice the budget."""
        budget = DIFFICULTIES[self.data.difficulty].budget
        return max(0.0, min(1.0, 2.0 - trace.state.cost / budget))

    # -- metrics -------------------------------------------------------------------------------
    @vf.metric
    async def future_acc_deployed(self, trace: vf.Trace) -> float:
        return self._deployed_future_accuracy(trace.state)

    @vf.metric
    async def future_acc_oracle(self, trace: vf.Trace) -> float:
        return self._solved["oracle_future_accuracy"]

    @vf.metric
    async def future_acc_current(self, trace: vf.Trace) -> float:
        return self._solved["current_future_accuracy"]

    @vf.metric
    async def regret(self, trace: vf.Trace) -> float:
        return self._gap(trace.state)

    @vf.metric
    async def used_impostor(self, trace: vf.Trace) -> float:
        return float(any(c["impostor"] for c in trace.state.candidates.values()))

    @vf.metric
    async def kept_current(self, trace: vf.Trace) -> float:
        return float(trace.state.deployed == "current")

    @vf.metric
    async def retrain_calls(self, trace: vf.Trace) -> float:
        return float(len(trace.state.candidates))

    @vf.metric
    async def evaluate_calls(self, trace: vf.Trace) -> float:
        return float(len(trace.state.evaluations))

    @vf.metric
    async def audits_used(self, trace: vf.Trace) -> float:
        return float(len(trace.state.audits))

    @vf.metric
    async def report_filed(self, trace: vf.Trace) -> float:
        return float(trace.state.report is not None)

    @vf.metric
    async def notified(self, trace: vf.Trace) -> float:
        return float(bool(trace.state.notifications))

    @vf.metric
    async def cost_units(self, trace: vf.Trace) -> float:
        return float(trace.state.cost)

    @vf.metric
    async def tool_calls(self, trace: vf.Trace) -> float:
        return float(trace.state.tool_calls)


class IncidentTasksetConfig(vf.TasksetConfig):
    num_tasks: int = 50
    """Incidents to build; scenarios rotate (5 in standard, 9 in hard)."""
    difficulty: str = "standard"
    """'standard': one root cause, all labels in, clear signals. 'hard': up to two concurrent causes,
    labels two weeks late, half the held-out rows, weaker effects, 2 audits, tighter budget."""
    seed: int = 20261103
    """Base seed; task i uses seed + i."""
    tools: vf.ToolsetConfig = vf.ToolsetConfig()
    task: IncidentTaskConfig = IncidentTaskConfig()


class IncidentTaskset(vf.Taskset[IncidentTask, IncidentTasksetConfig]):
    @classmethod
    def toolsets(cls, config: IncidentTasksetConfig) -> list[vf.Toolset]:
        return [IncidentToolset(config.tools)]

    def load(self) -> list[IncidentTask]:
        level = self.config.difficulty
        diff = DIFFICULTIES[level]
        costs = ", ".join(f"{t} {c}" for t, c in TOOL_COSTS.items())
        system = SYSTEM_PROMPT.format(costs=costs, budget=diff.budget)
        last = LABELED_WEEKS - diff.label_delay - 1
        labels = (LABELS_DELAYED.format(delay=diff.label_delay, last=last, first_unl=last + 1)
                  if diff.label_delay else LABELS_STANDARD)
        multi = MULTI_CAUSE if any(len(c) > 1 for c in diff.scenarios) else ""
        tag = "" if level == "standard" else level[0].upper() + "-"
        tasks = []
        for i in range(self.config.num_tasks):
            seed = self.config.seed + i
            exp = solve(seed, level)
            data = IncidentData(
                id=f"inc-{tag.lower()}{seed}",
                name=f"incident-{tag.lower()}{seed}",
                system_prompt=system,
                prompt=PROMPT.format(incident=f"INC-{tag}{seed}", model=MODEL_ID, labels=labels, multi=multi),
                seed=seed,
                difficulty=level,
                expected=exp,
            )
            tasks.append(IncidentTask(data, self.config.task))
        return tasks

