"""Bounded JSON-only synthetic policy evidence; replay never reads the network."""

import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ginseng.precision_artifact import _differences
from ginseng.provenance import digest
from ginseng.two_decision import (
    TRAIN_DOMAIN,
    VALIDATION_DOMAIN,
    VERSION,
    Action,
    Model,
    Policy,
    Scenarios,
    information_contract,
    select_policies,
    stream_identity,
    subset,
    summarize,
    training_trace,
    validate_policy,
    validate_scenarios,
)

MAX_BYTES = 8 * 1024**2


def policy_from_json(value):
    expected = {"kind", "model_identity", "root", "rules", "future_rules", "fallback"}
    if not isinstance(value, dict) or value.keys() != expected:
        raise ValueError("Invalid policy fields")
    return Policy(
        value["kind"],
        value["model_identity"],
        Action(**value["root"]),
        tuple((key, Action(**a)) for key, a in value["rules"]),
        tuple((key, Action(**a)) for key, a in value["future_rules"]),
        Action(**value["fallback"]),
    )


def capture(directory, payload):
    """Only this version's declared synthetic model is accepted; no personal input API."""
    if (
        payload["report"]["synthetic"] is not True
        or payload["model"]["version"] != VERSION
    ):
        raise ValueError("Only explicit synthetic experiment captures are supported")
    directory = Path(directory)
    if directory.exists():
        raise ValueError("Capture destination already exists")
    inputs = {k: v for k, v in payload.items() if k != "report"}
    manifest = dict(
        schema=1,
        version=VERSION,
        complete=True,
        input_identity=digest(inputs),
        output_digest=digest(payload["report"]),
        payload=payload,
    )
    manifest["integrity"] = digest(manifest)
    data = json.dumps(manifest, allow_nan=False, indent=2).encode()
    if len(data) > MAX_BYTES:
        raise ValueError("Capture exceeds 8 MiB")
    directory.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".two-decision-", dir=directory.parent))
    try:
        with (tmp / "experiment.json").open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, directory)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dict(
        path=str(directory),
        input_identity=manifest["input_identity"],
        output_digest=manifest["output_digest"],
        bytes=len(data),
    )


def load(directory):
    directory = Path(directory)
    file = directory / "experiment.json"
    try:
        if (
            directory.is_symlink()
            or file.is_symlink()
            or not 0 < file.stat().st_size <= MAX_BYTES
        ):
            raise ValueError("Unsafe capture path/size")
        m = json.loads(file.read_text())
        integrity = m.pop("integrity")
        if (
            digest(m) != integrity
            or m["schema"] != 1
            or m["version"] != VERSION
            or m["complete"] is not True
        ):
            raise ValueError("Corrupt or incompatible experiment")
        p = m["payload"]
        required_report = {
            "model",
            "model_identity",
            "synthetic",
            "summary",
            "information",
            "policies",
            "comparison",
            "training_replications",
            "solver_evidence",
            "training_review_nodes",
            "training_ledger",
            "training_trades",
            "training_streams",
            "validation_stream",
        }
        required_summary = {
            "training_paths",
            "validation_paths",
            "visible_horizon",
            "material_horizon",
            "review_day",
            "training_static_objective",
            "training_nonanticipative_objective",
            "training_hindsight_bound",
        }
        if (
            not isinstance(p["report"], dict)
            or not required_report <= p["report"].keys()
            or not isinstance(p["report"]["summary"], dict)
            or not required_summary <= p["report"]["summary"].keys()
        ):
            raise ValueError("Incomplete experiment report contract")
        if (
            digest({k: v for k, v in p.items() if k != "report"}) != m["input_identity"]
            or digest(p["report"]) != m["output_digest"]
        ):
            raise ValueError("Input/output integrity mismatch")
        model = Model(**p["model"])
        support = Scenarios(**p["support"])
        validate_scenarios(model, support)
        if (
            p["report"]["model_identity"] != model.identity
            or p["report"]["synthetic"] is not True
        ):
            raise ValueError("Model identity mismatch")
        training = p["training_draws"]
        validation = p["validation_draws"]
        if not 1 <= len(training) <= 10 or len(p["policies"]) != len(training):
            raise ValueError("Invalid replication count")
        for r, draw in enumerate([*training, validation]):
            stream = draw["stream"]
            domain = TRAIN_DOMAIN if r < len(training) else VALIDATION_DOMAIN
            rep = r if r < len(training) else 0
            if stream != stream_identity(stream["root_seed"], domain, rep):
                raise ValueError("Invalid declared stream")
            if (
                not isinstance(draw["indices"], list)
                or not 1 <= len(draw["indices"]) <= 65536
                or any(type(x) is not int for x in draw["indices"])
            ):
                raise ValueError("Invalid captured draw dimensions/type")
            subset(support, np.array(draw["indices"], dtype="int64"))
        from ginseng.decision_lab import independent_streams

        for d in training:
            independent_streams(d["stream"], validation["stream"])
        policies = []
        for rows in p["policies"]:
            if rows.keys() != {"static", "nonanticipative", "hindsight"}:
                raise ValueError("Incomplete policy comparison")
            parsed = {k: policy_from_json(v) for k, v in rows.items()}
            for k, policy in parsed.items():
                if k != policy.kind:
                    raise ValueError("Mislabeled policy")
                validate_policy(model, policy)
            policies.append(parsed)
        return p, model, support, policies
    except (KeyError, TypeError, OSError, OverflowError) as e:
        raise ValueError("Invalid two-decision artifact: " + str(e)) from e


def replay(directory):
    p, model, support, policies = load(directory)
    comparisons = []
    replications = []
    mismatches = []
    for key, actual in dict(
        model=asdict(model),
        information=information_contract(model),
        policies={k: asdict(v) for k, v in policies[0].items()},
        training_streams=[d["stream"] for d in p["training_draws"]],
        validation_stream=p["validation_draws"]["stream"],
    ).items():
        mismatches += _differences(
            json.loads(json.dumps(actual)), p["report"][key], key
        )
    declared_summary = p["report"]["summary"]
    for key, value in dict(
        training_paths=len(p["training_draws"][0]["indices"]),
        validation_paths=len(p["validation_draws"]["indices"]),
        visible_horizon=model.visible_horizon,
        material_horizon=model.material_horizon,
        review_day=model.review_day,
    ).items():
        mismatches += _differences(value, declared_summary[key], "summary." + key)
    validation = subset(
        support, np.array(p["validation_draws"]["indices"], dtype="int64")
    )
    for r, ps in enumerate(policies):
        training = subset(
            support, np.array(p["training_draws"][r]["indices"], dtype="int64")
        )
        _, evidence = select_policies(
            model, training
        )  # Training only; no holdout selection.
        if r == 0:
            mismatches += _differences(
                evidence, p["report"]["solver_evidence"], "solver_evidence"
            )
            for kind in ("static", "nonanticipative", "hindsight"):
                key = (
                    "training_hindsight_bound"
                    if kind == "hindsight"
                    else "training_" + kind + "_objective"
                )
                field = (
                    "enumerated_minimum_dollars"
                    if kind == "hindsight"
                    else "objective_dollars"
                )
                mismatches += _differences(
                    evidence[kind][field], declared_summary[key], "summary." + key
                )
            trace = training_trace(model, training, ps)
            for key, value in trace.items():
                mismatches += _differences(value, p["report"][key], key)
        for kind, policy in ps.items():
            summary = summarize(model, training, policy)
            if not np.isclose(
                summary["objective_dollars"],
                evidence[kind]["objective_dollars"],
                atol=1e-8,
                rtol=1e-12,
            ):
                mismatches.append(f"training[{r}].{kind}.finite_grid_objective")
            if summary["max_conservation_residual_dollars"] > 1e-8:
                mismatches.append(f"training[{r}].{kind}.accounting")
            replications.append(
                dict(
                    replication=r,
                    policy=kind,
                    **summary,
                    rules=[dict(node=k, **asdict(a)) for k, a in policy.rules],
                )
            )
            if r == 0:
                comparisons.extend(
                    [
                        dict(collection="training", policy=kind, **summary),
                        dict(
                            collection="independent_validation",
                            policy=kind,
                            **summarize(model, validation, policy),
                        ),
                    ]
                )
    mismatches += _differences(comparisons, p["report"]["comparison"], "comparison")
    mismatches += _differences(
        replications, p["report"]["training_replications"], "training_replications"
    )
    return dict(
        match=not mismatches,
        mismatches=mismatches,
        comparison=comparisons,
        scope="Exact captured finite support/draws, frozen holdout policies and training-grid objective equivalence. No validation optimization, network or regenerated randomness.",
    )
