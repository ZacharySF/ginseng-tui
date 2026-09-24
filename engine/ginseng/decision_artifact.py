"""Atomic decision manifests wrapping the existing validated prepared artifact."""

import json
import os
import shutil
import tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from ginseng import engine_artifact
from ginseng.decision_lab import fixed_interval, independent_streams
from ginseng.engine_artifact import ArtifactError
from ginseng.execution import ExecutionConfig
from ginseng.provenance import digest, source_fingerprint
from ginseng.verification import VERSION, ExecutablePlan, RiskContract, verify_plan

MAX_MANIFEST = 4 * 1024**2
NUMERICAL_TOLERANCES = dict(atol=1e-7, rtol=1e-10)


def streams_for_report(report):
    streams = {k: report["streams"][k] for k in ("training", "validation")}
    stability = report["stability"]
    if not 1 <= len(stability) <= 10:
        raise ArtifactError("Invalid replication table")
    for row in stability:
        if "prepared_reference" in row:
            name = row["prepared_reference"]
            if (
                type(row["replicate"]) is not int
                or not 1 <= row["replicate"] < 10
                or name != f"training_{row['replicate']}"
                or name in streams
            ):
                raise ArtifactError("Unsafe or duplicated replication reference")
            independent_streams(streams["training"], row["stream"])
            independent_streams(row["stream"], streams["validation"])
            streams[name] = row["stream"]
    return streams


def capture_decision(destination, run, *, allow_personal=False):
    if not run.report.get("synthetic") and not allow_personal:
        raise ArtifactError(
            "Personal input capture requires explicit allow_personal=True"
        )
    destination = Path(destination)
    if destination.exists():
        raise ArtifactError("Capture destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".decision-", dir=destination.parent))
    try:
        references = {}
        input_identities = {}
        streams = streams_for_report(run.report)
        for name in streams:
            engine_artifact.capture(
                temporary / name,
                run.prepared[name],
                dict(
                    opening_cash=run.controls["opening_cash"],
                    buffer=run.contract.operating_buffer,
                    coverage_target=run.contract.coverage_target,
                ),
                config=run.config,
                full=False,
                extra_inputs=dict(discretionary=run.discretionary[name]),
                model=streams[name],
            )
            content = (temporary / name / "manifest.json").read_bytes()
            input_identities[name] = json.loads(content)["input_identity"]
            references[name] = dict(
                path=name, sha256=sha256(content).hexdigest(), bytes=len(content)
            )
        manifest = dict(
            schema=1,
            complete=True,
            calculation=VERSION,
            calculation_source=source_fingerprint(),
            controls=run.controls,
            plan=asdict(run.plan),
            contract=asdict(run.contract),
            prepared=references,
            report=run.report,
            numerical_tolerances=NUMERICAL_TOLERANCES,
        )
        manifest["semantic_input_identity"] = digest(
            dict(
                controls=run.controls,
                plan=manifest["plan"],
                contract=manifest["contract"],
                prepared=input_identities,
                streams=run.report["streams"],
            )
        )
        manifest["output_integrity"] = digest(run.report)
        manifest["integrity"] = digest(manifest)
        data = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False).encode()
        if len(data) > MAX_MANIFEST:
            raise ArtifactError("Decision manifest exceeds resource limit")
        with (temporary / "decision.json").open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return dict(
        path=str(destination),
        semantic_input_identity=manifest["semantic_input_identity"],
        output_integrity=manifest["output_integrity"],
    )


def load_decision(directory):
    directory = Path(directory)
    try:
        file = directory / "decision.json"
        if (
            directory.is_symlink()
            or file.is_symlink()
            or file.stat().st_size > MAX_MANIFEST
        ):
            raise ArtifactError("Unsafe or oversized decision artifact")
        m = json.loads(file.read_text())
        integrity = m.pop("integrity")
        if (
            digest(m) != integrity
            or m["schema"] != 1
            or m["complete"] is not True
            or m["calculation"] != VERSION
        ):
            raise ArtifactError(
                "Corrupted, incomplete, or incompatible decision manifest"
            )
        if m["numerical_tolerances"] != NUMERICAL_TOLERANCES:
            raise ArtifactError("Unsupported numerical comparison contract")
        if digest(m["report"]) != m["output_integrity"]:
            raise ArtifactError("Output integrity mismatch")
        streams = streams_for_report(m["report"])
        if set(m["prepared"]) != set(streams):
            raise ArtifactError("Missing prepared inputs")
        independent_streams(
            m["report"]["streams"]["training"], m["report"]["streams"]["validation"]
        )
        loaded = {}
        total_bytes = 0
        # Validate both complete captures before doing any numerical replay.
        for name, ref in m["prepared"].items():
            if ref["path"] != name:
                raise ArtifactError("Unsafe prepared reference")
            folder = directory / name
            file = folder / "manifest.json"
            if (
                folder.is_symlink()
                or file.is_symlink()
                or file.stat().st_size != ref["bytes"]
                or ref["bytes"] > 1024**2
            ):
                raise ArtifactError("Invalid prepared manifest size/path")
            if sha256(file.read_bytes()).hexdigest() != ref["sha256"]:
                raise ArtifactError("Prepared manifest integrity mismatch")
            child = json.loads(file.read_text())
            total_bytes += sum(
                entry["file_bytes"] for entry in child["members"].values()
            )
            if total_bytes > engine_artifact.MAX_BYTES:
                raise ArtifactError("Decision artifact exceeds total byte budget")
            loaded[name] = engine_artifact.load(folder)
            p = loaded[name][1]
            if p.n_paths * p.visible_horizon * 8 * 10 > engine_artifact.MAX_BYTES:
                raise ArtifactError(
                    "Reference verification exceeds working-array budget"
                )
            if "discretionary" not in loaded[name][2]:
                raise ArtifactError("Missing action inputs")
        m["plan"]["withdrawals"] = tuple(tuple(a) for a in m["plan"]["withdrawals"])
        plan = ExecutablePlan(**m["plan"])
        contract = RiskContract(**m["contract"])
        semantic = digest(
            dict(
                controls=m["controls"],
                plan=asdict(plan),
                contract=asdict(contract),
                prepared={k: v[0]["input_identity"] for k, v in loaded.items()},
                streams=m["report"]["streams"],
            )
        )
        if semantic != m["semantic_input_identity"]:
            raise ArtifactError("Semantic input mismatch")
        for name, (meta, p, inputs, _) in loaded.items():
            if (
                meta["model"] != streams[name]
                or meta["scalars"]
                != dict(
                    opening_cash=m["controls"]["opening_cash"],
                    buffer=contract.operating_buffer,
                    coverage_target=contract.coverage_target,
                )
                or inputs["discretionary"].shape != (p.n_paths, p.visible_horizon)
            ):
                raise ArtifactError("Incompatible decision inputs")
        return m, loaded, plan, contract
    except (KeyError, TypeError, OSError, ValueError, OverflowError) as e:
        if isinstance(e, ArtifactError):
            raise
        raise ArtifactError("Invalid decision artifact: " + str(e)) from e


def numerical_differences(actual, expected, prefix=""):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return [prefix + ": keys"]
        return [
            d
            for k in expected
            for d in numerical_differences(actual[k], expected[k], prefix + "." + k)
        ]
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return [prefix + ": length"]
        return [
            d
            for i, (a, b) in enumerate(zip(actual, expected))
            for d in numerical_differences(a, b, f"{prefix}[{i}]")
        ]
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        return (
            []
            if isinstance(actual, (float, int))
            and np.isclose(actual, expected, atol=1e-7, rtol=1e-10)
            else [prefix]
        )
    return [] if actual == expected else [prefix]


def replay_decision(directory, config=ExecutionConfig()):
    m, loaded, plan, contract = load_decision(directory)
    mismatches = []
    results = {}
    executions = {}

    def compare(name, prepared, inputs, executable, expected):
        actual = asdict(
            verify_plan(
                prepared,
                m["controls"],
                executable,
                contract,
                inputs["weights"],
                inputs["discretionary"],
            )
        )
        if actual["identity"] != expected["identity"]:
            mismatches.append(name + ".identity")
        mismatches.extend(numerical_differences(actual, expected, name))
        results[name] = dict(
            verification=actual["status"],
            policy=actual["policy_status"],
            metrics=actual["metrics"],
        )
        return actual

    for name, (meta, p, inputs, outputs) in loaded.items():
        engine = engine_artifact.replay(Path(directory) / name, config)
        executions[name] = engine["execution"]
        if not engine["match"]:
            mismatches.append(name + ".engine")
        if name in ("training", "validation"):
            actual = compare(name, p, inputs, plan, m["report"][name])
        else:
            row = next(
                r
                for r in m["report"]["stability"]
                if r.get("prepared_reference") == name
            )
            replicate_plan = ExecutablePlan(
                **{
                    **row["actions"],
                    "withdrawals": tuple(
                        tuple(a) for a in row["actions"]["withdrawals"]
                    ),
                }
            )
            compare(name, p, inputs, replicate_plan, row["verification"])
        if name == "validation":
            interval = fixed_interval(
                actual["metrics"]["cash_failure_probability"], p.n_paths
            )
            mismatches.extend(
                numerical_differences(
                    interval, m["report"]["validation_interval"], "validation_interval"
                )
            )
            baseline = compare(
                "no_action_validation",
                p,
                inputs,
                ExecutablePlan(
                    spending_days=m["report"]["dimensions"]["visible_horizon"]
                ),
                m["report"]["no_action_validation"],
            )
            interval = fixed_interval(
                baseline["metrics"]["cash_failure_probability"], p.n_paths
            )
            mismatches.extend(
                numerical_differences(
                    interval, m["report"]["no_action_interval"], "no_action_interval"
                )
            )
    return dict(
        match=not mismatches,
        mismatches=mismatches,
        results=results,
        execution=executions,
        scope="Offline frozen-plan execution replay, including successful training replications and no-action comparison; no network and no optimizer invocation.",
        semantic_input_identity=m["semantic_input_identity"],
    )
