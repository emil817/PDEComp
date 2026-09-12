"""Opt-in wrapper for EDL's original LLM-guided discovery loop."""

import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np

from data.config import NATIVE_EDL_DEFAULTS, sindy_params
from utils.protocols import ProtocolSkippedError, ProtocolUnavailableError


EDL_SOURCE_ROOT = Path(__file__).resolve().parent / "EDL"


def _ensure_edl_import_paths():
    for path in (EDL_SOURCE_ROOT, EDL_SOURCE_ROOT / "evaluation"):
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)


def _openai_server_func(
    inputs,
    model="gpt-3.5-turbo",
    max_decode_steps=1024,
    temperature=1.0,
    seed=1,
    model_pretrain=None,
    tokenizer=None,
):
    """Call the remote model without importing EDL's local-LLM dependencies."""

    import openai

    prompts = [inputs] if isinstance(inputs, str) else list(inputs)
    outputs = []
    for prompt in prompts:
        completion = openai.ChatCompletion.create(
            model=model,
            temperature=temperature,
            max_tokens=max_decode_steps,
            messages=[{"role": "user", "content": prompt}],
            seed=seed,
        )
        outputs.append(completion.choices[0].message.content)
    return outputs


def _load_optimizer_functions():
    """Import EDL's search loop with a remote-only prompt client."""

    _ensure_edl_import_paths()
    if "optimzier_utils" in sys.modules:
        module = sys.modules["optimzier_utils"]
        return module.call_optimizer, module.organize

    original_prompt_utils = sys.modules.get("prompt_utils")
    prompt_utils = ModuleType("prompt_utils")
    prompt_utils.call_openai_server_func = _openai_server_func
    sys.modules["prompt_utils"] = prompt_utils
    try:
        from optimzier_utils import call_optimizer, organize
    finally:
        if original_prompt_utils is None:
            sys.modules.pop("prompt_utils", None)
        else:
            sys.modules["prompt_utils"] = original_prompt_utils
    return call_optimizer, organize


def _canonical_term(term):
    return (
        str(term)
        .replace("**", "^")
        .replace("*", " ")
        .replace("  ", " ")
        .strip()
    )


def _ordered_result_terms(equation):
    """Return terms in the coefficient order retained by EDL's evaluator."""

    expression_terms = [
        _canonical_term(term)
        for term in str(equation.exp_str).split(" + ")
        if term.strip()
    ]
    if len(expression_terms) == len(equation.coef):
        return expression_terms
    raise ValueError(
        "EDL result cannot be mapped to benchmark coefficients: "
        f"{len(expression_terms)} terms for {len(equation.coef)} coefficients"
    )


def _native_problem(data, x, t, filename):
    _ensure_edl_import_paths()
    from evaluation.sr_utils import Diff, Diff2, Diff3, Diff4, FiniteDiff, FiniteDiff2

    if isinstance(data, list):
        if len(data) != 1:
            raise ProtocolUnavailableError(
                "The opt-in EDL adapter currently exposes one scalar equation per LLM search"
            )
        data = data[0]
    values = np.asarray(data, dtype=float)
    target = sindy_params[filename].get("targets", [{"name": "u_t", "axis": "t", "order": 1}])[0]
    if target.get("axis", "t") != "t" or target.get("order", 1) not in (1, 2):
        raise ProtocolUnavailableError("EDL native adapter requires a first- or second-order time target")

    dt = float(np.asarray(t)[1] - np.asarray(t)[0])
    time_diff = FiniteDiff if target.get("order", 1) == 1 else FiniteDiff2
    if values.ndim == 1:
        lhs = time_diff(values, dt)
        features = {
            "u": values,
            "t": np.asarray(t, dtype=float),
        }
        if target.get("order", 1) > 1:
            features["u_t"] = FiniteDiff(values, dt)
        is_ode = True
    elif values.ndim == 2 and x is not None:
        native_u = values.T
        lhs = np.vstack([time_diff(row, dt) for row in native_u])
        features = {
            "u": native_u.reshape(-1),
            "x": np.repeat(np.asarray(x, dtype=float).reshape(-1, 1), native_u.shape[1], axis=1).reshape(-1),
            "t": np.repeat(np.asarray(t, dtype=float).reshape(1, -1), native_u.shape[0], axis=0).reshape(-1),
            "u_x": Diff(native_u, np.asarray(x), 0).reshape(-1),
            "u_xx": Diff2(native_u, np.asarray(x), 0).reshape(-1),
            "u_xxx": Diff3(native_u, np.asarray(x), 0).reshape(-1),
            "u_xxxx": Diff4(native_u, np.asarray(x), 0).reshape(-1),
        }
        lhs = lhs.reshape(-1)
        is_ode = False
    else:
        raise ProtocolUnavailableError("EDL native adapter supports scalar ODE and scalar 1D PDE data")

    return target.get("name", "u_t"), lhs.reshape(-1, 1), features, is_ode


@contextmanager
def _external_evaluator_data(lhs, features):
    import evaluation.scorer as scorer

    original = scorer.data_load
    scorer.data_load = lambda _name: (lhs, features)
    try:
        yield scorer
    finally:
        scorer.data_load = original


def run_native_edl(data, x, y, z, t, filename, options=None):
    """Run EDL's LLM loop when explicitly enabled and credentials are present."""

    config = {**NATIVE_EDL_DEFAULTS, **(options or {})}
    if options and options.get("max_iterations") is not None:
        config["max_epochs"] = int(options["max_iterations"])
    if not config.get("allow_external_llm", False):
        raise ProtocolSkippedError(
            "EDL native mode is configured but external LLM calls are disabled; pass --allow-external-llm to run it"
        )
    api_key = config.get("api_key") or os.environ.get("EDL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ProtocolSkippedError("EDL native mode requires EDL_API_KEY or OPENAI_API_KEY")

    import openai
    call_optimizer, organize = _load_optimizer_functions()

    target_name, lhs, features, is_ode = _native_problem(data, x, t, filename)
    feature_names = list(features)
    args = SimpleNamespace(
        LLM_name=config["llm_name"],
        N=int(config["samples_per_iteration"]),
        init_num=int(config["initial_samples"]),
        operators=config.get("operators", "{+, -, *, /, ^2}"),
        operands="{" + ", ".join(feature_names) + "}",
        optimize_type=config.get("optimize_type", "optimize_optimize"),
        evo_type=config.get("evo_type", "term"),
        reward_limit=float(config.get("reward_limit", 0.5)),
        sort="not_reverse",
        max_epoch=int(config["max_epochs"]),
        threshold=float(config.get("reward_threshold", 0.995)),
        data_name="ODE_external" if is_ode else filename,
        seed=int(config.get("seed", 1)),
        max_terms=int(config["max_terms"]),
        metric=config.get("metric", "sparse_reward"),
        metric_params=config.get("metric_params", [0.01]),
        mode=config.get("mode", "sparse_regression"),
        use_pqt=1,
        add_const=0,
    )
    openai.api_key = api_key
    if config.get("api_base"):
        openai.api_base = config["api_base"]
    llm_dict = {
        "name": config["llm_name"],
        "max_decode_steps": int(config.get("max_decode_length", 1024)),
        "temperature": float(config["temperature"]),
        "batch_size": 1,
        "model_pretrained": None,
        "tokenizer": None,
    }

    random.seed(args.seed)
    np.random.seed(args.seed)
    with _external_evaluator_data(lhs, features) as scorer:
        evaluator = scorer.Evaluator(
            "benchmark_external",
            metric=args.metric,
            metric_params=args.metric_params,
            max_terms=args.max_terms,
            mode=args.mode,
            add_const=args.add_const,
        )
        population, best, _, _ = call_optimizer(
            llm_dict, args, evaluator, call_type="initialization"
        )
        optimize_types = args.optimize_type.split("_")
        for epoch in range(args.max_epoch):
            optimize_type = optimize_types[epoch % len(optimize_types)]
            prompt, _ = organize(
                population,
                optimize_type,
                args.sort,
                evaluator.pq,
                use_pqt=bool(args.use_pqt),
                ode=is_ode,
            )
            evaluator.pq.push(population)
            population, best, _, _ = call_optimizer(
                llm_dict,
                args,
                evaluator,
                info=prompt,
                call_type=optimize_type,
            )
            if best.score >= args.threshold:
                break

    terms = _ordered_result_terms(best)
    coefficients = [float(value) for value in np.asarray(best.coef).reshape(-1)]
    return {
        "dataset": filename.split(".")[0],
        "targets": [target_name],
        "features": [terms],
        "coefficients": [coefficients[: len(terms)]],
        "library_sizes": {target_name: ""},
        "library_size": "",
        "model": best.exp_str,
        "equation_texts": [best.exp_str],
        "protocol": "native",
    }
