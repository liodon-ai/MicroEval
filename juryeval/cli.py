"""Command-line interface for juryeval."""

import argparse
import json
import sys


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="juryeval",
        description="Lightweight NLP/LLM evaluation toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser(
        "score", help="Score a single model output using LLM-as-Judge"
    )
    p_score.add_argument("--model", default="gpt-4", help="Judge model name")
    p_score.add_argument("--question", required=True, help="Input question/prompt")
    p_score.add_argument("--output", required=True, help="Model output to score")
    p_score.add_argument("--reference", help="Optional expected output")
    p_score.add_argument("--json", action="store_true", help="Output as JSON")

    p_compare = sub.add_parser(
        "compare", help="Compare two model outputs using LLM-as-Judge"
    )
    p_compare.add_argument("--model", default="gpt-4", help="Judge model name")
    p_compare.add_argument("--question", required=True, help="Input question/prompt")
    p_compare.add_argument("--output-a", required=True, help="First model output")
    p_compare.add_argument("--output-b", required=True, help="Second model output")
    p_compare.add_argument("--json", action="store_true", help="Output as JSON")

    p_eval = sub.add_parser(
        "evaluate", help="Run metrics on a dataset file"
    )
    p_eval.add_argument(
        "--metric",
        required=True,
        choices=[
            "classification",
            "translation",
            "summarization",
            "fluency",
            "pairwise_judge",
            "pointwise_judge",
        ],
        help="Metric to compute",
    )
    p_eval.add_argument("--predictions", required=True, help="Path to predictions file (JSON or JSONL)")
    p_eval.add_argument("--references", help="Path to references file (JSON or JSONL)")
    p_eval.add_argument("--model", default="gpt-4", help="Judge model (for judge metrics)")
    p_eval.add_argument(
        "--output", "-o", help="Output file (default: print to stdout)"
    )
    p_eval.add_argument("--format", choices=["json", "text"], default="text", help="Output format")

    p_cal = sub.add_parser(
        "calibrate", help="Run judge calibration analysis"
    )
    p_cal.add_argument("--model", default="gpt-4", help="Judge model name")
    p_cal.add_argument("--json", action="store_true", help="Output as JSON")

    p_prompt = sub.add_parser(
        "prompt", help="Analyze prompt sensitivity"
    )
    p_prompt.add_argument("--model", default="gpt-4", help="Model to test")
    p_prompt.add_argument("--question", required=True, help="Question to analyze")
    p_prompt.add_argument("--num-variants", type=int, default=5, help="Number of prompt variants")
    p_prompt.add_argument("--json", action="store_true", help="Output as JSON")

    return parser


def _cmd_score(args):
    from juryeval import PointwiseJudge

    judge = PointwiseJudge(model=args.model)
    result = judge.score(
        output=args.output,
        question=args.question,
        reference=args.reference,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Score:  {result.get('score', 'N/A'):.4f}")
        print(f"Reason: {result.get('reason', 'N/A')}")


def _cmd_compare(args):
    from juryeval import PairwiseJudge

    judge = PairwiseJudge(model=args.model)
    result = judge.compare(
        answer_a=args.output_a,
        answer_b=args.output_b,
        question=args.question,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Winner: {result.get('winner', 'N/A')}")
        print(f"Score:  {result.get('score', 'N/A'):.4f}")
        print(f"Reason: {result.get('reason', 'N/A')}")


def _load_file(path):
    """Load a JSON or JSONL file. Returns list of values."""
    with open(path) as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        elif first_char == "{":
            return [json.loads(line) for line in f]
        else:
            data = [line.strip() for line in f if line.strip()]
            return data


def _cmd_evaluate(args):
    preds = _load_file(args.predictions)

    if isinstance(preds[0], dict):
        pred_values = [p.get("output") or p.get("prediction") or p.get("text", "") for p in preds]
        ref_values = None
        if args.references:
            refs = _load_file(args.references)
            ref_values = [r.get("output") or r.get("reference") or r.get("text", "") for r in refs]
        elif "reference" in preds[0] or "expected_output" in preds[0]:
            ref_values = [p.get("reference") or p.get("expected_output", "") for p in preds]
        questions = [p.get("question") or p.get("input") or "" for p in preds]
    else:
        pred_values = preds
        ref_values = _load_file(args.references) if args.references else None
        questions = None

    metric = args.metric
    if metric == "classification":
        from juryeval import eval_classification

        if ref_values is None:
            print("error: --references required for classification metric", file=sys.stderr)
            sys.exit(1)
        result = eval_classification(pred_values, ref_values)
    elif metric == "translation":
        from juryeval import eval_translation

        if ref_values is None:
            print("error: --references required for translation metric", file=sys.stderr)
            sys.exit(1)
        result = eval_translation(pred_values, ref_values)
    elif metric == "summarization":
        from juryeval import eval_summarization

        if ref_values is None:
            print("error: --references required for summarization metric", file=sys.stderr)
            sys.exit(1)
        result = eval_summarization(pred_values, ref_values)
    elif metric == "fluency":
        from juryeval import perplexity, flesch_kincaid

        ppl = perplexity(pred_values) if len(pred_values) == 1 else [perplexity(p) for p in pred_values]
        fk = [flesch_kincaid(p) for p in pred_values]
        result = {"perplexity": ppl, "flesch_kincaid": fk}
    elif metric == "pairwise_judge":
        from juryeval import PairwiseJudge

        if ref_values is None:
            print("error: --references required for pairwise_judge metric", file=sys.stderr)
            sys.exit(1)
        judge = PairwiseJudge(model=args.model)
        results = []
        for i, (pred, ref) in enumerate(zip(pred_values, ref_values)):
            q = questions[i] if questions else ""
            r = judge.compare(answer_a=pred, answer_b=ref, question=q)
            results.append(r)
        winners = [r.get("winner") for r in results]
        result = {
            "results": results,
            "summary": {
                "total": len(results),
                "a_wins": winners.count("A"),
                "b_wins": winners.count("B"),
                "ties": winners.count("tie"),
            },
        }
    elif metric == "pointwise_judge":
        from juryeval import PointwiseJudge

        judge = PointwiseJudge(model=args.model)
        results = []
        for i, pred in enumerate(pred_values):
            q = questions[i] if questions else ""
            ref = ref_values[i] if ref_values else None
            r = judge.score(output=pred, question=q, reference=ref)
            results.append(r)
        scores = [r.get("score", 0) for r in results]
        result = {
            "results": results,
            "summary": {
                "count": len(scores),
                "mean": sum(scores) / len(scores) if scores else 0,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
            },
        }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(result, indent=2, default=str))


def _cmd_calibrate(args):
    from juryeval import PairwiseJudge, JudgeCalibration

    judge = PairwiseJudge(model=args.model)
    cal = JudgeCalibration()
    report = cal.evaluate(judge)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("Judge Calibration Report")
        print("=" * 40)
        for key, value in report.items():
            key_str = key.replace("_", " ").title()
            if isinstance(value, float):
                print(f"  {key_str}: {value:.4f}")
            else:
                print(f"  {key_str}: {value}")


def _cmd_prompt(args):
    from juryeval import PromptVariance

    def model_fn(prompt):
        from juryeval import PointwiseJudge

        judge = PointwiseJudge(model=args.model)
        result = judge.score(output="mock", question=prompt)
        return str(result.get("score", 0))

    pv = PromptVariance(model_fn=model_fn)
    report = pv.analyze(args.question, num_variants=args.num_variants)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("Prompt Sensitivity Analysis")
        print("=" * 40)
        print(f"  Variants:        {report.get('num_variants', 'N/A')}")
        print(f"  Output Length Mean:  {report.get('output_length_mean', 'N/A')}")
        print(f"  Outputs:          {report.get('outputs', [])}")


def main():
    parser = _build_parser()
    args = parser.parse_args()

    commands = {
        "score": _cmd_score,
        "compare": _cmd_compare,
        "evaluate": _cmd_evaluate,
        "calibrate": _cmd_calibrate,
        "prompt": _cmd_prompt,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
