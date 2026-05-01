import argparse
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm


BASE_DIR = Path(__file__).resolve().parent
STOCKS_SCRIPT = BASE_DIR / "stocks.py"
AGENT_WORKFLOW_SCRIPT = BASE_DIR / "agent_workflow.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full stocks + analyst pipeline.")

    parser.add_argument(
        "--skip-stocks",
        action="store_true",
        help="Skip stocks.py and run only the analyst stage.",
    )
    parser.add_argument(
        "--skip-agents",
        action="store_true",
        help="Skip agent_workflow.py and run only the screening/package stage.",
    )

    # stocks.py options
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="Force full metadata refresh in stocks.py.",
    )
    parser.add_argument(
        "--metadata-refresh-days",
        type=int,
        default=None,
        help="Override metadata cache TTL in days for stocks.py.",
    )
    parser.add_argument(
        "--refresh-insider",
        action="store_true",
        help="Force insider cache refresh in stocks.py.",
    )

    # agent_workflow.py options
    parser.add_argument("--top-n", type=int, default=0, help="Legacy shortcut for rank 1 through N.")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Legacy offset used with --top-n.",
    )
    parser.add_argument(
        "--start-rank",
        type=int,
        default=1,
        help="Starting screener rank for analyst selection.",
    )
    parser.add_argument(
        "--end-rank",
        type=int,
        default=3,
        help="Ending screener rank for analyst selection.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Concurrent analyst API calls.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override OpenAI model for agent_workflow.py.")
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        help="Reasoning effort for analyst calls.",
    )
    parser.add_argument(
        "--web-tool-type",
        type=str,
        default=None,
        help="Override web tool type for agent_workflow.py.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id override for analyst outputs.")
    parser.add_argument("--user-id", type=str, default="local-user", help="Application user id for session storage.")
    parser.add_argument("--user-email", type=str, default="", help="Optional user email for session storage.")
    parser.add_argument(
        "--include-feather",
        action="store_true",
        help="Include .feather files in analyst uploads.",
    )
    parser.add_argument(
        "--max-sec-html-files",
        type=int,
        default=4,
        help="Maximum SEC filing HTML files to attach per ticker.",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=1.5,
        help="Skip package files larger than this size in MB for analyst uploads.",
    )
    parser.add_argument(
        "--skip-tickers",
        type=str,
        default="",
        help="Comma-separated tickers to exclude from analyst runs.",
    )
    return parser.parse_args()


def run_step(label: str, cmd: list[str], pbar: tqdm) -> None:
    pbar.set_description(label)
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    pbar.update(1)


def resolve_selection_bounds(args: argparse.Namespace) -> tuple[int, int]:
    start_rank = max(1, int(args.start_rank))
    end_rank = max(start_rank, int(args.end_rank))
    if args.top_n > 0:
        start_rank = max(1, int(args.offset) + 1)
        end_rank = start_rank + int(args.top_n) - 1
    return start_rank, end_rank


def build_stocks_cmd(args: argparse.Namespace) -> list[str]:
    start_rank, end_rank = resolve_selection_bounds(args)
    cmd = [sys.executable, str(STOCKS_SCRIPT)]
    if args.refresh_metadata:
        cmd.append("--refresh-metadata")
    if args.metadata_refresh_days is not None:
        cmd.extend(["--metadata-refresh-days", str(args.metadata_refresh_days)])
    if args.refresh_insider:
        cmd.append("--refresh-insider")
    cmd.extend(
        [
            "--chart-start-rank",
            str(start_rank),
            "--chart-end-rank",
            str(end_rank),
            "--package-start-rank",
            str(start_rank),
            "--package-end-rank",
            str(end_rank),
        ]
    )
    return cmd


def build_agents_cmd(args: argparse.Namespace) -> list[str]:
    start_rank, end_rank = resolve_selection_bounds(args)

    cmd = [
        sys.executable,
        str(AGENT_WORKFLOW_SCRIPT),
        "--start-rank",
        str(start_rank),
        "--end-rank",
        str(end_rank),
        "--max-workers",
        str(args.max_workers),
        "--reasoning-effort",
        args.reasoning_effort,
        "--max-sec-html-files",
        str(args.max_sec_html_files),
        "--max-file-size-mb",
        str(args.max_file_size_mb),
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.web_tool_type:
        cmd.extend(["--web-tool-type", args.web_tool_type])
    if args.run_id:
        cmd.extend(["--run-id", args.run_id])
    if args.user_id:
        cmd.extend(["--user-id", args.user_id])
    if args.user_email:
        cmd.extend(["--user-email", args.user_email])
    if args.include_feather:
        cmd.append("--include-feather")
    if args.skip_tickers:
        cmd.extend(["--skip-tickers", args.skip_tickers])
    return cmd


def main() -> None:
    args = parse_args()

    if args.skip_stocks and args.skip_agents:
        raise SystemExit("Nothing to run: both --skip-stocks and --skip-agents were set.")

    total_steps = int(not args.skip_stocks) + int(not args.skip_agents)
    with tqdm(total=total_steps, desc="Pipeline", unit="stage") as pbar:
        if not args.skip_stocks:
            run_step("Running stocks.py", build_stocks_cmd(args), pbar)

        if not args.skip_agents:
            run_step("Running agent_workflow.py", build_agents_cmd(args), pbar)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
