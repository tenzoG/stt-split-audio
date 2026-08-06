#!/usr/bin/env python3
"""
AB Automation - production CLI orchestrator.

Reads the AB Google Sheet, picks rows whose "Uploaded Pecha Tool" cell is
empty, splits the workload across N users into contiguous ``Sr. no`` ranges
(auto-splitting to skip already-uploaded rows), and runs the existing 5-stage
pipeline once per range - updating only the ``AB`` entry inside ``var``.

The rest of the pipeline is untouched:
  * ``ab_config.json`` and ``ab_config_etext.json`` inherit FROM_ID/TO_ID from
    ``var`` via ``common_utils.load_config_from_file``.
  * ``run_all.sh`` still works as-is for manual runs.

Usage (from ``stt-split-audio/automation/``)::

    python run_ab_auto.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

# [Reason] Make sibling ../util available exactly like every other script here.
_HERE = Path(__file__).resolve().parent
_UTIL_DIR = str(_HERE.parent / "util")
if _UTIL_DIR not in sys.path:
    sys.path.append(_UTIL_DIR)

from common_utils import load_config_from_file  # noqa: E402

from pipeline import STAGES, run_stage  # noqa: E402
from sheet import (  # noqa: E402
    DURATION_COL,
    GROUP_TARGET_LABEL,
    SheetError,
    UserRange,
    build_user_ranges,
    drop_rows_without_duration,
    filter_group_a,
    filter_remaining,
    format_hhmmss,
    list_empty_ranges,
    load_ab_sheet,
    parse_durations,
    slice_by_sr_range,
)
from ui import (  # noqa: E402
    FinalSummary,
    ask_sr_range_and_users,
    banner,
    confirm_proceed,
    console,
    empty_ranges_panel,
    error_panel,
    final_summary_panel,
    gap_warning_panel,
    group_filter_panel,
    sheet_stats_panel,
    stage_failure_panel,
    user_summary_table,
)
from var_updater import VarUpdateError, update_ab_range  # noqa: E402


# [Reason] The AB config lives at a fixed relative path from this script.
AB_CONFIG_PATH: Path = _HERE.parent / "json_config" / "ab_config.json"


# ---------------------------------------------------------------------------
# Range execution
# ---------------------------------------------------------------------------
def _run_all_ranges(ranges: list[UserRange]) -> bool:
    """Run the 5-stage pipeline for every range. Returns True on full success."""
    total_users = len({r.user_index for r in ranges})
    stage_names = [s.name for s in STAGES]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        overall_task = progress.add_task(
            f"Overall (0/{len(ranges)} ranges)", total=len(ranges)
        )

        for idx, r in enumerate(ranges, start=1):
            sub_label = f".{r.sub_index}" if r.sub_index > 1 else ""
            range_label = f"User {r.user_index}/{total_users}{sub_label}  Range {r.from_id}-{r.to_id}"
            progress.console.rule(f"[bold]{range_label}[/bold]", style="cyan")

            # [Reason] Update var BEFORE running so both configs pick up the new range.
            try:
                update_ab_range(r.from_id, r.to_id)
            except VarUpdateError as exc:
                error_panel("var update failed", str(exc))
                return False
            progress.console.print(
                f"[green]var updated[/green]  AB.FROM_ID={r.from_id}  AB.TO_ID={r.to_id}"
            )

            stage_task = progress.add_task(
                f"{range_label}  waiting...", total=len(STAGES)
            )

            for stage_i, stage in enumerate(STAGES, start=1):
                progress.update(
                    stage_task,
                    description=f"{range_label}  Running: {stage.name} ({stage_i}/{len(STAGES)})",
                )
                # [Reason] Stream child output above the live progress region.
                result = run_stage(
                    stage,
                    on_line=lambda line: progress.console.log(line),
                )
                if not result.ok:
                    progress.update(
                        stage_task,
                        description=f"[red]{range_label}  FAILED at {stage.name}[/red]",
                    )
                    stage_failure_panel(stage.name, result.output_tail)
                    return False
                progress.advance(stage_task)

            progress.update(stage_task, description=f"[green]{range_label}  done[/green]")
            progress.advance(overall_task)
            progress.update(
                overall_task,
                description=f"Overall ({idx}/{len(ranges)} ranges)",
            )

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_sheet_id() -> str:
    """Read SHEET_ID from ab_config.json via the project's standard loader."""
    config = load_config_from_file(str(AB_CONFIG_PATH))
    sheet_id = config.get("SHEET_ID")
    if not sheet_id:
        raise SheetError('ab_config.json is missing "SHEET_ID".')
    return str(sheet_id)


def main() -> int:
    try:
        banner()

        # 1. Fetch and validate the sheet
        try:
            sheet_id = _load_sheet_id()
            df = load_ab_sheet(sheet_id)
        except SheetError as exc:
            error_panel("Sheet error", str(exc))
            return 2
        except Exception as exc:  # noqa: BLE001
            error_panel("Failed to load Google Sheet", f"{type(exc).__name__}: {exc}")
            return 2

        # [Reason] Restrict scope to Group A BEFORE any counting so other groups
        # never influence Total / Already Uploaded / Remaining, and inform the user.
        df, dropped_other_groups = filter_group_a(df)
        group_filter_panel(GROUP_TARGET_LABEL, dropped_other_groups)
        if df.empty:
            error_panel(
                "No Group A rows",
                "The sheet contains no rows tagged as Group A.",
            )
            return 0

        # [Reason] Rows without a Standard Duration are not real work items;
        # drop them BEFORE counting Total/Already Uploaded/Remaining so the
        # stats reflect only valid rows.
        df, dropped_no_duration = drop_rows_without_duration(df)

        remaining_df, stats = filter_remaining(df)
        # [Reason] Preview-parse durations so we can surface the invalid count
        # to the user BEFORE prompting for row/user counts.
        if not remaining_df.empty:
            _, invalid_all = parse_durations(remaining_df[DURATION_COL])
        else:
            invalid_all = 0
        sheet_stats_panel(stats, invalid_all, dropped_no_duration)

        if stats.remaining == 0:
            error_panel(
                "Nothing to do",
                'No rows are pending: every row already has "Uploaded Pecha Tool" set.',
            )
            return 0

        # 2. Show contiguous ranges of empty "Uploaded Pecha Tool" rows.
        empty_ranges = list_empty_ranges(remaining_df)
        empty_ranges_panel(empty_ranges)

        # 3. Ask for the Sr. no window + user count.
        min_sr = empty_ranges[0][0]
        max_sr = empty_ranges[-1][1]
        start_sr, end_sr, users = ask_sr_range_and_users(min_sr, max_sr)

        # [Reason] Restrict work to the user-specified Sr. no window.
        selected_df = slice_by_sr_range(remaining_df, start_sr, end_sr)
        if selected_df.empty:
            error_panel(
                "No rows in window",
                f"No remaining rows fall in Sr. no {start_sr}..{end_sr}.",
            )
            return 0

        # [Reason] Cap users at row count so no user gets 0 rows.
        if users > len(selected_df):
            console.print(
                f"[yellow]Only {len(selected_df)} row(s) in the selected window; "
                f"reducing users from {users} to {len(selected_df)}.[/yellow]\n"
            )
            users = len(selected_df)

        # 4. Build ranges and show plan
        ranges, had_gaps, invalid_selected = build_user_ranges(
            selected_df, len(selected_df), users
        )
        if not ranges:
            error_panel("No ranges", "Could not build any ranges from the selected rows.")
            return 2

        if had_gaps:
            gap_warning_panel()
        if invalid_selected:
            console.print(
                f"[yellow]Note:[/yellow] {invalid_selected} selected row(s) had unparseable "
                f"Standard Duration; counted as 00:00:00.\n"
            )

        user_summary_table(ranges)

        if not confirm_proceed():
            console.print("[dim]Aborted by user. No changes made.[/dim]")
            return 0

        # 4. Execute pipeline for every range
        success = _run_all_ranges(ranges)
        if not success:
            return 1

        # 5. Final summary
        final_summary_panel(
            FinalSummary(
                total_users=len({r.user_index for r in ranges}),
                ranges=[f"{r.from_id}-{r.to_id}" for r in ranges],
                skipped_uploaded=stats.already_uploaded,
                processed_rows=sum(r.rows for r in ranges),
                total_duration=sum((r.duration for r in ranges), pd.Timedelta(0)),
            )
        )
        return 0

    except KeyboardInterrupt:
        console.print()
        error_panel("Cancelled", "Run interrupted by user (Ctrl+C).")
        return 130
    except Exception as exc:  # noqa: BLE001
        # [Reason] Last-chance catch-all so users get a clear panel, not a raw traceback.
        error_panel("Unexpected error", f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
