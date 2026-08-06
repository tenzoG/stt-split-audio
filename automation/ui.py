"""
Rich-based terminal UI for the AB automation.

All user-facing output routes through the single ``console`` instance in this
module so styling stays consistent and no ``print()`` calls leak in from
elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt
from rich.table import Table
from rich.text import Text

from sheet import SheetStats, UserRange, format_hhmmss


# [Reason] Single shared Console so live regions (Progress) coordinate with prints.
console: Console = Console()


# --- Banners & panels ------------------------------------------------------
def banner() -> None:
    console.print()
    console.print(
        Panel.fit(
            Text("AB AUTOMATION", style="bold white", justify="center"),
            border_style="cyan",
            padding=(1, 8),
        )
    )
    console.print()


def group_filter_panel(group_label: str, dropped_other_groups: int) -> None:
    """Announce that the run is restricted to a single Group and show drop count."""
    line = Text.assemble(
        ("Processing ", "cyan"),
        (group_label, "bold cyan"),
        (" only.", "cyan"),
    )
    if dropped_other_groups:
        line.append(
            f"  (ignored {dropped_other_groups} row(s) from other groups)",
            style="dim",
        )
    console.print(Panel(line, border_style="cyan"))
    console.print()


def sheet_stats_panel(
    stats: SheetStats,
    invalid_durations: int,
    dropped_no_duration: int = 0,
) -> None:
    body = Table.grid(padding=(0, 2))
    body.add_column(style="bold cyan")
    body.add_column()
    body.add_row("Total Rows:", str(stats.total))
    body.add_row("Already Uploaded:", str(stats.already_uploaded))
    body.add_row("Remaining:", str(stats.remaining))
    # [Reason] Show how many sheet rows were ignored because they had no duration.
    if dropped_no_duration:
        body.add_row(
            Text("Ignored (no duration):", style="bold dim"),
            Text(f"{dropped_no_duration}", style="dim"),
        )
    if invalid_durations:
        body.add_row(
            Text("Invalid Durations:", style="bold yellow"),
            Text(f"{invalid_durations} (counted as 00:00:00)", style="yellow"),
        )
    console.print(Panel(body, title="Google Sheet Loaded", border_style="green"))
    console.print(
        f"[dim]Skipped {stats.already_uploaded} uploaded rows.[/dim]\n"
    )


def gap_warning_panel() -> None:
    msg = Text.assemble(
        ("Gap detected.\n\n", "bold yellow"),
        ("Some users' chunks would span already-uploaded rows.\n", "yellow"),
        ("Automatically split into multiple contiguous sub-ranges so ", "yellow"),
        ("uploaded rows are never re-processed.\n", "yellow"),
    )
    console.print(Panel(msg, title="WARNING", border_style="yellow"))
    console.print()


def error_panel(title: str, message: str) -> None:
    console.print(
        Panel(
            Text(message, style="red"),
            title=title,
            border_style="red",
        )
    )


def stage_failure_panel(stage_name: str, tail: str) -> None:
    console.print(
        Panel(
            Text(tail or "(no output captured)", style="red"),
            title=f"FAILED :: {stage_name}",
            border_style="red",
            subtitle="last output lines",
        )
    )


# --- Empty ranges display --------------------------------------------------
def empty_ranges_panel(ranges: Sequence[Tuple[int, int]]) -> None:
    """Render contiguous ranges of empty 'Uploaded Pecha Tool' Sr. no values."""
    if not ranges:
        console.print(
            Panel(
                Text("No remaining rows.", style="dim"),
                title="Empty 'Uploaded Pecha Tool' Ranges",
                border_style="cyan",
            )
        )
        console.print()
        return

    table = Table(
        title="Empty 'Uploaded Pecha Tool' Ranges",
        border_style="cyan",
        header_style="bold cyan",
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Sr. no Range", justify="center")
    table.add_column("Rows", justify="right")

    total_rows = 0
    for i, (start, end) in enumerate(ranges, start=1):
        rows = end - start + 1
        total_rows += rows
        label = f"{start}" if start == end else f"{start} - {end}"
        table.add_row(str(i), label, str(rows))

    console.print(table)
    console.print(
        f"[dim]{len(ranges)} contiguous range(s), {total_rows} row(s) total.[/dim]\n"
    )


# --- Prompts ---------------------------------------------------------------
def ask_sr_range_and_users(min_sr: int, max_sr: int) -> tuple[int, int, int]:
    """Prompt for start Sr. no, end Sr. no, and user count with bounds validation.

    Pressing Enter accepts the default (full window ``[min_sr, max_sr]``).
    """
    console.rule(style="dim")

    # [Reason] Defaults let the user process the whole remaining window with three Enters.
    while True:
        start_sr = IntPrompt.ask(
            f"[bold]Start Sr. no[/bold] [dim](range: {min_sr}-{max_sr})[/dim]",
            default=min_sr,
            console=console,
        )
        if start_sr < min_sr or start_sr > max_sr:
            console.print(
                f"[red]Start Sr. no must be between {min_sr} and {max_sr}.[/red]"
            )
            continue
        break

    while True:
        end_sr = IntPrompt.ask(
            f"[bold]End Sr. no[/bold] [dim](range: {start_sr}-{max_sr})[/dim]",
            default=max_sr,
            console=console,
        )
        if end_sr < start_sr:
            console.print(
                f"[red]End Sr. no must be >= start ({start_sr}).[/red]"
            )
            continue
        if end_sr > max_sr:
            console.print(f"[red]End Sr. no must be <= {max_sr}.[/red]")
            continue
        break

    while True:
        users = IntPrompt.ask(
            "[bold]Split among how many users?[/bold]",
            default=1,
            console=console,
        )
        if users <= 0:
            console.print("[red]Please enter a positive number.[/red]")
            continue
        break

    console.rule(style="dim")
    return start_sr, end_sr, users


def confirm_proceed() -> bool:
    return Confirm.ask("[bold]Proceed?[/bold]", default=True, console=console)


# --- Range summary table ---------------------------------------------------
def user_summary_table(ranges: Sequence[UserRange]) -> None:
    """Group sub-ranges by user and print a summary table."""
    # [Reason] Aggregate sub-ranges into one row per user for the summary view.
    per_user: dict[int, dict[str, object]] = {}
    for r in ranges:
        agg = per_user.setdefault(
            r.user_index,
            {"ranges": [], "rows": 0, "duration": pd.Timedelta(0)},
        )
        agg["ranges"].append(f"{r.from_id}-{r.to_id}")  # type: ignore[union-attr]
        agg["rows"] = int(agg["rows"]) + r.rows  # type: ignore[operator]
        agg["duration"] = agg["duration"] + r.duration  # type: ignore[operator]

    table = Table(title="User Summary", border_style="cyan", header_style="bold cyan")
    table.add_column("User", justify="center", style="bold")
    table.add_column("Range(s)")
    table.add_column("Rows", justify="right")
    table.add_column("Duration", justify="right")

    for user_i in sorted(per_user):
        agg = per_user[user_i]
        table.add_row(
            str(user_i),
            ", ".join(agg["ranges"]),  # type: ignore[arg-type]
            str(agg["rows"]),
            format_hhmmss(agg["duration"]),  # type: ignore[arg-type]
        )
    console.print(table)
    console.print()


# --- Final summary ---------------------------------------------------------
@dataclass
class FinalSummary:
    total_users: int
    ranges: List[str]
    skipped_uploaded: int
    processed_rows: int
    total_duration: pd.Timedelta


def final_summary_panel(summary: FinalSummary) -> None:
    body = Table.grid(padding=(0, 2))
    body.add_column(style="bold cyan")
    body.add_column()
    body.add_row("Users Processed:", str(summary.total_users))
    body.add_row("Ranges:", "\n".join(summary.ranges) if summary.ranges else "-")
    body.add_row("Skipped Uploaded Rows:", str(summary.skipped_uploaded))
    body.add_row("Processed Rows:", str(summary.processed_rows))
    body.add_row("Total Duration:", format_hhmmss(summary.total_duration))
    console.print()
    console.print(
        Panel(body, title="Completed Successfully", border_style="green")
    )
    console.print()
