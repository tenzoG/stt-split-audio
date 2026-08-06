"""
Runs the 5-stage AB pipeline for a single ``UserRange``.

Stages mirror ``run_all.sh`` exactly. The pipeline itself is unchanged; this
module only orchestrates ``subprocess`` calls, streams their output, and
reports success/failure per stage so the CLI can render progress.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple


# [Reason] Anchor relative paths (../foo/bar.py) to this file's directory so
# the subprocess sees the same layout run_all.sh assumes.
AUTOMATION_DIR: Path = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Stage:
    """One pipeline stage: the friendly name, script path, and its config."""

    name: str
    script: str
    config: str


# [Reason] Mirrors run_all.sh order exactly. Do not reorder without updating run_all.sh.
STAGES: Tuple[Stage, ...] = (
    Stage("yt_download.py", "../audio_download_and_split/yt_download.py", "../json_config/ab_config.json"),
    Stage("run_inference_text.py", "../inference_runner/run_inference_text.py", "../json_config/ab_config.json"),
    Stage("make_csv.py", "../make_db_csv/make_csv.py", "../json_config/ab_config.json"),
    Stage("download_doc.py", "../make_db_csv/download_doc.py", "../json_config/ab_config_etext.json"),
    Stage("transfer_text.py", "../make_db_csv/transfer_text.py", "../json_config/ab_config.json"),
)


@dataclass
class StageResult:
    """Result of a single stage run."""

    stage: Stage
    returncode: int
    output_tail: str  # last N lines of combined stdout/stderr

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _tail(lines: List[str], n: int = 60) -> str:
    return "".join(lines[-n:])


def run_stage(
    stage: Stage,
    on_line: Optional[Callable[[str], None]] = None,
) -> StageResult:
    """Execute one stage. ``on_line`` receives each output line (already stripped)."""
    proc = subprocess.Popen(
        ["python", stage.script, "--config", stage.config],
        cwd=str(AUTOMATION_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )
    buffered: List[str] = []
    assert proc.stdout is not None  # for type checkers; PIPE guarantees this
    try:
        for raw in proc.stdout:
            buffered.append(raw)
            if on_line is not None:
                on_line(raw.rstrip("\n"))
    finally:
        proc.stdout.close()
        proc.wait()

    return StageResult(
        stage=stage,
        returncode=proc.returncode,
        output_tail=_tail(buffered),
    )
