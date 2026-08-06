"""
Targeted regex-based updater for the shared ``stt-split-audio/var`` file.

The file mixes JSON with ``//`` line-comments (see ``common_utils._load_json_with_comments``),
so a naive ``json.dump`` would strip the department comments. This module rewrites
only the ``"AB"`` line and leaves everything else byte-identical.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


# [Reason] Resolve relative to this file so the updater works from any cwd.
VAR_PATH: Path = Path(__file__).resolve().parent.parent / "var"

# [Reason] Match the whole AB line with its indent + trailing comma so we can
# regenerate an identical-looking line and preserve document shape.
_AB_LINE_RE: re.Pattern[str] = re.compile(
    r'^(?P<indent>[ \t]*)"AB"\s*:\s*\{[^}]*\}(?P<trail>,?)',
    flags=re.MULTILINE,
)


class VarUpdateError(Exception):
    """Raised when the ``var`` file is missing or malformed."""


def _backup_once(var_path: Path) -> None:
    """Create ``var.bak`` on the first update in this run; keep subsequent updates.

    We don't want to overwrite the backup on every range in a multi-range run,
    otherwise the original starting state would be lost.
    """
    backup = var_path.with_suffix(var_path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(var_path, backup)


def update_ab_range(from_id: int, to_id: int, var_path: Path = VAR_PATH) -> None:
    """Rewrite ONLY the ``"AB"`` line inside ``var``.

    Raises ``VarUpdateError`` on any structural problem so the caller can abort
    before running the pipeline against a stale range.
    """
    if not var_path.exists():
        raise VarUpdateError(f"var file not found at {var_path}")

    text = var_path.read_text()
    if not _AB_LINE_RE.search(text):
        raise VarUpdateError('Could not locate the "AB" entry inside var.')

    _backup_once(var_path)

    # [Reason] Regenerate the AB line while preserving indent + trailing comma.
    def _replacement(match: re.Match[str]) -> str:
        return (
            f'{match.group("indent")}"AB": '
            f'{{ "FROM_ID": {from_id}, "TO_ID": {to_id} }}'
            f'{match.group("trail")}'
        )

    new_text, count = _AB_LINE_RE.subn(_replacement, text, count=1)
    if count != 1:
        raise VarUpdateError("Failed to update exactly one AB entry.")

    var_path.write_text(new_text)
