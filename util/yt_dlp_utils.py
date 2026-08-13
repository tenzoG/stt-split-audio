"""
Reusable yt-dlp audio download helpers with validation and client fallbacks.

Root cause context (yt-dlp >= 2025.11 / 2026.x):
  - Only the *deno* JS runtime is enabled by default.
  - Node must be explicitly enabled via ``--js-runtimes node[:PATH]``.
  - Node must be >= 22.0.0 (v20 on PATH is detected but marked unsupported).
  - Challenge solver scripts usually require ``--remote-components ejs:github``.
  Without these, yt-dlp often reports the misleading
  ``ERROR: [youtube] This video is not available`` even when the video plays in a browser.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


# [Reason] Known audio containers we accept after yt-dlp extract-audio
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".opus", ".webm", ".ogg", ".flac", ".aac"})

# [Reason] Default client fallback order for YouTube SABR / anti-bot format gaps
DEFAULT_PLAYER_CLIENTS: tuple[str, ...] = ("default", "android", "tv", "ios")

# [Reason] Browsers to try for cookie auth when the first pass fails without cookies
DEFAULT_COOKIE_BROWSER_FALLBACKS: tuple[str, ...] = ("chrome", "brave", "safari", "firefox", "edge")

# [Reason] yt-dlp binary names to try when PATH or venv layouts differ
YT_DLP_CANDIDATES: tuple[str, ...] = ("yt-dlp", "yt_dlp")

# [Reason] Matches yt-dlp.utils._jsruntime.NodeJsRuntime.MIN_SUPPORTED_VERSION
MIN_NODE_VERSION: tuple[int, ...] = (22, 0, 0)

# [Reason] Prefer challenge-solver related failures over the misleading "unavailable" message
_ERROR_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), message)
    for pattern, message in (
        (
            r"no supported javascript runtime|only deno is enabled by default",
            "No supported JavaScript runtime enabled. yt-dlp only enables deno by default; "
            "enable Node with --js-runtimes node (Node >= 22 required) or install Deno.",
        ),
        (
            r"remote component challenge solver|remote-components ejs",
            "EJS challenge solver script was skipped. Enable with --remote-components ejs:github "
            "(or set YT_DLP.remote_components).",
        ),
        (
            r"n challenge solving failed|sabr",
            "YouTube n-challenge / SABR extraction failed. Ensure a supported JS runtime "
            "(Node >= 22 or Deno) and --remote-components ejs:github are enabled; "
            "browser cookies may also help.",
        ),
        (r"only images are available|storyboard", "No downloadable media formats returned by YouTube (storyboard/images only)."),
        (r"requested format is not available", "Requested audio format is not available for this video."),
        (r"private video", "Video is private."),
        (r"video has been removed|has been deleted", "Video has been removed."),
        (
            r"not available in your country|available in your country|blocked in your country|region.?restrict",
            "Video is region-restricted.",
        ),
        (
            r"sign in to confirm|age.?restrict|confirm your age",
            "Video requires sign-in or is age-restricted. Configure cookies_from_browser or cookies_path.",
        ),
        (
            r"this video is not available|video is unavailable",
            "YouTube reported the video as unavailable to yt-dlp (often a JS-runtime / EJS / "
            "player-client issue — check diagnostics above; the video may still play in a browser).",
        ),
        (
            r"unable to download webpage|urlopen error|connection (reset|refused|timed out)|network is unreachable|temporary failure in name resolution",
            "Network failure while contacting YouTube.",
        ),
        (r"timed? ?out|timeout", "Download timed out."),
        (
            r"could not copy chrome cookie|could not find (chrome|firefox|brave|edge|safari)|unable to load cookies",
            "Failed to load browser cookies. Check cookies_from_browser / cookies_path.",
        ),
        (
            r"no such file or directory.*node|javascript runtime|js runtime|ejs",
            "JavaScript runtime missing or misconfigured. Set YT_DLP.js_runtime or install Node >= 22 / Deno.",
        ),
        (r"http error 403|403: forbidden", "YouTube returned HTTP 403 Forbidden."),
        (r"http error 404|404: not found", "YouTube returned HTTP 404 Not Found."),
        (r"unsupported url", "Unsupported or invalid URL."),
    )
)


ConfigLike = Mapping[str, Any]


@dataclass
class JsRuntimeSelection:
    """Detected or configured JS runtime for yt-dlp ``--js-runtimes``."""

    name: str
    path: Optional[str] = None
    version: Optional[str] = None
    supported: bool = True
    source: str = "auto"

    @property
    def runtime_arg(self) -> str:
        """Value passed to ``--js-runtimes`` (e.g. ``node:/abs/path``)."""
        if self.path:
            return f"{self.name}:{self.path}"
        return self.name

    def describe(self) -> str:
        ver = self.version or "unknown"
        path = self.path or "(PATH lookup)"
        status = "supported" if self.supported else "UNSUPPORTED"
        return f"{self.name} {ver} [{status}] @ {path} (via {self.source})"


@dataclass
class EnvironmentDiagnostics:
    """Runtime environment snapshot printed on download failure."""

    yt_dlp_cli_version: Optional[str] = None
    yt_dlp_module_version: Optional[str] = None
    yt_dlp_executable: Optional[str] = None
    python_executable: str = field(default_factory=lambda: sys.executable)
    path_env: str = field(default_factory=lambda: os.environ.get("PATH", ""))
    detected_node: Optional[str] = None
    detected_deno: Optional[str] = None
    selected_js_runtime: Optional[str] = None
    cookies_enabled: bool = False
    cookies_from_browser: Optional[str] = None
    cookies_path: Optional[str] = None
    remote_components: Optional[str] = None
    player_client: Optional[str] = None
    extractor_args: Optional[str] = None
    command: Optional[List[str]] = None

    def print_block(self) -> None:
        """Print actionable environment diagnostics."""
        print("\n===== yt-dlp diagnostics =====")
        print(f"Python executable:     {self.python_executable}")
        print(f"yt-dlp CLI version:    {self.yt_dlp_cli_version or '(unknown)'}")
        print(f"Python yt_dlp version: {self.yt_dlp_module_version or '(not importable)'}")
        print(f"yt-dlp executable:     {self.yt_dlp_executable or '(module fallback)'}")
        print(f"PATH:                  {self.path_env}")
        print(f"Detected node:         {self.detected_node or '(none)'}")
        print(f"Detected deno:         {self.detected_deno or '(none)'}")
        print(f"Selected JS runtime:   {self.selected_js_runtime or '(none enabled)'}")
        print(f"Cookies enabled:       {self.cookies_enabled}")
        print(f"cookies_from_browser:  {self.cookies_from_browser or '(none)'}")
        print(f"cookies_path:          {self.cookies_path or '(none)'}")
        print(f"remote_components:     {self.remote_components or '(none)'}")
        print(f"player_client:         {self.player_client or 'default'}")
        print(f"extractor_args:        {self.extractor_args or '(none)'}")
        if self.command:
            print(f"Exact command:         {_format_cmd(self.command)}")
        print("==============================\n")


@dataclass
class DownloadResult:
    """Outcome of a single URL download (possibly after multiple client attempts)."""

    success: bool
    filepath: Optional[str] = None
    reason: str = ""
    attempts: int = 0
    player_client: Optional[str] = None
    video_id: Optional[str] = None
    url: str = ""
    exit_code: Optional[int] = None
    duration_seconds: float = 0.0
    attempt_logs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result for logging or downstream consumers."""
        return asdict(self)


@dataclass
class PipelineSummary:
    """Aggregate counters for a multi-video download run."""

    downloaded: int = 0
    failed: int = 0
    skipped: int = 0
    retried: int = 0
    total_duration_seconds: float = 0.0

    def record_skip(self) -> None:
        """Count a row that was intentionally skipped (e.g. empty URL)."""
        self.skipped += 1

    def record(self, result: DownloadResult) -> None:
        """Update counters from one download result."""
        self.total_duration_seconds += result.duration_seconds
        if result.success:
            self.downloaded += 1
            if result.attempts > 1:
                self.retried += 1
        else:
            self.failed += 1
            if result.attempts > 1:
                self.retried += 1

    def print_report(self) -> None:
        """Print a final pipeline summary block."""
        print("\n" + "=" * 50)
        print("Download pipeline summary")
        print("=" * 50)
        print(f"Downloaded: {self.downloaded}")
        print(f"Failed:     {self.failed}")
        print(f"Skipped:    {self.skipped}")
        print(f"Retried:    {self.retried}")
        print(f"Total duration: {self.total_duration_seconds:.1f}s")
        print("=" * 50)


def extract_video_id(url: str) -> Optional[str]:
    """Extract a YouTube video id from common URL shapes, if present."""
    if not url:
        return None
    patterns = (
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:v=)([A-Za-z0-9_-]{11})",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_yt_dlp_config(config: Optional[ConfigLike] = None) -> Dict[str, Any]:
    """
    Normalize optional YT_DLP settings from a pipeline JSON config.

    Missing keys are omitted so callers can rely on defaults. Flat legacy keys
    (COOKIES_PATH, COOKIES_FROM_BROWSER) remain supported.
    """
    config = config or {}
    raw = dict(config.get("YT_DLP") or {})

    # [Reason] Preserve older cookie-only configs without a YT_DLP block
    if "cookies_path" not in raw and config.get("COOKIES_PATH"):
        raw["cookies_path"] = config["COOKIES_PATH"]
    if "cookies_from_browser" not in raw and config.get("COOKIES_FROM_BROWSER"):
        raw["cookies_from_browser"] = config["COOKIES_FROM_BROWSER"]

    return raw


def _parse_semver(version: str) -> Optional[tuple[int, ...]]:
    """Parse a dotted version string into an int tuple."""
    match = re.search(r"(\d+(?:\.\d+)*)", version or "")
    if not match:
        return None
    try:
        return tuple(int(part) for part in match.group(1).split("."))
    except ValueError:
        return None


def _node_version(path: str) -> Optional[str]:
    """Return ``node --version`` output (without leading v) or None."""
    try:
        completed = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    text = (completed.stdout or completed.stderr or "").strip()
    match = re.search(r"v?(\d+\.\d+\.\d+)", text)
    return match.group(1) if match else None


def _deno_version(path: str) -> Optional[str]:
    """Return deno version string or None."""
    try:
        completed = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    match = re.search(r"deno\s+(\S+)", (completed.stdout or ""), re.IGNORECASE)
    return match.group(1) if match else (completed.stdout or "").strip().splitlines()[:1][0] if completed.stdout else None


def _candidate_node_paths() -> List[str]:
    """Collect Node binaries from PATH, Homebrew, and nvm without hardcoding one machine path."""
    candidates: List[str] = []
    which_node = shutil.which("node")
    if which_node:
        candidates.append(which_node)

    home = Path.home()
    for fixed in (Path("/opt/homebrew/bin/node"), Path("/usr/local/bin/node")):
        if fixed.is_file():
            candidates.append(str(fixed))

    nvm_root = home / ".nvm" / "versions" / "node"
    if nvm_root.is_dir():
        for node_bin in sorted(nvm_root.glob("*/bin/node"), reverse=True):
            if node_bin.is_file():
                candidates.append(str(node_bin))

    for root in (
        home / ".fnm" / "node-versions",
        home / ".local" / "share" / "fnm" / "node-versions",
    ):
        if root.is_dir():
            for node_bin in sorted(root.glob("*/installation/bin/node"), reverse=True):
                if node_bin.is_file():
                    candidates.append(str(node_bin))

    # [Reason] Preserve order while de-duplicating resolved paths
    unique: List[str] = []
    seen = set()
    for path in candidates:
        try:
            key = str(Path(path).resolve())
        except OSError:
            key = path
        if key not in seen and Path(path).is_file():
            seen.add(key)
            unique.append(path)
    return unique


def detect_js_runtime(
    explicit: Optional[str] = None,
    *,
    prefer: str = "auto",
) -> Optional[JsRuntimeSelection]:
    """
    Detect a JS runtime suitable for yt-dlp YouTube challenges.

    ``explicit`` may be:
      - ``node`` / ``deno``
      - ``node:/abs/path``
      - ``/abs/path/to/node`` (treated as node)

    Auto mode prefers Deno (yt-dlp default), then the newest Node >= 22.
    """
    if explicit:
        value = str(explicit).strip()
        if ":" in value and not value.startswith(".") and not value.startswith("/"):
            name, path = value.split(":", 1)
            name = name.strip().lower()
            path = path.strip() or None
        elif os.path.sep in value or value.startswith("."):
            name = "node"
            path = value
        else:
            name = value.lower()
            path = None

        if name == "node":
            resolved = path or shutil.which("node")
            version = _node_version(resolved) if resolved else None
            vt = _parse_semver(version or "")
            supported = bool(vt and vt >= MIN_NODE_VERSION)
            return JsRuntimeSelection(
                name="node",
                path=resolved,
                version=version,
                supported=supported,
                source="config",
            )
        if name == "deno":
            resolved = path or shutil.which("deno")
            version = _deno_version(resolved) if resolved else None
            return JsRuntimeSelection(
                name="deno",
                path=resolved,
                version=version,
                supported=bool(resolved),
                source="config",
            )
        return JsRuntimeSelection(name=name, path=path, source="config")

    prefer = (prefer or "auto").lower()

    deno = shutil.which("deno")
    if prefer in {"auto", "deno"} and deno:
        version = _deno_version(deno)
        return JsRuntimeSelection(
            name="deno",
            path=deno,
            version=version,
            supported=True,
            source="auto-detect",
        )

    # [Reason] Prefer the newest supported Node when PATH points at an old nvm default (e.g. v20)
    best_node: Optional[JsRuntimeSelection] = None
    for node_path in _candidate_node_paths():
        version = _node_version(node_path)
        vt = _parse_semver(version or "")
        if not vt or vt < MIN_NODE_VERSION:
            continue
        candidate = JsRuntimeSelection(
            name="node",
            path=node_path,
            version=version,
            supported=True,
            source="auto-detect",
        )
        if best_node is None or (vt > (_parse_semver(best_node.version or "") or (0,))):
            best_node = candidate

    if best_node and prefer in {"auto", "node"}:
        return best_node

    # [Reason] Surface unsupported PATH node so diagnostics explain why auto-enable failed
    which_node = shutil.which("node")
    if which_node:
        version = _node_version(which_node)
        return JsRuntimeSelection(
            name="node",
            path=which_node,
            version=version,
            supported=False,
            source="auto-detect-unsupported",
        )
    return None


def resolve_yt_dlp_binary(explicit: Optional[str] = None) -> List[str]:
    """
    Return argv prefix that invokes yt-dlp.

    Prefers an explicit binary, then PATH ``yt-dlp``, then ``python -m yt_dlp``
    using the current interpreter (matches the active venv).
    """
    if explicit:
        return [explicit]
    for candidate in YT_DLP_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return [resolved]
    # [Reason] Use the same interpreter that imported this module / runs the pipeline
    return [sys.executable, "-m", "yt_dlp"]


def get_yt_dlp_versions(yt_dlp_bin: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(cli_version, module_version, executable_display)``."""
    argv = resolve_yt_dlp_binary(yt_dlp_bin)
    cli_version = None
    try:
        completed = subprocess.run(
            [*argv, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if completed.returncode == 0:
            cli_version = (completed.stdout or completed.stderr or "").strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass

    module_version = None
    try:
        import yt_dlp.version as yt_version  # type: ignore

        module_version = getattr(yt_version, "__version__", None)
    except Exception:
        module_version = None

    return cli_version, module_version, " ".join(argv)


def classify_download_error(output: str, exit_code: Optional[int] = None) -> str:
    """Map yt-dlp output / exit codes to a clear failure reason."""
    text = (output or "").strip()
    for pattern, message in _ERROR_PATTERNS:
        if pattern.search(text):
            return message
    if exit_code == 124:
        return "Download timed out."
    if not text:
        return f"yt-dlp failed with exit code {exit_code}."
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("ERROR:"):
            return stripped
    return f"yt-dlp failed with exit code {exit_code}."


def find_downloaded_audio(
    output_dir: Union[str, Path],
    file_stem: str,
    expected_ext: Optional[str] = None,
    *,
    newer_than: Optional[float] = None,
) -> Optional[Path]:
    """
    Locate a non-empty downloaded audio file for the given stem.

    When ``newer_than`` is set (epoch seconds), ignore stale files from prior runs.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None

    candidates: List[Path] = []
    if expected_ext:
        ext = expected_ext if expected_ext.startswith(".") else f".{expected_ext}"
        candidates.append(output_dir / f"{file_stem}{ext}")

    for path in sorted(output_dir.glob(f"{file_stem}.*")):
        if path.suffix.lower() in AUDIO_EXTENSIONS and path not in candidates:
            candidates.append(path)

    for path in candidates:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                continue
            if newer_than is not None and path.stat().st_mtime < newer_than:
                continue
            return path
        except OSError:
            continue
    return None


def validate_downloaded_file(
    filepath: Union[str, Path],
    *,
    expected_ext: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Verify a downloaded audio file is usable before splitting.

    Checks existence, non-empty size, and optional extension match.
    """
    path = Path(filepath)
    if not path.parent.is_dir():
        return False, f"Download directory does not exist: {path.parent}"
    if not path.is_file():
        return False, f"Downloaded audio file does not exist: {path}"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"Unable to stat downloaded file {path}: {exc}"
    if size <= 0:
        return False, f"Downloaded audio file is empty: {path}"
    if expected_ext:
        ext = expected_ext if expected_ext.startswith(".") else f".{expected_ext}"
        if path.suffix.lower() != ext.lower():
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                return False, f"Unexpected file extension for {path} (expected {ext})"
    elif path.suffix.lower() not in AUDIO_EXTENSIONS:
        return False, f"File is not a recognized audio extension: {path}"
    return True, "ok"


def _normalize_player_clients(clients: Optional[Sequence[str]]) -> List[Optional[str]]:
    """Convert config client names into yt-dlp extractor values (None = default)."""
    if not clients:
        clients = DEFAULT_PLAYER_CLIENTS
    normalized: List[Optional[str]] = []
    for client in clients:
        if client is None:
            normalized.append(None)
            continue
        name = str(client).strip().lower()
        if not name or name in {"default", "auto", "none"}:
            normalized.append(None)
        else:
            normalized.append(name)
    unique: List[Optional[str]] = []
    for client in normalized:
        if client not in unique:
            unique.append(client)
    return unique or [None]


def _build_extractor_args(
    player_client: Optional[str],
    extra_extractor_args: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Build a yt-dlp ``--extractor-args`` string for youtube (and optional extras)."""
    parts: List[str] = []
    youtube_bits: List[str] = []
    if player_client:
        youtube_bits.append(f"player_client={player_client}")

    if extra_extractor_args:
        for key, value in extra_extractor_args.items():
            if key == "youtube" and isinstance(value, Mapping):
                for sub_key, sub_val in value.items():
                    if sub_key == "player_client" and player_client:
                        continue
                    youtube_bits.append(f"{sub_key}={sub_val}")
            elif key == "youtube" and isinstance(value, str):
                youtube_bits.append(value)
            else:
                parts.append(f"{key}:{value}")

    if youtube_bits:
        parts.insert(0, "youtube:" + ",".join(youtube_bits))
    return ";".join(parts) if parts else None


def _format_cmd(cmd: Sequence[str]) -> str:
    """Shell-ish formatting for logs only (not used for execution)."""
    parts = []
    for part in cmd:
        if re.search(r"[\s\"']", part):
            parts.append("'" + part.replace("'", "'\\''") + "'")
        else:
            parts.append(part)
    return " ".join(parts)


def build_yt_dlp_command(
    url: str,
    output_template: str,
    *,
    audio_format: str = "wav",
    player_client: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    cookies_path: Optional[str] = None,
    js_runtime: Optional[JsRuntimeSelection] = None,
    extractor_args: Optional[Mapping[str, Any]] = None,
    remote_components: Optional[Sequence[str]] = None,
    extra_args: Optional[Sequence[str]] = None,
    yt_dlp_bin: Optional[str] = None,
    sample_rate: int = 16000,
    channels: int = 1,
    retries: Optional[int] = None,
) -> List[str]:
    """
    Build a yt-dlp argv list for audio extraction.

    Uses list form (no shell) to avoid quoting bugs and injection risks.
    """
    cmd: List[str] = resolve_yt_dlp_binary(yt_dlp_bin)

    cmd.extend(
        [
            "--no-playlist",
            "--extract-audio",
            "--audio-quality",
            "0",
            "--audio-format",
            audio_format,
            # [Reason] Force mono 16 kHz PCM-friendly audio for the STT splitter
            "--postprocessor-args",
            f"ffmpeg:-ar {sample_rate} -ac {channels}",
            "-f",
            "bestaudio/best",
            "-o",
            output_template,
            "--newline",
        ]
    )

    if retries is not None and retries >= 0:
        cmd.extend(["--retries", str(retries)])

    extractor = _build_extractor_args(player_client, extractor_args)
    if extractor:
        cmd.extend(["--extractor-args", extractor])

    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", str(cookies_from_browser)])
    elif cookies_path:
        cmd.extend(["--cookies", str(cookies_path)])

    if js_runtime and js_runtime.supported:
        # [Reason] Clear deno-only defaults when we intentionally select another runtime
        if js_runtime.name != "deno":
            cmd.append("--no-js-runtimes")
        cmd.extend(["--js-runtimes", js_runtime.runtime_arg])

    if remote_components:
        for component in remote_components:
            cmd.extend(["--remote-components", str(component)])

    if extra_args:
        cmd.extend(str(arg) for arg in extra_args)

    cmd.append(url)
    return cmd


def _run_yt_dlp(
    cmd: Sequence[str],
    *,
    timeout_seconds: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """Execute yt-dlp with the current process environment (inherits PATH)."""
    return subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        check=False,
        env=os.environ.copy(),
    )


def _log_attempt(
    *,
    video_id: Optional[str],
    url: str,
    attempt: int,
    player_client: Optional[str],
    exit_code: Optional[int],
    duration_seconds: float,
    filepath: Optional[str],
    failure_reason: Optional[str],
    cookies_from_browser: Optional[str] = None,
) -> None:
    """Emit a structured per-attempt log block."""
    client_label = player_client or "default"
    print("-" * 40)
    print(f"Video ID:        {video_id or 'unknown'}")
    print(f"URL:             {url}")
    print(f"Attempt number:  {attempt}")
    print(f"Player client:   {client_label}")
    print(f"Cookies browser: {cookies_from_browser or '(none)'}")
    print(f"Exit code:       {exit_code if exit_code is not None else 'n/a'}")
    print(f"Download duration: {duration_seconds:.1f}s")
    print(f"Downloaded file: {filepath or '(none)'}")
    if failure_reason:
        print(f"Failure reason:  {failure_reason}")
    print("-" * 40)


def _resolve_remote_components(yt_cfg: Mapping[str, Any]) -> List[str]:
    """Default to ejs:github unless explicitly disabled."""
    if yt_cfg.get("remote_components") is False or yt_cfg.get("enable_remote_components") is False:
        return []
    raw = yt_cfg.get("remote_components")
    if raw is None:
        return ["ejs:github"]
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw]


def _build_attempt_plan(
    clients: Sequence[Optional[str]],
    *,
    cookies_from_browser: Optional[str],
    cookies_path: Optional[str],
    cookie_fallbacks: Sequence[str],
    enable_cookie_fallback: bool,
) -> List[Tuple[Optional[str], Optional[str], Optional[str]]]:
    """
    Build ``(player_client, cookies_from_browser, cookies_path)`` attempts.

    Phase 1 uses configured cookies (if any). Phase 2 retries a short client
    list with browser cookies when the first phase had no cookie auth.
    """
    plan: List[Tuple[Optional[str], Optional[str], Optional[str]]] = []
    for client in clients:
        plan.append((client, cookies_from_browser, cookies_path))

    already_using_cookies = bool(cookies_from_browser or cookies_path)
    if enable_cookie_fallback and not already_using_cookies:
        # [Reason] Keep cookie retries bounded: default+android cover most auth cases
        cookie_clients: List[Optional[str]] = []
        for client in clients:
            label = client or "default"
            if label in {"default", "android"} and client not in cookie_clients:
                cookie_clients.append(client)
        if not cookie_clients:
            cookie_clients = [None, "android"]

        for browser in cookie_fallbacks:
            browser_name = str(browser).strip()
            if not browser_name:
                continue
            for client in cookie_clients:
                plan.append((client, browser_name, None))
    return plan


def collect_environment_diagnostics(
    *,
    yt_cfg: Mapping[str, Any],
    js_runtime: Optional[JsRuntimeSelection],
    player_client: Optional[str],
    cookies_from_browser: Optional[str],
    cookies_path: Optional[str],
    remote_components: Sequence[str],
    extractor_args_str: Optional[str],
    command: Optional[Sequence[str]],
) -> EnvironmentDiagnostics:
    """Build a diagnostics snapshot for failure reporting."""
    cli_ver, mod_ver, exe = get_yt_dlp_versions(yt_cfg.get("binary"))
    which_node = shutil.which("node")
    which_deno = shutil.which("deno")
    node_detail = which_node
    if which_node:
        ver = _node_version(which_node)
        node_detail = f"{which_node} (v{ver})" if ver else which_node
    return EnvironmentDiagnostics(
        yt_dlp_cli_version=cli_ver,
        yt_dlp_module_version=mod_ver,
        yt_dlp_executable=exe,
        detected_node=node_detail,
        detected_deno=which_deno,
        selected_js_runtime=js_runtime.describe() if js_runtime else None,
        cookies_enabled=bool(cookies_from_browser or cookies_path),
        cookies_from_browser=cookies_from_browser,
        cookies_path=cookies_path,
        remote_components=",".join(remote_components) if remote_components else None,
        player_client=player_client or "default",
        extractor_args=extractor_args_str,
        command=list(command) if command else None,
    )


def download_audio(
    url: str,
    output_dir: Union[str, Path],
    file_stem: str,
    config: Optional[ConfigLike] = None,
    *,
    audio_format: str = "wav",
    output_template: Optional[str] = None,
) -> DownloadResult:
    """
    Download audio for a single URL with optional client / cookie fallback retries.

    Returns a :class:`DownloadResult` and does **not** raise for normal yt-dlp
    failures. Success requires a non-empty audio file written during the attempt.
    """
    url = (url or "").strip()
    video_id = extract_video_id(url)
    started = time.monotonic()

    if not url:
        result = DownloadResult(
            success=False,
            reason="Empty or missing URL.",
            url=url,
            video_id=video_id,
            duration_seconds=0.0,
        )
        print("✗ Download failed")
        print("Reason:")
        print(result.reason)
        return result

    yt_cfg = get_yt_dlp_config(config)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template = output_template or yt_cfg.get("output_template") or str(out_dir / f"{file_stem}.%(ext)s")
    clients = _normalize_player_clients(yt_cfg.get("player_clients") or yt_cfg.get("player_client_fallback"))
    preferred = yt_cfg.get("player_client")
    if preferred and not yt_cfg.get("player_clients") and not yt_cfg.get("player_client_fallback"):
        preferred_norm = str(preferred).strip().lower()
        preferred_value = None if preferred_norm in {"default", "auto", "none"} else preferred_norm
        clients = [preferred_value] + [c for c in clients if c != preferred_value]

    cookies_from_browser = yt_cfg.get("cookies_from_browser")
    cookies_path = yt_cfg.get("cookies_path")
    if cookies_path and not Path(str(cookies_path)).is_file():
        result = DownloadResult(
            success=False,
            reason=f"Cookies file not found: {cookies_path}",
            url=url,
            video_id=video_id,
            duration_seconds=time.monotonic() - started,
        )
        print("✗ Download failed")
        print("Reason:")
        print(result.reason)
        return result

    js_runtime = detect_js_runtime(
        yt_cfg.get("js_runtime"),
        prefer=str(yt_cfg.get("js_runtime_prefer", "auto")),
    )
    if js_runtime and not js_runtime.supported:
        print(
            f"⚠ Detected JS runtime is unsupported for yt-dlp: {js_runtime.describe()}. "
            f"Need Node >= {'.'.join(map(str, MIN_NODE_VERSION))} or Deno. "
            "Will continue, but YouTube extraction will likely fail."
        )
        # [Reason] Do not pass an unsupported runtime; keep searching behavior honest
        js_runtime = None
    elif js_runtime:
        print(f"Using JS runtime: {js_runtime.describe()}")
    else:
        print(
            "⚠ No supported JS runtime detected (Deno, or Node >= 22). "
            "YouTube downloads will likely fail until one is installed."
        )

    remote_components = _resolve_remote_components(yt_cfg)
    timeout_seconds = yt_cfg.get("timeout_seconds")
    extra_args = yt_cfg.get("extra_args") or []
    extractor_args = yt_cfg.get("extractor_args")
    yt_dlp_bin = yt_cfg.get("binary")
    retries = yt_cfg.get("retries")
    enable_cookie_fallback = yt_cfg.get("cookie_fallback", True)
    cookie_fallbacks = yt_cfg.get("cookie_browser_fallbacks") or DEFAULT_COOKIE_BROWSER_FALLBACKS

    attempt_plan = _build_attempt_plan(
        clients,
        cookies_from_browser=cookies_from_browser,
        cookies_path=cookies_path,
        cookie_fallbacks=cookie_fallbacks,
        enable_cookie_fallback=bool(enable_cookie_fallback),
    )

    attempt_logs: List[Dict[str, Any]] = []
    last_reason = "Download failed."
    last_exit: Optional[int] = None
    last_client: Optional[str] = None
    last_cmd: Optional[List[str]] = None
    last_output = ""
    last_cookies_browser: Optional[str] = None
    last_cookies_path: Optional[str] = None
    diagnostics_printed = False

    for attempt_index, (player_client, attempt_cookies_browser, attempt_cookies_path) in enumerate(
        attempt_plan, start=1
    ):
        # [Reason] Remove stale zero-byte / partial files before each attempt
        for stale in out_dir.glob(f"{file_stem}.*"):
            if stale.suffix.lower() in AUDIO_EXTENSIONS:
                try:
                    if not stale.is_file() or stale.stat().st_size == 0:
                        stale.unlink(missing_ok=True)
                except OSError:
                    pass

        cmd = build_yt_dlp_command(
            url,
            template,
            audio_format=audio_format,
            player_client=player_client,
            cookies_from_browser=attempt_cookies_browser,
            cookies_path=attempt_cookies_path,
            js_runtime=js_runtime,
            extractor_args=extractor_args,
            remote_components=remote_components,
            extra_args=extra_args,
            yt_dlp_bin=yt_dlp_bin,
            retries=retries,
        )
        last_cmd = cmd
        last_cookies_browser = attempt_cookies_browser
        last_cookies_path = attempt_cookies_path

        attempt_started = time.monotonic()
        # [Reason] Ignore pre-existing audio from earlier runs when validating success
        mtime_floor = attempt_started - 1.0
        exit_code: Optional[int] = None
        combined_output = ""
        try:
            completed = _run_yt_dlp(cmd, timeout_seconds=timeout_seconds)
            exit_code = completed.returncode
            combined_output = "\n".join(
                part for part in (completed.stdout, completed.stderr) if part
            )
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            combined_output = (exc.stdout or "") + "\n" + (exc.stderr or "") + "\nDownload timed out."
        except FileNotFoundError:
            exit_code = 127
            combined_output = "yt-dlp executable not found. Install yt-dlp or set YT_DLP.binary."
        except OSError as exc:
            exit_code = 1
            combined_output = str(exc)

        attempt_duration = time.monotonic() - attempt_started
        downloaded = find_downloaded_audio(
            out_dir,
            file_stem,
            expected_ext=audio_format,
            newer_than=mtime_floor,
        )
        last_exit = exit_code
        last_client = player_client
        last_output = combined_output

        if exit_code == 0 and downloaded is not None:
            ok, reason = validate_downloaded_file(downloaded, expected_ext=audio_format)
            if ok:
                _log_attempt(
                    video_id=video_id,
                    url=url,
                    attempt=attempt_index,
                    player_client=player_client,
                    exit_code=exit_code,
                    duration_seconds=attempt_duration,
                    filepath=str(downloaded),
                    failure_reason=None,
                    cookies_from_browser=attempt_cookies_browser,
                )
                attempt_logs.append(
                    {
                        "attempt": attempt_index,
                        "player_client": player_client or "default",
                        "cookies_from_browser": attempt_cookies_browser,
                        "exit_code": exit_code,
                        "duration_seconds": attempt_duration,
                        "filepath": str(downloaded),
                        "success": True,
                        "command": cmd,
                    }
                )
                print("✓ Download succeeded")
                return DownloadResult(
                    success=True,
                    filepath=str(downloaded),
                    reason="ok",
                    attempts=attempt_index,
                    player_client=player_client or "default",
                    video_id=video_id,
                    url=url,
                    exit_code=exit_code,
                    duration_seconds=time.monotonic() - started,
                    attempt_logs=attempt_logs,
                )
            last_reason = reason
        elif exit_code == 0 and downloaded is None:
            last_reason = (
                "yt-dlp exited successfully but no new non-empty audio file was found "
                "(ignored any stale files from previous runs)."
            )
        else:
            last_reason = classify_download_error(combined_output, exit_code)

        _log_attempt(
            video_id=video_id,
            url=url,
            attempt=attempt_index,
            player_client=player_client,
            exit_code=exit_code,
            duration_seconds=attempt_duration,
            filepath=str(downloaded) if downloaded else None,
            failure_reason=last_reason,
            cookies_from_browser=attempt_cookies_browser,
        )
        attempt_logs.append(
            {
                "attempt": attempt_index,
                "player_client": player_client or "default",
                "cookies_from_browser": attempt_cookies_browser,
                "exit_code": exit_code,
                "duration_seconds": attempt_duration,
                "filepath": str(downloaded) if downloaded else None,
                "success": False,
                "reason": last_reason,
                "command": cmd,
            }
        )
        if combined_output.strip():
            tail = "\n".join(combined_output.strip().splitlines()[-20:])
            print(f"[yt-dlp output tail]\n{tail}")

        if not diagnostics_printed:
            # [Reason] Print full env diagnostics once after the first failure
            collect_environment_diagnostics(
                yt_cfg=yt_cfg,
                js_runtime=js_runtime,
                player_client=player_client,
                cookies_from_browser=attempt_cookies_browser,
                cookies_path=attempt_cookies_path,
                remote_components=remote_components,
                extractor_args_str=_build_extractor_args(player_client, extractor_args),
                command=cmd,
            ).print_block()
            diagnostics_printed = True

    print("✗ Download failed")
    print("Reason:")
    print(last_reason)
    if last_cmd:
        collect_environment_diagnostics(
            yt_cfg=yt_cfg,
            js_runtime=js_runtime,
            player_client=last_client,
            cookies_from_browser=last_cookies_browser,
            cookies_path=last_cookies_path,
            remote_components=remote_components,
            extractor_args_str=_build_extractor_args(last_client, extractor_args),
            command=last_cmd,
        ).print_block()
        if last_output.strip():
            print("[full yt-dlp stderr/stdout tail]")
            print("\n".join(last_output.strip().splitlines()[-40:]))

    return DownloadResult(
        success=False,
        filepath=None,
        reason=last_reason,
        attempts=len(attempt_plan),
        player_client=last_client or "default",
        video_id=video_id,
        url=url,
        exit_code=last_exit,
        duration_seconds=time.monotonic() - started,
        attempt_logs=attempt_logs,
    )


def directory_has_verified_audio(
    audio_dir: Union[str, Path],
    *,
    prefix: Optional[str] = None,
    expected_ext: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Return whether ``audio_dir`` contains at least one valid audio file to split.

    Used as a gate before ``split_audio_files()``.
    """
    path = Path(audio_dir)
    if not path.is_dir():
        return False, f"Download directory does not exist: {path}"

    matches: Iterable[Path]
    if expected_ext:
        ext = expected_ext if expected_ext.startswith(".") else f".{expected_ext}"
        matches = path.glob(f"*{ext}")
    else:
        matches = (p for p in path.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)

    found = False
    for file_path in matches:
        if not file_path.is_file():
            continue
        if prefix and not file_path.name.startswith(prefix):
            continue
        ok, _ = validate_downloaded_file(file_path, expected_ext=expected_ext)
        if ok:
            found = True
            break

    if not found:
        return False, f"No verified audio files found in {path}"
    return True, "ok"
