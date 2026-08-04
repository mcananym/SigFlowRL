"""Pure protocol helpers for reproducible chronological experiments."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_RESEARCH_SEEDS = (42, 123, 456, 789, 2026)


@dataclass(frozen=True)
class RollingOriginWindow:
    window_id: int
    train_end: date
    validation_end: date
    calibration_end: date
    test_start: date
    test_end: date


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def make_rolling_origin_windows(
    dates: Iterable[object],
    *,
    minimum_train_dates: int,
    validation_dates: int,
    calibration_dates: int,
    test_dates: int,
    step_dates: int | None = None,
    horizon_dates: int = 1,
) -> tuple[RollingOriginWindow, ...]:
    """Build expanding, purged rolling-origin windows on unique dates.

    A horizon-sized embargo separates adjacent sections, preventing a target
    that begins in one section from ending in the next. Every later window adds
    only past dates to its training history.
    """

    ordered = sorted({_as_date(value) for value in dates})
    if horizon_dates < 1:
        raise ValueError("horizon_dates must be positive")
    for name, value in (
        ("minimum_train_dates", minimum_train_dates),
        ("validation_dates", validation_dates),
        ("calibration_dates", calibration_dates),
        ("test_dates", test_dates),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    step = step_dates or test_dates
    if step < 1:
        raise ValueError("step_dates must be positive")

    embargo = horizon_dates - 1
    first_test_start = (
        minimum_train_dates
        + embargo
        + validation_dates
        + embargo
        + calibration_dates
        + embargo
    )
    windows: list[RollingOriginWindow] = []
    test_start_index = first_test_start
    window_id = 1
    while test_start_index + test_dates <= len(ordered):
        calibration_end_index = test_start_index - embargo - 1
        calibration_start_index = calibration_end_index - calibration_dates + 1
        validation_end_index = calibration_start_index - embargo - 1
        validation_start_index = validation_end_index - validation_dates + 1
        train_end_index = validation_start_index - embargo - 1
        if train_end_index + 1 < minimum_train_dates:
            raise AssertionError("Rolling window violates minimum training history")
        windows.append(RollingOriginWindow(
            window_id=window_id,
            train_end=ordered[train_end_index],
            validation_end=ordered[validation_end_index],
            calibration_end=ordered[calibration_end_index],
            test_start=ordered[test_start_index],
            test_end=ordered[test_start_index + test_dates - 1],
        ))
        window_id += 1
        test_start_index += step
    if not windows:
        required = first_test_start + test_dates
        raise ValueError(
            f"Need at least {required} unique dates for one rolling window; "
            f"received {len(ordered)}."
        )
    return tuple(windows)


def _git_commit(worktree: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_dirty(worktree: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=worktree,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def collect_reproducibility_manifest(
    worktree: Path,
    package_names: Sequence[str] = (
        "numpy", "pandas", "matplotlib", "torch", "scipy", "psutil"
    ),
) -> dict[str, object]:
    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_commit": _git_commit(worktree),
        "git_worktree_dirty": _git_dirty(worktree),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "research_seeds": list(DEFAULT_RESEARCH_SEEDS),
    }


def serialise_windows(
    windows: Sequence[RollingOriginWindow],
) -> list[dict[str, object]]:
    return [
        {key: value.isoformat() if isinstance(value, date) else value
         for key, value in asdict(window).items()}
        for window in windows
    ]
