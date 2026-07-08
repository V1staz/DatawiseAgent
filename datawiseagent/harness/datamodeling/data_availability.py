"""Label-blind public-data availability checks for DataModeling tasks.

Some benchmark copies contain Git LFS pointer text instead of real CSV bytes.
Those files are syntactically readable by ``pandas.read_csv`` as a one-column
table, which can make downstream routing misclassify a missing tabular task as
text modeling.  This module performs a cheap pre-modeling guard using only the
public train/test/sample paths from the task contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"
_PUBLIC_FILE_ATTRS = (
    ("train", "train_path"),
    ("test", "test_path"),
    ("sample_submission", "sample_submission_path"),
)


@dataclass(slots=True)
class PublicFileStatus:
    role: str
    path: str
    exists: bool
    bytes: int
    is_git_lfs_pointer: bool = False
    first_line: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DataAvailabilityReport:
    files: list[PublicFileStatus]
    lfs_pointer_files: list[str]
    missing_files: list[str]
    errors: list[str]
    usable_for_modeling: bool

    @property
    def train_available(self) -> bool:
        return _role_available(self.files, "train")

    @property
    def test_available(self) -> bool:
        return _role_available(self.files, "test")

    @property
    def sample_available(self) -> bool:
        return _role_available(self.files, "sample_submission")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "train_available": self.train_available,
                "test_available": self.test_available,
                "sample_available": self.sample_available,
            }
        )
        return payload

    def prompt_summary(self) -> str:
        if self.usable_for_modeling:
            return "public_data_status=usable; train/test/sample_submission are present and not Git LFS pointers."
        problems: list[str] = []
        if self.lfs_pointer_files:
            problems.append(f"git_lfs_pointers={self.lfs_pointer_files}")
        if self.missing_files:
            problems.append(f"missing_files={self.missing_files}")
        if self.errors:
            problems.append(f"errors={self.errors[:3]}")
        return "public_data_status=unusable_or_incomplete; " + "; ".join(problems)


def inspect_data_availability(contract: Any) -> DataAvailabilityReport:
    """Inspect public file availability without using hidden answer data."""

    statuses: list[PublicFileStatus] = []
    for role, attr in _PUBLIC_FILE_ATTRS:
        statuses.append(_inspect_public_file(role, Path(getattr(contract, attr))))

    lfs_pointer_files = [status.role for status in statuses if status.is_git_lfs_pointer]
    missing_files = [status.role for status in statuses if not status.exists]
    errors = [f"{status.role}:{status.error}" for status in statuses if status.error]
    usable = not lfs_pointer_files and not missing_files and not errors
    return DataAvailabilityReport(
        files=statuses,
        lfs_pointer_files=lfs_pointer_files,
        missing_files=missing_files,
        errors=errors,
        usable_for_modeling=usable,
    )


def _inspect_public_file(role: str, path: Path) -> PublicFileStatus:
    if not path.exists():
        return PublicFileStatus(role=role, path=str(path), exists=False, bytes=0)
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            first_line_bytes = f.readline(4096)
        first_line = first_line_bytes.decode("utf-8", errors="replace") if first_line_bytes else ""
        return PublicFileStatus(
            role=role,
            path=str(path),
            exists=True,
            bytes=size,
            is_git_lfs_pointer=first_line.strip() == _LFS_POINTER_PREFIX,
            first_line=first_line[:160],
        )
    except Exception as exc:  # pragma: no cover - defensive for filesystem edge cases.
        return PublicFileStatus(
            role=role,
            path=str(path),
            exists=True,
            bytes=0,
            error=f"{type(exc).__name__}:{str(exc)[:160]}",
        )


def _role_available(files: list[PublicFileStatus], role: str) -> bool:
    for status in files:
        if status.role == role:
            return status.exists and not status.is_git_lfs_pointer and not status.error
    return False
