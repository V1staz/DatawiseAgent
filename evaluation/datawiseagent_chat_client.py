"""Small HTTP client used by evaluation harness runners."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import httpx

BASE_URL = "http://127.0.0.1:8000"


def _json_or_raise(response: httpx.Response, request_name: str):
    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(
            f"{request_name} failed: status={response.status_code}, "
            f"body={response.text[:200]!r}"
        ) from exc


def create_user(username: str):
    response = httpx.post(f"{BASE_URL}/create_user/", json={"username": username}, timeout=None, trust_env=False)
    return _json_or_raise(response, "create_user")["user_id"]


def create_session(
    user_id: str,
    session_name: str,
    tool_mode: Literal["default", "dsbench", "datamodeling"] = "default",
    reset_llm_config: Optional[dict] = None,
    reset_code_executor: Optional[dict] = None,
):
    payload = {"tool_mode": tool_mode}
    if reset_llm_config is not None:
        payload["reset_llm_config"] = reset_llm_config
    if reset_code_executor is not None:
        payload["reset_code_executor"] = reset_code_executor
    response = httpx.post(
        f"{BASE_URL}/users/{user_id}/sessions/{session_name}",
        json=payload,
        timeout=None,
        trust_env=False,
    )
    return _json_or_raise(response, "create_session")["session_id"]


def upload_file(
    user_id: str,
    session_id: str,
    file_path: str,
    filename_to_save: Optional[str] = None,
):
    filename = filename_to_save or Path(file_path).name
    with open(file_path, "rb") as f:
        response = httpx.post(
            f"{BASE_URL}/upload/",
            data={"user_id": user_id, "session_id": session_id},
            files={"files": (filename, f)},
            timeout=None,
            trust_env=False,
        )
    return _json_or_raise(response, "upload_file")


def chat(
    user_id: str,
    session_id: str,
    query: str,
    work_mode: Literal["jupyter", "jupyter+script"] = "jupyter",
    timeout_seconds: int = 3600,
):
    response = httpx.post(
        f"{BASE_URL}/chat/",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
            "work_mode": work_mode,
        },
        timeout=timeout_seconds,
        trust_env=False,
    )
    return _json_or_raise(response, "chat")


def stop_session(user_id: str, session_id: str):
    response = httpx.delete(f"{BASE_URL}/users/{user_id}/sessions/{session_id}", timeout=120, trust_env=False)
    return _json_or_raise(response, "stop_session")
