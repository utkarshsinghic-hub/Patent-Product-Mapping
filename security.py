"""Authentication, rate limiting, quotas, audit log and upload guardrails.

Everything here runs on the server. The browser only ever receives rendered widgets.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

import ui
from pwhash import verify_password

log = logging.getLogger("p2p")
DATA_DIR = Path(os.environ.get("APP_DATA_DIR", Path(__file__).parent / ".data"))
AUDIT_FILE, QUOTA_FILE = DATA_DIR / "audit.jsonl", DATA_DIR / "quota.json"
_LOCK = threading.Lock()

MAX_FAILS, LOCK_SECONDS, IDLE_SECONDS = 5, 300, 30 * 60


# --------------------------------------------------------------------------- config (secrets or env, never the UI)
def get_secret(name: str, default: Any = "") -> Any:
    v = os.environ.get(name)
    if v not in (None, ""):
        return v
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def _auth_config():
    """users: {name: pbkdf2 hash}; admins: set(names). From st.secrets['auth'] or env AUTH_USERS_JSON / ADMIN_USERS."""
    users, admins = {}, set()
    try:
        a = st.secrets.get("auth", {})
        users = dict(a.get("users", {}))
        admins = set(a.get("admins", []))
    except Exception:
        pass
    if os.environ.get("AUTH_USERS_JSON"):
        try:
            users.update(json.loads(os.environ["AUTH_USERS_JSON"]))
        except Exception:
            log.error("AUTH_USERS_JSON is not valid JSON")
    if os.environ.get("ADMIN_USERS"):
        admins |= {x.strip() for x in os.environ["ADMIN_USERS"].split(",") if x.strip()}
    return users, admins


def get_limits() -> Dict[str, int]:
    lim = {"max_upload_mb": 25, "max_patents": 3000, "max_limitations": 60000, "max_claim_chars": 60000,
           "daily_review_calls": 1000, "calls_per_run": 400}
    try:
        for k, v in dict(st.secrets.get("limits", {})).items():
            if k in lim:
                lim[k] = int(v)
    except Exception:
        pass
    return lim


# --------------------------------------------------------------------------- audit log (metadata only, never content)
def audit(event: str, **fields):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, "session": st.session_state.get("sid", "-"),
           "user": st.session_state.get("user", {}).get("name", fields.pop("user", "-"))}
    rec.update({k: (str(v)[:120] if not isinstance(v, (int, float, bool)) else v) for k, v in fields.items()})
    try:
        with _LOCK:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with AUDIT_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    except Exception:
        log.exception("audit write failed")


def safe_error(user_message: str, exc: Optional[BaseException] = None) -> str:
    """Show the client a generic message with a reference id; the detail stays in the server log."""
    eid = uuid.uuid4().hex[:8].upper()
    log.error("error_id=%s user=%s", eid, st.session_state.get("user", {}).get("name", "-"), exc_info=exc)
    audit("error", error_id=eid, kind=type(exc).__name__ if exc else "-")
    return f"{user_message} (reference {eid})"


# --------------------------------------------------------------------------- login
@st.cache_resource
def _attempts() -> Dict[str, Dict[str, float]]:
    return {}


def _locked(key: str) -> int:
    rec = _attempts().get(key)
    return int(rec["until"] - time.time()) if rec and rec.get("until", 0) > time.time() else 0


def _fail(key: str):
    with _LOCK:
        rec = _attempts().setdefault(key, {"n": 0, "until": 0})
        rec["n"] += 1
        if rec["n"] >= MAX_FAILS:
            rec["until"], rec["n"] = time.time() + LOCK_SECONDS, 0


def login_gate() -> Dict[str, Any]:
    users, admins = _auth_config()
    st.session_state.setdefault("sid", uuid.uuid4().hex[:10])
    u = st.session_state.get("user")
    if u:
        if time.time() - st.session_state.get("last_active", 0) > IDLE_SECONDS:
            audit("session_timeout")
            st.session_state.clear()
            st.session_state["notice"] = "You were signed out after a period of inactivity."
            st.rerun()
        st.session_state["last_active"] = time.time()
        return u

    if not users:                                           # fail closed
        ui.login_card(None)
        st.error("The service is not configured. Please contact your administrator.")
        st.stop()

    notice = st.session_state.pop("notice", None)
    ui.login_card(notice)
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        with st.form("login", clear_on_submit=False):
            name = st.text_input("Username", autocomplete="username")
            pw = st.text_input("Password", type="password", autocomplete="current-password")
            go = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if go:
            key = (name or "").strip().lower()
            wait = max(_locked(key), _locked("session:" + st.session_state["sid"]))
            if wait:
                st.error(f"Too many attempts. Try again in {wait // 60 + 1} minute(s).")
            else:
                stored = users.get(name.strip()) if name else None
                ok = verify_password(pw, stored or "pbkdf2_sha256$310000$00$00")   # constant work for unknown users
                if stored and ok:
                    st.session_state["user"] = {"name": name.strip(), "role": "admin" if name.strip() in admins else "client"}
                    st.session_state["last_active"] = time.time()
                    audit("login_ok")
                    st.rerun()
                _fail(key)
                _fail("session:" + st.session_state["sid"])
                audit("login_failed", user=key[:40])
                time.sleep(1.0)
                st.error("Invalid username or password.")
    st.stop()


def logout():
    audit("logout")
    st.session_state.clear()
    st.rerun()


# --------------------------------------------------------------------------- daily review quota
def _quota_load() -> Dict[str, Any]:
    try:
        return json.loads(QUOTA_FILE.read_text())
    except Exception:
        return {}


def remaining_calls(user: Dict[str, Any], limits) -> Optional[int]:
    if user["role"] == "admin":
        return None
    used = _quota_load().get(date.today().isoformat(), {}).get(user["name"], 0)
    return max(0, limits["daily_review_calls"] - used)


def consume_calls(user: Dict[str, Any], n: int):
    if user["role"] == "admin" or n <= 0:
        return
    with _LOCK:
        q = _quota_load()
        today = date.today().isoformat()
        q = {today: q.get(today, {})}                       # drops older days
        q[today][user["name"]] = q[today].get(user["name"], 0) + n
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        QUOTA_FILE.write_text(json.dumps(q))


# --------------------------------------------------------------------------- upload guardrails
def validate_upload(name: str, data: bytes, limits) -> Optional[str]:
    ext = Path(name).suffix.lower()
    if ext not in (".xlsx", ".csv"):
        return "Only .xlsx and .csv files are accepted."
    if len(data) > limits["max_upload_mb"] * 1024 * 1024:
        return f"File is larger than {limits['max_upload_mb']} MB."
    if ext == ".xlsx":
        try:
            with zipfile.ZipFile(BytesIO(data)) as z:
                names = z.namelist()
                if any("vbaProject" in n for n in names):
                    return "Workbooks containing macros are not accepted."
                if sum(i.file_size for i in z.infolist()) > 400 * 1024 * 1024:
                    return "The workbook expands to an unreasonable size."
        except zipfile.BadZipFile:
            return "The file is not a valid .xlsx workbook."
    return None


INJECTION_RE = re.compile(r"(ignore (all |any )?(previous|prior|above) (instructions|rules)|disregard (the )?(system|above)|"
                          r"you are now|reveal (the |your )?(system )?prompt)", re.I)


def injection_hits(texts) -> int:
    return sum(1 for t in texts if INJECTION_RE.search(str(t)))
