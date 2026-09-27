"""'Sign in with Google' for the web app (OAuth 2.0 web flow with PKCE).

Needs a Google Cloud OAuth client of type **Web application** (not the Desktop one the CLI uses), with the
app's address as an authorised redirect URI. Settings (env vars or Streamlit secrets):
  GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, APP_URL (e.g. https://jobpilot.streamlit.app or http://localhost:8501)

Scopes: read Gmail + create drafts. The app never sends email.
"""
from __future__ import annotations

import json
import os

from . import accounts
from .gmail_client import SCOPES

os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")  # Google may return scopes in a different order


def configured() -> bool:
    return bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET") and os.getenv("APP_URL"))


def _redirect_uri() -> str:
    return os.getenv("APP_URL", "").rstrip("/") + "/"


def _flow(verifier: str):
    from google_auth_oauthlib.flow import Flow
    cfg = {"web": {"client_id": os.getenv("GOOGLE_CLIENT_ID"), "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                   "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                   "token_uri": "https://oauth2.googleapis.com/token", "redirect_uris": [_redirect_uri()]}}
    return Flow.from_client_config(cfg, scopes=SCOPES, redirect_uri=_redirect_uri(), code_verifier=verifier,
                                   autogenerate_code_verifier=False)


def sign_in_url(conn) -> str:
    state, verifier = accounts.new_oauth_state(conn)
    url, _ = _flow(verifier).authorization_url(access_type="offline", prompt="consent", state=state,
                                               include_granted_scopes="true")
    return url


def finish_sign_in(conn, code: str, state: str) -> tuple[int, str]:
    """Called when Google sends the user back with ?code=...&state=... -> (user_id, email)."""
    verifier = accounts.pop_oauth_state(conn, state)
    if not verifier:
        raise PermissionError("This sign-in link expired or wasn't started here. Please sign in again.")
    flow = _flow(verifier)
    flow.fetch_token(code=code)
    creds = flow.credentials
    granted = set(creds.granted_scopes or SCOPES)
    if not set(SCOPES) <= granted:
        raise PermissionError("JobPilot needs both boxes ticked on Google's screen (read Gmail + create drafts).")
    email = gmail_service_from_creds(creds).users().getProfile(userId="me").execute()["emailAddress"]
    uid = accounts.upsert_user(conn, email, creds.to_json())
    return uid, email


def gmail_service_from_creds(creds):
    from googleapiclient.discovery import build
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def gmail_service(conn, user_id: int):
    """A Gmail client for one user, refreshing (and re-saving) the access token when it has expired."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    raw = accounts.user_token(conn, user_id)
    if not raw:
        raise PermissionError("Not connected to Gmail. Please sign in again.")
    creds = Credentials.from_authorized_user_info(json.loads(raw), SCOPES)
    if not creds.valid and creds.refresh_token:
        creds.refresh(Request())
        accounts.save_token(conn, user_id, creds.to_json())
    return gmail_service_from_creds(creds)