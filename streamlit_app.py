"""JobPilot demo UI:  streamlit run streamlit_app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app import db
from app.llm import llm_label
from app.pipeline import process_email, set_draft_status
from app.reply_classifier import Email
from app.scam_detector import analyze_posting

DATA = Path(__file__).parent / "data"
st.set_page_config(page_title="JobPilot", page_icon="🧭", layout="wide")


@st.cache_resource
def get_conn():
    return db.connect()


conn = get_conn()
postings = json.loads((DATA / "sample_postings.json").read_text())
VERDICT = {"likely_scam": ("🚩 Likely scam", "error"), "suspicious": ("⚠️ Suspicious", "warning"),
           "looks_legit": ("✅ Looks legit", "success")}
SEV_ICON = {"critical": "🟥", "high": "🟧", "medium": "🟨", "low": "⬜"}
LABEL_ICON = {"REJECTED": "❌", "INTERVIEW_REQUEST": "📅", "INTERESTED": "🙂", "NEEDS_INFO": "📎",
              "AUTO_REPLY": "🤖", "UNCLEAR": "❓"}

with st.sidebar:
    st.title("🧭 JobPilot")
    st.caption("Scam screening + reply pipeline for a job search. Nothing is sent without you.")
    st.write(f"**Model:** `{llm_label()}`")
    use_llm = st.toggle("Use LLM layer", value=True, help="Off = deterministic rules only")
    st.divider()
    if st.button("Reset demo data"):
        from demo import seed
        db.reset(conn)
        seed(conn)
        st.success("Pipeline reset and seeded with sample outreach.")

tab_scam, tab_inbox, tab_board = st.tabs(["1 · Scam check", "2 · Process replies", "3 · Pipeline & drafts"])

# ------------------------------------------------------------------ scam check
with tab_scam:
    left, right = st.columns([1, 1.3])
    with left:
        choice = st.selectbox("Sample posting", ["(paste your own)"] + [p["title"] for p in postings])
        p = next((x for x in postings if x["title"] == choice), None)
        text = st.text_area("Posting text", value=p["text"] if p else "", height=300)
        c1, c2 = st.columns(2)
        website = c1.text_input("Company website", value=p["website"] if p else "")
        email = c2.text_input("Contact email", value=p["contact_email"] if p else "")
        run = st.button("Check posting", type="primary", disabled=not text.strip())
    with right:
        if run:
            with st.spinner("Checking..."):
                r = analyze_posting(text, p["company"] if p else "", website, email, use_llm=use_llm)
            label, kind = VERDICT[r.verdict]
            getattr(st, kind)(f"**{label}** · risk {r.risk_score}/100 · _{r.mode}_\n\n{r.summary}")
            for f in sorted(r.flags, key=lambda f: ["critical", "high", "medium", "low"].index(f.severity)):
                with st.expander(f"{SEV_ICON[f.severity]} {f.title}  ·  {f.category}  ·  {f.source}",
                                 expanded=f.severity in ("critical", "high")):
                    st.markdown(f"> {f.evidence}")
                    st.write(f.why)
            if r.green_flags:
                st.caption("Good signs: " + " · ".join(r.green_flags))

# ------------------------------------------------------------------ inbox
with tab_inbox:
    st.write("Reads replies, matches each to an application (thread → sender → domain), classifies it, "
             "checks it for scam signals, updates the pipeline and drafts a response for your review.")
    src = st.radio("Source", ["Sample replies", "My Gmail"], horizontal=True)
    scan = src == "My Gmail" and st.checkbox("Also scan my inbox for applications made via LinkedIn / job portals",
                                             value=True)
    window, window_error = None, None
    if scan:
        from datetime import date, timedelta
        from app.date_range import PRESETS, describe, gmail_query, resolve
        keys = list(PRESETS) + ["custom"]
        c1, c2 = st.columns([2, 1])
        choice = c1.selectbox("Period to scan", keys, index=keys.index("1m"),
                              format_func=lambda k: PRESETS[k][0] if k in PRESETS else "Custom range (max 1 year)")
        include_today = c2.checkbox("Include today", value=True, disabled=choice in ("today", "custom"))
        try:
            if choice == "custom":
                picked = st.date_input("From – to", value=(date.today() - timedelta(days=29), date.today()),
                                       min_value=date.today() - timedelta(days=5 * 366), max_value=date.today())
                if not (isinstance(picked, (tuple, list)) and len(picked) == 2):
                    raise ValueError("Pick both a start and an end date.")
                frm, to = resolve(start=picked[0], end=picked[1])
            else:
                frm, to = resolve(choice, include_today=include_today)
            window = gmail_query(frm, to)
            st.caption(f"Will scan: {describe(frm, to)}. Longer periods take longer the first time "
                       "(free AI limits); emails already checked are skipped on later runs.")
        except ValueError as exc:
            window_error = str(exc)
            st.error(window_error)
    if st.button("Process new replies", type="primary", disabled=bool(window_error)):
        if not db.pipeline(conn):
            from demo import seed
            seed(conn)
        if src == "My Gmail":
            from app.gmail_client import fetch_replies, get_service, sync_thread_dates
            svc = get_service()
            sync_thread_dates(svc, conn)
            emails = fetch_replies(svc, conn, days=150)
            if scan:
                from app.gmail_client import scan_inbox
                with st.spinner("Scanning inbox for LinkedIn / portal applications..."):
                    found = scan_inbox(svc, conn, use_llm=use_llm, window=window)
                n_new = sum(1 for r in found if r.get("created"))
                st.info(f"Inbox scan: {len(found)} emails checked, {n_new} new application(s) added.")
        else:
            emails = [Email(headers=e.pop("headers", None), **e)
                      for e in json.loads((DATA / "sample_replies.json").read_text())]
        from app.inbox_scan import ingest, is_noreply, is_platform
        results = [ingest(conn, e, use_llm=use_llm) if (is_platform(e.from_addr) or is_noreply(e.from_addr))
                   else process_email(conn, e, use_llm=use_llm) for e in emails]
        new = [r for r in results if not r.get("skipped")]
        st.success(f"Processed {len(new)} new, skipped {len(results) - len(new)} already seen.")
        for r in new:
            icon = LABEL_ICON.get(r["label"], "•")
            with st.container(border=True):
                st.markdown(f"**{icon} {r['label']}** ({r['confidence']:.2f}, {r['source']}) · "
                            f"{r['company'] or 'unmatched'} · _matched by {r['match']}_")
                st.caption(f"{r['from']} — {r['subject']}")
                st.write(r["reasoning"])
                if r["safety_flags"]:
                    st.error("Scam signals in this reply: " + ", ".join(r["safety_flags"]) + ". No draft created.")
                elif r["draft"]:
                    st.info("Draft created → review it in tab 3.")

# ------------------------------------------------------------------ board
with tab_board:
    from app.followups import create_drafts, days_since, refresh
    refresh(conn)
    if st.button("Draft follow-ups for applications with no reply"):
        with st.spinner("Drafting..."):
            made = create_drafts(conn, use_llm=use_llm)
        st.success(f"{len(made)} follow-up draft(s) created below.")
    rows = db.pipeline(conn)
    for r in rows:
        age = days_since(r.get("last_contact_at") or r.get("applied_at"))
        r["your_last_email"] = "-" if age is None else f"{age}d ago"
        r["last_label"] = r["last_label"] or "-"
    if rows:
        df = pd.DataFrame(rows)[["company", "role", "outreach_kind", "status", "last_label", "your_last_email", "priority", "next_action"]]
        c = st.columns(4)
        c[0].metric("Applications", len(df))
        c[1].metric("Interviews", int((df.status == "INTERVIEWING").sum()))
        c[2].metric("Needs you", int((df.priority == "high").sum()))
        c[3].metric("Rejected", int((df.status == "REJECTED").sum()))
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("Pipeline is empty. Use 'Reset demo data' in the sidebar, or process replies.")

    st.subheader("Drafts waiting for review")
    drafts = db.pending_drafts(conn)
    if not drafts:
        st.caption("Nothing waiting.")
    for d in drafts:
        with st.container(border=True):
            tag = "⏰ FOLLOW-UP (no reply)" if d.get("kind") == "followup" else f"{LABEL_ICON.get(d['label'], '')} {d['label']}"
            st.markdown(f"**{d['company']}** · {d['role']} · {tag}")
            st.caption(f"To {d['to_addr']} — {d['subject']}")
            body = st.text_area("Body", d["body"], key=f"body{d['id']}", height=170)
            from app.naming import unfilled_placeholders
            holes = unfilled_placeholders(body)
            if holes:
                st.warning("Fill these in before approving: " + ", ".join(holes))
            b1, b2, _ = st.columns([1, 1, 4])
            if b1.button("Approve → Gmail draft", key=f"ok{d['id']}", disabled=bool(holes)):
                gid = None
                try:
                    from app.gmail_client import create_gmail_draft, get_service
                    tid = d.get("thread_id")
                    if not tid and d.get("message_id"):
                        row = conn.execute("SELECT thread_id FROM messages WHERE id=?", (d["message_id"],)).fetchone()
                        tid = row["thread_id"] if row else None
                    gid = create_gmail_draft(get_service(), d["to_addr"], d["subject"], body,
                                             tid if tid and not tid.startswith("demo-") else None)
                    st.success("Saved to your Gmail Drafts. Send it from Gmail.")
                except Exception as exc:
                    st.warning(f"Approved locally (Gmail not connected: {exc.__class__.__name__}).")
                set_draft_status(conn, d["id"], "APPROVED", body=body, gmail_draft_id=gid)
                st.rerun()
            if b2.button("Discard", key=f"no{d['id']}"):
                set_draft_status(conn, d["id"], "DISCARDED")
                st.rerun()

    with st.expander("Agent activity log"):
        ev = conn.execute("SELECT ts, application_id, kind, detail FROM events ORDER BY id DESC LIMIT 50").fetchall()
        st.dataframe(pd.DataFrame([dict(e) for e in ev]), width="stretch", hide_index=True)