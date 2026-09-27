"""JobPilot for everyone: sign in with Google, pick a period, watch your job-search board fill in.

Run locally:   streamlit run web_app.py
Deploy:        see DEPLOY.md (Streamlit Community Cloud, free)
The single-user tools (demo.py, streamlit_app.py) keep working unchanged.
"""
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="JobPilot", page_icon="🧭", layout="wide")

# settings: Streamlit secrets (hosted) -> environment variables, so the app modules can read them
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass  # no secrets file: fall back to .env / environment

from app import accounts, auth, db, worker  # noqa: E402  (after secrets are in the environment)
from app.date_range import PRESETS, describe, gmail_query, resolve  # noqa: E402
from app.llm import llm_label  # noqa: E402
from app.naming import unfilled_placeholders  # noqa: E402
from app.pipeline import set_draft_status  # noqa: E402

LABEL_ICON = {"REJECTED": "❌", "INTERVIEW_REQUEST": "📅", "INTERESTED": "🙂", "NEEDS_INFO": "📎",
              "AUTO_REPLY": "🤖", "UNCLEAR": "❓"}


@st.cache_resource
def _start_worker():
    return worker.start()


_start_worker()
if "central" not in st.session_state:
    st.session_state.central = accounts.connect()
central = st.session_state.central


def allowed(email: str) -> bool:
    allow = [e.strip().lower() for e in os.getenv("ALLOWED_EMAILS", "").split(",") if e.strip()]
    return not allow or email.lower() in allow


# ------------------------------------------------------------------ coming back from Google
qp = st.query_params
if "code" in qp and "user" not in st.session_state:
    try:
        uid, email = auth.finish_sign_in(central, qp.get("code"), qp.get("state"))
        if not allowed(email):
            accounts.delete_user(central, uid)
            st.query_params.clear()
            st.error(f"{email} isn't on this app's tester list yet. Ask the owner to add you.")
            st.stop()
        st.session_state.user = {"id": uid, "email": email}
        st.query_params.clear()
        st.rerun()
    except Exception as exc:
        st.query_params.clear()
        st.error(f"Sign-in didn't work: {exc}")
elif "error" in qp:
    st.query_params.clear()
    st.warning("Sign-in was cancelled on Google's screen. Nothing was shared.")

# ------------------------------------------------------------------ signed out: landing page
if "user" not in st.session_state:
    st.title("🧭 JobPilot")
    st.subheader("Your job search, read and organised from your own Gmail.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Finds your applications**  \nLinkedIn, Indeed, Lever, Greenhouse, Ashby, direct emails, "
                "even ones you've forgotten.")
    c2.markdown("**Reads every reply**  \nRejected, interview request, needs info, auto-reply… with the "
                "next step for each, and scam warnings.")
    c3.markdown("**Drafts, never sends**  \nReplies and follow-ups are saved as Gmail drafts for you to "
                "review. Nothing is ever sent for you.")
    st.divider()
    if not auth.configured():
        st.error("Google sign-in isn't set up on this server yet (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, APP_URL). "
                 "See DEPLOY.md.")
        st.stop()
    if "signin_url" not in st.session_state:  # one sign-in link per visit (each holds a one-time security code)
        st.session_state.signin_url = auth.sign_in_url(central)
    st.link_button("Sign in with Google", st.session_state.signin_url, type="primary")
    st.caption("You'll be asked to allow two things: read your Gmail, and create drafts. JobPilot is in testing: "
               "only invited Google accounts can sign in. Your data stays in your own space on this server, and "
               "'Delete my data' removes it.")
    st.stop()

# ------------------------------------------------------------------ signed in
user = st.session_state.user
uid = user["id"]
if st.session_state.get("udb_for") != uid:  # the user's own database, opened once per visit
    st.session_state.udb, st.session_state.udb_for = accounts.user_db(uid), uid
udb = st.session_state.udb

with st.sidebar:
    st.title("🧭 JobPilot")
    st.write(f"Signed in as **{user['email']}**")
    name = st.text_input("Your name (signs your drafts)", value=db.get_setting(udb, "name", "") or "",
                         placeholder="e.g. Huzaifa")
    if name != (db.get_setting(udb, "name", "") or ""):
        db.set_setting(udb, "name", name.strip())
    st.caption(f"AI: `{llm_label()}`")
    st.divider()
    if st.button("Sign out"):
        for k in ("user", "udb", "udb_for", "signin_url"):
            st.session_state.pop(k, None)
        st.rerun()
    with st.popover("Delete my data"):
        st.write("Removes your board, drafts, scan history and the Gmail access this app stored. "
                 "(You can also remove access at myaccount.google.com → Security → Third-party apps.)")
        if st.button("Yes, delete everything", type="primary"):
            job = accounts.latest_job(central, uid)
            if job:
                accounts.cancel_job(central, job["id"])
            udb.close()
            accounts.delete_user(central, uid)
            for k in ("user", "udb", "udb_for", "signin_url"):
                st.session_state.pop(k, None)
            st.rerun()

tab_scan, tab_board, tab_drafts, tab_scam = st.tabs(["1 · Scan my inbox", "2 · My board", "3 · Drafts", "4 · Scam check"])

# ------------------------------------------------------------------ 1. scan
with tab_scan:
    job = accounts.latest_job(central, uid)
    active = bool(job and job["status"] in ("queued", "counting", "running"))

    keys = list(PRESETS) + ["custom"]
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Period to scan", keys, index=keys.index("1m"), disabled=active,
                          format_func=lambda k: PRESETS[k][0] if k in PRESETS else "Custom range (max 1 year)")
    include_today = c2.checkbox("Include today", value=True, disabled=active or choice in ("today", "custom"))
    window, label, problem = None, "", None
    try:
        if choice == "custom":
            from datetime import date, timedelta
            picked = st.date_input("From – to", value=(date.today() - timedelta(days=29), date.today()),
                                   max_value=date.today(), disabled=active)
            if not (isinstance(picked, (tuple, list)) and len(picked) == 2):
                raise ValueError("Pick both a start and an end date.")
            frm, to = resolve(start=picked[0], end=picked[1])
            label = f"Custom: {describe(frm, to)}"
        else:
            frm, to = resolve(choice, include_today=include_today)
            label = f"{PRESETS[choice][0]}: {describe(frm, to)}"
        window = gmail_query(frm, to)
        st.caption(f"Will scan {describe(frm, to)}. Emails already checked in earlier scans are skipped, "
                   "so re-running is quick.")
    except ValueError as exc:
        problem = str(exc)
        st.error(problem)

    if st.button("Start scan", type="primary", disabled=active or bool(problem)):
        worker.enqueue(central, uid, label, window)
        st.rerun()

    @st.fragment(run_every=2 if active else None)
    def live_progress():
        j = accounts.latest_job(central, uid)
        if not j:
            st.info("No scans yet. Pick a period and press Start scan. Tip: start with 'Last month'.")
            return
        p = accounts.progress(j)
        st.markdown(f"**{j['label']}**")
        if j["status"] in ("queued", "counting", "running"):
            st.progress(p["percent"] / 100, text=p["text"])
            if p.get("refining"):
                st.progress(p["refine_percent"] / 100, text=p["refine_text"])
                st.caption("Your board is already filled in. The AI is only re-checking the few emails the rules "
                           "weren't sure about; rows update as it goes.")
            if st.button("Stop", key="stop"):
                accounts.cancel_job(central, j["id"])
                st.rerun()
            st.caption("You can leave this page: the scan continues on the server. Your board fills in as it goes.")
        elif j["status"] == "done":
            st.progress(1.0, text=p["text"])
            st.success(j["message"] or "Done.")
        elif j["status"] == "cancelled":
            st.progress(p["percent"] / 100, text=p["text"])
            st.warning(j["message"])
        else:
            st.error(j["message"] or "The scan stopped.")
        if j["status"] not in ("queued", "counting", "running") and active:
            st.rerun()  # scan just finished: refresh the whole page so the buttons unlock

    live_progress()

# ------------------------------------------------------------------ 2. board
def render_board():
    from app.followups import create_drafts, days_since, refresh
    db.repair_statuses(udb)
    refresh(udb)
    rows = db.pipeline(udb)
    if not rows:
        st.info("Your board is empty. Run a scan in tab 1.")
    else:
        for r in rows:
            age = days_since(r.get("last_contact_at") or r.get("applied_at"))
            r["your_last_email"] = "—" if age is None else f"{age}d ago"
            r["last_label"] = r["last_label"] or "no reply yet"
        # ---- which period and which kind of outreach to show (hides rows, never deletes them)
        from datetime import date as _date, timedelta as _td
        from app.board_filters import custom_window, in_window, kind_matches
        p1, p2, p3 = st.columns([2, 1, 1.4])
        periods = ["all"] + list(PRESETS) + ["custom"]
        bperiod = p1.selectbox("Period", periods, key="board_period",
                               format_func=lambda k: "All time" if k == "all" else
                               PRESETS[k][0].replace(" (max)", "") if k in PRESETS else "Custom range")
        b_today = p2.checkbox("Include today", value=True, key="board_today",
                              disabled=bperiod in ("all", "today", "custom"))
        basis_label = p3.radio("Date means", ["Last activity", "When I applied"], horizontal=True, key="board_basis",
                               help="Last activity = your latest email or their latest reply, whichever is newer.")
        basis = "applied" if basis_label == "When I applied" else "activity"
        frm = to = None
        if bperiod == "custom":
            picked_dates = st.date_input("From – to", value=(_date.today() - _td(days=29), _date.today()),
                                         max_value=_date.today(), key="board_dates")
            if isinstance(picked_dates, (tuple, list)) and len(picked_dates) == 2:
                frm, to = custom_window(picked_dates[0], picked_dates[1])
        elif bperiod != "all":
            frm, to = resolve(bperiod, include_today=b_today)
        kinds = {"all": "All types", "application": "Applications", "cold": "Cold emails"}
        bkind = st.segmented_control("Type", list(kinds), default="all", key="board_kind",
                                     format_func=lambda k: kinds[k]) or "all"
        all_df = pd.DataFrame(rows)  # "Fix a row" can reach every row, whatever the filters show
        rows = [r for r in rows if in_window(r, frm, to, basis) and kind_matches(r, bkind)]
        if frm is not None:
            st.caption(f"{len(rows)} row(s) with {'activity' if basis == 'activity' else 'an application'} in "
                       f"{describe(frm, to)}" + ("" if bkind == "all" else f" · {kinds[bkind].lower()} only"))
        if not rows:
            st.info("Nothing in this period / type. Choose a longer period or 'All time'.")
            return

        df = pd.DataFrame(rows)
        # clickable counters: pick one and the table shows just those rows (no scrolling to find them)
        groups = {
            "All": df.index == df.index,
            "Needs you": df.priority == "high",
            "Interviews": df.status.isin(["INTERVIEWING", "INTERVIEWED", "OFFER"]),
            "Waiting on them": df.status == "WAITING_ON_THEM",
            "Rejected": df.status == "REJECTED",
            "No reply yet": df.status.isin(["APPLIED", "NO_RESPONSE"]),
        }
        labels = {name: f"{name}  {int(mask.sum())}" for name, mask in groups.items()}
        picked = st.segmented_control("Show", list(groups), default="All", key="board_filter",
                                      format_func=lambda n: labels[n], label_visibility="collapsed")
        picked = picked or "All"  # clicking the selected one again = show everything
        shown = df[groups[picked]]
        if picked != "All":
            st.caption(f"Showing {len(shown)} of {len(df)}: {picked.lower()}. Click **All** to see all.")
        view = shown[["id", "company", "role", "outreach_kind", "status", "last_label", "your_last_email",
                      "priority", "next_action"]].rename(columns={
                          "id": "#", "outreach_kind": "type", "last_label": "last reply", "your_last_email": "your last email"})
        view = view.assign(type=view["type"].map(lambda k: "cold email" if k == "cold" else "application"))
        if view.empty:
            st.info(f"Nothing under '{picked}' right now.")
        elif st.radio("View", ["Full text", "Interactive (sort, search, click a row)"], horizontal=True,
                      key="board_view", label_visibility="collapsed") == "Full text":
            # a plain table wraps long text, so the whole next step is readable without clicking
            plain = view.rename(columns={"#": "row", "next_action": "next step"}).reset_index(drop=True)
            # the plain table reads markdown: "$60–120k" would turn into a math formula, "-" into a bullet
            plain = plain.map(lambda v: v.replace("$", "\\$") if isinstance(v, str) else v)
            st.table(plain)
        else:
            st.caption("Click a row to read its full next step and details.")
            event = st.dataframe(
                view, width="stretch", hide_index=True, key="board_table",
                on_select="rerun", selection_mode="single-row",
                column_config={
                    "#": st.column_config.NumberColumn(width="small"),
                    "company": st.column_config.TextColumn(width="medium", pinned=True),
                    "role": st.column_config.TextColumn(width="medium"),
                    "type": st.column_config.TextColumn(width="small"),
                    "priority": st.column_config.TextColumn(width="small"),
                    "next_action": st.column_config.TextColumn("next step", width="large"),
                })
            sel = (event.selection.rows if event and hasattr(event, "selection") else []) or []
            if sel:
                r = shown.iloc[sel[0]]
                with st.container(border=True):
                    st.markdown(f"**#{int(r['id'])} · {r['company']}** · {r['role']}")
                    st.markdown(f"**Next step:** {r['next_action'] or '—'}")
                    d1, d2, d3, d4 = st.columns(4)
                    d1.markdown(f"**Status**  \n{r['status']}")
                    d2.markdown(f"**Last reply**  \n{r['last_label']}")
                    d3.markdown(f"**Your last email**  \n{r['your_last_email']}")
                    d4.markdown(f"**Contact**  \n{r.get('contact_email') or '—'}")
                    if r.get("subject"):
                        st.caption(f"Subject: {r['subject']}")
        with st.expander("Fix a row"):
            ids = [int(i) for i in all_df.id]
            rid = st.selectbox("Row #", ids, format_func=lambda i: f"#{i} · {all_df[all_df.id == i].company.iloc[0]}")
            f1, f2, f3 = st.columns(3)
            new_status = f1.selectbox("Set status", ["", "INTERVIEWING", "INTERVIEWED", "OFFER", "REJECTED", "CLOSED"])
            new_kind = f2.selectbox("Type", ["", "application", "cold"],
                                    format_func=lambda k: {"": "Choose an option", "cold": "cold email"}.get(k, k))
            new_name = f3.text_input("Company name")
            others = [i for i in ids if i != rid]
            merge_into = st.selectbox("Same application as another row? Merge it into:", [None] + others,
                                      format_func=lambda i: "—" if i is None else
                                      f"#{i} · {all_df[all_df.id == i].company.iloc[0]} · {all_df[all_df.id == i].role.iloc[0]}")
            if st.button("Save"):
                if new_status:
                    udb.execute("UPDATE applications SET status=?, priority='normal', next_action=? WHERE id=?",
                                (new_status, f"Set by you: {new_status.lower()}.", rid))
                if new_kind:
                    udb.execute("UPDATE applications SET outreach_kind=?, kind_locked=1 WHERE id=?", (new_kind, rid))
                if new_name.strip():
                    udb.execute("UPDATE applications SET company=? WHERE id=?", (new_name.strip(), rid))
                if merge_into:
                    db.merge_applications(udb, rid, merge_into)
                udb.commit()
                st.rerun()
            if st.checkbox(f"Delete row #{rid} (not a real application)"):
                if st.button("Delete it", type="primary"):
                    for t in ("drafts", "messages", "events"):
                        udb.execute(f"DELETE FROM {t} WHERE application_id=?", (rid,))
                    udb.execute("DELETE FROM applications WHERE id=?", (rid,))
                    udb.commit()
                    st.rerun()
        if st.button("Draft follow-ups for applications that are due"):
            with st.spinner("Drafting..."):
                made = create_drafts(udb)
            st.success(f"{len(made)} follow-up draft(s) added to tab 3.")


with tab_board:
    render_board()

# ------------------------------------------------------------------ 3. drafts
with tab_drafts:
    drafts = db.pending_drafts(udb)
    if not drafts:
        st.caption("Nothing waiting. Drafts appear here when a reply needs an answer, or when you draft follow-ups.")
    for d in drafts:
        with st.container(border=True):
            tag = "⏰ FOLLOW-UP" if d.get("kind") == "followup" else f"{LABEL_ICON.get(d['label'], '')} {d['label']}"
            st.markdown(f"**{d['company']}** · {d['role']} · {tag}")
            st.caption(f"To {d['to_addr']} — {d['subject']}")
            body = st.text_area("Body", d["body"], key=f"body{d['id']}", height=170)
            holes = unfilled_placeholders(body)
            if holes:
                st.warning("Fill these in before approving: " + ", ".join(holes))
            b1, b2, _ = st.columns([1, 1, 4])
            if b1.button("Save to Gmail drafts", key=f"ok{d['id']}", disabled=bool(holes)):
                from app.gmail_client import create_gmail_draft
                gid = create_gmail_draft(auth.gmail_service(central, uid), d["to_addr"], d["subject"], body,
                                         d.get("thread_id"))
                set_draft_status(udb, d["id"], "APPROVED", body=body, gmail_draft_id=gid)
                st.success("Saved in your Gmail Drafts. Open Gmail to send it.")
                st.rerun()
            if b2.button("Discard", key=f"no{d['id']}"):
                set_draft_status(udb, d["id"], "DISCARDED")
                st.rerun()

# ------------------------------------------------------------------ 4. scam check
with tab_scam:
    from app.scam_detector import analyze_posting
    text = st.text_area("Paste a job posting or a recruiter message", height=260)
    s1, s2 = st.columns(2)
    website = s1.text_input("Company website (optional)")
    contact = s2.text_input("Contact email (optional)")
    if st.button("Check it", type="primary", disabled=not text.strip()):
        with st.spinner("Checking..."):
            r = analyze_posting(text, "", website, contact)
        verdict = {"likely_scam": ("🚩 Likely scam", st.error), "suspicious": ("⚠️ Suspicious", st.warning),
                   "looks_legit": ("✅ Looks legit", st.success)}[r.verdict]
        verdict[1](f"**{verdict[0]}** · risk {r.risk_score}/100\n\n{r.summary}")
        for f in r.flags:
            st.markdown(f"- **{f.title}** ({f.severity}): {f.why}  \n  > {f.evidence}")