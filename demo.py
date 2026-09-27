"""JobPilot CLI.

  python demo.py seed                 # load sample outreach records into the pipeline
  python demo.py scam                 # run the scam detector on the sample postings
  python demo.py scam --file post.txt # ...or on your own posting
  python demo.py replies              # classify sample replies and update the pipeline
  python demo.py replies --gmail      # ...or read real replies from Gmail
  python demo.py import-sent --query "in:sent subject:internship"   # seed from your real sent mail
  python demo.py board                # show the pipeline
  python demo.py reset                # wipe the local DB
"""
import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from app import db
from app.llm import llm_label
from app.pipeline import process_email
from app.reply_classifier import Email
from app.scam_detector import analyze_posting

DATA = Path(__file__).parent / "data"
console = Console()
VERDICT_STYLE = {"likely_scam": "bold red", "suspicious": "bold yellow", "looks_legit": "bold green"}
LABEL_STYLE = {"REJECTED": "red", "INTERVIEW_REQUEST": "bold green", "INTERESTED": "green",
               "NEEDS_INFO": "cyan", "AUTO_REPLY": "dim", "UNCLEAR": "yellow"}


def seed(conn):
    postings = {p["id"]: p for p in json.loads((DATA / "sample_postings.json").read_text())}
    for o in json.loads((DATA / "sample_outreach.json").read_text()):
        if conn.execute("SELECT 1 FROM applications WHERE thread_id=?", (o["thread_id"],)).fetchone():
            continue
        score = None
        if o.get("posting_id") in postings:
            p = postings[o["posting_id"]]
            score = analyze_posting(p["text"], p["company"], p["website"], p["contact_email"], use_llm=False).risk_score
        db.add_application(conn, o["company"], o["role"], o["contact_email"], o["thread_id"], scam_score=score)
    console.print(f"[green]Seeded {conn.execute('SELECT COUNT(*) FROM applications').fetchone()[0]} applications.[/]")


def show_report(title, r):
    color = VERDICT_STYLE[r.verdict]
    body = f"[{color}]{r.verdict.replace('_', ' ').upper()}[/]  risk {r.risk_score}/100  ({r.mode})\n\n{escape(r.summary)}\n"
    for f in sorted(r.flags, key=lambda f: ["critical", "high", "medium", "low"].index(f.severity)):
        body += f"\n[bold]• \\[{f.severity}] {escape(f.title)}[/] ({f.category}, {f.source})\n  [italic]\"{escape(f.evidence)}\"[/]\n  {escape(f.why)}"
    if r.green_flags:
        body += "\n\n[green]Good signs:[/] " + escape(" ".join(r.green_flags))
    console.print(Panel(body, title=title, border_style=color.split()[-1]))


def cmd_scam(args):
    if args.file:
        text = Path(args.file).read_text()
        show_report(args.file, analyze_posting(text, website=args.website or "", contact_email=args.email or ""))
        return
    for p in json.loads((DATA / "sample_postings.json").read_text()):
        show_report(p["title"], analyze_posting(p["text"], p["company"], p["website"], p["contact_email"]))


def show_result(res):
    if res.get("skipped"):
        console.print(f"[dim]skip {res['gmail_id']}: {res['skipped']}[/]")
        return
    style = LABEL_STYLE.get(res["label"], "white")
    age = res.get("age_days")
    age_txt = f"  [dim]({age} days ago)[/]" if age is not None else ""
    new_tag = "[bold magenta]NEW[/] " if res.get("created") else ""
    head = f"{new_tag}[{style}]{res['label']}[/] ({res['confidence']:.2f}, {res['source']})  ←  {res['from']}{age_txt}\n{escape(res['subject'])}"
    body = f"{head}\n[dim]matched by {res['match']} → {res['company'] or 'no application'}[/]\n{escape(res['reasoning'])}"
    if res["safety_flags"]:
        body += f"\n[bold red]⚠ Scam signals in reply: {', '.join(res['safety_flags'])}. No draft created.[/]"
    if res["draft"]:
        body += f"\n\n[bold]Draft (pending your review):[/]\n{escape(res['draft']['body'])}"
    console.print(Panel(body, border_style=style.split()[-1]))


def cmd_replies(args, conn):
    if args.gmail:
        from app.gmail_client import fetch_replies, get_service, sync_thread_dates
        svc = get_service()
        sync_thread_dates(svc, conn)  # real dates + who spoke last, before classifying
        emails = fetch_replies(svc, conn, days=args.days)
        console.print(f"Fetched {len(emails)} messages from Gmail.")
    else:
        emails = [Email(headers=e.pop("headers", None), **e) for e in json.loads((DATA / "sample_replies.json").read_text())]
    from app.inbox_scan import ingest, is_noreply, is_platform
    for e in emails:
        # portal emails (LinkedIn, Lever, Greenhouse...) are matched by company + role, people's replies by thread
        show_result(ingest(conn, e) if (is_platform(e.from_addr) or is_noreply(e.from_addr)) else process_email(conn, e))


def cmd_board(conn):
    from app.followups import days_since, refresh
    refresh(conn)  # keep "no reply" next actions current (no LLM calls)
    t = Table(title="Application pipeline")
    for col in ("#", "Company", "Role", "Type", "Status", "Last reply", "Your last email", "Priority", "Next action"):
        t.add_column(col, overflow="fold")
    for r in db.pipeline(conn):
        pr = {"high": "[bold red]HIGH[/]", "low": "[dim]low[/]"}.get(r["priority"], "normal")
        age = days_since(r.get("last_contact_at") or r.get("applied_at"))
        t.add_row(str(r["id"]), r["company"], r["role"], r.get("outreach_kind") or "-", r["status"], r["last_label"] or "-",
                  "-" if age is None else f"{age}d ago", pr,
                  escape(r["next_action"] or ""))
    console.print(t)
    n = len(db.pending_drafts(conn))
    if n:
        console.print(f"[bold]{n} draft(s) waiting for your review[/] (open the Streamlit app to approve/edit).")


def main():
    ap = argparse.ArgumentParser(description="JobPilot demo CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scam"); s.add_argument("--file"); s.add_argument("--website"); s.add_argument("--email")
    r = sub.add_parser("replies"); r.add_argument("--gmail", action="store_true"); r.add_argument("--days", type=int, default=30)
    i = sub.add_parser("import-sent"); i.add_argument("--query", default="in:sent newer_than:60d subject:(application OR internship OR intern)")
    rn = sub.add_parser("rename", help="fix a company name: rename 19 \"Dexterz Technologies\"")
    rn.add_argument("id", type=int); rn.add_argument("company")
    fu = sub.add_parser("follow-ups"); fu.add_argument("--redo", action="store_true",
                                                      help="discard pending follow-up drafts and write them again")
    fu.add_argument("--include-old", action="store_true", help="also draft for old/cold threads the rules would close")
    si = sub.add_parser("scan-inbox", help="find applications made via LinkedIn / job portals from your inbox")
    si.add_argument("--range", default="1y", choices=["today", "3d", "7d", "1m", "3m", "6m", "9m", "1y"],
                    help="how far back: today | 3d | 7d | 1m | 3m | 6m | 9m | 1y (default, max)")
    si.add_argument("--exclude-today", action="store_true", help="stop at midnight this morning")
    si.add_argument("--from", dest="date_from", help="custom range start, YYYY-MM-DD (max 1 year long)")
    si.add_argument("--to", dest="date_to", help="custom range end, YYYY-MM-DD (default: today)")
    si.add_argument("--days", type=int, help=argparse.SUPPRESS)  # old option, still accepted
    si.add_argument("--max", type=int, default=800)
    si.add_argument("--recheck-ignored", action="store_true", help="look again at emails ignored by earlier scans")
    si.add_argument("--only", help='just the emails matching a Gmail search, e.g. --only "from:hr@menzies-ras.com.pk"')
    rp = sub.add_parser("reprocess", help="reprocess 17  (bring back a discarded draft)"); rp.add_argument("id", type=int)
    ss = sub.add_parser("set-status", help="set-status 26 INTERVIEWED  (record an outcome yourself)")
    ss.add_argument("id", type=int)
    ss.add_argument("status", type=str.upper, choices=["APPLIED", "INTERVIEWING", "INTERVIEWED", "OFFER", "REJECTED", "CLOSED"])
    pk = sub.add_parser("peek", help='peek "subject:\\"Web3 Product Intern at Gate\\""  (debug one email)')
    pk.add_argument("query"); pk.add_argument("-n", type=int, default=1)
    dl = sub.add_parser("delete", help="delete 29  (remove a row that isn't a real application)"); dl.add_argument("id", type=int)
    sk = sub.add_parser("set-kind", help="set-kind 13 cold   |   set-kind 8 application")
    sk.add_argument("id", type=int); sk.add_argument("kind", choices=["cold", "application"])
    for name in ("seed", "board", "reset", "demo", "clear-replies", "clean-roles", "sync-dates"):
        sub.add_parser(name)
    args = ap.parse_args()

    console.print(f"[dim]LLM: {llm_label()}[/]")
    conn = db.connect()
    if args.cmd == "seed":
        seed(conn)
    elif args.cmd == "scam":
        cmd_scam(args)
    elif args.cmd == "replies":
        cmd_replies(args, conn)
    elif args.cmd == "import-sent":
        from app.gmail_client import get_service, import_sent_applications
        console.print(f"Imported {import_sent_applications(get_service(), conn, args.query, db.add_application)} applications.")
    elif args.cmd == "board":
        cmd_board(conn)
    elif args.cmd == "rename":
        conn.execute("UPDATE applications SET company=? WHERE id=?", (args.company, args.id)); conn.commit()
        console.print(f"Application {args.id} renamed to {args.company}.")
    elif args.cmd == "scan-inbox":
        from datetime import date as _date, timedelta
        from app.date_range import describe, gmail_query, resolve
        from app.gmail_client import get_service, scan_inbox
        try:
            if args.days:
                frm, to = resolve(start=_date.today() - timedelta(days=min(args.days, 366) - 1), end=_date.today())
            else:
                frm, to = resolve(args.range, include_today=not args.exclude_today,
                                  start=_date.fromisoformat(args.date_from) if args.date_from else None,
                                  end=_date.fromisoformat(args.date_to) if args.date_to else (
                                      _date.today() if args.date_from else None))
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            return
        console.print(f"Scanning: {describe(frm, to)}")
        results = scan_inbox(get_service(), conn, max_results=args.max, recheck_ignored=args.recheck_ignored,
                             only=args.only, window=gmail_query(frm, to))
        new = [r for r in results if r.get("created")]
        matched = [r for r in results if not r.get("skipped") and not r.get("created")]
        ignored = [r for r in results if r.get("skipped") == "not about your application"]
        deferred = [r for r in results if str(r.get("skipped", "")).startswith("deferred")]
        for r in new + matched:
            show_result(r)
        console.print(f"[bold]{len(results)} inbox emails checked:[/] {len(new)} new application(s) found, "
                      f"{len(matched)} update(s) to existing ones, {len(ignored)} ignored (alerts, newsletters...)"
                      + (f", {len(deferred)} left for next run (AI limit)." if deferred else "."))
        cmd_board(conn)
    elif args.cmd == "reprocess":  # forget one application's processed replies so 'replies --gmail' redoes them
        conn.execute("DELETE FROM drafts WHERE application_id=? AND status='PENDING_REVIEW'", (args.id,))
        conn.execute("DELETE FROM drafts WHERE application_id=? AND status='DISCARDED'", (args.id,))
        conn.execute("DELETE FROM messages WHERE application_id=?", (args.id,))
        conn.execute("UPDATE applications SET status='APPLIED', priority='normal' WHERE id=?", (args.id,))
        conn.commit()
        console.print(f"Application {args.id} reset. Now run both:\n  python demo.py replies --gmail --days 150\n"
                      "  python demo.py scan-inbox   (portal / HR emails that aren't in your own thread)")
    elif args.cmd == "set-status":  # record outcomes the agent can't see (an interview happened, an offer came by phone)
        notes = {"INTERVIEWED": ("normal", "Interviewed. Wait for their decision; a short thank-you note is good practice."),
                 "OFFER": ("high", "Offer received. Review it carefully before accepting."),
                 "REJECTED": ("normal", "Closed."), "CLOSED": ("low", "Closed by you."),
                 "APPLIED": ("normal", "Applied."), "INTERVIEWING": ("normal", "Interview process ongoing.")}
        prio, note = notes[args.status]
        conn.execute("UPDATE applications SET status=?, priority=?, next_action=?, last_update=? WHERE id=?",
                     (args.status, prio, note, db.now(), args.id))
        conn.execute("UPDATE drafts SET status='SUPERSEDED' WHERE application_id=? AND status='PENDING_REVIEW'", (args.id,))
        db.log_event(conn, args.id, "status_set_by_user", {"status": args.status}); conn.commit()
        cmd_board(conn)
    elif args.cmd == "peek":  # debug: exactly what the app reads from an email, and how rules classify it
        from app.gmail_client import _x, get_service, to_email
        from app.inbox_scan import rules_extract
        from app.reply_classifier import classify
        svc = get_service()
        ids = _x(svc.users().messages().list(userId="me", q=args.query, maxResults=args.n)).get("messages", [])
        if not ids:
            console.print("No email matches that Gmail search.")
        for m in ids:
            e = to_email(_x(svc.users().messages().get(userId="me", id=m["id"], format="full")))
            info, cls = rules_extract(e), classify(e, use_llm=False)
            console.print(Panel(
                f"[bold]{escape(e.subject)}[/]\nfrom {escape(e.from_name)} <{e.from_addr}>  ·  {e.received_at[:10]}\n\n"
                f"[dim]what the app reads ({len(e.body)} chars):[/]\n{escape(e.body[:900])}\n\n"
                f"[cyan]scan rules →[/] related={info['related']} company={info['company']!r} role={info['role']!r}\n"
                f"[cyan]reply rules →[/] {cls.label} ({cls.confidence:.2f}) {escape(cls.reasoning)}",
                title=m["id"]))
    elif args.cmd == "delete":  # remove a row that isn't a real application (and its drafts/messages)
        for t in ("drafts", "messages", "events"):
            conn.execute(f"DELETE FROM {t} WHERE application_id=?", (args.id,))
        conn.execute("DELETE FROM applications WHERE id=?", (args.id,)); conn.commit()
        console.print(f"Deleted application {args.id}.")
    elif args.cmd == "set-kind":
        conn.execute("UPDATE applications SET outreach_kind=?, kind_locked=1 WHERE id=?", (args.kind, args.id))
        conn.commit()
        cmd_board(conn)
    elif args.cmd == "sync-dates":  # real sent dates + follow-ups already sent, from Gmail
        from app.gmail_client import get_service, sync_thread_dates
        console.print(f"Synced dates for {sync_thread_dates(get_service(), conn)} applications."); cmd_board(conn)
    elif args.cmd == "follow-ups":  # drafts for applications with no reply
        from app.followups import create_drafts
        made = create_drafts(conn, redo=args.redo, include_old=args.include_old)
        for m in made:
            console.print(Panel(escape(m["body"]), title=f"Follow-up · {m['company']} · applied {m['age']}d ago",
                                border_style="cyan"))
        console.print(f"[bold]{len(made)} follow-up draft(s) created[/], waiting for your review in the Streamlit app.")
    elif args.cmd == "clean-roles":  # tidy role names imported from email subjects
        from app.reply_classifier import clean_role
        for r in conn.execute("SELECT id, role FROM applications").fetchall():
            conn.execute("UPDATE applications SET role=? WHERE id=?", (clean_role(r["role"]), r["id"]))
        conn.commit(); console.print("Role names cleaned.")
    elif args.cmd == "clear-replies":  # keep applications, forget processed replies so they can be re-classified
        db.clear_replies(conn); console.print("Replies cleared. Run 'replies --gmail' again to re-classify.")
    elif args.cmd == "reset":
        db.reset(conn); console.print("DB cleared.")
    elif args.cmd == "demo":  # the whole 5-minute story in one go
        db.reset(conn); seed(conn); cmd_scam(argparse.Namespace(file=None))
        cmd_replies(argparse.Namespace(gmail=False), conn); cmd_board(conn)


if __name__ == "__main__":
    main()