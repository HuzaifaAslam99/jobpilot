# Putting JobPilot online (free, up to 100 testers)

The web version is `web_app.py`: people sign in with Google, pick a period, and get their own board.
The single-user tools (`demo.py`, `streamlit_app.py`) are unchanged and keep working.

Total time: about 30 minutes. Cost: free.

---

## 1. Try it on your own computer first

### 1a. Create a "Web application" sign-in client
The CLI's `credentials.json` is a *Desktop* client. The website needs a separate *Web* client in the
**same** Google Cloud project, which already has the Gmail API enabled.

1. Go to console.cloud.google.com, open your project, then **APIs & Services → Credentials**.
2. Click **Create credentials → OAuth client ID**.
3. Set **Application type** to **Web application** and the name to "JobPilot web".
4. Under **Authorized redirect URIs**, add `http://localhost:8501/`. Keep the slash at the end.
5. Click **Create**, then copy the **Client ID** and **Client secret**.

### 1b. Add the settings to `.env`
```
GOOGLE_CLIENT_ID=...apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-...
APP_URL=http://localhost:8501
```
Locally you can leave `TOKEN_KEY` empty. The app then creates one in `data/.token_key`.

### 1c. Run it
```
pip install -r requirements.txt
streamlit run web_app.py
```
Sign in with a Google account that is on your **Test users** list (see step 3).

---

## 2. Put it online (Streamlit Community Cloud)

1. **Push the project to GitHub.** A private repository is fine. First check that `.gitignore` keeps these
   out of the repository: `.env`, `credentials.json`, `token.json`, `data/`, `.streamlit/secrets.toml`.
   Run `git status` before your first commit and make sure none of them are listed.
2. Go to **share.streamlit.io**, sign in with GitHub, and click **Create app**.
3. Choose your repository and branch, set **Main file** to `web_app.py`, and pick an app URL, for example
   `jobpilot-huzaifa`.
4. Open **Advanced settings → Secrets** and paste the contents of `.streamlit/secrets.toml.example`
   with your real values filled in:
   - `APP_URL`: your app's address, e.g. `https://jobpilot-huzaifa.streamlit.app`
   - `TOKEN_KEY`: make one with
     `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
     Keep it somewhere safe. If you change it, everyone has to sign in again.
5. Click **Deploy**.
6. In Google Cloud, open the Web client again and add your app's address as a second redirect URI, e.g.
   `https://jobpilot-huzaifa.streamlit.app/`. Keep the slash at the end.

---

## 3. Invite testers (up to 100)

1. In Google Cloud, go to **Google Auth Platform → Audience**. On some consoles this is
   **OAuth consent screen**.
2. Leave **Publishing status** set to **Testing**.
3. Under **Test users**, click **Add users** and enter your testers' Gmail addresses (the CTO, friends).
   Only these accounts can sign in.
4. Testers will see "Google hasn't verified this app". That's normal in Testing mode: they click
   **Continue** and tick both boxes (read Gmail, create drafts).

For an extra lock inside the app, you can also set `ALLOWED_EMAILS` in secrets.

---

## Good to know (limits of the free setup)

- **One scanner for everyone.** It gives each user a batch of emails in turn, so a small scan is never
  stuck behind someone's year-long backlog.
- **AI limits are shared** across all users (your Groq and Gemini keys). Several things keep usage low:
  rules come first, portal templates are cached so they're read once for everyone, and unclear emails
  go to the AI 6 at a time. When every model is at its limit, emails wait for a later scan. Nothing is
  guessed wrongly.
- **Community Cloud storage is not permanent.** When the app restarts (you push new code, or it wakes
  from sleep after a long quiet period), `data/` can be wiped. Users then sign in again and rescan, and
  the template cache refills. Scans that were running continue from where they stopped while the app
  stays up. For permanent storage later, move the databases to a host with a persistent disk, or to a
  hosted database.
- **Going public** (more than 100 users, no "unverified" warning) needs Google's verification for Gmail
  access: a privacy policy, a domain, and a yearly security assessment. Only do that once people find it
  useful.
- **Privacy.** Each user's data lives in their own database file. Google tokens are stored encrypted.
  "Delete my data" in the sidebar removes everything for that user. The app never sends email; it only
  creates drafts.