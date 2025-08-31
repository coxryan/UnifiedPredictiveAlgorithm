# UPA-F

Turn-key predictions and live edge report with a React dashboard, static portal, and GitHub Pages deploy.

## What's here
- `app/` — React + Vite + Tailwind dashboard
- `data/` — CSVs used by the dashboard and portal
- `portal/` — lightweight static page linking to CSVs
- `.github/workflows/deploy.yml` — CI that builds and deploys to Pages

## Quick start
```bash
git init
git add .
git commit -m "UPA-F: initial portal + dashboard"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```
Then in GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

The app will be served at: `https://<your-username>.github.io/<repo-name>/`

---
Generated 2025-08-31 21:45 UTC
