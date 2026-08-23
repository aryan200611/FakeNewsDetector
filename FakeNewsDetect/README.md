# STYLEPRINT

A fake-news **style** detector — an end-to-end ML project spanning data
cleaning, model training, a Flask API, and a connected web frontend.

STYLEPRINT classifies news text as stylistically closer to "real" (wire-style
journalism) or "fake" (hyperpartisan/clickbait) writing, and shows *why* —
surfacing the specific words and phrases that pushed the verdict each way.

**This project is as much about rigorous evaluation as it is about the model
itself.** The README below documents real data-leakage issues found and
fixed during development, and an honest accuracy number that's notably lower
than the naive one — see [Methodology & Honest Evaluation](#methodology--honest-evaluation).

## Architecture

```
┌─────────────────┐      POST /predict       ┌──────────────────────┐
│  frontend/       │ ───────────────────────▶ │  backend/            │
│  styleprint.html │                          │  Flask API           │
│  (static, no     │ ◀─────────────────────── │  serving a 20,000-   │
│  build step)     │   verdict + confidence   │  feature scikit-learn│
└─────────────────┘   + contributing terms    │  logistic regression │
                                               │  model               │
                                               └──────────────────────┘
                                                          ▲
                                                          │ trained by
                                                          │
                                               ┌──────────────────────┐
                                               │  notebook/            │
                                               │  training pipeline,   │
                                               │  leakage analysis,    │
                                               │  cross-subject eval   │
                                               └──────────────────────┘
```

- **`notebook/`** — the full training pipeline: data cleaning, TF-IDF +
  logistic regression / SGD / Naive Bayes comparison, and two evaluation
  passes (a standard random split, and a stricter cross-subject holdout).
- **`backend/`** — a Flask API that loads the trained model and serves
  predictions. Model weights never ship to the client.
- **`frontend/`** — a single static HTML file with no build step. Calls the
  backend's `/predict` endpoint and renders the verdict, a confidence gauge,
  and the top contributing terms.

## Demo

1. Start the backend (see `backend/README.md` for full setup):
   ```bash
   cd backend
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   python app.py
   ```
2. Open `frontend/styleprint.html` directly in a browser.
3. Paste an article and click **Analyze Signal**.

## Deployment

This project is set up to deploy fully for free. Two pieces, deployed
separately:

### 1. Deploy the backend (Render)

Full instructions are in [`backend/README.md`](backend/README.md#deploying-to-render-free-tier).
Short version: push to GitHub → connect the repo on [render.com](https://render.com)
→ Render auto-detects `backend/render.yaml` → you get a live URL like
`https://styleprint-backend-xxxx.onrender.com`.

### 2. Deploy the frontend (GitHub Pages)

1. In `frontend/styleprint.html`, find this line near the top of the
   `<script>` block:
   ```js
   : 'https://fakenewsdetector-2-lf31.onrender.com'; 
   ```
   Replace it with your actual Render URL from step 1.
2. Commit and push that change.
3. On GitHub, go to your repo's **Settings → Pages**.
4. Under **Source**, select the branch (e.g. `main`) and set the folder to
   `/frontend` if that option is available, or use GitHub's Pages settings
   to serve from a `docs/` folder / root, adjusting your repo layout as
   needed (GitHub Pages only serves one directory per site — see note
   below).
5. Save. GitHub gives you a live URL like
   `https://yourusername.github.io/styleprint/`.
6. Visit it, paste some text, and confirm it calls your live backend.

**Note on GitHub Pages folder limitation:** GitHub Pages serves from either
the repo root or a `/docs` folder, not an arbitrary subfolder like
`/frontend`. The simplest fix: copy `frontend/styleprint.html` to
`docs/index.html` at the repo root, then point Pages at the `/docs` folder.
Alternatively, deploy the frontend to Netlify or Vercel instead, which both
let you set `frontend` as the publish directory directly with no
restructuring — drag-and-drop the `frontend` folder onto
[app.netlify.com/drop](https://app.netlify.com/drop) for the fastest path.

### 3. Lock down CORS (recommended once both are live)

In Render's dashboard, add an environment variable to the backend service:
`ALLOWED_ORIGIN` = your GitHub Pages / Netlify URL exactly (e.g.
`https://yourusername.github.io`). This stops other websites from calling
your API. Redeploy for it to take effect.

### What I could and couldn't do for you

I don't have the ability to create accounts or push code on your behalf —
GitHub, Render, and Netlify all require your own login. What's already done
for you: `render.yaml`, `Procfile`, and `Dockerfile` are written and the
backend has been tested to run correctly with the `$PORT` environment
variable Render injects; the frontend auto-detects localhost vs. production
so only one placeholder line needs updating. I wasn't able to test the
exact `gunicorn` production command in my sandbox (no internet access to
install it there), but it's the standard, widely-used invocation and
matches what Render/Railway expect by default.

## Methodology & Honest Evaluation

This dataset (Kaggle's Fake and Real News Dataset, ~45k articles) is known
to contain several data leaks that make naive accuracy numbers misleading.
This project explicitly finds and addresses them rather than reporting an
inflated score:

| Leak found | Evidence | Fix |
|---|---|---|
| `subject`/`date` columns perfectly separate the classes | Every REAL article tagged `politicsNews`/`worldnews`; every FAKE article tagged something else | Dropped as model features |
| Reuters wire dateline (`WASHINGTON (Reuters) -`) | Present in ~99% of REAL articles, ~0% of FAKE | Stripped via regex |
| "Reuters" mentioned elsewhere in body text | e.g. "told Reuters" | Stripped |
| Photo credit boilerplate (`Photo by X/Getty Images`) | ~37% of FAKE articles, 0% of REAL | Stripped |
| `pic.twitter.com` embed links | ~15% of FAKE articles, 0% of REAL | Stripped |

**Two accuracy numbers are reported, deliberately:**
- **~98%** — standard random train/test split. Inflated: train and test sets
  share the same subject categories, so the model partly succeeds by
  recognizing subject-specific phrasing it's already seen.
- **~93%** — cross-subject holdout. The model is trained on some article
  subjects (e.g. `politicsNews`, `News`, `politics`) and evaluated on
  **entirely unseen** subjects (`worldnews`, `left-news`, `Middle-east`,
  etc.). This is the more honest measure of generalization, and the one to
  trust.

**What the model actually detects:** writing *style* — formal wire-service
register vs. casual/opinionated register — not factual accuracy. It has no
access to external facts or sources. A true claim written in a casual tone
can be flagged FAKE; a false claim written in neutral wire-style prose can
pass as REAL. The frontend surfaces this directly with a deliberately-chosen
edge-case example that demonstrates the failure mode, rather than only
showing cases where the model looks good.

## Tech Stack

- **ML**: Python, scikit-learn (TF-IDF + Logistic Regression), pandas
- **Backend**: Flask, joblib
- **Frontend**: vanilla HTML/CSS/JS (no framework, no build step)
- **Data**: [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

## Project Structure

```
.
├── notebook/
│   └── fake_news_detector.ipynb   # training, leakage analysis, evaluation
├── backend/
│   ├── app.py                     # Flask API
│   ├── train_model.py             # reproducible training script
│   ├── model.joblib                # pre-trained model artifact
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── Procfile                   # for Render/Railway/Heroku-style deploys
│   ├── render.yaml                # Render Blueprint (auto-config)
│   └── README.md                  # backend-specific setup + deploy docs
├── frontend/
│   └── styleprint.html            # static web UI
├── LICENSE
└── README.md                      # you are here
```

## Limitations

See the in-app disclaimer and the notebook's evaluation sections for the
full discussion. In short: this is a style-pattern classifier trained on a
narrow, dated (2016–2017 US political news) dataset from a small number of
sources. It is a reasonable triage/flagging tool and a demonstration of ML
engineering practices (leakage detection, honest evaluation, full-stack
integration) — not a production fact-checking system.

## License

MIT — see [LICENSE](LICENSE).
