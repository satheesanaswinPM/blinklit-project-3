# Theme Explorer (optional / advanced)

React + Vite UI backed by `discovery_engine.api` (FastAPI).

**This is not the primary product surface.** For category-discovery research demos and stakeholder review, use the Streamlit console from the repo root:

```bash
python -m streamlit run app.py
```

## When to use this

- Deep BERTopic / SQLite Theme Explorer browsing
- API integration experiments against `/themes`, `/stats`, etc.

## Run

```bash
# from repo root — API
uvicorn discovery_engine.api:app --reload

# from frontend/
npm install
npm run dev
```

Requires a prior `python -m scripts.run_pipeline` (or equivalent) so the SQLite insight store is populated.
