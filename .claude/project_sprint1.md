---
name: Sprint 1-2 complete — Sprint 3-4 next
description: Learning platform Azure PoC — full status, working decisions, and what to build next
type: project
---

Sprints 1 and 2 are fully working end-to-end. Dashboard connects to live Fabric SQL.

**Why:** PoC for learning platform modernization on Microsoft Azure — Fabric + Foundry + AI Search.

---

## What is working right now

- `uv run streamlit run ui/app.py` launches the dashboard
- Fabric SQL authenticates via service principal token injection (see auth notes below)
- All 4 KPI cards, funnel, heatmap, radar, scatter, at-risk table render from live gold tables
- Dropout risk scores + intervention suggestions appear in the at-risk table

---

## File map (complete, as of Sprint 2)

```
scripts/
  01_generate_synthetic_data.py   — 200 courses / 500 learners / 107K xAPI events (seed 42)
  02_upload_to_fabric.py          — uploads data/raw/ to OneLake Files/raw/

src/
  config.py                       — load_dotenv + typed env getters (NOTE: dashboard does NOT import this)
  fabric/
    onelake.py                    — OneLakeClient (azure-storage-file-datalake)
    sql_client.py                 — FabricSQLClient (pyodbc + AAD token injection)
  analytics/
    queries.py                    — FabricAnalytics: 11 SQL methods over gold/silver/dim tables
    predictions.py                — DropoutPredictor: local GBM, thread-safe, in-memory cached

notebooks/
  01_xapi_ingestion.ipynb         — Fabric PySpark: bronze → silver → gold (8 tables)
  03_dropout_prediction.ipynb     — Fabric Data Science: GBM + MLflow + model registry

ui/
  app.py                          — st.navigation entry; auto-writes stub pages 2 and 3
  pages/
    1_analytics_dashboard.py      — full Streamlit dashboard
    2_ai_consultant.py            — STUB (Sprint 3)
    3_course_generator.py         — STUB (Sprint 4)
```

---

## Critical: Auth + connection (hard-won, do not change)

**Problem:** `Authentication=ActiveDirectoryServicePrincipal` in the ODBC connection string
returns error 18456 (login failed) with Fabric, even with correct credentials.

**Solution in `src/fabric/sql_client.py`:**
1. Acquire AAD token via `azure-identity.ClientSecretCredential` for scope `https://database.windows.net/.default`
2. Encode as `struct.pack("<I{n}s", len(token_utf16), token_utf16)`
3. Pass to pyodbc via `attrs_before={1256: token_bytes}` (SQL_COPT_SS_ACCESS_TOKEN)
4. Connection string has NO Authentication= key when using token injection

Auth selection in `FabricSQLClient.from_env()`:
- AZURE_TENANT_ID + AZURE_CLIENT_ID + AZURE_CLIENT_SECRET set → token injection (service principal) ← live env uses this
- None of the above, not CI → ActiveDirectoryInteractive
- CI=true → ActiveDirectoryDefault

**`load_dotenv()` lives in `sql_client.py` at module import**, not in `config.py`. This is intentional — the Streamlit process never imports `config.py`, so `.env` must be loaded wherever the SQL client is first imported.

---

## Env vars (all set in .env)

```
FABRIC_SQL_ENDPOINT=eb63sf3asooevb6esz7xdpqp4q-lpzl7cf34eielgcq47csfdcagm.datawarehouse.fabric.microsoft.com
FABRIC_LAKEHOUSE_NAME=xapi_lh
FABRIC_WORKSPACE_ID=88bff25b-e1bb-4510-9850-e7c5228c4033
FABRIC_LAKEHOUSE_ID=65160143-e420-4a45-9f65-62844c5ed14b
AZURE_TENANT_ID=17b97d20-9360-4a9c-87c4-967f71be0fe4
AZURE_CLIENT_ID=faaa4a17-898c-45c3-8077-bb27f1ee8574
AZURE_CLIENT_SECRET=<in .env>
FOUNDRY_PROJECT_ENDPOINT=<needs filling>
MODEL_DEPLOYMENT_NAME=gpt-4o-mini
EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
AZURE_SEARCH_ENDPOINT=https://learning-poc-search.search.windows.net
AZURE_SEARCH_ADMIN_KEY=<needs filling>
AZURE_SEARCH_INDEX_NAME=course-catalog
```

---

## Fabric tables (live in xapi_lh database)

Bronze: `bronze_xapi_events`, `dim_courses`, `dim_learners`
Silver: `silver_xapi_events` (partitioned by course_id)
Gold:   `gold_course_completion`, `gold_module_dropoff`, `gold_department_skills`, `gold_engagement_hourly`

Columns used by queries — see `src/analytics/queries.py` for exact SQL.
Key: `silver_xapi_events` has columns: `learner_email, learner_name, verb, course_id, module_id, department, course_format, score, course_completed (bool), ts (timestamp)`.

---

## Bugs fixed (do not reintroduce)

- `Styler.applymap` → `Styler.map` (pandas 2.1+ rename). Already fixed in dashboard.
- Streamlit `icon=":bar_chart:"` shortcodes not accepted in 1.56 → use literal Unicode `"📊"`.
- `FabricAnalytics.from_env()` no longer takes an `interactive` parameter.

---

## Infra notes

- **uv** manages the venv at `.venv`. All commands: `uv run python ...`, `uv run streamlit run ui/app.py`
- **ODBC Driver 18 for SQL Server** installed via `winget install Microsoft.msodbcsql.18`
- Driver auto-detected in `_detect_driver()` — falls back to Driver 17 if 18 not present
- Python 3.14.2 in the uv venv (uv chose latest)

---

## Sprint 3 — What to build (AI Search + Foundry Agent)

Read `POC_AZURE_NATIVE_PLAN.md` Tasks 3.1–3.5 for full spec. Summary:

### 3.1 `src/search/index_manager.py`
Create Azure AI Search index `course-catalog` with:
- Fields: course_id (key), title, description, department, skill_level, format, duration_hours, skills_taught (Collection), completion_rate, avg_score, content_vector (3072-dim for text-embedding-3-large)
- VectorSearch: HnswAlgorithmConfiguration
- SemanticSearch: title as title field, description as content field
- SDK: `azure-search-documents>=11.6.0`, `SearchIndexClient`, `AzureKeyCredential`

### 3.2 `src/search/indexer.py` + `scripts/05_index_courses_ai_search.py`
- Load `data/seed_courses.json`
- For each course: build text = "Title: ... Department: ... Level: ... Skills: ... Description: ..."
- Embed via Foundry: `AIProjectClient(endpoint=FOUNDRY_PROJECT_ENDPOINT).inference.get_azure_openai_client()` → `.embeddings.create(model=EMBEDDING_DEPLOYMENT_NAME)`
- Enrich with live analytics: `FabricAnalytics.get_course_stats(course_id)` → add completion_rate, avg_score
- Upload to AI Search via `SearchClient.upload_documents()`
- Batch size 50 to avoid token limits

### 3.3 `src/search/searcher.py`
Hybrid search (vector + keyword + filters):
- `SearchClient.search(search_text, vector_queries=[VectorizedQuery(...)], filter=..., select=..., query_type="semantic", semantic_configuration_name="semantic-config")`

### 3.4 `src/foundry/agent.py` + `scripts/06_create_foundry_agent.py`
Create Educational Consultant agent:
- `AIProjectClient.agents.create_agent(model=MODEL_DEPLOYMENT_NAME, instructions=SYSTEM_PROMPT, tools=[AzureAISearchTool(...)])`
- System prompt: recommends only courses from search results, explains why, flags prerequisites, suggests learning paths
- Store agent ID somewhere (env var or file) for the chat UI to reuse

### 3.5 `src/foundry/chat.py`
Thread-based chat using Foundry Agent Service:
- `project.agents.threads.create()` → `messages.create()` → `runs.create_and_process()` → `messages.list()`
- Expose `chat(thread_id, message, agent_id) → (reply_text, thread_id)`

### 3.6 `ui/pages/2_ai_consultant.py`
Replace the stub. Features:
- Sidebar: learner picker (from `FabricAnalytics.get_courses()` / `dim_learners`)
- When learner selected: fetch profile + skill gaps + completed courses from Fabric SQL, pass as system context
- `st.chat_message` / `st.chat_input` UI with streaming if possible
- Expandable "Sources" showing which AI Search results the agent used

---

## Sprint 4 — What to build (Course Generator)

Read `POC_AZURE_NATIVE_PLAN.md` Tasks 4.1–4.3. Summary:

### 4.1 `src/foundry/course_generator.py`
Three functions using direct Foundry model inference (no agent, just chat completions):
- `generate_course_outline(objectives, department, duration_hours, skill_gaps) → dict` — returns structured JSON with modules, assessments
- `generate_quiz(module_content, difficulty, count) → list[dict]` — returns questions + GIFT format
- `get_course_improvement_suggestions(course_id, analytics) → str` — reads funnel data, returns analysis

### 4.2 Gap analysis function in `course_generator.py`
- `gap_analysis(department, analytics) → str` — reads skill gaps + existing courses, recommends 3 new courses

### 4.3 `ui/pages/3_course_generator.py`
Replace the stub. Four tabs:
1. Course Outline Generator — form → call generate_course_outline → display structured output
2. Quiz Generator — paste module content → generate questions
3. Course Improvement — pick course from dropdown → show funnel + AI suggestions
4. Gap Analysis — pick department → show gaps + recommended new courses

---

## How to run

```bash
cd c:/code/learning-ai
uv run streamlit run ui/app.py
```
