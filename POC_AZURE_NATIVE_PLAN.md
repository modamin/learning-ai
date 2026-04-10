# Learning Platform Modernization — Azure-Native PoC Plan

## Overview

This plan builds the Analytics and AI pillars directly on **Microsoft Fabric**, **Microsoft Foundry**, and **Azure AI Search**. No local substitutes. You'll have real Azure infrastructure running at the end.

---

## Prerequisites (Manual Setup Before Coding)

Complete these in the Azure Portal / Fabric Portal / Foundry Portal before handing off to Claude Code.

### 1. Azure Subscription & Resource Group
```bash
az login
az group create --name rg-learning-poc --location eastus2
```

### 2. Microsoft Fabric
- Go to https://app.fabric.microsoft.com
- Create a **Fabric Workspace**: `learning-poc-ws`
- Inside the workspace, create:
  - A **Lakehouse**: `learning_lakehouse`
  - A **Data Warehouse**: `learning_dw` (optional, can use Lakehouse SQL endpoint)
- Note down the workspace ID and lakehouse ID (visible in URL)
- Get a **Fabric REST API token** or use service principal:
  ```bash
  az account get-access-token --resource https://analysis.windows.net/powerbi/api
  ```

### 3. Microsoft Foundry
- Go to https://ai.azure.com (ensure "New Foundry" toggle is ON)
- Click **Create an agent** → creates an account + project
- Deploy a model: **gpt-4o-mini** (for speed) and **text-embedding-3-large**
- Note down:
  - `PROJECT_ENDPOINT`: `https://<resource>.services.ai.azure.com/api/projects/<project>`
  - `MODEL_DEPLOYMENT_NAME`: e.g., `gpt-4o-mini`
  - `EMBEDDING_DEPLOYMENT_NAME`: e.g., `text-embedding-3-large`

### 4. Azure AI Search
```bash
az search service create \
  --name learning-poc-search \
  --resource-group rg-learning-poc \
  --sku basic \
  --location eastus2

# Get the admin key
az search admin-key show \
  --service-name learning-poc-search \
  --resource-group rg-learning-poc
```
Note the **endpoint** (`https://learning-poc-search.search.windows.net`) and **admin key**.

### 5. Environment Variables
Create `.env`:
```
# Foundry
FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
MODEL_DEPLOYMENT_NAME=gpt-4o-mini
EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large

# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://learning-poc-search.search.windows.net
AZURE_SEARCH_ADMIN_KEY=<key>
AZURE_SEARCH_INDEX_NAME=course-catalog

# Fabric
FABRIC_WORKSPACE_ID=<guid>
FABRIC_LAKEHOUSE_ID=<guid>
FABRIC_SQL_ENDPOINT=<from lakehouse SQL analytics endpoint>

# Auth
AZURE_TENANT_ID=<tenant>
AZURE_CLIENT_ID=<optional, for service principal>
AZURE_CLIENT_SECRET=<optional>
```

---

## Project Structure

```
learning-platform-poc/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── seed_courses.json               # 200 synthetic courses
│   ├── seed_learners.json              # 500 synthetic learners
│   └── seed_xapi_events.json           # 50K+ synthetic xAPI statements
├── scripts/
│   ├── 01_generate_synthetic_data.py   # Generate all seed data
│   ├── 02_upload_to_fabric.py          # Upload seed data → Fabric Lakehouse via OneLake API
│   ├── 03_create_fabric_tables.py      # Create medallion tables (bronze/silver/gold) via SQL endpoint
│   ├── 04_run_fabric_pipeline.py       # Transform bronze→silver→gold using Fabric SQL
│   ├── 05_index_courses_ai_search.py   # Embed courses → Azure AI Search vector index
│   ├── 06_create_foundry_agent.py      # Create the Educational Consultant agent in Foundry
│   └── 07_demo.py                      # End-to-end demo runner
├── notebooks/                          # Fabric notebooks (upload to workspace)
│   ├── 01_xapi_ingestion.ipynb         # PySpark: bronze → silver
│   ├── 02_analytics_gold.ipynb         # PySpark: silver → gold aggregations
│   ├── 03_dropout_prediction.ipynb     # Fabric Data Science: train ML model
│   └── 04_course_embeddings.ipynb      # Generate embeddings, push to AI Search
├── src/
│   ├── config.py                       # Load env vars, Azure credentials
│   ├── fabric/
│   │   ├── onelake.py                  # OneLake file upload/download (Azure SDK)
│   │   ├── sql_client.py               # Query Fabric SQL analytics endpoint (pyodbc/JDBC)
│   │   └── pipeline.py                 # Transform logic callable from scripts or notebooks
│   ├── search/
│   │   ├── index_manager.py            # Create/update Azure AI Search index schema
│   │   ├── indexer.py                  # Embed courses and push to index
│   │   └── searcher.py                 # Hybrid search (vector + keyword + filters)
│   ├── foundry/
│   │   ├── agent.py                    # Create & manage Foundry Agent (azure-ai-projects SDK)
│   │   ├── chat.py                     # Chat with agent (thread/run pattern)
│   │   └── course_generator.py         # GenAI course dev (direct model calls via Foundry)
│   ├── analytics/
│   │   ├── queries.py                  # SQL queries against Fabric gold tables
│   │   ├── engagement.py               # Engagement analytics
│   │   ├── completion.py               # Completion funnels
│   │   ├── skill_gaps.py               # Skill gap analysis
│   │   └── predictions.py             # Call Fabric ML model endpoint
│   └── api/
│       └── main.py                     # FastAPI: wraps all components
├── ui/
│   ├── app.py                          # Streamlit multi-page app
│   └── pages/
│       ├── 1_analytics_dashboard.py    # Charts from Fabric SQL queries
│       ├── 2_ai_consultant.py          # Chat with Foundry Agent
│       └── 3_course_generator.py       # GenAI course dev tools
├── powerbi/
│   └── learning_analytics.pbix         # Optional: Power BI template connecting to Fabric
└── infra/
    └── main.bicep                      # Optional: IaC for all Azure resources
```

---

## Requirements

```
# requirements.txt
azure-identity>=1.19.0
azure-ai-projects>=2.0.0              # Foundry Agent SDK
azure-search-documents>=11.6.0        # AI Search SDK
azure-storage-file-datalake>=12.17.0   # OneLake / ADLS Gen2 access
pyodbc>=5.1.0                          # Fabric SQL endpoint queries
pandas>=2.2.0
plotly>=5.24.0
streamlit>=1.40.0
fastapi>=0.115.0
uvicorn>=0.32.0
scikit-learn>=1.5.0
python-dotenv>=1.0.0
pydantic>=2.9.0
```

---

## Sprint-by-Sprint Implementation

### Sprint 1: Synthetic Data + Upload to Fabric (Day 1)

#### Task 1.1 — Generate Synthetic Data
**File:** `scripts/01_generate_synthetic_data.py`

Generate the same seed data as before (200 courses, 500 learners, 50K xAPI events) with intentional patterns. Output as JSON files in `data/`.

Key patterns to bake in:
- Module 3-4 drop-off in 30% of courses (visible in completion funnels)
- Engineering dept has Python/ML skill gaps
- Blended courses have 25% higher completion than self-paced
- Peak engagement at 10am and 2pm
- Some courses have very low scores on specific quizzes

#### Task 1.2 — Upload to Fabric Lakehouse
**File:** `scripts/02_upload_to_fabric.py` and `src/fabric/onelake.py`

Upload seed data to the Lakehouse Files section via the OneLake ADLS Gen2 API:

```python
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

credential = DefaultAzureCredential()
service_client = DataLakeServiceClient(
    account_url="https://onelake.dfs.fabric.microsoft.com",
    credential=credential
)
fs_client = service_client.get_file_system_client(file_system=WORKSPACE_ID)
dir_client = fs_client.get_directory_client(f"{LAKEHOUSE_ID}/Files/raw")

# Upload each JSON file
for filename in ["seed_courses.json", "seed_learners.json", "seed_xapi_events.json"]:
    file_client = dir_client.get_file_client(filename)
    with open(f"data/{filename}", "rb") as f:
        file_client.upload_data(f, overwrite=True)
```

#### Task 1.3 — Create Medallion Tables
**File:** `scripts/03_create_fabric_tables.py` and `src/fabric/sql_client.py`

Connect to the Fabric Lakehouse SQL analytics endpoint (read-only) or use the Fabric REST API / notebooks to create Delta tables:

**Option A — Fabric Notebook** (`notebooks/01_xapi_ingestion.ipynb`):
```python
# In Fabric notebook (PySpark)
df_events = spark.read.json("Files/raw/seed_xapi_events.json")
df_events.write.format("delta").mode("overwrite").saveAsTable("bronze_xapi_events")

df_courses = spark.read.json("Files/raw/seed_courses.json")
df_courses.write.format("delta").mode("overwrite").saveAsTable("dim_courses")

df_learners = spark.read.json("Files/raw/seed_learners.json")
df_learners.write.format("delta").mode("overwrite").saveAsTable("dim_learners")
```

**Option B — Fabric REST API** (from local machine):
Use the Fabric Items API to create and execute notebooks programmatically. Claude Code can generate the notebook JSON and upload it.

#### Task 1.4 — Transform Bronze → Silver → Gold
**File:** `notebooks/02_analytics_gold.ipynb`

```python
# Silver: validate, normalize, enrich
df_silver = spark.sql("""
    SELECT
        actor.mbox AS learner_email,
        verb.id AS verb,
        object.id AS object_id,
        object.definition.name AS object_name,
        result.score.scaled AS score,
        result.duration AS duration,
        timestamp,
        context.extensions.department AS department,
        context.extensions.course_id AS course_id,
        context.extensions.module_id AS module_id
    FROM bronze_xapi_events
    WHERE actor.mbox IS NOT NULL
""")
df_silver.write.format("delta").mode("overwrite").saveAsTable("silver_xapi_events")

# Gold: aggregations
spark.sql("""
    CREATE OR REPLACE TABLE gold_course_completion AS
    SELECT
        course_id,
        COUNT(DISTINCT learner_email) as enrolled,
        COUNT(DISTINCT CASE WHEN verb = 'completed' THEN learner_email END) as completed,
        ROUND(AVG(CASE WHEN verb = 'scored' THEN score END), 2) as avg_score,
        ROUND(AVG(CAST(duration AS DOUBLE) / 60), 1) as avg_duration_hours
    FROM silver_xapi_events
    GROUP BY course_id
""")

spark.sql("""
    CREATE OR REPLACE TABLE gold_module_dropoff AS
    SELECT
        course_id,
        module_id,
        COUNT(DISTINCT learner_email) as learners_reached,
        COUNT(DISTINCT CASE WHEN verb = 'completed' THEN learner_email END) as learners_completed
    FROM silver_xapi_events
    GROUP BY course_id, module_id
    ORDER BY course_id, module_id
""")

spark.sql("""
    CREATE OR REPLACE TABLE gold_department_skills AS
    SELECT
        d.department,
        s.skill_name,
        ROUND(AVG(s.proficiency_level), 2) as avg_proficiency,
        COUNT(*) as learner_count
    FROM dim_learners l
    LATERAL VIEW EXPLODE(skills) AS s
    JOIN (SELECT DISTINCT learner_email, department FROM silver_xapi_events) d
        ON l.email = d.learner_email
    GROUP BY d.department, s.skill_name
""")

spark.sql("""
    CREATE OR REPLACE TABLE gold_engagement_hourly AS
    SELECT
        course_id,
        module_id,
        HOUR(timestamp) as hour_of_day,
        DAYOFWEEK(timestamp) as day_of_week,
        COUNT(*) as event_count,
        COUNT(DISTINCT learner_email) as unique_learners
    FROM silver_xapi_events
    WHERE verb IN ('interacted', 'progressed', 'launched')
    GROUP BY course_id, module_id, HOUR(timestamp), DAYOFWEEK(timestamp)
""")
```

**Verify:** Query the SQL analytics endpoint from local machine:
```python
import pyodbc
conn = pyodbc.connect(
    f"Driver={{ODBC Driver 18 for SQL Server}};"
    f"Server={FABRIC_SQL_ENDPOINT};"
    f"Database={LAKEHOUSE_NAME};"
    f"Authentication=ActiveDirectoryInteractive;"
)
cursor = conn.cursor()
cursor.execute("SELECT * FROM gold_course_completion LIMIT 5")
print(cursor.fetchall())
```

---

### Sprint 2: Analytics Engine + Dashboard (Day 2)

#### Task 2.1 — Analytics Query Layer
**File:** `src/analytics/queries.py`

All queries run against the Fabric SQL analytics endpoint:

```python
class FabricAnalytics:
    def __init__(self, connection_string: str):
        self.conn = pyodbc.connect(connection_string)

    def get_completion_funnel(self, course_id: str) -> list[dict]:
        query = """
            SELECT module_id, learners_reached, learners_completed,
                   ROUND(CAST(learners_completed AS FLOAT) / learners_reached * 100, 1) as pct
            FROM gold_module_dropoff
            WHERE course_id = ?
            ORDER BY module_id
        """
        return self._fetch(query, [course_id])

    def get_engagement_heatmap(self, course_id: str) -> list[dict]:
        query = """
            SELECT module_id, hour_of_day, SUM(event_count) as events
            FROM gold_engagement_hourly
            WHERE course_id = ?
            GROUP BY module_id, hour_of_day
        """
        return self._fetch(query, [course_id])

    def get_department_skill_gaps(self, department: str) -> list[dict]:
        query = """
            SELECT skill_name, avg_proficiency, learner_count,
                   3.5 as target,  -- org target proficiency
                   ROUND(3.5 - avg_proficiency, 2) as gap
            FROM gold_department_skills
            WHERE department = ?
            ORDER BY gap DESC
        """
        return self._fetch(query, [department])

    def get_course_ranking(self) -> list[dict]:
        query = """
            SELECT c.course_id, d.title, c.enrolled, c.completed,
                   ROUND(CAST(c.completed AS FLOAT) / c.enrolled * 100, 1) as completion_rate,
                   c.avg_score
            FROM gold_course_completion c
            JOIN dim_courses d ON c.course_id = d.course_id
            ORDER BY completion_rate DESC
        """
        return self._fetch(query)

    def get_at_risk_learners(self, course_id: str) -> list[dict]:
        query = """
            SELECT s.learner_email, l.name, l.department,
                   COUNT(DISTINCT s.module_id) as modules_started,
                   MAX(s.timestamp) as last_activity,
                   DATEDIFF(day, MAX(s.timestamp), GETDATE()) as days_inactive,
                   AVG(s.score) as avg_score
            FROM silver_xapi_events s
            JOIN dim_learners l ON s.learner_email = l.email
            WHERE s.course_id = ?
              AND s.verb != 'completed'
            GROUP BY s.learner_email, l.name, l.department
            HAVING DATEDIFF(day, MAX(s.timestamp), GETDATE()) > 7
            ORDER BY days_inactive DESC
        """
        return self._fetch(query, [course_id])

    def get_overview_kpis(self) -> dict:
        query = """
            SELECT
                (SELECT COUNT(DISTINCT learner_email) FROM silver_xapi_events) as total_learners,
                (SELECT COUNT(DISTINCT learner_email) FROM silver_xapi_events
                 WHERE timestamp > DATEADD(month, -1, GETDATE())) as active_this_month,
                (SELECT ROUND(AVG(CAST(completed AS FLOAT) / enrolled * 100), 1)
                 FROM gold_course_completion WHERE enrolled > 10) as avg_completion_rate,
                (SELECT COUNT(*) FROM dim_courses) as total_courses
        """
        return self._fetch_one(query)
```

#### Task 2.2 — Dropout Prediction Model
**File:** `notebooks/03_dropout_prediction.ipynb` (Fabric Data Science)

```python
# In Fabric notebook with MLflow tracking
import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = spark.sql("""
    SELECT
        learner_email, course_id,
        COUNT(*) as total_events,
        COUNT(DISTINCT module_id) as modules_touched,
        AVG(score) as avg_score,
        COUNT(DISTINCT DATE(timestamp)) as active_days,
        DATEDIFF(day, MIN(timestamp), MAX(timestamp)) as span_days,
        MAX(CASE WHEN verb = 'completed' THEN 1 ELSE 0 END) as completed
    FROM silver_xapi_events
    GROUP BY learner_email, course_id
    HAVING COUNT(*) >= 5
""").toPandas()

X = df[['total_events', 'modules_touched', 'avg_score', 'active_days', 'span_days']]
y = df['completed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("dropout_prediction")
with mlflow.start_run():
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "dropout_model")
    print(f"Accuracy: {accuracy:.3f}")

# Register model in Fabric
mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/dropout_model", "dropout_predictor")
```

#### Task 2.3 — Streamlit Analytics Dashboard
**File:** `ui/pages/1_analytics_dashboard.py`

Connects to Fabric SQL endpoint and renders Plotly charts:
- KPI cards (total learners, active, completion rate, courses)
- Completion funnel (horizontal bar)
- Engagement heatmap (module × hour, Plotly heatmap)
- Department skill gap radar chart
- Course ranking scatter (completion rate vs avg score)
- At-risk learners table

---

### Sprint 3: Azure AI Search + Foundry Agent (Day 3-4)

#### Task 3.1 — Create Azure AI Search Index
**File:** `src/search/index_manager.py`

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticSearch, SemanticPrioritizedFields, SemanticField
)
from azure.core.credentials import AzureKeyCredential

def create_course_index(endpoint: str, key: str, index_name: str):
    client = SearchIndexClient(endpoint, AzureKeyCredential(key))

    fields = [
        SearchField(name="course_id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="description", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="department", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchField(name="skill_level", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchField(name="format", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="duration_hours", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        SearchField(name="skills_taught", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True, searchable=True),
        SearchField(name="completion_rate", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        SearchField(name="avg_score", type=SearchFieldDataType.Double, sortable=True),
        SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=3072,
                    vector_search_profile_name="vector-profile"),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
        profiles=[VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw-config")]
    )

    semantic = SemanticSearch(configurations=[
        SemanticConfiguration(name="semantic-config", prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="description")]
        ))
    ])

    index = SearchIndex(name=index_name, fields=fields,
                        vector_search=vector_search, semantic_search=semantic)
    client.create_or_update_index(index)
```

#### Task 3.2 — Embed Courses and Index
**File:** `scripts/05_index_courses_ai_search.py` and `src/search/indexer.py`

```python
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.search.documents import SearchClient

# Use Foundry to generate embeddings
project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())
openai_client = project.inference.get_azure_openai_client()

def embed_course(course: dict) -> list[float]:
    text = f"Title: {course['title']}\nDepartment: {course['department']}\n" \
           f"Level: {course['skill_level']}\nSkills: {', '.join(course['skills_taught'])}\n" \
           f"Description: {course['description']}\nDuration: {course['duration_hours']} hours"
    response = openai_client.embeddings.create(input=[text], model=EMBEDDING_DEPLOYMENT_NAME)
    return response.data[0].embedding

# Embed all courses and upload to AI Search
search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))
documents = []
for course in courses:
    vector = embed_course(course)
    doc = {**course, "content_vector": vector}
    # Enrich with analytics from Fabric
    stats = fabric_analytics.get_course_stats(course["course_id"])
    if stats:
        doc["completion_rate"] = stats["completion_rate"]
        doc["avg_score"] = stats["avg_score"]
    documents.append(doc)

search_client.upload_documents(documents=documents)
```

#### Task 3.3 — Create Foundry Educational Consultant Agent
**File:** `scripts/06_create_foundry_agent.py` and `src/foundry/agent.py`

```python
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    AzureAISearchTool,
    AzureAISearchToolResource,
    AzureAISearchResource,
)

project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())

# Connect AI Search as a tool for the agent
search_tool = AzureAISearchTool(azure_ai_search=AzureAISearchResource(
    index_name=SEARCH_INDEX_NAME,
    endpoint=SEARCH_ENDPOINT,
    authentication={"type": "system_assigned_managed_identity"},  # or API key
))

SYSTEM_PROMPT = """You are an AI Educational Consultant for NovaTech Learning Corp.
You help learners find courses, understand skill gaps, and plan learning journeys.

RULES:
- Only recommend courses from the search results provided — never invent courses.
- Explain WHY each course matches the learner's goals.
- Consider the learner's department, current skills, and completed courses.
- Include time commitment and format for each recommendation.
- Suggest a sequenced learning path when recommending multiple courses.
- Flag prerequisites the learner hasn't completed.
- Be encouraging but honest about time commitments.

For each recommendation include:
1. Course title + why it's relevant
2. Key skills they'll gain
3. Time commitment and format
4. Prerequisites needed
"""

agent = project.agents.create_version(
    agent_name="educational-consultant",
    definition=PromptAgentDefinition(
        model=MODEL_DEPLOYMENT_NAME,
        instructions=SYSTEM_PROMPT,
        tools=[search_tool],
    )
)
print(f"Agent created: {agent.name} v{agent.version}")
```

#### Task 3.4 — Chat with Agent
**File:** `src/foundry/chat.py`

```python
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())
openai_client = project.inference.get_azure_openai_client()

def chat_with_consultant(message: str, learner_context: str = None, history: list = None):
    """Chat with the Educational Consultant agent."""
    messages = history or []

    # Prepend learner context if available
    if learner_context and not history:
        messages.append({
            "role": "system",
            "content": f"Current learner context:\n{learner_context}"
        })

    messages.append({"role": "user", "content": message})

    response = openai_client.chat.completions.create(
        model=MODEL_DEPLOYMENT_NAME,
        messages=messages,
        # The agent's AI Search tool is invoked automatically
        # when the agent determines it needs course data
    )

    assistant_msg = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg, messages
```

**For full agent features (memory, tools, threads):**
```python
# Using Foundry Agent Service directly
agent_client = project.agents

# Create a thread (conversation session)
thread = agent_client.threads.create()

# Add user message
agent_client.messages.create(thread_id=thread.id, role="user", content=message)

# Run the agent
run = agent_client.runs.create(thread_id=thread.id, agent_id=agent.id)

# Poll for completion
while run.status in ("queued", "in_progress"):
    run = agent_client.runs.get(thread_id=thread.id, run_id=run.id)

# Get response
messages = agent_client.messages.list(thread_id=thread.id)
```

#### Task 3.5 — AI Consultant Chat UI
**File:** `ui/pages/2_ai_consultant.py`

Streamlit chat UI that:
- Sidebar: pick a learner from `dim_learners` (fetched from Fabric SQL)
- When learner selected, query Fabric for their profile + skill gaps + completed courses
- Pass learner context to agent
- Display chat with streaming
- Show "Sources" expandable (which courses were retrieved from AI Search)

---

### Sprint 4: GenAI Course Development (Day 4-5)

#### Task 4.1 — Course Generator
**File:** `src/foundry/course_generator.py`

Uses Foundry model catalog directly (no agent needed, just inference):

```python
def generate_course_outline(objectives: str, department: str, duration_hours: int,
                            skill_gaps: list[dict] = None) -> dict:
    """Generate a structured course outline using Foundry Models."""

    gap_context = ""
    if skill_gaps:
        gap_context = f"\n\nSkill gaps for {department} department:\n"
        for gap in skill_gaps[:5]:
            gap_context += f"- {gap['skill_name']}: current {gap['avg_proficiency']}/5, target 3.5/5\n"

    prompt = f"""Create a detailed course outline for the following:

LEARNING OBJECTIVES:
{objectives}

TARGET AUDIENCE: {department} department
TARGET DURATION: {duration_hours} hours
{gap_context}

Respond in JSON with this structure:
{{
  "title": "Course Title",
  "description": "2-3 sentence description",
  "skill_level": "beginner|intermediate|advanced",
  "duration_hours": {duration_hours},
  "modules": [
    {{
      "module_number": 1,
      "title": "Module Title",
      "duration_minutes": 45,
      "learning_objectives": ["obj1", "obj2"],
      "content_outline": ["topic1", "topic2"],
      "assessment": {{
        "type": "quiz|assignment|discussion",
        "description": "Brief description",
        "question_count": 5
      }}
    }}
  ],
  "prerequisites": ["course title or skill"],
  "skills_taught": ["skill1", "skill2"]
}}"""

    response = openai_client.chat.completions.create(
        model=MODEL_DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def generate_quiz(module_content: str, difficulty: str, count: int) -> list[dict]:
    """Generate quiz questions in Moodle GIFT format."""
    prompt = f"""Generate {count} quiz questions for this module content:

{module_content}

Difficulty: {difficulty}

Respond in JSON array:
[{{
  "question": "Question text?",
  "type": "multiple_choice",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "correct_answer": "A",
  "explanation": "Why this is correct",
  "gift_format": "::Q1::Question text{{=Correct answer ~Wrong1 ~Wrong2 ~Wrong3}}"
}}]"""

    response = openai_client.chat.completions.create(
        model=MODEL_DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def get_course_improvement_suggestions(course_id: str, analytics: FabricAnalytics) -> str:
    """Analyze course analytics and suggest improvements using GenAI."""
    funnel = analytics.get_completion_funnel(course_id)
    course = analytics.get_course_detail(course_id)

    prompt = f"""Analyze this course's performance data and suggest improvements:

COURSE: {course['title']}
COMPLETION RATE: {course.get('completion_rate', 'N/A')}%
AVG SCORE: {course.get('avg_score', 'N/A')}

MODULE DROP-OFF FUNNEL:
{json.dumps(funnel, indent=2)}

Identify:
1. Which modules have the worst drop-off and why that might be
2. Specific actionable improvements for struggling modules
3. Whether the course should be restructured (split, merged, reordered)
4. Assessment quality issues (if avg scores suggest questions are too hard/easy)
"""
    response = openai_client.chat.completions.create(
        model=MODEL_DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

#### Task 4.2 — Gap Analysis
Query Fabric for skill gaps, then ask Foundry to recommend new courses:

```python
def gap_analysis(department: str, analytics: FabricAnalytics) -> str:
    gaps = analytics.get_department_skill_gaps(department)
    existing = analytics.get_courses_by_department(department)

    prompt = f"""You are an L&D strategist. Analyze skill gaps and recommend new courses.

DEPARTMENT: {department}

CURRENT SKILL GAPS (proficiency out of 5, target is 3.5):
{json.dumps(gaps[:10], indent=2)}

EXISTING COURSES COVERING THIS DEPARTMENT:
{json.dumps(existing[:15], indent=2)}

Provide:
1. Top 3 skills that need new courses (not already covered)
2. For each, a recommended course with: title, objectives, duration, format
3. Expected impact on closing the gap
"""
    response = openai_client.chat.completions.create(
        model=MODEL_DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

#### Task 4.3 — Course Generator UI
**File:** `ui/pages/3_course_generator.py`

Four tabs as specified in the original plan, all hitting Foundry + Fabric.

---

### Sprint 5: Integration, API, Polish (Day 5-6)

#### Task 5.1 — FastAPI
**File:** `src/api/main.py`

Wraps all components. Same endpoints as original plan. Adds health checks for Fabric SQL endpoint and Foundry Agent.

#### Task 5.2 — Power BI Report (Optional)
**File:** `powerbi/learning_analytics.pbix`

Create a Power BI report connected to the Fabric Lakehouse via Direct Lake mode. Include the same visuals as the Streamlit dashboard. This is the "production" version of the dashboard.

#### Task 5.3 — End-to-End Demo Script
**File:** `scripts/07_demo.py`

Automated script that runs the full demo sequence:
1. Query Fabric for KPIs
2. Show completion funnel for a struggling course
3. Chat with the Foundry agent as a learner
4. Generate a course outline based on skill gaps
5. Print timing metrics (query latency, agent response time)

---

## Azure Service Mapping

| PoC Component | Azure Service | SDK / Access Method |
|---|---|---|
| Data Lake | Fabric Lakehouse (OneLake) | `azure-storage-file-datalake` → `onelake.dfs.fabric.microsoft.com` |
| ETL | Fabric Notebooks (PySpark) | Upload `.ipynb` to workspace, run via portal or REST API |
| SQL Analytics | Fabric SQL Endpoint | `pyodbc` with AAD auth |
| ML Model | Fabric Data Science (MLflow) | MLflow in Fabric notebooks |
| Dashboards | Streamlit (PoC) / Power BI (prod) | `pyodbc` queries → Plotly charts |
| Vector Search | Azure AI Search | `azure-search-documents` SDK |
| LLM | Foundry Model Catalog | `azure-ai-projects` SDK → `.inference.get_azure_openai_client()` |
| Embeddings | Foundry Model Catalog | Same SDK, `text-embedding-3-large` deployment |
| AI Agent | Foundry Agent Service | `azure-ai-projects` SDK → `.agents` |
| Agent Tools | AI Search via Foundry Tools | `AzureAISearchTool` in agent definition |

---

## Estimated Azure Costs (PoC / 1 month)

| Service | SKU | Est. Cost |
|---|---|---|
| Fabric | F2 capacity (trial free) | $0 (trial) or ~$263/mo |
| Azure AI Search | Basic | ~$75/mo |
| Foundry / Azure OpenAI | GPT-4o-mini + embeddings | ~$20-50/mo (PoC volume) |
| Total | | **~$100-400/mo** (or free with Fabric trial) |

---
