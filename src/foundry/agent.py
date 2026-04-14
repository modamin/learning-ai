"""
Sprint 3 — Task 3.4: Educational Consultant agent management + OpenAI client factory.

Auth note
---------
AIProjectClient.get_openai_client() uses scope https://ai.azure.com/.default, which
requires the 'Azure AI Developer' role on the AI Services resource.

get_openai_client() in this module uses scope https://cognitiveservices.azure.com/.default,
which only requires 'Cognitive Services OpenAI User' — a lighter role that most SPs already
have.  It derives the Azure OpenAI base URL from FOUNDRY_PROJECT_ENDPOINT automatically.

Required Azure role (assign once in Azure Portal):
  Resource : AI Services resource (e.g. admin-2125-resource)
  Role     : Cognitive Services OpenAI User  (or Azure AI Developer)
  Assignee : service principal object ID shown in the 401 error
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import re as _re

import requests as _requests
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MCPTool, PromptAgentDefinition
from azure.identity import ClientSecretCredential, DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

logger = logging.getLogger(__name__)

AGENT_NAME = "educational-consultant"

# Scope used by the direct AzureOpenAI client (requires Cognitive Services OpenAI User)
_COGNITIVESERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"

SYSTEM_PROMPT = """\
You are an AI Educational Consultant for NovaTech Learning Corp.
You help learners find courses, understand skill gaps, and plan personalised learning journeys.

RULES:
- Only recommend courses that appear in the <search_results> context block provided.
  Never invent course titles or IDs.
- For each recommendation explain WHY it matches the learner's goals or skill gaps.
- Consider the learner's department, current skill proficiency, and completed courses.
- Always include: time commitment, format (self-paced / blended / instructor-led), and prerequisites.
- When recommending multiple courses, suggest a sequenced learning path.
- Flag any prerequisites the learner has not yet completed.
- Be encouraging but honest about workload.

For each recommended course include:
  1. Course title and ID
  2. Why it's relevant to this learner
  3. Key skills they will gain
  4. Time commitment and format
  5. Prerequisites needed (if any)
  6. Suggested position in their learning path
"""

FOUNDRY_IQ_SYSTEM_PROMPT = """\
You are an AI Educational Consultant for NovaTech Learning Corp.
You help learners find courses, understand skill gaps, and plan personalised learning journeys.

RULES:
- Use the knowledge base tool to search for relevant courses. Always search before recommending.
- Only recommend courses you found via the knowledge base. Never invent course titles or IDs.
- For each recommendation explain WHY it matches the learner's goals or skill gaps.
- Consider the learner's department, current skill proficiency, and completed courses (provided in the user message).
- Always include: time commitment, format (self-paced / blended / instructor-led), and prerequisites.
- When recommending multiple courses, suggest a sequenced learning path.
- Flag any prerequisites the learner has not yet completed.
- Be encouraging but honest about workload.
- If the knowledge base does not contain relevant courses, say "I don't know" rather than inventing recommendations.
- Include citations for every course you recommend using the format: 【message_idx:search_idx†source_name】

For each recommended course include:
  1. Course title and ID
  2. Why it's relevant to this learner
  3. Key skills they will gain
  4. Time commitment and format
  5. Prerequisites needed (if any)
  6. Suggested position in their learning path
"""


# ── Credential helpers ────────────────────────────────────────────────────────

def _get_credential():
    tenant    = os.environ.get("AZURE_TENANT_ID")
    client_id = os.environ.get("AZURE_CLIENT_ID")
    secret    = os.environ.get("AZURE_CLIENT_SECRET")
    if tenant and client_id and secret:
        return ClientSecretCredential(tenant, client_id, secret)
    return DefaultAzureCredential()


def get_project_client() -> AIProjectClient:
    """Return an authenticated AIProjectClient from env vars."""
    return AIProjectClient(
        endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        credential=_get_credential(),
    )


def _parse_openai_base_url(foundry_endpoint: str) -> str:
    """
    Derive the Azure OpenAI base URL from the Foundry project endpoint.

    Foundry endpoint:  https://<resource>.services.ai.azure.com/api/projects/<project>
    OpenAI base URL:   https://<resource>.services.ai.azure.com

    Keeps the host, strips the path.
    """
    match = re.match(r"(https://[^/]+)", foundry_endpoint.rstrip("/"))
    if not match:
        raise ValueError(f"Cannot parse base URL from FOUNDRY_PROJECT_ENDPOINT: {foundry_endpoint!r}")
    return match.group(1)


def get_openai_client():
    """
    Return an authenticated openai.AzureOpenAI client for embeddings and chat.

    Uses scope https://cognitiveservices.azure.com/.default so the SP only needs
    'Cognitive Services OpenAI User' (not the broader 'Azure AI Developer' role).

    Required env vars: FOUNDRY_PROJECT_ENDPOINT
    Optional env vars: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
    """
    from openai import AzureOpenAI

    foundry_endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
    base_url = _parse_openai_base_url(foundry_endpoint)
    credential = _get_credential()

    token_provider = get_bearer_token_provider(credential, _COGNITIVESERVICES_SCOPE)

    client = AzureOpenAI(
        azure_endpoint=base_url,
        azure_ad_token_provider=token_provider,
        api_version="2024-12-01-preview",
    )
    logger.info("AzureOpenAI client ready. Base URL: %s", base_url)
    return client


# ── FoundryIQ helpers ─────────────────────────────────────────────────────────

def _parse_project_resource_id() -> str:
    """
    Build the ARM resource ID for the Foundry project from env vars.

    Foundry endpoint: https://{account}-resource.services.ai.azure.com/api/projects/{project}
    ARM resource ID:  /subscriptions/{sub}/resourceGroups/{rg}/providers/
                      Microsoft.MachineLearningServices/workspaces/{account}/projects/{project}
    """
    endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"].rstrip("/")
    match = _re.match(r"https://([^.]+)-resource\.services\.ai\.azure\.com/api/projects/([^/]+)", endpoint)
    if not match:
        raise ValueError(f"Cannot parse account/project from FOUNDRY_PROJECT_ENDPOINT: {endpoint!r}")
    account_name = match.group(1)
    project_name = match.group(2)

    sub = os.environ["AZURE_SUBSCRIPTION_ID"]
    rg  = os.environ["AZURE_RESOURCE_GROUP"]
    return (
        f"/subscriptions/{sub}/resourceGroups/{rg}"
        f"/providers/Microsoft.MachineLearningServices"
        f"/workspaces/{account_name}/projects/{project_name}"
    )


def create_project_connection(
    connection_name: str | None = None,
    kb_name: str | None = None,
    credential=None,
) -> str:
    """
    Create or update the ARM project connection that gives the Foundry agent
    access to the FoundryIQ knowledge base MCP endpoint.

    Returns the full connection resource ID string to pass to MCPTool.project_connection_id.

    Required env vars: FOUNDRY_PROJECT_ENDPOINT, AZURE_SUBSCRIPTION_ID,
                       AZURE_RESOURCE_GROUP, FOUNDRY_KB_CONNECTION_NAME,
                       FOUNDRY_IQ_KB_NAME, AZURE_SEARCH_ENDPOINT
    """
    connection_name = connection_name or os.environ["FOUNDRY_KB_CONNECTION_NAME"]
    kb_name         = kb_name or os.environ["FOUNDRY_IQ_KB_NAME"]
    search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/")
    project_resource_id = _parse_project_resource_id()

    credential = credential or _get_credential()
    token_provider = get_bearer_token_provider(credential, "https://management.azure.com/.default")
    token = token_provider()

    mcp_target = f"{search_endpoint}/knowledgebases/{kb_name}/mcp?api-version=2025-11-01-preview"
    arm_url = (
        f"https://management.azure.com{project_resource_id}"
        f"/connections/{connection_name}?api-version=2025-10-01-preview"
    )

    response = _requests.put(
        arm_url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "name": connection_name,
            "type": "Microsoft.MachineLearningServices/workspaces/connections",
            "properties": {
                "authType": "ProjectManagedIdentity",
                "category": "RemoteTool",
                "target": mcp_target,
                "isSharedToAll": True,
                "audience": "https://search.azure.com/",
                "metadata": {"ApiType": "Azure"},
            },
        },
    )
    response.raise_for_status()
    action = "updated" if response.status_code == 200 else "created"
    connection_id = f"{project_resource_id}/connections/{connection_name}"
    logger.info("Project connection '%s' %s. ID: %s", connection_name, action, connection_id)
    return connection_id


# ── Agent management ──────────────────────────────────────────────────────────

def create_or_update_agent(
    project: AIProjectClient | None = None,
    model: str | None = None,
    agent_name: str = AGENT_NAME,
    use_foundry_iq: bool = False,
    kb_connection_id: str | None = None,
) -> str:
    """
    Register (or update) the Educational Consultant agent in Foundry.
    Returns the agent name used as a stable identifier.

    Note: this call requires 'Azure AI Developer' role on the AI Services resource.

    Set use_foundry_iq=True to attach the FoundryIQ MCP knowledge base tool.
    kb_connection_id should be the full ARM connection resource ID returned by
    create_project_connection().
    """
    project = project or get_project_client()
    model   = model or os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4o-mini")

    if use_foundry_iq:
        kb_name         = os.environ.get("FOUNDRY_IQ_KB_NAME", "")
        search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
        mcp_url = f"{search_endpoint}/knowledgebases/{kb_name}/mcp?api-version=2025-11-01-preview"
        mcp_tool = MCPTool(
            server_label="course-catalog-kb",
            server_url=mcp_url,
            require_approval="never",
            allowed_tools=["knowledge_base_retrieve"],
            project_connection_id=kb_connection_id or os.environ.get("FOUNDRY_KB_CONNECTION_NAME", ""),
        )
        definition = PromptAgentDefinition(
            model=model,
            instructions=FOUNDRY_IQ_SYSTEM_PROMPT,
            tools=[mcp_tool],
        )
        description = "NovaTech AI Educational Consultant — FoundryIQ MCP knowledge base"
    else:
        definition = PromptAgentDefinition(
            model=model,
            instructions=SYSTEM_PROMPT,
        )
        description = "NovaTech AI Educational Consultant — recommends courses based on learner profile"

    result = project.agents.create_version(
        agent_name=agent_name,
        definition=definition,
        description=description,
    )

    logger.info(
        "Agent '%s' registered. Version: %s  Model: %s",
        result.name, result.version, model,
    )
    _save_agent_name(result.name)
    return result.name


def get_agent_name() -> str:
    if os.environ.get("FOUNDRY_AGENT_NAME"):
        return os.environ["FOUNDRY_AGENT_NAME"]
    name_file = Path(__file__).resolve().parents[2] / ".agent_name"
    if name_file.exists():
        return name_file.read_text().strip()
    return AGENT_NAME


def _save_agent_name(name: str) -> None:
    name_file = Path(__file__).resolve().parents[2] / ".agent_name"
    name_file.write_text(name)
