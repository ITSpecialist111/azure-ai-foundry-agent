import azure.functions as func
import logging
import json
import os
import time
import uuid
import requests
from azure.identity import DefaultAzureCredential, AzureError


#  ──────────────────────────────────────────────────────────────────────────
#  Helpers
#  ──────────────────────────────────────────────────────────────────────────
def _parse_conn_string() -> tuple[str, str]:
    """
    Expected format in AIProjectConnString:
        https://<endpoint>; <project_id>; <resource_group>; <project_name>
    Only the first two segments are required for the REST calls.
    """
    conn = os.environ.get("AIProjectConnString", "")
    if not conn:
        raise ValueError("Environment variable AIProjectConnString is missing.")
    parts = [p.strip() for p in conn.split(";")]
    if len(parts) < 2:
        raise ValueError("AIProjectConnString is not in '<endpoint>;<project_id>;…' format.")
    endpoint = parts[0]
    if not endpoint.startswith("http"):
        endpoint = "https://" + endpoint
    endpoint = endpoint.rstrip("/")            # → https://eastus.api.azureml.ms
    project_id = parts[1]                      # → bfec7165-…
    return endpoint, project_id


def _get_auth_headers() -> dict:
    cred = DefaultAzureCredential()
    token = cred.get_token("https://cognitiveservices.azure.com/.default")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token.token}"
    }


#  ──────────────────────────────────────────────────────────────────────────
#  Azure Function
#  ──────────────────────────────────────────────────────────────────────────
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="agent_httptrigger")
def agent_httptrigger(req: func.HttpRequest) -> func.HttpResponse:          # noqa: C901
    """
    Query parameters or JSON body:
        message   – (required) user text
        agentid   – (required) Azure AI Agent / Assistant ID  (e.g. asst_abc123)
        threadid  – (optional) re-use an existing thread
    """
    # ── 1. Input handling
    message = req.params.get("message")
    agent_id = req.params.get("agentid")
    thread_id = req.params.get("threadid")

    if not message or not agent_id:
        try:
            body = req.get_json()
        except ValueError:
            body = {}
        message = message or body.get("message")
        agent_id = agent_id or body.get("agentid")
        thread_id = thread_id or body.get("threadid")

    if not message or not agent_id:
        return func.HttpResponse(
            "Pass both 'message' and 'agentid' as query parameters or in JSON.",
            status_code=400
        )

    try:
        # ── 2. Connection info & auth header
        endpoint, project_id = _parse_conn_string()
        base = f"{endpoint}/openai/projects/{project_id}/agents/{agent_id}"
        headers = _get_auth_headers()

        # ── 3. Create a thread if none supplied
        if not thread_id:
            r = requests.post(f"{base}/threads", headers=headers, json={})
            r.raise_for_status()
            thread_id = r.json()["id"]
            logging.info("Created new thread %s", thread_id)

        # ── 4. Add the user message
        msg_payload = {"role": "user", "content": message}
        r = requests.post(f"{base}/threads/{thread_id}/messages",
                          headers=headers, json=msg_payload)
        r.raise_for_status()

        # ── 5. Kick off a run
        r = requests.post(f"{base}/threads/{thread_id}/runs",
                          headers=headers, json={})
        r.raise_for_status()
        run_id = r.json()["id"]

        # ── 6. Poll run status (≤ 30 s)
        status_url = f"{base}/threads/{thread_id}/runs/{run_id}"
        for _ in range(30):
            run = requests.get(status_url, headers=headers).json()
            if run["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(1)

        if run["status"] != "completed":
            return func.HttpResponse(f"Run status: {run['status']}", status_code=500)

        # ── 7. Fetch assistant messages (latest first)
        r = requests.get(f"{base}/threads/{thread_id}/messages",
                         headers=headers, params={"order": "desc"})
        r.raise_for_status()
        messages = r.json().get("data", [])
        assistant_reply = next(
            (
                " ".join(
                    part["text"]["value"]
                    for part in m["content"]
                    if part.get("type") == "text"
                )
            )
            for m in messages if m["role"] == "assistant"
        )
        return func.HttpResponse(assistant_reply, status_code=200,
                                 mimetype="text/plain")

    # ── 8. Error handling
    except AzureError as ex:
        logging.error("Azure identity error: %s", ex)
        return func.HttpResponse(f"Auth error: {ex}", status_code=500)

    except requests.HTTPError as ex:
        logging.error("HTTP %s – %s", ex.response.status_code, ex.response.text)
        return func.HttpResponse(ex.response.text, status_code=ex.response.status_code)

    except Exception as ex:
        logging.error("Unexpected: %s", ex)
        return func.HttpResponse(f"Internal Server Error: {ex}", status_code=500)
