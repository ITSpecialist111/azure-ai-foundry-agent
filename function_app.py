import azure.functions as func
import logging
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
import json
import os

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="agent_httptrigger")
def agent_httptrigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    message = req.params.get('message')
    agentid = req.params.get('agentid')
    threadid = req.params.get('threadid')
    
    if not message or not agentid:
        try:
            req_body = req.get_json()
        except ValueError:
            req_body = None

        if req_body:
            message = req_body.get('message')
            agentid = req_body.get('agentid')
            threadid = req_body.get('threadid')

    if not message or not agentid:
        return func.HttpResponse(
            "Pass in a message and agentid in the query string or in the request body for a personalized response.",
            status_code=400
        )

    conn_str = os.environ.get("AIProjectConnString", "")
    if not conn_str:
        logging.error("AIProjectConnString is not set in local.settings.json or environment variables.")
        return func.HttpResponse(
            "Internal Server Error: Missing AIProjectConnString.",
            status_code=500
        )

    try:
        project_client = AIProjectClient(
            endpoint=conn_str.split(";")[0],
            project_id=conn_str.split(";")[1],
            resource_group_name=conn_str.split(";")[2],
            project_name=conn_str.split(";")[3],
            credential=DefaultAzureCredential()
        )


        agent = project_client.agents.get_agent(agentid)
        if not agent:
            logging.error(f"Agent with ID {agentid} not found.")
            return func.HttpResponse(
                f"Agent with ID {agentid} not found.",
                status_code=404
            )

        thread = project_client.agents.get_thread(threadid) if threadid else None
        if not thread:
            thread = project_client.agents.create_thread()

        project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=message,
        )

        project_client.agents.create_and_process_run(
            thread_id=thread.id,
            agent_id=agent.id
        )

        messages = project_client.agents.list_messages(thread_id=thread.id)
        assistant_messages = [m for m in messages.data if m["role"] == "assistant"]
        if assistant_messages:
            assistant_message = assistant_messages[-1]
            assistant_text = " ".join(
                part["text"]["value"] for part in assistant_message["content"] if "text" in part
            )
        else:
            assistant_text = "No assistant message found."

        return func.HttpResponse(
            assistant_text,
            status_code=200,
            mimetype="text/plain"
        )
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return func.HttpResponse(
            "Internal Server Error: " + str(e),
            status_code=500
        )
