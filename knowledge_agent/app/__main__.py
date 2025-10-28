import logging
import os
import sys

import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks.base_push_notification_sender import BasePushNotificationSender
from a2a.server.tasks.inmemory_push_notification_config_store import InMemoryPushNotificationConfigStore


from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from app.agent import KnowledgeAgent
from app.agent_executor import KnowledgeAgentExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


def main():
    """Starts Knowledge's Agent server."""
    # host = "0.0.0.0" 
    host = "localhost" 
    port = 8002  
    try:
        if not os.getenv("GROQ_API_KEY"):
            raise MissingAPIKeyError("GROQ_API_KEY environment variable not set.")

        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        
        skill = AgentSkill(
            id="answer_user_questions",
            name="Answer User Questions",
            description="Answers user queries by using internal LLM or external tools as needed",
            tags=["general knowledge", "cybersecurity", "general Q&A"],
            examples=[
                "Tell me about React JS",
                "How does a firewall work?",
                "Tell me about CVE-2025-1234",
                "What is MITRE ATT&CK T1059?",
                "Explain ransomware techniques"
            ],
        )
        agent_card = AgentCard(
            name="Knowledge_Agent",
            description="Answers user questions using LLM reasoning and external knowledge tools",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=KnowledgeAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=KnowledgeAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        httpx_client = httpx.AsyncClient()
        config_store = InMemoryPushNotificationConfigStore()
        request_handler = DefaultRequestHandler(
            agent_executor=KnowledgeAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=config_store,
            push_sender=BasePushNotificationSender(
                httpx_client=httpx_client, 
                config_store=config_store
            ),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
