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
from app.agent import DetectionAgent
from app.agent_executor import DetectionAgentExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


def main():
    """Starts Detection Agent server."""
    # host = "0.0.0.0" 
    host = "localhost" 
    port = 8001  # changed port to avoid collision
    try:
        if not os.getenv("VIRUSTOTAL_API_KEY"):
            raise MissingAPIKeyError("VIRUSTOTAL_API_KEY environment variable not set.")

        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        
        skills = [
            AgentSkill(
                id="detect_malicious_url",
                name="Detect Malicious URL",
                description="Detects malware or malicious content from URLs using VirusTotal API",
                tags=["security", "malware", "threat detection"],
                examples=["Check this URL for malware: http://example.com/malicious"]
            ),
            AgentSkill(
                id="detect_malicious_file",
                name="Detect Malicious File",
                description="Scans files for malware using VirusTotal API",
                tags=["security", "malware", "threat detection"],
                examples=["Scan this file for viruses: report.pdf"]
            ),
            AgentSkill(
                id="detect_malicious_email",
                name="Detect Malicious Email",
                description="Analyzes emails for threats",
                tags=["security", "malware", "threat detection"],
                examples=["Analyze this email for threats"]
            )
        ]

        agent_card = AgentCard(
            name="Detection_Agent",
            description="Detects malware and malicious content using VirusTotal API",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=DetectionAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=DetectionAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=skills,
        )

        httpx_client = httpx.AsyncClient()
        config_store = InMemoryPushNotificationConfigStore()
        request_handler = DefaultRequestHandler(
            agent_executor=DetectionAgentExecutor(),
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

