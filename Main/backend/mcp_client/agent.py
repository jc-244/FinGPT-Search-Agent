# backend/mcp_client/agent.py

import json
import os
import shlex
from typing import Optional, List
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from agents import Agent
from agents.mcp import MCPServerSse, MCPServerStdio
from agents.model_settings import ModelSettings
import logging

# Load .env from the backend root directory
from pathlib import Path
backend_dir = Path(__file__).resolve().parent.parent
load_dotenv(backend_dir / '.env')

# Import model configuration resolver
import sys
sys.path.insert(0, str(backend_dir))
from datascraper.models_config import get_model_config

# Import Playwright tools
try:
    from .playwright_tools import (
        navigate_to_url,
        get_page_text,
        click_element,
        fill_form_field,
        press_enter,
        get_current_url,
        wait_for_element,
        extract_links,
        cleanup_browser
    )
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    logging.warning("Playwright tools not available. Install playwright to enable browser automation.")
    PLAYWRIGHT_AVAILABLE = False

USER_ONLY_MODELS = {"o3-mini"}

# Default prompt for standard usage
DEFAULT_PROMPT = (
    "You are a helpful financial assistant. "
    "You have access to tools that can help you answer questions. "
    "ALWAYS use the available tools when they are relevant to the user's request. "
)

# Enhanced prompt when Playwright tools are enabled
PLAYWRIGHT_PROMPT = (
    "You are a helpful financial assistant with web browsing capabilities. "
    "You can navigate websites, extract information, click elements, and fill forms. "
    "When asked to research financial information:\n"
    "1. Use navigate_to_url to visit relevant pages\n"
    "2. Use get_page_text to extract content\n"
    "3. Use click_element and fill_form_field for interactive elements\n"
    "4. Use extract_links to find relevant pages to explore\n"
    "5. Use wait_for_element when pages have dynamic content\n"
    "Use tools when they help answer the user's question. If you can answer from knowledge, do so."
)

def create_domain_restricted_prompt(domain: str, current_url: str) -> str:
    """Create a prompt for domain-restricted navigation."""
    return (
        f"You are a financial assistant helping the user navigate and understand: {domain}\n"
        f"The user is currently on: {current_url}\n\n"
        f"IMPORTANT - You can ONLY navigate within {domain}. Never attempt to visit external domains.\n\n"
        "To answer user questions, follow this workflow:\n"
        f"1. FIRST: Navigate to {current_url} using navigate_to_url() to see what's currently on the page\n"
        "2. THEN: Use get_page_text() to read the content\n"
        "3. IF the answer is on the current page: Answer the question directly\n"
        "4. IF the answer is NOT on the current page:\n"
        f"   - Use extract_links() to find relevant pages on {domain}\n"
        "   - Navigate to the most relevant page using navigate_to_url()\n"
        "   - Use get_page_text() to extract the information\n"
        "   - Continue navigating as needed to find the answer\n\n"
        "You MUST actively use navigation tools to find information. Don't say 'I don't know' "
        "without first navigating and searching the website.\n\n"
        "Available tools:\n"
        "- navigate_to_url(url): Visit a page within the domain\n"
        "- get_page_text(): Read content from current page\n"
        "- extract_links(): Get all links from current page\n"
        "- click_element(selector): Click elements on the page\n"
        "- fill_form_field(selector, value): Fill input fields\n"
    )

@asynccontextmanager
async def create_fin_agent(model: str = "gpt-4o",
                          system_prompt: Optional[str] = None,
                          use_playwright: bool = False,
                          restricted_domain: Optional[str] = None,
                          current_url: Optional[str] = None,
                          user_timezone: Optional[str] = None,
                          user_time: Optional[str] = None):
    """
    Create a financial agent with optional Playwright browser automation tools.

    Args:
        model: The OpenAI model to use (e.g., 'gpt-4o', 'o4-mini')
        system_prompt: Custom system prompt (if None, uses default based on tools)
        use_playwright: Whether to enable Playwright browser automation tools
        restricted_domain: Restrict navigation to this domain (e.g., "finance.yahoo.com")
        current_url: Current webpage URL for context
        user_timezone: User's IANA timezone (e.g., "America/New_York")
        user_time: User's current time in ISO format

    Yields:
        Agent instance configured with appropriate tools
    """
    # Select appropriate prompt based on context
    if system_prompt:
        instructions = system_prompt
    elif use_playwright and PLAYWRIGHT_AVAILABLE and restricted_domain:
        # Domain-restricted mode: emphasize staying on current website
        instructions = create_domain_restricted_prompt(restricted_domain, current_url or restricted_domain)
    elif use_playwright and PLAYWRIGHT_AVAILABLE:
        # General Playwright mode
        instructions = PLAYWRIGHT_PROMPT
    else:
        instructions = DEFAULT_PROMPT

    # Add timezone and time information to instructions
    if user_timezone or user_time:
        from datetime import datetime
        import pytz

        time_info_parts = []
        if user_timezone and user_time:
            try:
                # Parse ISO time and convert to user's timezone
                utc_time = datetime.fromisoformat(user_time.replace('Z', '+00:00'))
                user_tz = pytz.timezone(user_timezone)
                local_time = utc_time.astimezone(user_tz)

                time_info_parts.append(f"User's timezone: {user_timezone}")
                time_info_parts.append(f"Current local time for user: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            except Exception as e:
                logging.warning(f"Error formatting time info in agent: {e}")
                if user_timezone:
                    time_info_parts.append(f"User's timezone: {user_timezone}")
        elif user_timezone:
            time_info_parts.append(f"User's timezone: {user_timezone}")

        if time_info_parts:
            instructions = f"{instructions}\n\n[TIME CONTEXT]: {' | '.join(time_info_parts)}"

    # Resolve model ID to actual model name via models_config
    model_config = get_model_config(model)
    if not model_config:
        logging.warning(f"Model ID '{model}' not found in config, using as-is")
        actual_model = model
    else:
        actual_model = model_config["model_name"]
        logging.info(f"Model resolution: {model} -> {actual_model}")

    # Build tools list
    tools: List = []

    # Initialize MCP servers if configured
    connected_mcp_servers: List = []
    mcp_servers_for_agent: List = []

    async def _cleanup_mcp_servers():
        """Cleanup MCP servers that were connected before an initialization error."""
        for server in connected_mcp_servers:
            try:
                await server.cleanup()
            except Exception as cleanup_error:
                logging.error(f"Error during MCP server cleanup: {cleanup_error}")

    mcp_url_env = os.getenv("MCP_SERVER_URL", "").strip()
    if mcp_url_env:
        headers = None
        headers_json = os.getenv("MCP_SERVER_HEADERS_JSON", "").strip()
        if headers_json:
            try:
                headers = json.loads(headers_json)
            except json.JSONDecodeError as decode_error:
                logging.error(f"Failed to parse MCP_SERVER_HEADERS_JSON: {decode_error}")

        timeout = os.getenv("MCP_SERVER_TIMEOUT")
        sse_read_timeout = os.getenv("MCP_SERVER_SSE_READ_TIMEOUT")

        for url in [value.strip() for value in mcp_url_env.split(",") if value.strip()]:
            params = {"url": url}
            if headers:
                params["headers"] = headers
            if timeout:
                try:
                    params["timeout"] = float(timeout)
                except ValueError:
                    logging.error("MCP_SERVER_TIMEOUT must be a number of seconds; ignoring value.")
            if sse_read_timeout:
                try:
                    params["sse_read_timeout"] = float(sse_read_timeout)
                except ValueError:
                    logging.error("MCP_SERVER_SSE_READ_TIMEOUT must be a number of seconds; ignoring value.")

            server = MCPServerSse(params)
            try:
                await server.connect()
                logging.info(f"Connected to MCP SSE server '{server.name}' ({url})")
                connected_mcp_servers.append(server)
            except Exception as connect_error:
                logging.error(f"Failed to connect to MCP SSE server at {url}: {connect_error}")
                await _cleanup_mcp_servers()
                raise

    mcp_command = os.getenv("MCP_SERVER_COMMAND", "").strip()
    if mcp_command:
        args_env = os.getenv("MCP_SERVER_ARGS", "").strip()
        args = shlex.split(args_env) if args_env else []

        env_json = os.getenv("MCP_SERVER_ENV_JSON", "").strip()
        env_vars = None
        if env_json:
            try:
                env_vars = json.loads(env_json)
            except json.JSONDecodeError as decode_error:
                logging.error(f"Failed to parse MCP_SERVER_ENV_JSON: {decode_error}")

        cwd = os.getenv("MCP_SERVER_CWD", "").strip() or None
        encoding = os.getenv("MCP_SERVER_ENCODING", "").strip() or None
        encoding_error_handler = os.getenv("MCP_SERVER_ENCODING_ERROR_HANDLER", "").strip() or None

        params = {"command": mcp_command}
        if args:
            params["args"] = args
        if env_vars:
            params["env"] = env_vars
        if cwd:
            params["cwd"] = cwd
        if encoding:
            params["encoding"] = encoding
        if encoding_error_handler:
            params["encoding_error_handler"] = encoding_error_handler

        stdio_server = MCPServerStdio(params)
        try:
            await stdio_server.connect()
            logging.info(f"Connected to MCP stdio server '{stdio_server.name}'")
            if mcp_command == "docker" and any("sec-edgar-mcp" in arg for arg in ([mcp_command] + args)):
                logging.info("[MCP DEBUG] Connected to SEC EDGAR MCP server via docker")
            connected_mcp_servers.append(stdio_server)
        except Exception as connect_error:
            logging.error(
                f"Failed to connect to MCP stdio server using command '{mcp_command}': {connect_error}"
            )
            await _cleanup_mcp_servers()
            raise

    if connected_mcp_servers:
        mcp_servers_for_agent = connected_mcp_servers

    if use_playwright and PLAYWRIGHT_AVAILABLE:
        # Store domain restriction in global context for tools to access
        from . import playwright_tools
        playwright_tools._RESTRICTED_DOMAIN = restricted_domain
        playwright_tools._CURRENT_URL = current_url

        tools = [
            navigate_to_url,
            get_page_text,
            click_element,
            fill_form_field,
            press_enter,
            get_current_url,
            wait_for_element,
            extract_links
        ]
        domain_info = f" (restricted to {restricted_domain})" if restricted_domain else ""
        logging.info(f"Playwright tools enabled{domain_info}: {len(tools)} tools available")
    elif use_playwright and not PLAYWRIGHT_AVAILABLE:
        logging.error("Playwright requested but not available. Running without browser tools.")

    try:
        # Create agent with or without tools
        agent = Agent(
            name="FinGPT Search Agent",
            instructions=instructions,
            model=actual_model,  # Remember to use resolved model name, not frontend ID
            tools=tools if tools else [],
            mcp_servers=mcp_servers_for_agent,
            model_settings=ModelSettings(
                tool_choice="auto" if tools else None
            )
        )

        yield agent

    finally:
        # Cleanup MCP servers
        for server in connected_mcp_servers:
            try:
                await server.cleanup()
                logging.info(f"Disconnected MCP server '{server.name}'")
            except Exception as cleanup_error:
                logging.error(
                    f"Error disconnecting MCP server '{getattr(server, 'name', 'unknown')}': {cleanup_error}"
                )

        # Cleanup browser if it was used
        if use_playwright and PLAYWRIGHT_AVAILABLE:
            try:
                # Clear domain restriction
                from . import playwright_tools
                playwright_tools._RESTRICTED_DOMAIN = None
                playwright_tools._CURRENT_URL = None

                await cleanup_browser()
            except Exception as e:
                logging.error(f"Browser cleanup error: {e}")
