from email.quoprimime import header_decode
import os
import time
from datetime import datetime, timedelta
import uuid
import re
from typing import Optional, AsyncIterator
import base64
from random import choice
import hashlib
import asyncio
import json
import httpx
import undetected_chromedriver as uc
from pathlib import Path

from models import Session, ChatCompletionRequest
from settings import (BASE_URL,API_URL,RETRY_WAIT_SECONDS,NEW_SESSION_RETRIES,HEADERS,PROXY_PROTOCOL,PROXY_HOST,PROXY_PORT,PROXY_AUTH,PROXY_USERNAME,PROXY_PASSWORD)

# cookies and user agent to bypass Cloudflare and ChatGPT rate limit
_cookies = None
_user_agent = None

# HTTPX client with predefined configuration
client = httpx.AsyncClient()
client.headers.update(HEADERS)
client.verify = False

if os.getenv("PROXY") == "true":
    proxies = {
        "http://": f"{PROXY_PROTOCOL}://{PROXY_HOST}:{PROXY_PORT}",
        "https://": f"{PROXY_PROTOCOL}://{PROXY_HOST}:{PROXY_PORT}",
    }
    
    if PROXY_AUTH == "true":
        auth = httpx.BasicAuth(PROXY_USERNAME, PROXY_PASSWORD)
    else:
        auth = None

    client.proxies = proxies
    client.auth = auth


def extract_version_and_os(user_agent: str) -> tuple[str, str]:
    """Extract OS and Chrome Browser Version from Chrome User Agent String."""
    # Extract Chrome version
    chrome_version_pattern = r"Chrome\/(\d+)"
    chrome_version_match = re.search(chrome_version_pattern, user_agent)
    chrome_version = chrome_version_match.group(1) if chrome_version_match else "112"

    # Extract operating system
    os_pattern = r"\(([^;]+)"
    os_match = re.search(os_pattern, user_agent)
    operating_system = os_match.group(1) if os_match else None

    # Extract the general name of the operating system
    operating_system = operating_system.split(' ')[0] if operating_system else "Windows"

    return (chrome_version, operating_system)

def get_cookies_and_user_agent(url, timeout=60):
    """
    Navigates to the given URL, solves Cloudflare captcha, and returns cookies and user agent.

    Args:
        url (str):  The URL to navigate to.
        timeout (int, optional): Timeout in seconds for captcha solving. Defaults to 60.

    Returns:
         tuple: A tuple containing:
            - list: A list of cookies as dictionaries.
            - str: The user agent of the browser.
    """
    options = uc.ChromeOptions()
    #options.add_argument("--window-position=-2000,0")  # Start Chrome Hidden
    options.add_argument("--start-minimized")  # Start Chrome minimized
    # user_data_dir=Path(__file__).parent / "chrome_data", no_sandbox=False, user_multi_procs=True, use_subprocess=False
    driver = uc.Chrome(options=options,header_decode=False)
    driver.get(url)

    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            driver.quit()
            raise TimeoutError(
                "Timeout: Cloudflare captcha  not solved within the time limit."
            )

        try:
            driver.find_element(by=uc.By.ID, value="challenge-form")
            time.sleep(1)  # Wait for captcha to be solved
        except Exception:
            break  # Captcha solved

    cookies = driver.get_cookies()
    user_agent = driver.execute_script("return navigator.userAgent")

    driver.quit()  # Close the driver

    return cookies, user_agent

async def get_new_session(retries: int = 0) -> Optional[Session]:
    """Gets a new session ID and token from the OpenAI API."""
    global _cookies, _user_agent

    try:
        if not _cookies or not _user_agent:
            _cookies , _user_agent = get_cookies_and_user_agent(BASE_URL)
        
        _version , _operation_system = extract_version_and_os(_user_agent)
        
        device_id = str(uuid.uuid4())
        
        _headers = {
            **HEADERS,
            "user-agent": _user_agent,
            "sec-ch-ua": f'"{_user_agent}";v="{_version}"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": f'"{_operation_system}"',
            "oai-device-id": device_id,
        }

        # Generate the cookies dictionary
        cookies = {cookie['name']: cookie['value'] for cookie in _cookies}
        
        response = await client.post(
            f"{BASE_URL}/backend-anon/sentinel/chat-requirements", headers=_headers,
            cookies= cookies
        )

        response.raise_for_status()
        
        session_data = response.json()
        session_data["device_id"] = device_id
        session_data["headers"] = _headers
        session_data["cookies"] = cookies
        
        return Session(**session_data)
    except Exception as e:
        print(f"Error getting a new session (attempt {retries+1}/{NEW_SESSION_RETRIES}): {str(e)}")
        
        if retries < NEW_SESSION_RETRIES:
            await asyncio.sleep(RETRY_WAIT_SECONDS)
            
            _cookies = None
            _user_agent = None
            
            return await get_new_session(retries + 1)
        
        return None

def generate_proof_token(seed: str, diff: str, user_agent: str) -> str:
    """Generates a proof token for the OpenAI API."""
    cores = [2, 4, 6, 8, 12, 16, 24, 28, 32]
    screens = [3000, 4000, 6000, 8000]

    core = choice(cores)
    screen = choice(screens)

    now = datetime.utcnow() - timedelta(hours=8)
    parse_time = now.strftime("%a, %d %b %Y %H:%M:%S GMT-0500 (Eastern Time)")

    config = [core + screen, parse_time, 4294705152, 0, user_agent]

    diff_len = len(diff) // 2

    for i in range(100000):
        config[3] = i
        json_data = json.dumps(config)
        base = base64.b64encode(json_data.encode()).decode()
        hash_value = hashlib.sha3_512((seed + base).encode()).hexdigest()

        if hash_value[:diff_len] <= diff:
            result = "gAAAAAB" + base
            return result

    fallback_base = base64.b64encode(seed.encode()).decode()
    return "gAAAAABwQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D" + fallback_base

async def send_chat_completion_request(request: ChatCompletionRequest, session: Session) -> AsyncIterator[str]:
    """Sends a chat completion request to the OpenAI API, handling streaming if requested."""

    _proof_token = generate_proof_token(
        session.proofofwork["seed"],
        session.proofofwork["difficulty"],
        session.headers.get("user-agent"),
    )

    _openai_request_body = {
        "action": "next",
        "messages": [
            {
                "author": {"role": m.role},
                "content": {"content_type": "text", "parts": [m.content]},
            }
            for m in request.messages
        ],
        "parent_message_id": str(uuid.uuid4()),
        "model": "gpt-3.5-turbo",  # request.model,
        "timezone_offset_min": -180,
        "suggestions": [],
        "history_and_training_disabled": True,
        "conversation_mode": {"kind": "primary_assistant"},
        "websocket_request_id": str(uuid.uuid4()),
    }

    _request_headers = {
        **session.headers,
        "openai-sentinel-chat-requirements-token": session.token,
        "openai-sentinel-proof-token": _proof_token,
    }

    async with client.stream(
        "POST",
        API_URL,
        json=_openai_request_body,
        headers=_request_headers,
        cookies=session.cookies
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_lines():
            yield chunk