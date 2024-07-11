import os
from dotenv import load_dotenv
from pathlib import Path

env_file = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_file, override=True)

# Constants for server and API configuration
PORT = int(os.getenv("SERVER_PORT", "3040"))
# Load API key from environment variable
API_KEY = os.getenv("API_KEY")

# Constants for API configuration
BASE_URL = "https://chat.openai.com"
API_URL = f"{BASE_URL}/backend-anon/conversation"
TOKEN_REFRESH_INTERVAL = int(
    os.getenv("TOKEN_REFRESH_INTERVAL", "60")
)  # Interval to refresh token in seconds
RETRY_WAIT_SECONDS = int(
    os.getenv("RETRY_WAIT_SECONDS", "6")
)  # Wait time in seconds after an error on getting session
NEW_SESSION_RETRIES = int(os.getenv("NEW_SESSION_RETRIES", "5"))

# Headers for API requests
HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "content-type": "application/json",
    "oai-language": "en-US",
    "origin": BASE_URL,
    "pragma": "no-cache",
    "referer": BASE_URL,
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
}

PROXY_PROTOCOL = os.getenv("PROXY_PROTOCOL")
PROXY_HOST = os.getenv("PROXY_HOST")
PROXY_PORT = os.getenv("PROXY_PORT")
PROXY_AUTH = os.getenv("PROXY_AUTH")
PROXY_USERNAME = os.getenv("PROXY_USERNAME")
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")

# Embedding Model Name & Path
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
