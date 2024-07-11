from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a single message in a conversation."""

    role: str
    content: str


class MessageData:
    def __init__(self, deserialized_msg):
        _message = deserialized_msg.get("message", {})

        if not _message:
            _message = {}

        self.status = _message.get("status", None)
        self.role = _message.get("author", {}).get("role", None)
        self.content = _message.get("content", {}).get("parts", [None])[0]
        self.finish_type = (
            _message.get("metadata", {}).get("finish_details", {}).get("type", None)
        )
        self.is_complete = _message.get("metadata", {}).get("is_complete", False)
        self.error = _message.get("error", None)

    def has_message(self):
        """Checks if the deserialized message contains a valid 'message' key."""
        return bool(
            self.status or self.role or self.content or self.finish_type or self.error
        )


class ChatCompletionRequest(BaseModel):
    """Represents a chat completion request to the OpenAI API."""

    messages: List[Message]
    model: str = "gpt-3.5-turbo"
    stream: bool = False


class ChatCompletionChunk(BaseModel):
    """Represents a single chat completion chunk in a streaming response."""

    id: str
    created: int
    model: str
    object: str
    choices: List[Dict]


class Session(BaseModel):
    """Represents a session  object for interacting with the OpenAI API."""

    device_id: str
    persona: str
    arkose: dict
    turnstile: dict
    proofofwork: dict
    token: str
    headers: dict
    cookies: dict


class EmbeddingsRequest(BaseModel):
    """Represents an embeddings request to the OpenAI API."""

    input: Union[str, List[str], List[List[int]]] = Field(
        ..., description="Input text to embed, encoded as a string or array of tokens."
    )
    model: str = Field(..., description="ID of the model to use.")
    encoding_format: str = Field(
        "float", description="The format to return the embeddings in."
    )
    dimensions: int = Field(
        None,
        description="The number of dimensions the resulting output embeddings should have.",
    )
    user: str = Field(
        None, description="A unique identifier representing your end-user."
    )


class Embedding(BaseModel):
    """Represents an embedding vector returned by embedding endpoint."""

    object: str = Field("embedding", description="The object type.")
    index: int = Field(
        ..., description="The index of the embedding in the list of embeddings."
    )
    embedding: List[float] = Field(
        ..., description="The embedding vector, which is a list of floats."
    )


class EmbeddingsResponse(BaseModel):
    """Represents a response from the OpenAI embeddings endpoint."""

    object: str = Field("list", description="The object type.")
    data: List[Embedding] = Field(..., description="A list of embedding objects.")
    model: str = Field(
        ..., description="The ID of the model used to generate the embeddings."
    )
    usage: dict = Field(..., description="The usage statistics for the request.")
