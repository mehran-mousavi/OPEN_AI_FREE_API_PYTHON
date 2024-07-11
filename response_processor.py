import json
import re
from typing import AsyncIterator, Tuple, Optional, List
import time
import random

from models import ChatCompletionChunk, MessageData, Message


# Function to generate a random completion ID
def generate_completion_id(prefix: str = "chatcmpl-") -> str:
    """Generates a unique identifier for a chat  completion.
    
    The ID is a string that begins with the given prefix (default "chatcmpl-")
    followed by  28 random alphanumeric characters. 

    Args:
        prefix: An optional prefix for the ID (default "chatcmpl-").

    Returns :
            A unique chat completion ID.
    """
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    length = 28
    for _ in range(length):
        prefix += random.choice(characters)
    return prefix


async def process_message_chunks(
    request_messages: List[Message],
    response_stream: AsyncIterator[str],
) -> AsyncIterator[Tuple[Optional[str], str, Optional[str]]]:
    """
     Processes message chunks from the response stream, yielding new content and finish reason.

    Args:
        response: The asynchronous response object.

    Yields:
        Tuple[str, str]: A tuple containing the new content  and the finish reason.
    """
    
    # Initialize the response full content container
    accumulated_response_text = ""  

    async for message_chunk in response_stream:
        if message_chunk.strip("\n").strip().startswith("data: [DONE]"):
            break

        if not message_chunk.strip("\n").strip().startswith("data:"):
            continue

        if re.match(r"^data:\s\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{6}$", message_chunk):
            continue

        try:
            _deserialized_msg = json.loads(message_chunk.strip("\n").strip().strip("data:").strip())

            _message_data = MessageData(_deserialized_msg)

            if not _message_data.has_message():
                continue

            if _message_data.error:
                yield None, "stop", _message_data.error
                break

            content = _message_data.content

            # Exclude requested messages from response
            request_contents = {msg.content for msg in request_messages}
            if content in request_contents:
                continue


            completion_chunk = content[len(accumulated_response_text) :]

            finish_reason = None  # Initialize finish_reason
            if _message_data.status == "finished_successfully":
                finish_reason = _message_data.finish_type

            yield completion_chunk, finish_reason, _message_data.error

            if finish_reason:  # Check if there's a finish reason
                break  # Exit the loop if the response is finished

            accumulated_response_text += completion_chunk  # Update tracked content

        except Exception as e:
            print(f"Error on chunk processing: {str(e)}")
            print(str(message_chunk))
            raise e


async def stream_response_generator(
    request_messages: List[Message],
    response_stream: AsyncIterator[str],
) -> AsyncIterator[ChatCompletionChunk]:

    completion_id = generate_completion_id()
    response_generation_time = int(time.time())
    chunk_index = 0

    async for content_chunk, finish_reason, error in process_message_chunks(
        request_messages, response_stream
    ):
        yield ChatCompletionChunk(
            **{
                "id": completion_id,
                "created": response_generation_time,
                "model": "gpt-3.5-turbo",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "finish_reason": finish_reason,
                        "index": chunk_index,
                        "delta": {
                            "content": content_chunk if not error else error,
                        },
                    }
                ],
            }
        )
        chunk_index += 1


async def process_streaming_request(
    request_messages: List[Message],
    response_generator: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Processes a streaming chat completion  response, yielding ChatCompletionChunk objects."""
    async for stream_response_chunk in stream_response_generator(
        request_messages, response_generator
    ):
        yield f"data: {json.dumps(stream_response_chunk.model_dump())}\n\n"


async def process_normal_request(
    request_messages: List[Message],
    response_generator: AsyncIterator[str],
) -> str:
    """Processes a non-streaming chat completion response."""
    finish_reason = None
    response_buffer = ""

    async for new_content, _finish_reason, error in process_message_chunks(
        request_messages, response_generator
    ):
        finish_reason = _finish_reason
        response_buffer += new_content if not error else error

    request_id = generate_completion_id()
    created = int(time.time())

    response_data = {
        "id": request_id,
        "created": created,
        "model": "gpt-3.5-turbo",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": finish_reason,
                "index": 0,
                "message": {
                    "content": response_buffer,
                    "role": "assistant",
                },
            }
        ],
    }

    return json.dumps(response_data)
