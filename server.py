import torch
from transformers import AutoModel

import uvicorn
import traceback

# from setting_loader import PORT, API_KEY
from fastapi import FastAPI, HTTPException, Header, Request, Response, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from models import (
    ChatCompletionRequest,
    EmbeddingsRequest,
    Embedding,
    EmbeddingsResponse,
)

from api_client import get_new_session, send_chat_completion_request
from response_processor import process_streaming_request, process_normal_request
from settings import PORT, API_KEY, EMBEDDING_MODEL


app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = AutoModel.from_pretrained(
    EMBEDDING_
    MODEL,
    trust_remote_code=True,
).to(device)


# Config Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],  # This should include "OPTIONS"
    allow_headers=["*"],
)


@app.options("/v1/embeddings", status_code=status.HTTP_200_OK)
async def embeddings_options():
    """Accept completions Request"""
    return JSONResponse(content={"message": "success"})


@app.post("/v1/embeddings")
async def handle_embeddings(request: Request, authorization: str = Header(None)):
    """Handles embeddings requests."""

    # Check API key authorization
    provided_api_key = authorization.split(" ")[1] if authorization else None
    if provided_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Incorrect API key")

    try:
        # Parse request body
        body = await request.json()
        embedding_request = EmbeddingsRequest(**body)

        # pylint: disable=not-an-iterable
        # Handle different input types
        if isinstance(embedding_request.input, str):
            inputs = [embedding_request.input]
        elif hasattr(embedding_request.input, "__iter__"):  # Check if it's iterable
            if all(isinstance(x, str) for x in embedding_request.input):
                inputs = embedding_request.input
            elif all(isinstance(x, list) for x in embedding_request.input):
                inputs = [" ".join(map(str, x)) for x in embedding_request.input]
            else:
                raise ValueError("Invalid input type")
        else:
            raise ValueError("Invalid input type")

        # Calculate embeddings
        with torch.no_grad():
            embeddings = embedding_model.encode(inputs)

        # Create EmbeddingsResponse
        embedding_response = EmbeddingsResponse(
            object="list",
            data=[
                Embedding(object="embedding", index=i, embedding=embedding.tolist())
                for i, embedding in enumerate(embeddings)
            ],
            model=embedding_request.model,
            usage={
                "prompt_tokens": len(inputs),
                "total_tokens": len(inputs),
            },  # adjust usage stats as needed
        )

        # Return serialized EmbeddingsResponse as JSON
        return JSONResponse(
            content=embedding_response.model_dump_json(), media_type="application/json"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}") from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {e}"
        ) from e


@app.options("/v1/chat/completions", status_code=status.HTTP_200_OK)
async def chat_completions_options():
    """Accept completions Request"""
    return JSONResponse(content={"message": "success"})


@app.post("/v1/chat/completions")
async def handle_chat_completion(request: Request, authorization: str = Header(None)):
    """Handles chat completion requests."""

    # Check API key authorization
    provided_api_key = authorization.split(" ")[1] if authorization else None
    if provided_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Incorrect API key")

    try:
        # Parse request body
        body = await request.json()
        chat_completion_request = ChatCompletionRequest(**body)

        # Get a new session
        session = await get_new_session()
        if not session:
            raise HTTPException(
                status_code=503, detail="Failed to obtain a session from OpenAI API"
            )

        # Send chat completion request
        response = send_chat_completion_request(chat_completion_request, session)

        # Process response based on streaming
        if chat_completion_request.stream:
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
            return StreamingResponse(
                process_streaming_request(chat_completion_request.messages, response),
                media_type="text/event-stream",
                headers=headers,
            )
        else:
            return Response(
                content=await process_normal_request(
                    chat_completion_request.messages, response
                ),
                media_type="application/json",
                headers={"Content-Type": "application/json"},
            )
    except Exception as e:
        print(f"Error handling chat completion: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred..."
        ) from e


# Start the server
if __name__ == "__main__":
    print(f"💡 Server is running at http://localhost:{PORT}")
    print(f"🔗 Local Base URL: http://localhost:{PORT}/v1")
    print(f"🔗 Local Endpoint: http://localhost:{PORT}/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
