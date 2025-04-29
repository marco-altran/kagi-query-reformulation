import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LlamaRAMCache

# --- Configuration ---
REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
FILENAME = "qwen2.5-0.5b-instruct-q4_0.gguf"
N_GPU_LAYERS = 0
N_CTX = 1024
TEMPERATURE = 0.5
TOP_P = 0.95
MAX_TOKENS = 20

PREFIX = (
    "You are an assistant that rewrites user questions into concise search-engine queries.\n"
    "Write one to up to three MAXIMUM search queries. Have them be diverse from one another.\n"
    "<example>\n<question>\nWhat are some ways to do fast query reformulation</question>\n<search_queries>\n"
    "Fast query reformulation techniques\n"
    "Query expansion methods\n"
    "Query refinement strategies\n"
    "</search_queries>\n</example>\n<question>\n"
)
SUFFIX = "\n</question>\n<search_queries>\n"

llm: Llama = None
system_message: dict = None
cache = LlamaRAMCache()

class QueryRequest(BaseModel):
    question: str
    desired_max_latency: float = 100  # ms, default to 100 if not set

class QueryResponse(BaseModel):
    queries: List[str]
    elapsed_time: float  # ms

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, system_message
    # Download the model file
    print(f"Downloading model '{FILENAME}' from repo '{REPO_ID}'...")
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        print(f"Model downloaded/found at: {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file was not found at: {model_path}")

    # Initialize the Llama model
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=False,
        use_mmap=True,
        use_mlock=True,
        batch_size=128,
        cache=cache
    )

    # Prepare system message and warm up
    system_message = {"role": "system", "content": PREFIX}
    print("Warming up the model…")
    # Warm up the system prompt
    _ = llm.create_chat_completion(
        messages=[system_message, {"role": "user", "content": " " + SUFFIX}],
        max_tokens=2,
        temperature=0.0,
    )
    # Warm up all the rewrite_query variables
    rewrite_query("Hi", 100)
    print("Model warm-up completed.")
    yield

app = FastAPI(lifespan=lifespan)

def rewrite_query(question: str, desired_max_latency: float):
    """
    Reformulates a question into up to three search queries.
    """
    # Calculate dynamic max tokens based on question length
    dynamic_max_tokens = min(MAX_TOKENS, max(12, MAX_TOKENS - (len(question) // 40)))
    user_msg = {"role": "user", "content": question + SUFFIX}

    start = time.perf_counter()
    # Stream response to enforce time constraints
    stream = llm.create_chat_completion(
        messages=[system_message, user_msg],
        max_tokens=dynamic_max_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=["\n\n", "</", "\n</", "\n\n\n", "</search_queries>"],
        stream=True,
    )

    text = ""
    newline_count = 0
    for chunk in stream:
        # each chunk is a dict with delta content
        delta = chunk["choices"][0]["delta"].get("content", "")
        text += delta
        # After initial token, start timer
        elapsed = (time.perf_counter() - start) * 1000
        if elapsed > 0.9 * desired_max_latency:
            # Stop collecting queries if the time spent is above 90% of desired_max_latency
            break
        if "\n" in delta:
            newline_count += 1
            if elapsed > 0.75 * desired_max_latency or newline_count >= 3:
                break
    try:
        stream.close()
    except Exception:
        pass

    # Parse queries or fallback to original question
    if not text.strip():
        queries = [question]
    else:
        queries = [q for q in text.strip().split("\n") if q]

    elapsed_time = (time.perf_counter() - start) * 1000
    return queries, elapsed_time

@app.post("/rewrite", response_model=QueryResponse)
def api_rewrite(request: QueryRequest):
    """
    HTTP endpoint to rewrite a user question into search queries.
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Call helper to get reformulated queries
    queries, elapsed_time = rewrite_query(request.question, request.desired_max_latency)
    return QueryResponse(queries=queries, elapsed_time=elapsed_time)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
