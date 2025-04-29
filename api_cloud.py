import os
import time

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware

load_dotenv()

# --- Configuration ---
REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
FILENAME = "qwen2.5-0.5b-instruct-q4_0.gguf"
N_GPU_LAYERS = -1
N_CTX = 1024
TEMPERATURE = 0.5
TOP_P = 0.95
MAX_TOKENS = 20
ENDPOINT_URL_GCP = "https://j3j65dinj9n008pj.us-east4.gcp.endpoints.huggingface.cloud/v1/"
API_KEY = os.getenv("HF_API_KEY", "hf_XXXXX")

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

system_message: dict = {"role": "system", "content": PREFIX}

class QueryRequest(BaseModel):
    question: str
    desired_max_latency: float = 100  # ms, default to 100 if not set

class QueryResponse(BaseModel):
    queries: List[str]
    elapsed_time: float  # ms

client = OpenAI(
        base_url=ENDPOINT_URL_GCP,
        api_key=API_KEY
    )

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

def rewrite_query(question: str, desired_max_latency: float):
    """
    Reformulates a question into up to three search queries.
    """
    # Calculate dynamic max tokens based on question length
    dynamic_max_tokens = min(MAX_TOKENS, max(12, MAX_TOKENS - (len(question) // 40)))
    user_msg = {"role": "user", "content": question + SUFFIX}

    start = time.perf_counter()
    newline_count = 0

    # Stream response to enforce time constraints
    stream = client.chat.completions.create(
        model=REPO_ID,
        messages=[system_message, user_msg],
        max_tokens=dynamic_max_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stream=True,
        stop=["</"]
    )

    text = ""
    for message in stream:
        # each chunk is a dict with delta content
        delta = getattr(message.choices[0].delta, "content", "")
        if not isinstance(delta, str):
            continue
        text += delta
        # After initial token, start timer
        elapsed = (time.perf_counter() - start) * 1000
        if elapsed > 0.9 * desired_max_latency:
            # Stop collecting queries if the time spent is above 90% of desired_max_latency
            print("Emergency break.")
            break
        if "\n" in delta:
            newline_count += 1
            if elapsed > 0.75 * desired_max_latency or newline_count >= 3:
                print("Time limit exceeded, breaking stream.")
                break
    stream.close()

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
    queries, elapsed_time = rewrite_query(request.question, request.desired_max_latency)
    return QueryResponse(queries=queries, elapsed_time=elapsed_time)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
