from fastapi import FastAPI
from pydantic import BaseModel
import requests
import time
import json
from fastapi.responses import StreamingResponse
from datetime import datetime
from uuid import uuid4
from typing import Optional

app = FastAPI(
    title="LLM Performance Lab",
    description="FastAPI + Ollama + Llama 3.1",
    version="1.0"
)

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.1:8b"

performance_logs = []
# =========================================================
# Request Models
# =========================================================

class ChatRequest(BaseModel):
    prompt: str
    temperature: float = 0.2
    num_predict: int = 300


class SummarizeRequest(BaseModel):
    text: str
    num_predict: int = 300


class TestCaseRequest(BaseModel):
    api_description: str
    num_predict: int = 500

class PerformanceMetrics(BaseModel):

    request_id: str
    timestamp: str
    endpoint: str
    model: str

    input_tokens: int
    output_tokens: int
    total_tokens: int

    ttft_sec: Optional[float]

    prompt_processing_time_sec: float
    generation_time_sec: float
    total_latency_sec: float

    output_tokens_per_sec: float

    status: str

# =========================================================
# Root
# =========================================================

@app.get("/")
def root():

    return {
        "application": "LLM Performance Lab",
        "model": MODEL_NAME,
        "engine": "Ollama",
        "status": "running"
    }


# =========================================================
# Health Check
# =========================================================

@app.get("/health")
def health():

    try:

        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=5
        )

        if response.status_code == 200:

            return {
                "status": "healthy",
                "ollama": "running",
                "model": MODEL_NAME
            }

        return {
            "status": "unhealthy",
            "ollama": "not responding"
        }

    except Exception as e:

        return {
            "status": "unhealthy",
            "error": str(e)
        }


# =========================================================
# Chat
# =========================================================

@app.post("/api/chat")
def chat(request: ChatRequest):

    start_time = time.perf_counter()

    payload = {
        "model": MODEL_NAME,
        "prompt": request.prompt,
        "stream": False,
        "options": {
            "temperature": request.temperature,
            "num_predict": request.num_predict
        }
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    result = response.json()

    print(result)

    end_time = time.perf_counter()

    # --------------------------------------------------
    # Ollama metrics
    # --------------------------------------------------

    total_duration_ns = result.get("total_duration", 0)
    load_duration_ns = result.get("load_duration", 0)
    prompt_eval_duration_ns = result.get(
        "prompt_eval_duration", 0
    )
    eval_duration_ns = result.get(
        "eval_duration", 0
    )

    prompt_tokens = result.get(
        "prompt_eval_count", 0
    )

    output_tokens = result.get(
        "eval_count", 0
    )

    # --------------------------------------------------
    # Convert nanoseconds → seconds
    # --------------------------------------------------

    total_duration_sec = (
        total_duration_ns / 1_000_000_000
    )

    load_duration_sec = (
        load_duration_ns / 1_000_000_000
    )

    prompt_eval_duration_sec = (
        prompt_eval_duration_ns / 1_000_000_000
    )

    eval_duration_sec = (
        eval_duration_ns / 1_000_000_000
    )

    # --------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------

    total_tokens = (
        prompt_tokens + output_tokens
    )

    if prompt_eval_duration_sec > 0:

        prompt_tokens_per_sec = (
            prompt_tokens /
            prompt_eval_duration_sec
        )

    else:

        prompt_tokens_per_sec = 0


    if eval_duration_sec > 0:

        output_tokens_per_sec = (
            output_tokens /
            eval_duration_sec
        )

    else:

        output_tokens_per_sec = 0


    return {

        # -------------------------------
        # Response
        # -------------------------------

        "response": result.get("response"),

        "model": result.get("model"),

        # -------------------------------
        # Token Metrics
        # -------------------------------

        "input_tokens": prompt_tokens,

        "output_tokens": output_tokens,

        "total_tokens": total_tokens,

        # -------------------------------
        # Latency Metrics
        # -------------------------------

        "total_latency_sec":
            total_duration_sec,

        "model_load_time_sec":
            load_duration_sec,

        "prompt_processing_time_sec":
            prompt_eval_duration_sec,

        "generation_time_sec":
            eval_duration_sec,

        # -------------------------------
        # Throughput
        # -------------------------------

        "prompt_tokens_per_sec":
            round(
                prompt_tokens_per_sec,
                2
            ),

        "output_tokens_per_sec":
            round(
                output_tokens_per_sec,
                2
            ),

        # -------------------------------
        # Application measurement
        # -------------------------------

        "application_latency_sec":
            round(
                end_time - start_time,
                4
            )
    }

# =========================================================
# Chat Stream
# =========================================================
@app.post("/api/chat-stream")
def chat_stream(request: ChatRequest):

    request_id = str(uuid4())

    timestamp = datetime.now().isoformat()

    def generate():

        start_time = time.perf_counter()

        first_token_time = None

        input_tokens = 0
        output_tokens = 0

        prompt_processing_time = 0
        generation_time = 0

        total_latency = 0

        try:

            payload = {
                "model": MODEL_NAME,
                "prompt": request.prompt,
                "stream": True,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.num_predict
                }
            }

            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                stream=True,
                timeout=300
            )

            response.raise_for_status()

            for line in response.iter_lines():

                if not line:
                    continue

                data = json.loads(line)

                token = data.get(
                    "response",
                    ""
                )

                # -------------------------------------
                # TTFT
                # -------------------------------------

                if token and first_token_time is None:

                    first_token_time = (
                        time.perf_counter()
                    )

                    ttft = (
                        first_token_time -
                        start_time
                    )

                    print(
                        f"[{request_id}] "
                        f"TTFT = {ttft:.4f}s"
                    )

                # -------------------------------------
                # Send token
                # -------------------------------------

                if token:

                    yield token

                # -------------------------------------
                # Final Ollama metrics
                # -------------------------------------

                if data.get("done"):

                    end_time = time.perf_counter()

                    total_latency = (
                        end_time -
                        start_time
                    )

                    input_tokens = data.get(
                        "prompt_eval_count",
                        0
                    )

                    output_tokens = data.get(
                        "eval_count",
                        0
                    )
                    

                    prompt_processing_time = (
                        data.get(
                            "prompt_eval_duration",
                            0
                        ) / 1_000_000_000
                    )

                    generation_time = (
                        data.get(
                            "eval_duration",
                            0
                        ) / 1_000_000_000
                    )

                    # ---------------------------------
                    # Tokens/sec
                    # ---------------------------------

                    if generation_time > 0:

                        tokens_per_sec = (
                            output_tokens /
                            generation_time
                        )

                    else:

                        tokens_per_sec = 0

                    # ---------------------------------
                    # TTFT
                    # ---------------------------------

                    if first_token_time:

                        ttft = (
                            first_token_time -
                            start_time
                        )

                    else:

                        ttft = None

                    # ---------------------------------
                    # Metrics object
                    # ---------------------------------

                    metrics = {

                        "request_id":
                            request_id,

                        "timestamp":
                            timestamp,

                        "endpoint":
                            "/api/chat-stream",

                        "model":
                            MODEL_NAME,

                        "input_tokens":
                            input_tokens,

                        "output_tokens":
                            output_tokens,

                        "total_tokens":
                            input_tokens +
                            output_tokens,

                        "ttft_sec":
                            round(ttft, 4)
                            if ttft is not None
                            else None,

                        "prompt_processing_time_sec":
                            round(
                                prompt_processing_time,
                                4
                            ),

                        "generation_time_sec":
                            round(
                                generation_time,
                                4
                            ),

                        "total_latency_sec":
                            round(
                                total_latency,
                                4
                            ),

                        "output_tokens_per_sec":
                            round(
                                tokens_per_sec,
                                2
                            ),

                        "status":
                            "success"
                    }

                    # ---------------------------------
                    # Store metrics
                    # ---------------------------------

                    performance_logs.append(
                        metrics
                    )

                    print(
                        "===================================="
                    )

                    print(
                        f"Request ID      : "
                        f"{request_id}"
                    )

                    print(
                        f"Input Tokens    : "
                        f"{input_tokens}"
                    )

                    print(
                        f"Output Tokens   : "
                        f"{output_tokens}"
                    )

                    print(
                        f"TTFT            : "
                        f"{ttft:.4f}s"
                        if ttft is not None
                        else "TTFT            : N/A"
                    )

                    print(
                        f"Total Latency   : "
                        f"{total_latency:.4f}s"
                    )

                    print(
                        f"Generation      : "
                        f"{generation_time:.4f}s"
                    )

                    print(
                        f"Tokens/sec      : "
                        f"{tokens_per_sec:.2f}"
                    )

                    print(
                        "===================================="
                    )

        except Exception as e:

            metrics = {

                "request_id":
                    request_id,

                "timestamp":
                    timestamp,

                "endpoint":
                    "/api/chat-stream",

                "model":
                    MODEL_NAME,

                "input_tokens":
                    0,

                "output_tokens":
                    0,

                "total_tokens":
                    0,

                "ttft_sec":
                    None,

                "prompt_processing_time_sec":
                    0,

                "generation_time_sec":
                    0,

                "total_latency_sec":
                    round(
                        time.perf_counter() -
                        start_time,
                        4
                    ),

                "output_tokens_per_sec":
                    0,

                "status":
                    "failed"
            }

            performance_logs.append(
                metrics
            )

            yield f"ERROR: {str(e)}"

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "X-Request-ID": request_id
        }
    )
# =========================================================
# Summarization
# =========================================================

@app.post("/api/summarize")
def summarize(request: SummarizeRequest):

    prompt = f"""
You are a professional summarization assistant.

Summarize the following text.

TEXT:

{request.text}

Provide a concise summary.
"""

    payload = {

        "model": MODEL_NAME,

        "prompt": prompt,

        "stream": False,

        "options": {
            "temperature": 0.1,
            "num_predict": request.num_predict
        }
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    result = response.json()
    

    return {

        "summary": result.get("response"),

        "model": result.get("model"),

        "prompt_tokens":
            result.get("prompt_eval_count"),

        "output_tokens":
            result.get("eval_count"),

        "total_duration_ns":
            result.get("total_duration"),

        "prompt_eval_duration_ns":
            result.get("prompt_eval_duration"),

        "eval_duration_ns":
            result.get("eval_duration")
    }


# =========================================================
# Performance Test Generator
# =========================================================

@app.post("/api/generate-test")
def generate_test(request: TestCaseRequest):

    prompt = f"""
You are an expert performance test engineer.

Generate a performance test plan for the following API:

{request.api_description}

Include:

1. Test Scenario
2. User Load
3. Ramp-up
4. Duration
5. TPS
6. Response Time SLA
7. Throughput
8. Performance Risks
9. Monitoring Metrics

Provide the result in a structured format.
"""

    payload = {

        "model": MODEL_NAME,

        "prompt": prompt,

        "stream": False,

        "options": {
            "temperature": 0.2,
            "num_predict": request.num_predict
        }
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    result = response.json()

    return {

        "test_plan": result.get("response"),

        "model": result.get("model"),

        "prompt_tokens":
            result.get("prompt_eval_count"),

        "output_tokens":
            result.get("eval_count"),

        "total_duration_ns":
            result.get("total_duration"),

        "eval_duration_ns":
            result.get("eval_duration")
    }

# =========================================================
# Performance Test Generator
# =========================================================


@app.get("/api/metrics")
def get_metrics():

    return {
        "total_requests":
            len(performance_logs),

        "requests":
            performance_logs
    }