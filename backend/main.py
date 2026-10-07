from fastapi import FastAPI
from pydantic import BaseModel
import requests
import time
import json
from fastapi.responses import StreamingResponse
from datetime import datetime
from uuid import uuid4
from typing import Optional, Literal
from fastapi import HTTPException
from pathlib import Path
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI(
    title="LLM Performance Lab",
    description="FastAPI + Ollama + Llama 3.1 / Google Gemini",
    version="1.0"
)

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.1:8b"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Retry transient Gemini errors (e.g. 503 "high demand") with backoff
gemini_client = (
    genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=5,
                initial_delay=2,
                max_delay=20,
                # 429 is left out: the free-tier limit is per day,
                # so retrying within seconds only adds delay.
                http_status_codes=[500, 502, 503, 504]
            )
        )
    )
    if GEMINI_API_KEY
    else None
)

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

    # Not exposed by cloud APIs such as Gemini → None
    model_load_time_sec: Optional[float]
    prompt_processing_time_sec: Optional[float]
    generation_time_sec: float
    total_latency_sec: float

    output_tokens_per_sec: float
    tpot_ms: float

    context_window: Optional[int]
    context_window_usage_pct: Optional[float]

    status: str


class EvaluateRequest(BaseModel):
    request_id: str
    prompt: str
    response: str


# Structured output schema for the LLM judge

class ClaimCheck(BaseModel):
    claim: str
    verdict: Literal["supported", "unsupported"]
    reason: str


class JudgeResult(BaseModel):
    relevance: int
    accuracy: int
    completeness: int
    clarity: int
    claims: list[ClaimCheck]
    summary: str


# A response "passes" quality if its average score AND its accuracy
# score are at least this (1-10); accuracy is checked separately so a
# fluent but wrong answer cannot pass on relevance/clarity alone.
QUALITY_PASS_THRESHOLD = 7

JUDGE_THINKING_LEVEL = os.getenv("JUDGE_THINKING_LEVEL", "minimal")

# =========================================================
# Context window helpers
# =========================================================

def get_ollama_context_window():

    # The model supports 131k tokens, but Ollama runs it with
    # a smaller num_ctx; /api/ps reports the one actually in use.
    try:

        response = requests.get(
            f"{OLLAMA_URL}/api/ps",
            timeout=5
        )

        for model in response.json().get("models", []):

            if model.get("name") == MODEL_NAME:

                return model.get("context_length")

    except Exception:

        pass

    return None


def context_window_usage_pct(tokens, window):

    if not window:

        return None

    return round(tokens / window * 100, 4)

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

        model_load_time = 0
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

                    model_load_time = (
                        data.get(
                            "load_duration",
                            0
                        ) / 1_000_000_000
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
                    # TPOT (time per output token)
                    # ---------------------------------

                    if output_tokens > 0:

                        tpot = (
                            generation_time /
                            output_tokens
                        )

                    else:

                        tpot = 0

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
                    # Context window usage
                    # ---------------------------------

                    context_window = (
                        get_ollama_context_window()
                    )

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

                        "model_load_time_sec":
                            round(
                                model_load_time,
                                4
                            ),

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

                        "tpot_ms":
                            round(
                                tpot * 1000,
                                2
                            ),

                        "context_window":
                            context_window,

                        "context_window_usage_pct":
                            context_window_usage_pct(
                                input_tokens +
                                output_tokens,
                                context_window
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

                "model_load_time_sec":
                    0,

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

                "tpot_ms":
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
# Gemini Chat Stream
# =========================================================

@app.post("/api/gemini-chat-stream")
def gemini_chat_stream(request: ChatRequest):

    request_id = str(uuid4())

    timestamp = datetime.now().isoformat()

    def generate():

        start_time = time.perf_counter()

        first_token_time = None

        usage = None

        try:

            if gemini_client is None:

                raise RuntimeError(
                    "GEMINI_API_KEY is not set in .env"
                )

            # Gemini counts "thinking" tokens against
            # max_output_tokens; keep thinking minimal so the
            # whole budget goes to the visible answer, like
            # Llama 3.1.
            config = types.GenerateContentConfig(
                temperature=request.temperature,
                max_output_tokens=request.num_predict,
                thinking_config=types.ThinkingConfig(
                    thinking_level="minimal"
                ),
                automatic_function_calling=(
                    types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                )
            )

            stream = gemini_client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=request.prompt,
                config=config
            )

            for chunk in stream:

                token = chunk.text or ""

                # -------------------------------------
                # TTFT
                # -------------------------------------

                if token and first_token_time is None:

                    first_token_time = (
                        time.perf_counter()
                    )

                    print(
                        f"[{request_id}] "
                        f"TTFT = {first_token_time - start_time:.4f}s"
                    )

                if token:

                    yield token

                # Usage metadata is cumulative; the last
                # chunk carries the final counts.
                if chunk.usage_metadata:

                    usage = chunk.usage_metadata

            end_time = time.perf_counter()

            total_latency = end_time - start_time

            input_tokens = (
                usage.prompt_token_count or 0
                if usage else 0
            )

            output_tokens = (
                usage.candidates_token_count or 0
                if usage else 0
            )

            thinking_tokens = (
                usage.thoughts_token_count or 0
                if usage else 0
            )

            # ---------------------------------
            # Gemini does not report load / prompt
            # processing / generation durations, so
            # generation time is measured here as the
            # time from first token to end of stream.
            # ---------------------------------

            if first_token_time:

                ttft = first_token_time - start_time

                generation_time = end_time - first_token_time

            else:

                ttft = None

                generation_time = 0

            if generation_time > 0:

                tokens_per_sec = (
                    output_tokens /
                    generation_time
                )

            else:

                tokens_per_sec = 0

            if output_tokens > 0:

                tpot = (
                    generation_time /
                    output_tokens
                )

            else:

                tpot = 0

            metrics = {
                "request_id": request_id,
                "timestamp": timestamp,
                "endpoint": "/api/gemini-chat-stream",
                "model": GEMINI_MODEL,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
                "total_tokens":
                    input_tokens +
                    output_tokens +
                    thinking_tokens,
                "ttft_sec":
                    round(ttft, 4)
                    if ttft is not None
                    else None,
                "model_load_time_sec": None,
                "prompt_processing_time_sec": None,
                "generation_time_sec":
                    round(generation_time, 4),
                "total_latency_sec":
                    round(total_latency, 4),
                "output_tokens_per_sec":
                    round(tokens_per_sec, 2),
                "tpot_ms":
                    round(tpot * 1000, 2),
                "status": "success"
            }

            performance_logs.append(metrics)

            print(f"[{request_id}] Gemini metrics: {metrics}")

        except Exception as e:

            performance_logs.append({
                "request_id": request_id,
                "timestamp": timestamp,
                "endpoint": "/api/gemini-chat-stream",
                "model": GEMINI_MODEL,
                "input_tokens": 0,
                "output_tokens": 0,
                "thinking_tokens": 0,
                "total_tokens": 0,
                "ttft_sec": None,
                "model_load_time_sec": None,
                "prompt_processing_time_sec": None,
                "generation_time_sec": 0,
                "total_latency_sec":
                    round(
                        time.perf_counter() -
                        start_time,
                        4
                    ),
                "output_tokens_per_sec": 0,
                "tpot_ms": 0,
                "status": "failed"
            })

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


# =========================================================
# Quality Evaluation (LLM-as-judge)
# =========================================================

def judge_with_gemini(judge_prompt):

    if gemini_client is None:

        raise RuntimeError("GEMINI_API_KEY is not set in .env")

    result = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=judge_prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=JudgeResult,
            thinking_config=types.ThinkingConfig(
                thinking_level=JUDGE_THINKING_LEVEL
            ),
            automatic_function_calling=(
                types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            )
        )
    )

    if result.parsed is None:

        raise RuntimeError("Gemini returned an unparseable evaluation")

    return result.parsed


def judge_with_ollama(judge_prompt):

    # Ollama structured output: "format" takes a JSON schema
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": judge_prompt,
            "stream": False,
            "format": JudgeResult.model_json_schema(),
            "options": {
                "temperature": 0
            }
        },
        timeout=300
    )

    response.raise_for_status()

    return JudgeResult.model_validate_json(
        response.json()["response"]
    )


@app.post("/api/evaluate")
def evaluate(request: EvaluateRequest):

    judge_prompt = f"""
You are a strict evaluator of answers produced by an AI assistant.

QUESTION:
{request.prompt}

ANSWER:
{request.response}

Tasks:

1. Extract the factual claims made in the ANSWER (at most 15).
   Opinions, greetings and formatting are not claims.
   Mark each claim "supported" if it is correct and consistent
   with the question, or "unsupported" if it is false,
   fabricated, or cannot be verified.

2. Score the ANSWER from 1 (worst) to 10 (best) on:
   - relevance: does it address the question that was asked?
   - accuracy: are its statements correct?
   - completeness: does it fully answer? (cut-off answers score lower)
   - clarity: is it clear and well organised?

3. Give a one-sentence summary of your verdict.
"""

    start_time = time.perf_counter()

    # Gemini is the preferred (independent, stronger) judge. If it is
    # unavailable (no key, quota exhausted, overloaded) fall back to
    # the local Llama model so quality metrics are still produced.
    judge_fallback_reason = None

    try:

        judge = judge_with_gemini(judge_prompt)

        judge_model = GEMINI_MODEL

    except Exception as gemini_error:

        judge_fallback_reason = str(gemini_error).split("\n")[0][:200]

        print(
            f"[{request.request_id}] Gemini judge failed, "
            f"falling back to {MODEL_NAME}: {judge_fallback_reason}"
        )

        try:

            judge = judge_with_ollama(judge_prompt)

            judge_model = MODEL_NAME

        except Exception as ollama_error:

            raise HTTPException(
                status_code=503,
                detail=(
                    f"Gemini judge failed ({judge_fallback_reason}); "
                    f"Ollama judge failed ({ollama_error})"
                )
            )

    eval_latency = time.perf_counter() - start_time

    # ---------------------------------
    # Quality metrics
    # ---------------------------------

    unsupported_claims = sum(
        1 for c in judge.claims
        if c.verdict == "unsupported"
    )

    if judge.claims:

        hallucination_rate = (
            unsupported_claims /
            len(judge.claims) * 100
        )

    else:

        hallucination_rate = 0

    quality_score = (
        judge.relevance +
        judge.accuracy +
        judge.completeness +
        judge.clarity
    ) / 4

    evaluation = {
        "judge_model": judge_model,
        "judge_fallback_reason": judge_fallback_reason,
        "hallucination_rate_pct":
            round(hallucination_rate, 2),
        "claims_total": len(judge.claims),
        "claims_unsupported": unsupported_claims,
        "quality_score": round(quality_score, 2),
        "quality_pass":
            quality_score >= QUALITY_PASS_THRESHOLD
            and judge.accuracy >= QUALITY_PASS_THRESHOLD,
        "relevance": judge.relevance,
        "accuracy": judge.accuracy,
        "completeness": judge.completeness,
        "clarity": judge.clarity,
        "judge_summary": judge.summary,
        "claims": [c.model_dump() for c in judge.claims],
        "eval_latency_sec": round(eval_latency, 4)
    }

    # Attach to the request's performance record so
    # /api/metrics holds performance + quality together.
    for record in performance_logs:

        if record["request_id"] == request.request_id:

            record["quality"] = evaluation

            break

    return evaluation


@app.get("/api/metrics")
def get_metrics():

    return {
        "total_requests":
            len(performance_logs),

        "requests":
            performance_logs
    }