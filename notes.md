# AI Knowledge Assistant” locally using Llama + vLLM + FastAPI + a simple UI, then use JMeter/k6 to performance-test it.

## AI Performance Engineering Assistant

                 AI Assistant
                     │
       ┌─────────────┼──────────────┐
       │             │              │
       ▼             ▼              ▼
       Ask          Summarize     Generate Test Script
       Question     Document


## Architecture



                         ┌──────────────────┐
                         │   User / Browser │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   Frontend UI    │
                         │ Streamlit/React  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   FastAPI        │
                         │  Application     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │      vLLM        │
                         │  Inference API   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Llama Model      │
                         │ Local / Private  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ GPU / CPU        │
                         └──────────────────┘


       Performance Testing
       ───────────────────

       JMeter / k6
            │
            ▼
       FastAPI / vLLM
            │
            ├── TTFT
            ├── TTFT
            ├── E2E Latency
            ├── Tokens/sec
            ├── Throughput
            ├── Concurrency
            ├── Error %
            └── Queue time

       Infrastructure
            │
            ├── GPU utilization
            ├── GPU memory
            ├── CPU
            ├── RAM
            └── KV Cache


## App architecture

                 ┌─────────────────┐
                 │    Browser      │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   Streamlit     │
                 │    :8501        │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    FastAPI      │
                 │    :8080        │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │     Ollama      │
                 │    :11434       │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    Llama 3.1    │
                 └────────┬────────┘
                          │
                          ▼
                       CPU/GPU


uvicorn main:app --host 0.0.0.0 --port 8080 --reload