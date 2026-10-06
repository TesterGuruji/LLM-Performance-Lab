import streamlit as st
import requests
import time



BACKEND_URL = "http://localhost:8080"


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="LLM Performance Lab",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# Header
# =========================================================

st.title("🤖 LLM Performance Lab")

st.write(
    "Local Llama 3.1 application powered by "
    "Ollama + FastAPI"
)


# =========================================================
# Sidebar
# =========================================================

st.sidebar.header("Select Operation")

operation = st.sidebar.selectbox(
    "Operation",
    [
        "Chat",
        "Streaming Chat",
        "Gemini Streaming Chat",
        "Summarize",
        "Generate Performance Test"
    ]
)


# =========================================================
# CHAT
# =========================================================

if operation == "Chat":

    st.header("💬 Chat with Llama 3.1")

    prompt = st.text_area(
        "Enter your question",
        height=150,
        placeholder="Ask something..."
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.2,
        0.1
    )

    num_predict = st.number_input(
        "Maximum output tokens",
        min_value=50,
        max_value=2000,
        value=300
    )

    if st.button("Send"):

        if not prompt:

            st.warning(
                "Please enter a prompt."
            )

        else:

            payload = {

                "prompt": prompt,

                "temperature":
                    temperature,

                "num_predict":
                    num_predict
            }

            with st.spinner(
                "Llama is generating..."
            ):

                response = requests.post(

                    f"{BACKEND_URL}/api/chat",

                    json=payload,

                    timeout=300
                )

            if response.status_code == 200:

                result = response.json()

                st.subheader("Response")

                st.write(
                    result["response"]
                )

                # -----------------------------------------
                # Metrics
                # -----------------------------------------

                
                st.subheader("LLM Performance Metrics")

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Input Tokens",
                        result.get("input_tokens", 0)
                    )

                with col2:

                    st.metric(
                        "Output Tokens",
                        result.get("output_tokens", 0)
                    )

                with col3:

                    st.metric(
                        "Total Tokens",
                        result.get("total_tokens", 0)
                    )

                with col4:

                    st.metric(
                        "Total Latency",
                        f'{result.get("total_latency_sec", 0):.2f}s'
                    )


                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Model Load",
                        f'{result.get("model_load_time_sec", 0):.2f}s'
                    )

                with col2:

                    st.metric(
                        "Prompt Processing",
                        f'{result.get("prompt_processing_time_sec", 0):.2f}s'
                    )

                with col3:

                    st.metric(
                        "Generation",
                        f'{result.get("generation_time_sec", 0):.2f}s'
                    )

                with col4:

                    st.metric(
                        "Output Tokens/sec",
                        result.get("output_tokens_per_sec", 0)
                    )

                with st.expander("API Response"):

                    st.json(result)

            else:

                st.error(
                    response.text
                )


# =========================================================
# SUMMARIZE
# =========================================================

elif operation == "Summarize":

    st.header("📝 Summarize Text")

    text = st.text_area(
        "Enter text",
        height=300
    )

    num_predict = st.number_input(
        "Maximum output tokens",
        min_value=50,
        max_value=2000,
        value=300
    )

    if st.button("Summarize"):

        if not text:

            st.warning(
                "Please enter some text."
            )

        else:

            payload = {

                "text": text,

                "num_predict":
                    num_predict
            }

            with st.spinner(
                "Generating summary..."
            ):

                response = requests.post(

                    f"{BACKEND_URL}/api/summarize",

                    json=payload,

                    timeout=300
                )

            if response.status_code == 200:

                result = response.json()

                st.subheader("Summary")

                st.write(
                    result["summary"]
                )

                st.subheader(
                    "Inference Metrics"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Input Tokens",
                        result.get(
                            "prompt_tokens",
                            0
                        )
                    )

                with col2:

                    st.metric(
                        "Output Tokens",
                        result.get(
                            "output_tokens",
                            0
                        )
                    )

                with st.expander("API Response"):

                    st.json(result)

            else:

                st.error(
                    response.text
                )


# =========================================================
# PERFORMANCE TEST GENERATOR
# =========================================================

elif operation == "Generate Performance Test":

    st.header(
        "🧪 Performance Test Generator"
    )

    api_description = st.text_area(

        "Describe your API",

        height=250,

        placeholder="""
Example:

POST /api/payment

Request:

{
    "customerId": "123",
    "amount": 100
}

Expected TPS: 100
        """
    )

    num_predict = st.number_input(

        "Maximum output tokens",

        min_value=100,

        max_value=3000,

        value=500
    )

    if st.button(
        "Generate Test Plan"
    ):

        if not api_description:

            st.warning(
                "Please describe your API."
            )

        else:

            payload = {

                "api_description":
                    api_description,

                "num_predict":
                    num_predict
            }

            with st.spinner(
                "Generating performance test plan..."
            ):

                response = requests.post(

                    f"{BACKEND_URL}/api/generate-test",

                    json=payload,

                    timeout=300
                )

            if response.status_code == 200:

                result = response.json()

                st.subheader(
                    "Generated Performance Test Plan"
                )

                st.write(
                    result["test_plan"]
                )

                with st.expander("API Response"):

                    st.json(result)

            else:

                st.error(
                    response.text
                )

# =========================================================
# STREAMING CHAT
# =========================================================


elif operation in ("Streaming Chat", "Gemini Streaming Chat"):

    is_gemini = operation == "Gemini Streaming Chat"

    stream_endpoint = (
        "/api/gemini-chat-stream"
        if is_gemini
        else "/api/chat-stream"
    )

    # Gemini does not report some timings → show N/A
    def format_seconds(value):

        return "N/A" if value is None else f"{value:.2f}s"

    st.header(
        "✨ Gemini Streaming Chat"
        if is_gemini
        else "⚡ Streaming Chat"
    )

    prompt = st.text_area(
        "Enter your question",
        height=150
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.2,
        0.1,
        key="stream_temperature"
    )

    num_predict = st.number_input(
        "Maximum output tokens",
        min_value=50,
        max_value=2000,
        value=300,
        key="stream_tokens"
    )

    if st.button("Generate Response"):

        if not prompt:

            st.warning(
                "Please enter a prompt."
            )

        else:

            payload = {
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": num_predict
            }

            st.subheader("Response")

            response_placeholder = st.empty()

            full_response = ""

            start_time = time.perf_counter()

            first_token_time = None

            try:

                with requests.post(
                    f"{BACKEND_URL}{stream_endpoint}",
                    json=payload,
                    stream=True,
                    timeout=300
                ) as response:

                    response.raise_for_status()

                    request_id = response.headers.get(
                        "X-Request-ID"
                    )

                    for chunk in response.iter_content(
                        chunk_size=None,
                        decode_unicode=True
                    ):

                        if chunk:

                            if first_token_time is None:

                                first_token_time = (
                                    time.perf_counter()
                                )

                            full_response += chunk

                            response_placeholder.markdown(
                                full_response
                            )

                end_time = time.perf_counter()

                # --------------------------------------
                # Calculate TTFT
                # --------------------------------------

                if first_token_time:

                    ttft = (
                        first_token_time -
                        start_time
                    )

                else:

                    ttft = 0

                total_latency = (
                    end_time -
                    start_time
                )
                
                # --------------------------------------
                # Metrics
                # --------------------------------------

                st.subheader(
                    "Streaming Metrics"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "TTFT",
                        f"{ttft:.3f} sec"
                    )

                with col2:

                    st.metric(
                        "Total Latency",
                        f"{total_latency:.3f} sec"
                    )

                # --------------------------------------
                # LLM metrics recorded by the backend
                # --------------------------------------

                metrics_response = requests.get(
                    f"{BACKEND_URL}/api/metrics",
                    timeout=10
                )

                metrics_response.raise_for_status()

                llm_metrics = next(
                    (
                        m for m in
                        metrics_response.json()["requests"]
                        if m["request_id"] == request_id
                    ),
                    {}
                )

                st.subheader("LLM Performance Metrics")

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Input Tokens",
                        llm_metrics.get("input_tokens", 0)
                    )

                with col2:

                    st.metric(
                        "Output Tokens",
                        llm_metrics.get("output_tokens", 0)
                    )

                with col3:

                    st.metric(
                        "Total Tokens",
                        llm_metrics.get("total_tokens", 0)
                    )

                with col4:

                    st.metric(
                        "Model Load",
                        format_seconds(llm_metrics.get("model_load_time_sec", 0))
                    )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Prompt Processing",
                        format_seconds(llm_metrics.get("prompt_processing_time_sec", 0))
                    )

                with col2:

                    st.metric(
                        "Generation",
                        format_seconds(llm_metrics.get("generation_time_sec", 0))
                    )

                with col3:

                    st.metric(
                        "Output Tokens/sec",
                        llm_metrics.get("output_tokens_per_sec", 0)
                    )

                with col4:

                    st.metric(
                        "TPOT",
                        f'{llm_metrics.get("tpot_ms", 0):.2f} ms'
                    )

                with st.expander("API Response"):

                    st.caption(f"{stream_endpoint} (raw stream)")

                    st.code(full_response, language=None)

                    st.caption("/api/metrics (this request)")

                    st.json(llm_metrics)

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )
