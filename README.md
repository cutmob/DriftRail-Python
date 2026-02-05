# DriftRail Python SDK

[![PyPI version](https://badge.fury.io/py/driftrail.svg)](https://pypi.org/project/driftrail/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI Safety & Observability Platform — Monitor, classify, and audit every LLM interaction.

## Installation

```bash
pip install driftrail

# For async support
pip install driftrail[async]
```

## Quick Start

```python
from driftrail import DriftRail

client = DriftRail(
    api_key="dr_live_...",
    app_id="my-app"
)

# Log an LLM interaction
response = client.ingest(
    model="claude-sonnet-4",
    provider="anthropic",
    input={"prompt": "What is the capital of France?"},
    output={"text": "The capital of France is Paris."}
)

print(f"Event ID: {response.event_id}")
```

## Inline Guardrails

Block dangerous outputs BEFORE they reach users:

```python
from driftrail import DriftRail

client = DriftRail(api_key="...", app_id="my-app")

# Get response from your LLM
llm_response = your_llm_call(user_prompt)

# Guard it before returning to user
result = client.guard(
    output=llm_response,
    input=user_prompt,
    mode="strict"  # or "permissive"
)

if result.allowed:
    return result.output  # May be redacted if PII was found
else:
    print(f"Blocked: {[t.reason for t in result.triggered]}")
    return "Sorry, I can't help with that."
```

### Guard Modes

- `strict` (default): Blocks on medium+ risk (PII, moderate toxicity, prompt injection)
- `permissive`: Only blocks on high risk (severe toxicity, high-risk injection)

### Fail-Open vs Fail-Closed

```python
# Fail-open (default): If DriftRail is unavailable, content is allowed through
client = DriftRail(api_key="...", app_id="...", guard_mode="fail_open")

# Fail-closed: If DriftRail is unavailable, raises exception
client = DriftRail(api_key="...", app_id="...", guard_mode="fail_closed")

try:
    result = client.guard(output=llm_response)
except GuardBlockedError as e:
    print(f"Blocked: {e.result.triggered}")
```

## Async Usage

```python
import asyncio
from driftrail import DriftRailAsync

async def main():
    async with DriftRailAsync(api_key="...", app_id="my-app") as client:
        response = await client.ingest(
            model="claude-3",
            provider="anthropic",
            input={"prompt": "Hello"},
            output={"text": "Hi there!"}
        )

asyncio.run(main())
```

## Fire-and-Forget (Non-blocking)

```python
# Won't block your main thread
client.ingest_async(
    model="gpt-4o",
    provider="openai",
    input={"prompt": "..."},
    output={"text": "..."}
)
```

> ⚠️ **Serverless Warning**: Do not use `ingest_async()` in AWS Lambda, Google Cloud Functions, or other serverless environments. Use the synchronous `ingest()` method instead.

## With Metadata

```python
import time

start = time.time()
# ... your LLM call ...
latency = int((time.time() - start) * 1000)

client.ingest(
    model="gpt-4o",
    provider="openai",
    input={"prompt": "..."},
    output={"text": "..."},
    metadata={
        "latency_ms": latency,
        "tokens_in": 50,
        "tokens_out": 150,
        "temperature": 0.7
    }
)
```

## With RAG Sources

```python
client.ingest(
    model="claude-3.5-haiku",
    provider="anthropic",
    input={
        "prompt": "What does our refund policy say?",
        "retrieved_sources": [
            {"id": "doc-123", "content": "Refunds are available within 30 days..."},
            {"id": "doc-456", "content": "Contact support for refund requests..."}
        ]
    },
    output={"text": "According to our policy, refunds are available within 30 days..."}
)
```

## Enterprise Features

```python
from driftrail import DriftRailEnterprise

client = DriftRailEnterprise(api_key="...", app_id="my-app")

# Incident management
stats = client.get_incident_stats()

# Compliance status
compliance = client.get_compliance_status()

# Model leaderboard
leaderboard = client.get_model_leaderboard(metric="avg_risk_score")

# Brand safety checks
violations = client.check_brand_safety("Some AI output text")
```

## Documentation

- [Full Documentation](https://docs.driftrail.com)
- [API Reference](https://docs.driftrail.com/api)
- [Dashboard](https://app.driftrail.com)
- [GitHub Repository](https://github.com/cutmob/DriftRail-Python)

## License

MIT
