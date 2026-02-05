# DriftRail Python SDK

Official Python SDK for [DriftRail](https://driftrail.com) - AI Safety & Observability Platform.

[![PyPI version](https://badge.fury.io/py/driftrail.svg)](https://badge.fury.io/py/driftrail)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Installation

```bash
pip install driftrail

# With async support
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
    model="gpt-4o",
    provider="openai",
    input={"prompt": "What is the capital of France?"},
    output={"text": "The capital of France is Paris."},
    metadata={"latency_ms": 250, "tokens_in": 10, "tokens_out": 8}
)

print(f"Event ID: {response.event_id}")
```

## Inline Guardrails

Block dangerous outputs before they reach users:

```python
result = client.guard(
    output=llm_response,
    input=user_prompt,
    mode="strict"  # or "permissive"
)

if result.allowed:
    return result.output  # May be redacted
else:
    return "Sorry, I can't help with that."
```

## Async Client

```python
from driftrail import DriftRailAsync

async with DriftRailAsync(api_key="...", app_id="my-app") as client:
    response = await client.ingest(
        model="gpt-4o",
        provider="openai",
        input={"prompt": "Hello"},
        output={"text": "Hi there!"}
    )
```

## Enterprise Features

The `DriftRailEnterprise` client provides access to all dashboard features:

```python
from driftrail import DriftRailEnterprise

client = DriftRailEnterprise(api_key="dr_live_...", app_id="my-app")
```

### Incidents

```python
# List incidents
incidents = client.list_incidents(
    status=["open", "investigating"],
    severity=["high", "critical"]
)

# Create incident
incident = client.create_incident(
    title="High risk outputs detected",
    severity="high",
    incident_type="safety",
    description="Multiple high-risk outputs in production"
)

# Get stats
stats = client.get_incident_stats()
```

### Compliance

```python
# Get compliance status
status = client.get_compliance_status()

# Generate compliance report
report = client.generate_compliance_report(
    framework="soc2",
    format="pdf",
    include_evidence=True
)

# Get compliance score
score = client.get_compliance_score()
```

### Drift Detection

```python
# Get drift metrics
metrics = client.get_drift_metrics()

# Get drift alerts
alerts = client.get_drift_alerts(severity="critical", unresolved=True)

# Acknowledge/resolve alerts
client.acknowledge_alert(alert_id)
client.resolve_alert(alert_id, notes="Fixed in v2.1")

# Get drift score
score = client.get_drift_score()

# Get predictions
predictions = client.get_drift_predictions()
```

### Distributed Tracing

```python
# Start a trace
trace = client.start_trace(
    app_id="my-app",
    name="chat-completion",
    user_id="user-123",
    metadata={"session": "abc"}
)

# Add spans
span = client.start_span(
    trace_id=trace["trace_id"],
    name="llm-call",
    span_type="llm",
    model="gpt-4o"
)

# End span with results
client.end_span(
    span_id=span["span_id"],
    output={"text": "Response"},
    tokens_in=100,
    tokens_out=50
)

# End trace
client.end_trace(trace["trace_id"])
```

### Prompt Management

```python
# Create prompt
prompt = client.create_prompt(
    name="customer-support",
    content="You are a helpful assistant...",
    variables=["customer_name", "issue"],
    tags=["support", "production"]
)

# Create version
version = client.create_prompt_version(
    prompt_id=prompt["prompt_id"],
    content="Updated prompt content...",
    commit_message="Improved tone"
)

# Deploy to environment
client.deploy_prompt_version(
    version_id=version["version_id"],
    environment="prod"
)

# Get deployed prompt
deployed = client.get_deployed_prompt(prompt["prompt_id"], "prod")
```

### Evaluations

```python
# Create dataset
dataset = client.create_dataset(
    name="qa-test-set",
    schema_type="qa"
)

# Add items
client.add_dataset_items(dataset["dataset_id"], [
    {"input": {"question": "What is 2+2?"}, "expected_output": {"answer": "4"}},
    {"input": {"question": "Capital of Japan?"}, "expected_output": {"answer": "Tokyo"}}
])

# Run evaluation
run = client.create_eval_run(
    dataset_id=dataset["dataset_id"],
    model="gpt-4o",
    evaluators=[
        {"name": "correctness", "type": "llm_judge"},
        {"name": "exact_match", "type": "exact_match"}
    ]
)
```

### Semantic Caching

```python
# Check cache
result = client.cache_lookup(input="What is the weather?", model="gpt-4o")

if result["hit"]:
    return result["output"]
else:
    # Call LLM and store
    response = call_llm(...)
    client.cache_store(
        input="What is the weather?",
        output=response,
        model="gpt-4o"
    )
```

### Agent Simulations

```python
# Create simulation
sim = client.create_simulation(
    name="booking-flow",
    scenario="User wants to book a flight to Paris",
    persona={"name": "Traveler", "traits": ["impatient", "budget-conscious"]},
    success_criteria=[
        {"name": "booking_complete", "description": "Flight successfully booked"}
    ],
    max_turns=20
)

# Run simulation
run = client.run_simulation(sim["simulation_id"])

# Get results
results = client.get_simulation_run(run["run_id"])
```

### Integrations

```python
# Create Slack integration
client.create_integration(
    type="slack",
    webhook_url="https://hooks.slack.com/...",
    channel_name="#alerts",
    events=["high_risk", "incident", "drift_alert"]
)

# Test integration
client.test_integration(webhook_url, "slack")
```

### Model Analytics

```python
# Get summary
summary = client.get_model_analytics_summary()

# Get historical logs
logs = client.get_historical_logs(
    model="gpt-4o",
    min_risk_score=0.7,
    limit=100
)

# Get model benchmarks
benchmarks = client.get_model_benchmarks(model="gpt-4o")

# Record model switch
client.record_model_switch(
    app_id="my-app",
    new_model="gpt-4o",
    new_provider="openai",
    previous_model="gpt-4",
    switch_reason="Cost optimization"
)
```

### Executive Dashboard

```python
# Get executive metrics
metrics = client.get_executive_metrics(period="7d")

# Get/update KPI targets
targets = client.get_kpi_targets()
client.update_kpi_targets({
    "max_high_risk_percent": 5.0,
    "target_latency_ms": 500
})

# Export metrics
export = client.export_executive_metrics(period="30d", format="pdf")
```

### Guardrails

```python
# List guardrails
guardrails = client.get_guardrails()

# Create guardrail
client.create_guardrail(
    name="block-competitors",
    rule_type="blocked_terms",
    action="block",
    config={"terms": ["competitor1", "competitor2"]}
)

# Get stats
stats = client.get_guardrail_stats()
```

### Custom Detections

```python
# Create custom detection
client.create_custom_detection(
    name="financial-advice",
    detection_type="semantic",
    config={"description": "Detects unauthorized financial advice"},
    severity="high"
)
```

### Benchmarks

```python
# Get industry benchmark report
report = client.get_benchmark_report(industry="fintech")

# Set your industry
client.set_tenant_industry("fintech")
```

### Exports

```python
# Create export
export = client.create_export(
    export_type="events",
    format="csv",
    date_from="2024-01-01",
    date_to="2024-01-31"
)

# Check status
status = client.get_export_status(export["export_id"])
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | Required | Your DriftRail API key |
| `app_id` | Required | Your application identifier |
| `base_url` | `https://api.driftrail.com` | API base URL |
| `timeout` | `30` | Request timeout in seconds |
| `fail_open` | `True` | Don't raise on errors |
| `guard_mode` | `fail_open` | Guard behavior on block |

## Error Handling

```python
from driftrail import GuardBlockedError

client = DriftRail(
    api_key="...",
    app_id="my-app",
    guard_mode="fail_closed"  # Raise on blocked content
)

try:
    result = client.guard(output=response)
except GuardBlockedError as e:
    print(f"Blocked: {e.result.triggered}")
```

## Type Hints

Full type hints are available for all methods and responses:

```python
from driftrail import (
    DriftRail,
    IngestResponse,
    GuardResult,
    Incident,
    DriftAlert,
    Trace,
    Span,
    # ... and many more
)
```

## Links

- [Documentation](https://driftrail.com/docs)
- [API Reference](https://driftrail.com/api-reference)

## License

MIT
