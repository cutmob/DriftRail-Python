"""
DriftRail Type Definitions
Complete type definitions for all SDK features.
"""

from typing import Optional, List, Dict, Any, Literal, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime

Provider = Literal["openai", "google", "anthropic", "other"]


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


@dataclass
class SourceRef:
    id: str
    type: Optional[str] = None
    content: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ToolCall:
    id: str
    type: Literal["function"] = "function"
    function: Dict[str, str] = field(default_factory=dict)


@dataclass
class InputPayload:
    prompt: str
    messages: Optional[List[Message]] = None
    retrieved_sources: Optional[List[SourceRef]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"prompt": self.prompt}
        if self.messages:
            d["messages"] = [asdict(m) for m in self.messages]
        if self.retrieved_sources:
            d["retrievedSources"] = [asdict(s) for s in self.retrieved_sources]
        return d


@dataclass
class OutputPayload:
    text: str
    tool_calls: Optional[List[ToolCall]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"text": self.text}
        if self.tool_calls:
            d["toolCalls"] = [asdict(t) for t in self.tool_calls]
        return d


@dataclass
class Metadata:
    latency_ms: Optional[int] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    temperature: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.latency_ms is not None:
            d["latencyMs"] = self.latency_ms
        if self.tokens_in is not None:
            d["tokensIn"] = self.tokens_in
        if self.tokens_out is not None:
            d["tokensOut"] = self.tokens_out
        if self.temperature is not None:
            d["temperature"] = self.temperature
        return d


@dataclass
class IngestPayload:
    model: str
    provider: Provider
    input: InputPayload
    output: OutputPayload
    metadata: Optional[Metadata] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "model": self.model,
            "provider": self.provider,
            "input": self.input.to_dict(),
            "output": self.output.to_dict(),
        }
        if self.metadata:
            d["metadata"] = self.metadata.to_dict()
        return d


@dataclass
class IngestResponse:
    success: bool
    event_id: Optional[str] = None
    job_id: Optional[str] = None
    error: Optional[str] = None
    duplicate: bool = False


# Guard types for inline guardrails

GuardMode = Literal["strict", "permissive"]
GuardAction = Literal["allow", "block", "redact", "warn"]


@dataclass
class GuardTriggered:
    type: Literal["classification", "guardrail"]
    name: str
    reason: str


@dataclass
class GuardClassification:
    risk_score: int
    pii_detected: bool
    pii_types: List[str]
    toxicity_detected: bool
    toxicity_severity: str
    prompt_injection_detected: bool
    prompt_injection_risk: str


@dataclass
class GuardResult:
    """Result from inline guard check."""
    allowed: bool
    action: GuardAction
    output: str
    triggered: List[GuardTriggered]
    latency_ms: int
    fallback: bool
    classification: Optional[GuardClassification] = None
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GuardResult":
        triggered = [
            GuardTriggered(
                type=t.get("type", "guardrail"),
                name=t.get("name", "Unknown"),
                reason=t.get("reason", ""),
            )
            for t in data.get("triggered", [])
        ]
        
        classification = None
        if data.get("classification"):
            c = data["classification"]
            classification = GuardClassification(
                risk_score=c.get("risk_score", 0),
                pii_detected=c.get("pii", {}).get("detected", False),
                pii_types=c.get("pii", {}).get("types", []),
                toxicity_detected=c.get("toxicity", {}).get("detected", False),
                toxicity_severity=c.get("toxicity", {}).get("severity", "none"),
                prompt_injection_detected=c.get("prompt_injection", {}).get("detected", False),
                prompt_injection_risk=c.get("prompt_injection", {}).get("risk", "none"),
            )
        
        return cls(
            allowed=data.get("allowed", True),
            action=data.get("action", "allow"),
            output=data.get("output", ""),
            triggered=triggered,
            latency_ms=data.get("latency_ms", 0),
            fallback=data.get("fallback", False),
            classification=classification,
        )


class GuardBlockedError(Exception):
    """Raised when content is blocked in fail_closed mode."""
    def __init__(self, result: GuardResult):
        self.result = result
        reasons = [t.reason for t in result.triggered]
        super().__init__(f"Content blocked: {'; '.join(reasons)}")


# ============ Incident Types ============

@dataclass
class Incident:
    incident_id: str
    title: str
    severity: Literal["low", "medium", "high", "critical"]
    status: Literal["open", "investigating", "mitigating", "resolved", "closed"]
    incident_type: str
    created_at: str
    description: Optional[str] = None


@dataclass
class IncidentStats:
    open: int
    investigating: int
    critical: int
    mttr_hours: float


# ============ Compliance Types ============

@dataclass
class ComplianceFramework:
    name: str
    enabled: bool
    status: str


@dataclass
class ComplianceStatus:
    frameworks: List[ComplianceFramework]
    pending_reviews: int
    recent_violations: int
    data_region: str


@dataclass
class ComplianceScore:
    compliance_score: float
    total_events: int
    high_risk_events: int
    high_risk_percentage: float
    date_range: Dict[str, str]
    rating: str


@dataclass
class ComplianceReportMeta:
    report_id: str
    framework: str
    date_from: str
    date_to: str
    compliance_score: float
    findings_count: int
    generated_at: str
    generated_by: str


@dataclass
class CustomControl:
    control_id: str
    name: str
    description: str
    category: str
    evidence_type: str
    keywords: List[str]


# ============ Executive Dashboard Types ============

@dataclass
class KpiTargets:
    max_high_risk_percent: float
    target_latency_ms: int
    target_sla_percent: float
    target_mttr_hours: float
    monthly_budget_usd: float
    max_critical_events_daily: int
    target_avg_risk_score: float
    sla_latency_threshold_ms: int
    max_open_incidents: int
    cost_per_1k_tokens: float


@dataclass
class ExecutiveMetrics:
    period: str
    health_score: float
    summary: Dict[str, Any]
    cost_metrics: Dict[str, Any]
    incident_metrics: Dict[str, Any]
    risk_trends: List[Dict[str, Any]]
    volume_trends: List[Dict[str, Any]]
    model_breakdown: List[Dict[str, Any]]
    compliance_status: Dict[str, Any]
    kpi_status: List[Dict[str, Any]]


# ============ Model Analytics Types ============

@dataclass
class HistoricalLog:
    event_id: str
    model: str
    provider: str
    environment: str
    timestamp: str
    latency_ms: int
    tokens_in: int
    tokens_out: int
    risk_score: float


@dataclass
class ModelSwitch:
    switch_id: str
    app_id: str
    previous_model: Optional[str]
    new_model: str
    previous_provider: Optional[str]
    new_provider: str
    switch_reason: Optional[str]
    environment: Optional[str]
    switched_at: str


@dataclass
class EnvironmentStats:
    environment: str
    event_count: int
    avg_latency_ms: float
    avg_risk_score: float
    error_rate: float
    models: List[str]


@dataclass
class ModelBenchmark:
    benchmark_id: str
    model: str
    environment: Optional[str]
    avg_latency_ms: float
    p95_latency_ms: float
    avg_risk_score: float
    error_rate: float
    throughput: float
    sample_count: int
    calculated_at: str


@dataclass
class ModelAnalyticsSummary:
    total_models: int
    total_events_7d: int
    avg_latency_ms: float
    top_model: Optional[str]
    environments: List[str]
    recent_switches: int


# ============ Drift & Alerts Types ============

@dataclass
class DriftMetrics:
    health_score: float
    current: Dict[str, float]
    baselines: Dict[str, float]
    last_updated: str


@dataclass
class DriftAlert:
    alert_id: str
    app_id: str
    model: Optional[str]
    alert_type: str
    severity: Literal["critical", "warning", "info"]
    current_value: float
    baseline_value: float
    deviation_percent: float
    created_at: str
    acknowledged_at: Optional[str]
    resolved_at: Optional[str]


@dataclass
class DriftTrend:
    date: str
    avg_deviation: float
    alert_count: int
    metrics: Dict[str, float]


@dataclass
class DriftBaseline:
    baseline_id: str
    app_id: str
    model: Optional[str]
    valid_from: str
    avg_risk_score: Optional[float]
    sample_count: int
    risk_distribution: Optional[Dict[str, int]]


@dataclass
class DriftScore:
    overall_score: float
    status: Literal["healthy", "warning", "critical"]
    by_model: List[Dict[str, Any]]


# ============ Drift V3 Types ============

@dataclass
class NotificationChannel:
    channel_id: str
    app_id: str
    channel_type: Literal["email", "slack", "webhook", "pagerduty", "teams"]
    name: str
    config: Dict[str, Any]
    severity_filter: List[str]
    is_enabled: bool
    created_at: str


@dataclass
class DriftSegment:
    segment_id: str
    app_id: str
    name: str
    description: Optional[str]
    filter_criteria: Dict[str, Any]
    is_active: bool
    created_at: str


@dataclass
class CorrelationEvent:
    correlation_id: str
    app_id: str
    primary_metric: str
    correlated_metric: str
    correlation_coefficient: float
    lag_minutes: int
    detected_at: str
    is_active: bool


@dataclass
class DistributionSnapshot:
    snapshot_id: str
    app_id: str
    metric_type: str
    distribution_data: Dict[str, Any]
    anomaly_score: float
    is_anomaly: bool
    captured_at: str


@dataclass
class SeasonalityPattern:
    pattern_id: str
    app_id: str
    metric_type: str
    day_of_week: int
    hour_of_day: int
    expected_value: float
    std_deviation: float
    sample_count: int


@dataclass
class BaselineStatistics:
    app_id: str
    model: Optional[str]
    metric_type: str
    mean: float
    std_dev: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float
    sample_count: int
    calculated_at: str


# ============ Tracing Types ============

SpanType = Literal["llm", "tool", "retrieval", "chain", "agent", "custom"]


@dataclass
class Trace:
    trace_id: str
    app_id: str
    name: Optional[str]
    start_time: str
    end_time: Optional[str]
    duration_ms: Optional[int]
    status: Literal["running", "completed", "error"]
    metadata: Dict[str, Any]
    tags: List[str]
    user_id: Optional[str]
    session_id: Optional[str]


@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    span_type: SpanType
    start_time: str
    end_time: Optional[str]
    duration_ms: Optional[int]
    status: Literal["running", "completed", "error"]
    model: Optional[str]
    provider: Optional[str]
    tokens_in: Optional[int]
    tokens_out: Optional[int]
    cost_usd: Optional[float]
    input: Optional[Dict[str, Any]]
    output: Optional[Dict[str, Any]]


# ============ Prompt Management Types ============

@dataclass
class Prompt:
    prompt_id: str
    name: str
    description: Optional[str]
    tags: List[str]
    is_active: bool
    created_at: str
    updated_at: str


@dataclass
class PromptVersion:
    version_id: str
    prompt_id: str
    version: int
    content: str
    variables: List[str]
    model_config: Dict[str, Any]
    is_published: bool
    is_default: bool
    commit_message: Optional[str]
    created_at: str


# ============ Evaluation Types ============

@dataclass
class EvalDataset:
    dataset_id: str
    name: str
    description: Optional[str]
    schema_type: Literal["qa", "conversation", "classification", "custom"]
    item_count: int
    tags: List[str]
    is_active: bool


@dataclass
class EvalDatasetItem:
    item_id: str
    dataset_id: str
    input: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]]
    context: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class EvalRun:
    run_id: str
    dataset_id: str
    name: Optional[str]
    status: Literal["pending", "running", "completed", "failed"]
    model: Optional[str]
    total_items: int
    completed_items: int
    passed_items: int
    failed_items: int
    avg_score: Optional[float]
    created_at: str


@dataclass
class EvalResult:
    result_id: str
    run_id: str
    item_id: str
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]]
    expected_output: Optional[Dict[str, Any]]
    scores: Dict[str, Dict[str, Any]]
    overall_score: Optional[float]
    passed: Optional[bool]
    latency_ms: Optional[int]


# ============ Cache Types ============

@dataclass
class CacheSettings:
    is_enabled: bool
    similarity_threshold: float
    ttl_seconds: int
    max_entries: int
    embedding_model: str


@dataclass
class CacheStats:
    total_entries: int
    total_hits: int
    total_tokens_saved: int
    total_cost_saved_usd: float
    hit_rate: float


@dataclass
class CacheLookupResult:
    hit: bool
    cache_id: Optional[str] = None
    output: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    reason: Optional[str] = None


# ============ Simulation Types ============

@dataclass
class Simulation:
    simulation_id: str
    name: str
    description: Optional[str]
    persona: Optional[Dict[str, Any]]
    scenario: str
    success_criteria: List[Dict[str, str]]
    max_turns: int
    model: Optional[str]
    status: Literal["draft", "ready", "running", "completed"]
    tags: List[str]


@dataclass
class SimulationRun:
    run_id: str
    simulation_id: str
    status: Literal["running", "completed", "failed", "stopped"]
    turn_count: int
    success: Optional[bool]
    criteria_results: List[Dict[str, Any]]
    summary: Optional[str]
    total_latency_ms: Optional[int]
    total_tokens: Optional[int]
    started_at: str
    completed_at: Optional[str]


@dataclass
class SimulationTurn:
    turn_id: str
    run_id: str
    turn_number: int
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_calls: Optional[List[Dict[str, Any]]]
    tool_results: Optional[List[Dict[str, Any]]]
    latency_ms: Optional[int]
    tokens_in: Optional[int]
    tokens_out: Optional[int]


@dataclass
class SimulationStats:
    total_simulations: int
    total_runs: int
    success_rate: float
    avg_turns: float


# ============ Integration Types ============

@dataclass
class Integration:
    integration_id: str
    type: Literal["slack", "teams", "discord"]
    webhook_url: str
    channel_name: Optional[str]
    events: List[str]
    is_active: bool
    created_at: str


# ============ Benchmark Types ============

@dataclass
class BenchmarkMetric:
    metric_name: str
    your_value: float
    industry_avg: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_90: float
    your_percentile: float
    rating: Literal["excellent", "good", "average", "below_average", "needs_attention"]
    comparison: str


@dataclass
class BenchmarkReport:
    tenant_id: str
    industry: str
    generated_at: str
    date_range: Dict[str, str]
    metrics: List[BenchmarkMetric]
    overall_rating: Literal["excellent", "good", "average", "below_average", "needs_attention"]
    summary: str


@dataclass
class Industry:
    id: str
    name: str
    description: str


# ============ Guardrail Types ============

@dataclass
class Guardrail:
    guardrail_id: str
    name: str
    description: Optional[str]
    rule_type: str
    action: Literal["flag", "block", "redact", "warn"]
    config: Dict[str, Any]
    is_active: bool
    priority: int
    created_at: str


@dataclass
class GuardrailStats:
    total_triggers: int
    by_guardrail: List[Dict[str, Any]]
    by_action: List[Dict[str, Any]]


# ============ Retention Types ============

@dataclass
class RetentionPolicy:
    policy_id: str
    name: str
    description: Optional[str]
    is_active: bool
    data_type: str
    retention_days: int
    compliance_hold: bool


@dataclass
class RetentionSummary:
    active_policies: int
    total_records_deleted: int
    last_execution: Optional[str]
    compliance_holds: int


# ============ Audit Types ============

@dataclass
class AuditLog:
    log_id: str
    action: str
    resource_type: str
    resource_id: Optional[str]
    user_id: Optional[str]
    ip_address: Optional[str]
    details: Dict[str, Any]
    created_at: str


# ============ Events Types ============

@dataclass
class Event:
    event_id: str
    app_id: str
    model: str
    provider: str
    input_prompt: str
    output_text: str
    risk_score: float
    latency_ms: int
    tokens_in: int
    tokens_out: int
    created_at: str
    classifications: Optional[Dict[str, Any]] = None


# ============ Custom Detection Types ============

@dataclass
class CustomDetection:
    detection_id: str
    name: str
    description: Optional[str]
    detection_type: Literal["keyword", "regex", "semantic", "llm_judge"]
    config: Dict[str, Any]
    severity: Literal["low", "medium", "high", "critical"]
    is_active: bool
    created_at: str


# ============ Webhook Types ============

@dataclass
class Webhook:
    webhook_id: str
    url: str
    events: List[str]
    is_active: bool
    secret: Optional[str]
    created_at: str
