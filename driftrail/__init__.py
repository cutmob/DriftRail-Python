"""
DriftRail Python SDK
AI Safety & Observability Platform

Usage:
    from driftrail import DriftRail, DriftRailEnterprise
    
    # Basic client
    client = DriftRail(api_key="dr_live_...", app_id="my-app")
    
    # Enterprise client with full features
    enterprise = DriftRailEnterprise(api_key="dr_live_...", app_id="my-app")
"""

from .client import DriftRail, DriftRailAsync, DriftRailEnterprise
from .types import (
    # Core types
    Provider,
    Message,
    SourceRef,
    ToolCall,
    InputPayload,
    OutputPayload,
    Metadata,
    IngestPayload,
    IngestResponse,
    # Guard types
    GuardMode,
    GuardAction,
    GuardTriggered,
    GuardClassification,
    GuardResult,
    GuardBlockedError,
    # Incident types
    Incident,
    IncidentStats,
    # Compliance types
    ComplianceFramework,
    ComplianceStatus,
    ComplianceScore,
    ComplianceReportMeta,
    CustomControl,
    # Executive types
    KpiTargets,
    ExecutiveMetrics,
    # Model Analytics types
    HistoricalLog,
    ModelSwitch,
    EnvironmentStats,
    ModelBenchmark,
    ModelAnalyticsSummary,
    # Drift types
    DriftMetrics,
    DriftAlert,
    DriftTrend,
    DriftBaseline,
    DriftScore,
    # Drift V3 types
    NotificationChannel,
    DriftSegment,
    CorrelationEvent,
    DistributionSnapshot,
    SeasonalityPattern,
    BaselineStatistics,
    # Tracing types
    SpanType,
    Trace,
    Span,
    # Prompt types
    Prompt,
    PromptVersion,
    # Evaluation types
    EvalDataset,
    EvalDatasetItem,
    EvalRun,
    EvalResult,
    # Cache types
    CacheSettings,
    CacheStats,
    CacheLookupResult,
    # Simulation types
    Simulation,
    SimulationRun,
    SimulationTurn,
    SimulationStats,
    # Integration types
    Integration,
    # Benchmark types
    BenchmarkMetric,
    BenchmarkReport,
    Industry,
    # Guardrail types
    Guardrail,
    GuardrailStats,
    # Retention types
    RetentionPolicy,
    RetentionSummary,
    # Audit types
    AuditLog,
    # Event types
    Event,
    # Custom Detection types
    CustomDetection,
    # Webhook types
    Webhook,
)

__version__ = "2.1.0"
__all__ = [
    # Clients
    "DriftRail",
    "DriftRailAsync",
    "DriftRailEnterprise",
    # Core types
    "Provider",
    "Message",
    "SourceRef",
    "ToolCall",
    "InputPayload",
    "OutputPayload",
    "Metadata",
    "IngestPayload",
    "IngestResponse",
    # Guard types
    "GuardMode",
    "GuardAction",
    "GuardTriggered",
    "GuardClassification",
    "GuardResult",
    "GuardBlockedError",
    # Incident types
    "Incident",
    "IncidentStats",
    # Compliance types
    "ComplianceFramework",
    "ComplianceStatus",
    "ComplianceScore",
    "ComplianceReportMeta",
    "CustomControl",
    # Executive types
    "KpiTargets",
    "ExecutiveMetrics",
    # Model Analytics types
    "HistoricalLog",
    "ModelSwitch",
    "EnvironmentStats",
    "ModelBenchmark",
    "ModelAnalyticsSummary",
    # Drift types
    "DriftMetrics",
    "DriftAlert",
    "DriftTrend",
    "DriftBaseline",
    "DriftScore",
    # Drift V3 types
    "NotificationChannel",
    "DriftSegment",
    "CorrelationEvent",
    "DistributionSnapshot",
    "SeasonalityPattern",
    "BaselineStatistics",
    # Tracing types
    "SpanType",
    "Trace",
    "Span",
    # Prompt types
    "Prompt",
    "PromptVersion",
    # Evaluation types
    "EvalDataset",
    "EvalDatasetItem",
    "EvalRun",
    "EvalResult",
    # Cache types
    "CacheSettings",
    "CacheStats",
    "CacheLookupResult",
    # Simulation types
    "Simulation",
    "SimulationRun",
    "SimulationTurn",
    "SimulationStats",
    # Integration types
    "Integration",
    # Benchmark types
    "BenchmarkMetric",
    "BenchmarkReport",
    "Industry",
    # Guardrail types
    "Guardrail",
    "GuardrailStats",
    # Retention types
    "RetentionPolicy",
    "RetentionSummary",
    # Audit types
    "AuditLog",
    # Event types
    "Event",
    # Custom Detection types
    "CustomDetection",
    # Webhook types
    "Webhook",
]
