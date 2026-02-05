"""
DriftRail Type Definitions
"""

from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field, asdict

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
