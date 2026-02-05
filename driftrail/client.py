"""
DriftRail Client - Complete SDK with all enterprise features.
"""

import json
from typing import Optional, Dict, Any, Union, List
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor

from .types import (
    IngestPayload, IngestResponse, InputPayload, OutputPayload, 
    Metadata, Provider, GuardResult, GuardBlockedError
)

DEFAULT_BASE_URL = "https://api.driftrail.com"


class DriftRail:
    """Synchronous DriftRail client."""

    def __init__(self, api_key: str, app_id: str, base_url: str = DEFAULT_BASE_URL,
                 timeout: int = 30, fail_open: bool = True, guard_mode: str = "fail_open"):
        self.api_key = api_key
        self.app_id = app_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.fail_open = fail_open
        self.guard_mode = guard_mode
        self._executor = ThreadPoolExecutor(max_workers=4)

    def ingest(self, model: str, provider: Provider, input: Union[InputPayload, Dict[str, Any]],
               output: Union[OutputPayload, Dict[str, Any]], metadata: Optional[Union[Metadata, Dict[str, Any]]] = None) -> IngestResponse:
        if isinstance(input, dict):
            input_payload = InputPayload(prompt=input.get("prompt", ""), messages=input.get("messages"),
                retrieved_sources=input.get("retrievedSources") or input.get("retrieved_sources"))
        else:
            input_payload = input
        if isinstance(output, dict):
            output_payload = OutputPayload(text=output.get("text", ""),
                tool_calls=output.get("toolCalls") or output.get("tool_calls"))
        else:
            output_payload = output
        metadata_payload = None
        if metadata:
            if isinstance(metadata, dict):
                metadata_payload = Metadata(latency_ms=metadata.get("latencyMs") or metadata.get("latency_ms"),
                    tokens_in=metadata.get("tokensIn") or metadata.get("tokens_in"),
                    tokens_out=metadata.get("tokensOut") or metadata.get("tokens_out"),
                    temperature=metadata.get("temperature"))
            else:
                metadata_payload = metadata
        payload = IngestPayload(model=model, provider=provider, input=input_payload, output=output_payload, metadata=metadata_payload)
        return self._send_ingest(payload)

    def ingest_async(self, model: str, provider: Provider, input: Union[InputPayload, Dict[str, Any]],
                     output: Union[OutputPayload, Dict[str, Any]], metadata: Optional[Union[Metadata, Dict[str, Any]]] = None) -> None:
        self._executor.submit(self.ingest, model, provider, input, output, metadata)


    def guard(self, output: str, input: Optional[str] = None, mode: str = "strict", timeout_ms: int = 100) -> GuardResult:
        url = f"{self.base_url}/api/guard"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}", "X-App-Id": self.app_id}
        payload = {"output": output, "input": input or "", "mode": mode, "timeout_ms": min(timeout_ms, 500), "app_id": self.app_id}
        try:
            data = json.dumps(payload).encode("utf-8")
            req = Request(url, data=data, headers=headers, method="POST")
            with urlopen(req, timeout=max(1, timeout_ms / 1000 + 1)) as response:
                result = GuardResult.from_dict(json.loads(response.read().decode("utf-8")))
                if self.guard_mode == "fail_closed" and not result.allowed:
                    raise GuardBlockedError(result)
                return result
        except GuardBlockedError:
            raise
        except HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            if self.guard_mode == "fail_closed":
                raise Exception(f"Guard API error: HTTP {e.code}: {error_body}")
            return GuardResult(allowed=True, action="allow", output=output, triggered=[], latency_ms=0, fallback=True, error=f"HTTP {e.code}: {error_body}")
        except Exception as e:
            if self.guard_mode == "fail_closed":
                raise Exception(f"Guard API error: {e}")
            return GuardResult(allowed=True, action="allow", output=output, triggered=[], latency_ms=0, fallback=True, error=str(e))

    def _send_ingest(self, payload: IngestPayload) -> IngestResponse:
        url = f"{self.base_url}/ingest"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}", "X-App-Id": self.app_id}
        try:
            data = json.dumps(payload.to_dict()).encode("utf-8")
            req = Request(url, data=data, headers=headers, method="POST")
            with urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                return IngestResponse(success=result.get("success", False), event_id=result.get("event_id"),
                    job_id=result.get("job_id"), duplicate=result.get("duplicate", False))
        except HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            if self.fail_open:
                return IngestResponse(success=False, error=f"HTTP {e.code}: {error_body}")
            raise
        except URLError as e:
            if self.fail_open:
                return IngestResponse(success=False, error=f"Network error: {e.reason}")
            raise
        except Exception as e:
            if self.fail_open:
                return IngestResponse(success=False, error=str(e))
            raise

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    def __enter__(self) -> "DriftRail":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class DriftRailAsync:
    """Async DriftRail client using aiohttp."""

    def __init__(self, api_key: str, app_id: str, base_url: str = DEFAULT_BASE_URL, timeout: int = 30, fail_open: bool = True):
        self.api_key = api_key
        self.app_id = app_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.fail_open = fail_open
        self._session: Any = None

    async def _get_session(self) -> Any:
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    async def ingest(self, model: str, provider: Provider, input: Union[InputPayload, Dict[str, Any]],
                     output: Union[OutputPayload, Dict[str, Any]], metadata: Optional[Union[Metadata, Dict[str, Any]]] = None) -> IngestResponse:
        import aiohttp
        if isinstance(input, dict):
            input_payload = InputPayload(prompt=input.get("prompt", ""), messages=input.get("messages"),
                retrieved_sources=input.get("retrievedSources") or input.get("retrieved_sources"))
        else:
            input_payload = input
        if isinstance(output, dict):
            output_payload = OutputPayload(text=output.get("text", ""), tool_calls=output.get("toolCalls") or output.get("tool_calls"))
        else:
            output_payload = output
        metadata_payload = None
        if metadata:
            if isinstance(metadata, dict):
                metadata_payload = Metadata(latency_ms=metadata.get("latencyMs") or metadata.get("latency_ms"),
                    tokens_in=metadata.get("tokensIn") or metadata.get("tokens_in"),
                    tokens_out=metadata.get("tokensOut") or metadata.get("tokens_out"), temperature=metadata.get("temperature"))
            else:
                metadata_payload = metadata
        payload = IngestPayload(model=model, provider=provider, input=input_payload, output=output_payload, metadata=metadata_payload)
        url = f"{self.base_url}/ingest"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}", "X-App-Id": self.app_id}
        try:
            session = await self._get_session()
            async with session.post(url, json=payload.to_dict(), headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                result = await response.json()
                if response.status >= 400:
                    if self.fail_open:
                        return IngestResponse(success=False, error=f"HTTP {response.status}: {result.get('error', 'Unknown')}")
                    raise Exception(f"HTTP {response.status}: {result}")
                return IngestResponse(success=result.get("success", False), event_id=result.get("event_id"),
                    job_id=result.get("job_id"), duplicate=result.get("duplicate", False))
        except Exception as e:
            if self.fail_open:
                return IngestResponse(success=False, error=str(e))
            raise

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "DriftRailAsync":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()



class DriftRailEnterprise(DriftRail):
    """Enterprise DriftRail client with full monitoring features."""

    def _api_request(self, endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        try:
            body = json.dumps(data).encode("utf-8") if data else None
            req = Request(url, data=body, headers=headers, method=method)
            with urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            raise Exception(f"HTTP {e.code}: {error_body}")

    # Incidents
    def list_incidents(self, status: Optional[List[str]] = None, severity: Optional[List[str]] = None, limit: int = 50) -> Dict[str, Any]:
        params = []
        if status: params.append(f"status={','.join(status)}")
        if severity: params.append(f"severity={','.join(severity)}")
        params.append(f"limit={limit}")
        return self._api_request(f"/api/incidents?{'&'.join(params)}")

    def create_incident(self, title: str, severity: str, incident_type: str, description: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request("/api/incidents", "POST", {"title": title, "severity": severity, "incident_type": incident_type, "description": description})

    def update_incident_status(self, incident_id: str, status: str, comment: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/incidents/{incident_id}/status", "PATCH", {"status": status, "comment": comment})

    def get_incident_stats(self) -> Dict[str, Any]:
        return self._api_request("/api/incidents/stats")

    # Compliance
    def get_compliance_status(self) -> Dict[str, Any]:
        return self._api_request("/api/compliance/status")

    def get_compliance_score(self) -> Dict[str, Any]:
        return self._api_request("/api/compliance-reports/score")

    def get_compliance_reports(self) -> Dict[str, Any]:
        return self._api_request("/api/compliance-reports/reports")

    def get_compliance_frameworks(self) -> Dict[str, Any]:
        return self._api_request("/api/compliance-reports/frameworks")

    def generate_compliance_report(self, framework: str, format: str = "json", include_evidence: bool = True, include_ai_analysis: bool = False) -> Dict[str, Any]:
        return self._api_request("/api/compliance-reports/reports", "POST", {"framework": framework, "format": format, "include_evidence": include_evidence, "include_ai_analysis": include_ai_analysis})

    def create_custom_framework(self, name: str, controls: List[Dict[str, Any]], description: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request("/api/compliance-reports/frameworks/custom", "POST", {"name": name, "description": description, "controls": controls})

    # Model Comparison
    def get_model_leaderboard(self, metric: str = "avg_risk_score") -> Dict[str, Any]:
        return self._api_request(f"/api/models/leaderboard?metric={metric}")

    def create_model_comparison(self, name: str, model_a: str, model_b: str) -> Dict[str, Any]:
        return self._api_request("/api/models/comparisons", "POST", {"name": name, "model_a": model_a, "model_b": model_b})

    # Exports
    def create_export(self, export_type: str, format: str = "json", date_from: Optional[str] = None, date_to: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request("/api/exports", "POST", {"export_type": export_type, "format": format, "date_from": date_from, "date_to": date_to})

    def get_export_status(self, export_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/exports/{export_id}")

    # Brand Safety
    def check_brand_safety(self, text: str, location: str = "output") -> Dict[str, Any]:
        return self._api_request("/api/brand-safety/check", "POST", {"text": text, "location": location})

    def create_brand_safety_rule(self, name: str, rule_type: str, config: Dict[str, Any], action: str = "flag", severity: str = "medium") -> Dict[str, Any]:
        return self._api_request("/api/brand-safety/rules", "POST", {"name": name, "rule_type": rule_type, "config": config, "action": action, "severity": severity})

    # Executive Dashboard
    def get_executive_metrics(self, period: str = "7d") -> Dict[str, Any]:
        return self._api_request(f"/api/executive?period={period}")

    def get_kpi_targets(self) -> Dict[str, Any]:
        return self._api_request("/api/executive/targets")

    def update_kpi_targets(self, targets: Dict[str, Any]) -> Dict[str, Any]:
        return self._api_request("/api/executive/targets", "PUT", targets)

    def export_executive_metrics(self, period: str = "7d", format: str = "json") -> Dict[str, Any]:
        return self._api_request("/api/executive/export", "POST", {"period": period, "format": format})


    # Model Analytics
    def get_model_analytics_summary(self) -> Dict[str, Any]:
        return self._api_request("/api/model-analytics/summary")

    def get_historical_logs(self, model: Optional[str] = None, environment: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, min_risk_score: Optional[float] = None, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        params = [f"limit={limit}", f"offset={offset}"]
        if model: params.append(f"model={model}")
        if environment: params.append(f"environment={environment}")
        if start_time: params.append(f"start_time={start_time}")
        if end_time: params.append(f"end_time={end_time}")
        if min_risk_score is not None: params.append(f"min_risk_score={min_risk_score}")
        return self._api_request(f"/api/model-analytics/logs?{'&'.join(params)}")

    def get_model_switches(self, app_id: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        params = [f"limit={limit}"]
        if app_id: params.append(f"app_id={app_id}")
        return self._api_request(f"/api/model-analytics/switches?{'&'.join(params)}")

    def record_model_switch(self, app_id: str, new_model: str, new_provider: str, previous_model: Optional[str] = None, previous_provider: Optional[str] = None, switch_reason: Optional[str] = None, environment: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request("/api/model-analytics/switches", "POST", {"app_id": app_id, "new_model": new_model, "new_provider": new_provider, "previous_model": previous_model, "previous_provider": previous_provider, "switch_reason": switch_reason, "environment": environment})

    def get_environment_comparison(self, model: Optional[str] = None, app_id: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        params = [f"days={days}"]
        if model: params.append(f"model={model}")
        if app_id: params.append(f"app_id={app_id}")
        return self._api_request(f"/api/model-analytics/environments?{'&'.join(params)}")

    def get_model_benchmarks(self, model: Optional[str] = None, environment: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        params = [f"limit={limit}"]
        if model: params.append(f"model={model}")
        if environment: params.append(f"environment={environment}")
        return self._api_request(f"/api/model-analytics/benchmarks?{'&'.join(params)}")

    def calculate_model_benchmark(self, model: str, environment: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        return self._api_request("/api/model-analytics/benchmarks/calculate", "POST", {"model": model, "environment": environment, "days": days})

    # Drift & Alerts
    def get_drift_metrics(self, app_id: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/alerts/metrics{'?app_id=' + app_id if app_id else ''}")

    def get_drift_alerts(self, severity: Optional[str] = None, unresolved: bool = False, app_id: Optional[str] = None, alert_type: Optional[str] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        params = [f"limit={limit}", f"offset={offset}"]
        if severity: params.append(f"severity={severity}")
        if unresolved: params.append("unresolved=true")
        if app_id: params.append(f"app_id={app_id}")
        if alert_type: params.append(f"alert_type={alert_type}")
        return self._api_request(f"/api/alerts?{'&'.join(params)}")

    def acknowledge_alert(self, alert_id: str) -> Dict[str, Any]:
        return self._api_request("/api/alerts", "PATCH", {"alert_id": alert_id, "action": "acknowledge"})

    def resolve_alert(self, alert_id: str, notes: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request("/api/alerts", "PATCH", {"alert_id": alert_id, "action": "resolve", "notes": notes})

    def get_drift_trends(self, days: int = 7) -> Dict[str, Any]:
        return self._api_request(f"/api/alerts/trends?days={days}")

    def get_drift_baselines(self, app_id: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/alerts/baselines{'?app_id=' + app_id if app_id else ''}")

    def refresh_baseline(self, baseline_id: str) -> Dict[str, Any]:
        return self._api_request("/api/alerts/baselines", "POST", {"baseline_id": baseline_id, "action": "refresh"})


    # Drift Detection V3
    def get_model_drift_comparison(self, app_id: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/drift/models{'?app_id=' + app_id if app_id else ''}")

    def get_drift_heatmap(self, days: int = 30, app_id: Optional[str] = None) -> Dict[str, Any]:
        params = [f"days={days}"]
        if app_id: params.append(f"app_id={app_id}")
        return self._api_request(f"/api/drift/heatmap?{'&'.join(params)}")

    def get_drift_thresholds(self, app_id: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/drift/thresholds{'?app_id=' + app_id if app_id else ''}")

    def update_drift_thresholds(self, thresholds: Dict[str, Dict[str, Any]], app_id: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request("/api/drift/thresholds", "PUT", {"thresholds": thresholds, "app_id": app_id})

    def get_baseline_history(self, app_id: Optional[str] = None, model: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        params = [f"limit={limit}"]
        if app_id: params.append(f"app_id={app_id}")
        if model: params.append(f"model={model}")
        return self._api_request(f"/api/drift/baselines/history?{'&'.join(params)}")

    def get_drift_deployment_correlations(self, days: int = 30, app_id: Optional[str] = None) -> Dict[str, Any]:
        params = [f"days={days}"]
        if app_id: params.append(f"app_id={app_id}")
        return self._api_request(f"/api/drift/correlations?{'&'.join(params)}")

    def record_deployment(self, app_id: str, deployment_type: Optional[str] = None, version: Optional[str] = None, description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._api_request("/api/drift/deployments", "POST", {"app_id": app_id, "deployment_type": deployment_type, "version": version, "description": description, "metadata": metadata})

    def get_drift_predictions(self, app_id: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/drift/predictions{'?app_id=' + app_id if app_id else ''}")

    def get_drift_score(self, app_id: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/drift/score{'?app_id=' + app_id if app_id else ''}")

    def get_notification_channels(self, app_id: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/drift/notifications/channels{'?app_id=' + app_id if app_id else ''}")

    def create_notification_channel(self, channel_type: str, name: str, config: Dict[str, Any], severity_filter: Optional[List[str]] = None, is_enabled: bool = True, app_id: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request("/api/drift/notifications/channels", "POST", {"channel_type": channel_type, "name": name, "config": config, "severity_filter": severity_filter or ["critical", "warning"], "is_enabled": is_enabled, "app_id": app_id})

    def update_notification_channel(self, channel_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        return self._api_request(f"/api/drift/notifications/channels/{channel_id}", "PUT", updates)

    def delete_notification_channel(self, channel_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/drift/notifications/channels/{channel_id}", "DELETE")

    def get_drift_segments(self, app_id: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/drift/segments{'?app_id=' + app_id if app_id else ''}")

    def create_drift_segment(self, name: str, filter_criteria: Dict[str, Any], description: Optional[str] = None, app_id: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request("/api/drift/segments", "POST", {"name": name, "description": description, "filter_criteria": filter_criteria, "app_id": app_id})

    def get_correlation_events(self, app_id: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        params = [f"days={days}"]
        if app_id: params.append(f"app_id={app_id}")
        return self._api_request(f"/api/drift/correlations/events?{'&'.join(params)}")

    def get_distribution_analysis(self, app_id: Optional[str] = None, metric_type: Optional[str] = None) -> Dict[str, Any]:
        params = []
        if app_id: params.append(f"app_id={app_id}")
        if metric_type: params.append(f"metric_type={metric_type}")
        return self._api_request(f"/api/drift/distribution{'?' + '&'.join(params) if params else ''}")

    def get_seasonality_patterns(self, app_id: Optional[str] = None, metric_type: Optional[str] = None) -> Dict[str, Any]:
        params = []
        if app_id: params.append(f"app_id={app_id}")
        if metric_type: params.append(f"metric_type={metric_type}")
        return self._api_request(f"/api/drift/seasonality{'?' + '&'.join(params) if params else ''}")

    def get_baseline_statistics(self, app_id: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        params = []
        if app_id: params.append(f"app_id={app_id}")
        if model: params.append(f"model={model}")
        return self._api_request(f"/api/drift/statistics{'?' + '&'.join(params) if params else ''}")


    # Distributed Tracing
    def start_trace(self, app_id: str, name: Optional[str] = None, user_id: Optional[str] = None, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._api_request("/api/traces", "POST", {"app_id": app_id, "name": name, "user_id": user_id, "session_id": session_id, "metadata": metadata or {}, "tags": tags or []})

    def end_trace(self, trace_id: str, status: str = "completed") -> Dict[str, Any]:
        return self._api_request(f"/api/traces/{trace_id}", "PATCH", {"status": status})

    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/traces/{trace_id}")

    def list_traces(self, app_id: Optional[str] = None, status: Optional[str] = None, user_id: Optional[str] = None, session_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        params = [f"limit={limit}", f"offset={offset}"]
        if app_id: params.append(f"app_id={app_id}")
        if status: params.append(f"status={status}")
        if user_id: params.append(f"user_id={user_id}")
        if session_id: params.append(f"session_id={session_id}")
        return self._api_request(f"/api/traces?{'&'.join(params)}")

    def start_span(self, trace_id: str, name: str, span_type: str, parent_span_id: Optional[str] = None, model: Optional[str] = None, provider: Optional[str] = None, input: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._api_request("/api/traces/spans", "POST", {"trace_id": trace_id, "name": name, "span_type": span_type, "parent_span_id": parent_span_id, "model": model, "provider": provider, "input": input, "metadata": metadata})

    def end_span(self, span_id: str, status: str = "completed", status_message: Optional[str] = None, output: Optional[Dict[str, Any]] = None, tokens_in: Optional[int] = None, tokens_out: Optional[int] = None, cost_usd: Optional[float] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/traces/spans/{span_id}", "PATCH", {"status": status, "status_message": status_message, "output": output, "tokens_in": tokens_in, "tokens_out": tokens_out, "cost_usd": cost_usd})

    # Prompt Management
    def create_prompt(self, name: str, description: Optional[str] = None, content: Optional[str] = None, variables: Optional[List[str]] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._api_request("/api/prompts", "POST", {"name": name, "description": description, "content": content, "variables": variables or [], "tags": tags or []})

    def get_prompt(self, prompt_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/prompts/{prompt_id}")

    def list_prompts(self, is_active: Optional[bool] = None, tags: Optional[List[str]] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        params = [f"limit={limit}", f"offset={offset}"]
        if is_active is not None: params.append(f"is_active={str(is_active).lower()}")
        if tags: params.append(f"tags={','.join(tags)}")
        return self._api_request(f"/api/prompts?{'&'.join(params)}")

    def create_prompt_version(self, prompt_id: str, content: str, variables: Optional[List[str]] = None, model_config: Optional[Dict[str, Any]] = None, commit_message: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/prompts/{prompt_id}/versions", "POST", {"content": content, "variables": variables or [], "model_config": model_config or {}, "commit_message": commit_message})

    def deploy_prompt_version(self, version_id: str, environment: str) -> Dict[str, Any]:
        return self._api_request("/api/prompts/deploy", "POST", {"version_id": version_id, "environment": environment})

    def get_deployed_prompt(self, prompt_id: str, environment: str) -> Dict[str, Any]:
        return self._api_request(f"/api/prompts/{prompt_id}/deployed?environment={environment}")

    def rollback_prompt(self, prompt_id: str, environment: str, version_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/prompts/{prompt_id}/rollback", "POST", {"environment": environment, "version_id": version_id})


    # Evaluation Framework
    def create_dataset(self, name: str, description: Optional[str] = None, schema_type: str = "qa", tags: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._api_request("/api/evaluations/datasets", "POST", {"name": name, "description": description, "schema_type": schema_type, "tags": tags or []})

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/evaluations/datasets/{dataset_id}")

    def list_datasets(self, schema_type: Optional[str] = None, is_active: Optional[bool] = None, limit: int = 50) -> Dict[str, Any]:
        params = [f"limit={limit}"]
        if schema_type: params.append(f"schema_type={schema_type}")
        if is_active is not None: params.append(f"is_active={str(is_active).lower()}")
        return self._api_request(f"/api/evaluations/datasets?{'&'.join(params)}")

    def add_dataset_items(self, dataset_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._api_request(f"/api/evaluations/datasets/{dataset_id}/items", "POST", {"items": items})

    def create_eval_run(self, dataset_id: str, evaluators: List[Dict[str, Any]], name: Optional[str] = None, model: Optional[str] = None, prompt_version_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._api_request("/api/evaluations/runs", "POST", {"dataset_id": dataset_id, "name": name, "model": model, "prompt_version_id": prompt_version_id, "evaluators": evaluators, "config": config or {}})

    def get_eval_run(self, run_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/evaluations/runs/{run_id}")

    def list_eval_runs(self, dataset_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        params = [f"limit={limit}"]
        if dataset_id: params.append(f"dataset_id={dataset_id}")
        if status: params.append(f"status={status}")
        return self._api_request(f"/api/evaluations/runs?{'&'.join(params)}")

    def submit_eval_result(self, run_id: str, item_id: str, output: Dict[str, Any], scores: Dict[str, Dict[str, Any]], latency_ms: Optional[int] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/evaluations/runs/{run_id}/results", "POST", {"item_id": item_id, "output": output, "scores": scores, "latency_ms": latency_ms})

    # Semantic Caching
    def get_cache_settings(self) -> Dict[str, Any]:
        return self._api_request("/api/cache/settings")

    def update_cache_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        return self._api_request("/api/cache/settings", "PUT", settings)

    def get_cache_stats(self) -> Dict[str, Any]:
        return self._api_request("/api/cache/stats")

    def cache_lookup(self, input: Union[str, Dict[str, Any]], model: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request("/api/cache/lookup", "POST", {"input": input, "model": model})

    def cache_store(self, input: Union[str, Dict[str, Any]], output: str, model: str, provider: Optional[str] = None, app_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._api_request("/api/cache/store", "POST", {"input": input, "output": output, "model": model, "provider": provider, "app_id": app_id, "metadata": metadata})

    def invalidate_cache(self, cache_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/cache/{cache_id}", "DELETE")

    def clear_cache(self, model: Optional[str] = None, app_id: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request("/api/cache/clear", "POST", {"model": model, "app_id": app_id})


    # Agent Simulation
    def create_simulation(self, name: str, scenario: str, description: Optional[str] = None, persona: Optional[Dict[str, Any]] = None, success_criteria: Optional[List[Dict[str, str]]] = None, max_turns: int = 10, model: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._api_request("/api/simulations", "POST", {"name": name, "scenario": scenario, "description": description, "persona": persona, "success_criteria": success_criteria or [], "max_turns": max_turns, "model": model, "tags": tags or []})

    def get_simulation(self, simulation_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/simulations/{simulation_id}")

    def list_simulations(self, status: Optional[str] = None, tags: Optional[List[str]] = None, limit: int = 50) -> Dict[str, Any]:
        params = [f"limit={limit}"]
        if status: params.append(f"status={status}")
        if tags: params.append(f"tags={','.join(tags)}")
        return self._api_request(f"/api/simulations?{'&'.join(params)}")

    def run_simulation(self, simulation_id: str, max_turns: Optional[int] = None, model: Optional[str] = None) -> Dict[str, Any]:
        config = {}
        if max_turns: config["max_turns"] = max_turns
        if model: config["model"] = model
        return self._api_request(f"/api/simulations/{simulation_id}/run", "POST", config)

    def get_simulation_run(self, run_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/simulations/runs/{run_id}")

    def add_simulation_turn(self, run_id: str, turn_number: int, role: str, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None, tool_results: Optional[List[Dict[str, Any]]] = None, latency_ms: Optional[int] = None, tokens_in: Optional[int] = None, tokens_out: Optional[int] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/simulations/runs/{run_id}/turns", "POST", {"turn_number": turn_number, "role": role, "content": content, "tool_calls": tool_calls, "tool_results": tool_results, "latency_ms": latency_ms, "tokens_in": tokens_in, "tokens_out": tokens_out})

    def complete_simulation_run(self, run_id: str, success: bool, criteria_results: List[Dict[str, Any]], summary: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/simulations/runs/{run_id}/complete", "POST", {"success": success, "criteria_results": criteria_results, "summary": summary})

    def get_simulation_stats(self) -> Dict[str, Any]:
        return self._api_request("/api/simulations/stats")

    # Integrations
    def get_integrations(self) -> Dict[str, Any]:
        return self._api_request("/api/integrations")

    def create_integration(self, type: str, webhook_url: str, channel_name: Optional[str] = None, events: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._api_request("/api/integrations", "POST", {"type": type, "webhook_url": webhook_url, "channel_name": channel_name, "events": events or ["high_risk", "incident"]})

    def test_integration(self, webhook_url: str, type: str) -> Dict[str, Any]:
        return self._api_request("/api/integrations/test", "POST", {"webhook_url": webhook_url, "type": type})

    def update_integration(self, integration_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        return self._api_request(f"/api/integrations/{integration_id}", "PATCH", updates)

    def delete_integration(self, integration_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/integrations/{integration_id}", "DELETE")

    # Benchmarks
    def get_industries(self) -> Dict[str, Any]:
        return self._api_request("/api/benchmarks/industries")

    def get_benchmark_report(self, industry: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request(f"/api/benchmarks{'?industry=' + industry if industry else ''}")

    def set_tenant_industry(self, industry: str) -> Dict[str, Any]:
        return self._api_request("/api/benchmarks/industry", "PATCH", {"industry": industry})


    # Retention Policies
    def get_retention_policies(self) -> Dict[str, Any]:
        return self._api_request("/api/retention")

    def get_retention_summary(self) -> Dict[str, Any]:
        return self._api_request("/api/retention/summary")

    def create_retention_policy(self, name: str, data_type: str, retention_days: int, description: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request("/api/retention", "POST", {"name": name, "data_type": data_type, "retention_days": retention_days, "description": description})

    # Guardrails
    def get_guardrails(self) -> Dict[str, Any]:
        return self._api_request("/api/guardrails")

    def get_guardrail_stats(self) -> Dict[str, Any]:
        return self._api_request("/api/guardrails/stats")

    def create_guardrail(self, name: str, rule_type: str, action: str, description: Optional[str] = None, config: Optional[Dict[str, Any]] = None, priority: int = 0) -> Dict[str, Any]:
        return self._api_request("/api/guardrails", "POST", {"name": name, "description": description, "rule_type": rule_type, "action": action, "config": config or {}, "priority": priority})

    def update_guardrail(self, guardrail_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        return self._api_request(f"/api/guardrails/{guardrail_id}", "PATCH", updates)

    def delete_guardrail(self, guardrail_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/guardrails/{guardrail_id}", "DELETE")

    # Audit Logs
    def get_audit_logs(self, action: Optional[str] = None, resource_type: Optional[str] = None, user_id: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        params = [f"limit={limit}", f"offset={offset}"]
        if action: params.append(f"action={action}")
        if resource_type: params.append(f"resource_type={resource_type}")
        if user_id: params.append(f"user_id={user_id}")
        if start_time: params.append(f"start_time={start_time}")
        if end_time: params.append(f"end_time={end_time}")
        return self._api_request(f"/api/audit?{'&'.join(params)}")

    # Events
    def get_events(self, app_id: Optional[str] = None, model: Optional[str] = None, min_risk_score: Optional[float] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        params = [f"limit={limit}", f"offset={offset}"]
        if app_id: params.append(f"app_id={app_id}")
        if model: params.append(f"model={model}")
        if min_risk_score is not None: params.append(f"min_risk_score={min_risk_score}")
        if start_time: params.append(f"start_time={start_time}")
        if end_time: params.append(f"end_time={end_time}")
        return self._api_request(f"/api/events?{'&'.join(params)}")

    def get_event(self, event_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/events/{event_id}")

    def get_live_events(self, limit: int = 50) -> Dict[str, Any]:
        return self._api_request(f"/api/events/live?limit={limit}")

    # Custom Detections
    def get_custom_detections(self) -> Dict[str, Any]:
        return self._api_request("/api/detections")

    def create_custom_detection(self, name: str, detection_type: str, config: Dict[str, Any], severity: str = "medium", description: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request("/api/detections", "POST", {"name": name, "description": description, "detection_type": detection_type, "config": config, "severity": severity})

    def update_custom_detection(self, detection_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        return self._api_request(f"/api/detections/{detection_id}", "PATCH", updates)

    def delete_custom_detection(self, detection_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/detections/{detection_id}", "DELETE")

    # Webhooks
    def get_webhooks(self) -> Dict[str, Any]:
        return self._api_request("/api/webhooks")

    def create_webhook(self, url: str, events: List[str], secret: Optional[str] = None) -> Dict[str, Any]:
        return self._api_request("/api/webhooks", "POST", {"url": url, "events": events, "secret": secret})

    def update_webhook(self, webhook_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        return self._api_request(f"/api/webhooks/{webhook_id}", "PATCH", updates)

    def delete_webhook(self, webhook_id: str) -> Dict[str, Any]:
        return self._api_request(f"/api/webhooks/{webhook_id}", "DELETE")

    # Classifications
    def get_classifications(self, event_id: Optional[str] = None, classification_type: Optional[str] = None, min_score: Optional[float] = None, limit: int = 100) -> Dict[str, Any]:
        params = [f"limit={limit}"]
        if event_id: params.append(f"event_id={event_id}")
        if classification_type: params.append(f"type={classification_type}")
        if min_score is not None: params.append(f"min_score={min_score}")
        return self._api_request(f"/api/classifications?{'&'.join(params)}")

    # Stats
    def get_stats(self, period: str = "7d", app_id: Optional[str] = None) -> Dict[str, Any]:
        params = [f"period={period}"]
        if app_id: params.append(f"app_id={app_id}")
        return self._api_request(f"/api/stats?{'&'.join(params)}")
