"""
DriftRail Client - Sync and Async implementations
"""

import json
from typing import Optional, Dict, Any, Union
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor

from .types import (
    IngestPayload, IngestResponse, InputPayload, OutputPayload, 
    Metadata, Provider, GuardResult, GuardBlockedError
)

DEFAULT_BASE_URL = "https://api.driftrail.com"


class DriftRail:
    """
    Synchronous DriftRail client.
    
    Usage:
        client = DriftRail(api_key="dr_live_...", app_id="my-app")
        
        response = client.ingest(
            model="gpt-5",
            provider="openai",
            input={"prompt": "Hello"},
            output={"text": "Hi there!"}
        )
    """

    def __init__(
        self,
        api_key: str,
        app_id: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 30,
        fail_open: bool = True,
        guard_mode: str = "fail_open",
    ):
        """
        Initialize DriftRail client.
        
        Args:
            api_key: Your DriftRail API key (dr_live_... or dr_test_...)
            app_id: Your application identifier
            base_url: API base URL (default: https://api.driftrail.com)
            timeout: Request timeout in seconds
            fail_open: If True, errors are logged but don't raise exceptions
            guard_mode: "fail_open" (default) or "fail_closed" for guard() calls
        """
        self.api_key = api_key
        self.app_id = app_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.fail_open = fail_open
        self.guard_mode = guard_mode
        self._executor = ThreadPoolExecutor(max_workers=4)

    def ingest(
        self,
        model: str,
        provider: Provider,
        input: Union[InputPayload, Dict[str, Any]],
        output: Union[OutputPayload, Dict[str, Any]],
        metadata: Optional[Union[Metadata, Dict[str, Any]]] = None,
    ) -> IngestResponse:
        """
        Ingest an LLM interaction for classification.
        
        Args:
            model: Model name (e.g., "gpt-5", "claude-3")
            provider: Provider name ("openai", "google", "anthropic", "other")
            input: Input payload with prompt and optional messages/sources
            output: Output payload with text and optional tool calls
            metadata: Optional metadata (latency, tokens, temperature)
            
        Returns:
            IngestResponse with event_id and job_id on success
        """
        # Build payload
        if isinstance(input, dict):
            input_payload = InputPayload(
                prompt=input.get("prompt", ""),
                messages=input.get("messages"),
                retrieved_sources=input.get("retrievedSources") or input.get("retrieved_sources"),
            )
        else:
            input_payload = input

        if isinstance(output, dict):
            output_payload = OutputPayload(
                text=output.get("text", ""),
                tool_calls=output.get("toolCalls") or output.get("tool_calls"),
            )
        else:
            output_payload = output

        metadata_payload = None
        if metadata:
            if isinstance(metadata, dict):
                metadata_payload = Metadata(
                    latency_ms=metadata.get("latencyMs") or metadata.get("latency_ms"),
                    tokens_in=metadata.get("tokensIn") or metadata.get("tokens_in"),
                    tokens_out=metadata.get("tokensOut") or metadata.get("tokens_out"),
                    temperature=metadata.get("temperature"),
                )
            else:
                metadata_payload = metadata

        payload = IngestPayload(
            model=model,
            provider=provider,
            input=input_payload,
            output=output_payload,
            metadata=metadata_payload,
        )

        return self._send_ingest(payload)

    def ingest_async(
        self,
        model: str,
        provider: Provider,
        input: Union[InputPayload, Dict[str, Any]],
        output: Union[OutputPayload, Dict[str, Any]],
        metadata: Optional[Union[Metadata, Dict[str, Any]]] = None,
    ) -> None:
        """
        Ingest asynchronously (fire-and-forget).
        Does not block the main thread.
        """
        self._executor.submit(self.ingest, model, provider, input, output, metadata)

    def guard(
        self,
        output: str,
        input: Optional[str] = None,
        mode: str = "strict",
        timeout_ms: int = 100,
    ) -> GuardResult:
        """
        Inline guardrail check - blocks dangerous outputs before they reach users.
        
        Args:
            output: The LLM output text to check
            input: Optional user input/prompt for context
            mode: "strict" (block on medium+ risk) or "permissive" (block on high only)
            timeout_ms: Classification timeout in ms (default 100, max 500)
            
        Returns:
            GuardResult with allowed, action, output (possibly redacted), triggered
            
        Raises:
            GuardBlockedError: If guard_mode="fail_closed" and content is blocked
        """
        url = f"{self.base_url}/api/guard"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-App-Id": self.app_id,
        }
        
        payload = {
            "output": output,
            "input": input or "",
            "mode": mode,
            "timeout_ms": min(timeout_ms, 500),
            "app_id": self.app_id,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = Request(url, data=data, headers=headers, method="POST")
            guard_timeout = max(1, timeout_ms / 1000 + 1)
            
            with urlopen(req, timeout=guard_timeout) as response:
                result_data = json.loads(response.read().decode("utf-8"))
                result = GuardResult.from_dict(result_data)
                
                if self.guard_mode == "fail_closed" and not result.allowed:
                    raise GuardBlockedError(result)
                
                return result

        except GuardBlockedError:
            raise
            
        except HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            if self.guard_mode == "fail_closed":
                raise Exception(f"Guard API error: HTTP {e.code}: {error_body}")
            return GuardResult(
                allowed=True, action="allow", output=output, triggered=[],
                latency_ms=0, fallback=True, error=f"HTTP {e.code}: {error_body}",
            )

        except (URLError, Exception) as e:
            if self.guard_mode == "fail_closed":
                raise Exception(f"Guard API error: {e}")
            return GuardResult(
                allowed=True, action="allow", output=output, triggered=[],
                latency_ms=0, fallback=True, error=str(e),
            )

    def _send_ingest(self, payload: IngestPayload) -> IngestResponse:
        """Send ingest request to API."""
        url = f"{self.base_url}/ingest"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-App-Id": self.app_id,
        }

        try:
            data = json.dumps(payload.to_dict()).encode("utf-8")
            req = Request(url, data=data, headers=headers, method="POST")
            
            with urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                return IngestResponse(
                    success=result.get("success", False),
                    event_id=result.get("event_id"),
                    job_id=result.get("job_id"),
                    duplicate=result.get("duplicate", False),
                )

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
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=False)

    def __enter__(self) -> "DriftRail":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()



class DriftRailAsync:
    """
    Async DriftRail client using aiohttp.
    
    Usage:
        async with DriftRailAsync(api_key="...", app_id="my-app") as client:
            response = await client.ingest(...)
    """

    def __init__(
        self,
        api_key: str,
        app_id: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 30,
        fail_open: bool = True,
    ):
        self.api_key = api_key
        self.app_id = app_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.fail_open = fail_open
        self._session: Any = None

    async def _get_session(self) -> Any:
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                raise ImportError("aiohttp is required for async client: pip install driftrail[async]")
        return self._session

    async def ingest(
        self,
        model: str,
        provider: Provider,
        input: Union[InputPayload, Dict[str, Any]],
        output: Union[OutputPayload, Dict[str, Any]],
        metadata: Optional[Union[Metadata, Dict[str, Any]]] = None,
    ) -> IngestResponse:
        """Async ingest - see DriftRail.ingest for documentation."""
        import aiohttp

        if isinstance(input, dict):
            input_payload = InputPayload(
                prompt=input.get("prompt", ""),
                messages=input.get("messages"),
                retrieved_sources=input.get("retrievedSources") or input.get("retrieved_sources"),
            )
        else:
            input_payload = input

        if isinstance(output, dict):
            output_payload = OutputPayload(
                text=output.get("text", ""),
                tool_calls=output.get("toolCalls") or output.get("tool_calls"),
            )
        else:
            output_payload = output

        metadata_payload = None
        if metadata:
            if isinstance(metadata, dict):
                metadata_payload = Metadata(
                    latency_ms=metadata.get("latencyMs") or metadata.get("latency_ms"),
                    tokens_in=metadata.get("tokensIn") or metadata.get("tokens_in"),
                    tokens_out=metadata.get("tokensOut") or metadata.get("tokens_out"),
                    temperature=metadata.get("temperature"),
                )
            else:
                metadata_payload = metadata

        payload = IngestPayload(
            model=model,
            provider=provider,
            input=input_payload,
            output=output_payload,
            metadata=metadata_payload,
        )

        url = f"{self.base_url}/ingest"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-App-Id": self.app_id,
        }

        try:
            session = await self._get_session()
            async with session.post(
                url,
                json=payload.to_dict(),
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                result = await response.json()
                if response.status >= 400:
                    if self.fail_open:
                        return IngestResponse(
                            success=False,
                            error=f"HTTP {response.status}: {result.get('error', 'Unknown error')}",
                        )
                    raise Exception(f"HTTP {response.status}: {result}")

                return IngestResponse(
                    success=result.get("success", False),
                    event_id=result.get("event_id"),
                    job_id=result.get("job_id"),
                    duplicate=result.get("duplicate", False),
                )

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
    """
    Enterprise DriftRail client with monitoring features.
    
    Includes: Incidents, Compliance, Model Comparison, Exports, Brand Safety
    """

    def _api_request(self, endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an API request."""
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            body = json.dumps(data).encode("utf-8") if data else None
            req = Request(url, data=body, headers=headers, method=method)
            
            with urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        except HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            raise Exception(f"HTTP {e.code}: {error_body}")

    def list_incidents(
        self,
        status: Optional[list] = None,
        severity: Optional[list] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List incidents with optional filters."""
        params = []
        if status:
            params.append(f"status={','.join(status)}")
        if severity:
            params.append(f"severity={','.join(severity)}")
        params.append(f"limit={limit}")
        query = "&".join(params)
        return self._api_request(f"/api/incidents?{query}")

    def create_incident(
        self,
        title: str,
        severity: str,
        incident_type: str,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new incident."""
        return self._api_request("/api/incidents", "POST", {
            "title": title,
            "severity": severity,
            "incident_type": incident_type,
            "description": description,
        })

    def get_incident_stats(self) -> Dict[str, Any]:
        """Get incident statistics."""
        return self._api_request("/api/incidents/stats")

    def get_compliance_status(self) -> Dict[str, Any]:
        """Get compliance framework status."""
        return self._api_request("/api/compliance/status")

    def get_model_leaderboard(self, metric: str = "avg_risk_score") -> Dict[str, Any]:
        """Get model performance leaderboard."""
        return self._api_request(f"/api/models/leaderboard?metric={metric}")

    def create_export(
        self,
        export_type: str,
        format: str = "json",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a data export job."""
        return self._api_request("/api/exports", "POST", {
            "export_type": export_type,
            "format": format,
            "date_from": date_from,
            "date_to": date_to,
        })

    def check_brand_safety(self, text: str, location: str = "output") -> Dict[str, Any]:
        """Check text against brand safety rules."""
        return self._api_request("/api/brand-safety/check", "POST", {
            "text": text,
            "location": location,
        })
