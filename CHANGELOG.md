# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-02-05

### Added
- Complete enterprise feature parity with Node.js SDK
- **Distributed Tracing**: `start_trace()`, `end_trace()`, `start_span()`, `end_span()`, `list_traces()`
- **Prompt Management**: `create_prompt()`, `create_prompt_version()`, `deploy_prompt_version()`, `rollback_prompt()`
- **Evaluation Framework**: `create_dataset()`, `add_dataset_items()`, `create_eval_run()`, `submit_eval_result()`
- **Semantic Caching**: `cache_lookup()`, `cache_store()`, `get_cache_stats()`, `clear_cache()`
- **Agent Simulations**: `create_simulation()`, `run_simulation()`, `add_simulation_turn()`, `complete_simulation_run()`
- **Drift Detection V3**: `get_drift_score()`, `get_drift_predictions()`, `get_drift_heatmap()`, `get_seasonality_patterns()`
- **Notification Channels**: `create_notification_channel()`, `update_notification_channel()`
- **Drift Segments**: `create_drift_segment()`, `get_drift_segments()`
- **Model Analytics**: `get_model_analytics_summary()`, `get_historical_logs()`, `record_model_switch()`, `calculate_model_benchmark()`
- **Executive Dashboard**: `get_executive_metrics()`, `get_kpi_targets()`, `update_kpi_targets()`, `export_executive_metrics()`
- **Compliance Reports**: `generate_compliance_report()`, `create_custom_framework()`, `get_compliance_score()`
- **Integrations**: `create_integration()`, `test_integration()`, Slack/Teams/Discord support
- **Benchmarks**: `get_benchmark_report()`, `set_tenant_industry()`, industry comparison
- **Guardrails Management**: `create_guardrail()`, `update_guardrail()`, `get_guardrail_stats()`
- **Custom Detections**: `create_custom_detection()`, keyword/regex/semantic/llm_judge types
- **Retention Policies**: `create_retention_policy()`, `get_retention_summary()`
- **Audit Logs**: `get_audit_logs()` with filtering
- **Events API**: `get_events()`, `get_event()`, `get_live_events()`
- **Webhooks**: `create_webhook()`, `update_webhook()`
- **Classifications**: `get_classifications()` with filtering
- 60+ new type definitions for all enterprise features

### Changed
- Comprehensive type hints for all new methods
- Updated README with full enterprise documentation

## [2.0.0] - 2025-02-04

### Added
- Inline guardrails with `guard()` method for real-time content blocking
- `GuardResult` and `GuardBlockedError` types for guardrail responses
- `guard_mode` parameter for fail-open/fail-closed behavior
- `DriftRailEnterprise` client with incident, compliance, and brand safety features
- Python 3.13 support
- `py.typed` marker for PEP 561 compliance

### Changed
- Improved type hints throughout the codebase
- Updated classifiers for better PyPI discoverability

## [1.0.0] - 2024-12-01

### Added
- Initial release
- `DriftRail` synchronous client
- `DriftRailAsync` async client with aiohttp
- `ingest()` and `ingest_async()` methods
- Support for OpenAI, Anthropic, Google, and other providers
- Metadata tracking (latency, tokens, temperature)
- RAG source tracking
- Fail-open architecture by default
