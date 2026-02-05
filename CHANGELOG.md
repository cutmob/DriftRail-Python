# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
