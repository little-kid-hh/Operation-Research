# Public Release Notes

This package is the public-shareable edition of the exploratory XGB-guided HybridSVM no-seed run.

Edits relative to the frozen internal archive:

- converted machine-local absolute paths to package-relative paths;
- updated `repro_bundle/code/*` defaults so the copied bundle can run after unzip;
- sanitized local Python site-package paths from stored traceback text;
- preserved experiment metrics, accepted/rejected decisions, prompts, responses, and generated feature code.

What was not changed:

- model outputs;
- trial statuses;
- reported metrics;
- accepted feature bank contents.
