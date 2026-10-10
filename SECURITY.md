# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/prxcode/tinyserve/security/advisories/new)
rather than in a public issue. Include steps to reproduce and the affected commit. You can
expect an initial reply within a week.

## Supported versions

Only the latest commit on `main` is supported. There are no maintained release branches.

## Deployment notes

tinyserve is a research and portfolio project, not a hardened service.

- The HTTP API has no authentication or rate limiting per client. Run it on a private network
  or behind a reverse proxy that provides both.
- Request size limits (prompt length, `max_tokens`, queue depth) exist to protect GPU memory,
  not to stop a determined attacker.
- Weights are loaded with `safetensors`, which does not execute code. Pickle-based checkpoints
  (`.bin`, `.pt`) are deliberately not supported.
- Never commit credentials. AWS access for deployment should come from an instance role or
  your local AWS profile, not from files in this repository.
