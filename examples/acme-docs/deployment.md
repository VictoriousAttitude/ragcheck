# Deployment runbook

Releases ship through a staged rollout controlled by the deploy pipeline.

## Rollout stages

A rollout is a four-stage progression: canary, ten percent, fifty percent,
and full fleet. Each stage bakes for thirty minutes while automated health
probes compare error rates against the previous release. A failing probe
halts the rollout and pages the release owner.

## Rollbacks

Rollbacks reuse the same pipeline in reverse and complete within ten minutes.
The last three releases are kept warm so a rollback never waits on an image
build. Database migrations are forward-only; a rollback never reverts schema
changes.

## Freeze windows

The deploy freeze is active during the last week of each quarter and during
company-wide incidents. Emergency fixes during a freeze require sign-off from
the on-call incident commander.
