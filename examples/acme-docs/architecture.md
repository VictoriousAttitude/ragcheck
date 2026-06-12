# Acme platform architecture

The platform consists of a gateway, a scheduler, and a fleet of workers.

## Gateway

The gateway is the single entry point for all external traffic. It terminates
TLS, authenticates requests, and applies per-tenant rate limits before routing
calls to internal services. Rate limits are enforced with a sliding window of
one minute.

## Scheduler

The scheduler is a leader-elected service that assigns jobs to workers. Job
priority is computed from tenant tier and queue age. When no worker accepts a
job within five minutes, the scheduler escalates it to the on-call channel.

## Workers

Workers are stateless containers that execute jobs pulled from the queue.
Each worker reports a heartbeat every ten seconds; three missed heartbeats
mark the worker as unhealthy and drain its jobs back to the queue.
