# Authentication

Access control is enforced at the gateway for every request.

## Tokens

Signed tokens are issued by the identity service and expire after one hour.
Refresh happens transparently as long as the session remains active. The
identity service rotates signing keys weekly without downtime.

## Service accounts

A service account is a non-human identity used by automated clients. Service
account credentials live in the secrets manager and are rotated every ninety
days. Creating a service account requires approval from the resource owner.

## Permissions

Permissions follow a role-based model. Roles are defined per project, and a
principal may hold several roles at once. Permission checks are cached for
thirty seconds, so revocations can take up to half a minute to propagate.
