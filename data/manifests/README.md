# Manifests

Only publish manifests after the real dataset schema and split provenance have
been verified. Do not treat the upstream `seen` flag as a training split.
Every manifest used by an experiment is hashed and recorded with that run.
