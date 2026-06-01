# 17. Training store is public (`training-data` branch); private store retired

- **Status:** Accepted
- **Date:** 2026-05-31
- **Amends:** ADR-0009 (which made the raw store private).
- **Relates to:** ADR-0007 (training-data branch), ADR-0010 (ACIS dropped), ADR-0015 (store).

## Context

ADR-0009 made the raw training store **private** (separate repo, `DATA_REPO_TOKEN`) **solely
because of ACIS's unsettled redistribution rights**. ADR-0010 then dropped ACIS and retargeted
to ECCC (`envcanada` observations + HRDPS), both **redistributable with attribution**. The
store therefore now holds only redistributable data — triggering ADR-0009's own escape hatch
("the raw store could be made public … if the ambiguity resolves").

## Decision

The training store is **public**, committed to a `training-data` branch in the main repo by
the hourly inference Action via the built-in `GITHUB_TOKEN`. The separate private repo and
`DATA_REPO_TOKEN` are **retired** — no external setup is required to collect data.
**Attribution remains mandatory** (the published forecast carries it; the store is raw ECCC
data, redistributable with attribution).

## Consequences

- Zero-setup hourly data collection (no private repo, no PAT/secret).
- The inference Action manages the `training-data` branch as **state**: it force-pushes a
  single commit of the current store each run (not accumulating append commits), keeping git
  history bounded. The branch is derived, forward-regenerable data; provenance is in each row
  (`issue_time`/`written_at`), not git history (ADR-0018).
- Training (subsystem 3) reads the `training-data` branch.
- ADR-0007's "fourth artifact home" reverts to the public `training-data` branch it originally
  described; ADR-0009's private-store consequence is superseded for the ECCC-only deployment.
- If a future deployment reintroduces a redistribution-restricted source, the private-store
  decision (ADR-0009) would have to be revisited for that deployment.
