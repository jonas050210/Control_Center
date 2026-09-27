# Future Roblox Player adapter — boundary and status

**Status: NOT implemented.** There is no Roblox connection, no Roblox Studio
plugin, no `RobloxPlayerAdapter` class, and no network protocol to any
Roblox service anywhere in this repository. This document exists only to
define the *boundary* a future implementation would need to respect, and to
be explicit about what does and does not exist today so nobody mistakes the
interface hook for a working integration.

## Why this exists now

The long-term goal (stated in the project's roadmap) is to eventually train
or evaluate a policy against a real Roblox game ("TTK Testing"-style combat)
using an external adapter, while reusing the same PPO policy trained in this
Godot-based simulator. For that to be possible without retraining the
observation/action heads, the Roblox adapter must produce/consume the exact
same semantic contract as SandboxAI's Godot environment — not a
similar-looking one.

## The interface

`python/sandboxai/contract.py` defines:

- `OBSERVATION_SPEC` / `OBSERVATION_FIELD_COUNT` / `OBSERVATION_LOW` /
  `OBSERVATION_HIGH` — the exact 33-float, `[-1, 1]`-normalized observation
  shape and per-field semantics (mirrors
  `docs/OBSERVATION_ACTION_CONTRACT.md`).
- `ACTION_SPEC` / `ACTION_NVEC` — the `MultiDiscrete([3,3,3,3,2])` action
  shape and per-field semantics.
- `GameAdapter` — an `abc.ABC` with `reset(seed) -> observation`,
  `step(action) -> (observation, reward, done, info)`, and `close()`. This is
  the seam a future adapter implements. `GodotBatchClient` /
  `GodotVecEnv` (in `python/sandboxai/godot_env.py`) do **not** currently
  inherit from `GameAdapter` — they predate it and have a slightly different
  (vectorized, N-environments-at-once) surface for SB3 compatibility.
  Aligning them, or writing a thin per-environment `GameAdapter` wrapper
  around `GodotGymEnv`, is future work, not done in this milestone.

## What a real implementation would need to do (not attempted here)

1. **Connect to a live Roblox session.** Roblox does not expose a documented,
   free, local, stdin/stdout-style headless simulation bridge equivalent to
   Godot's `--headless --script`. A realistic integration would likely need
   either a Roblox Studio plugin driving `RemoteEvent`s/`RemoteFunction`s to
   an out-of-process trainer, or Roblox's official Open Cloud APIs (subject
   to Roblox's own terms, rate limits, and possibly cost depending on usage)
   — evaluating exactly which mechanism is viable is unresearched and
   unimplemented here.
2. **Compute the same 33 fields from Roblox's game state**, using only
   information a Roblox player character could access (own
   `Humanoid`/`CFrame`/health, other characters' relative position/health/
   aliveness within some plausible detection radius) — no server-only
   internals.
3. **Translate the 5-field `MultiDiscrete` action into Roblox input** (e.g.
   `Humanoid:Move()`, camera rotation, a fire `RemoteEvent`), respecting the
   same weapon-cooldown-style semantics so the policy's learned timing still
   applies.
4. **Match units/normalization.** Roblox's default unit (studs) and
   coordinate conventions are not the same as Godot's meters; a real adapter
   must convert distances/velocities into the same normalized ranges this
   contract defines, likely re-deriving sensible "arena size"/"max distance"
   constants for the target Roblox place rather than reusing SandboxAI's
   literal 10 m arena half-extent.
5. **Validate empirically** that a policy trained in the Godot simulator
   transfers usefully to the Roblox adapter's observations before trusting
   any evaluation numbers from it — sim-to-sim/sim-to-real transfer is not
   guaranteed just because the vector shapes match.

None of the above is implemented, tested, or even prototyped in this
repository. Treat this document as an interface contract and a task list for
a future milestone, not as evidence of working Roblox integration.
