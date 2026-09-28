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
  `OBSERVATION_HIGH` — the exact 84-float, `[-1, 1]`-normalized observation
  shape and per-field semantics (mirrors
  `docs/OBSERVATION_ACTION_CONTRACT.md`).
- `ACTION_SPEC` / `ACTION_NVEC` — the `MultiDiscrete([3,3,3,3,2,2])` action
  shape and per-field semantics.
- `OBSERVATION_GROUPS` — the observation split into the six semantic
  channels an adapter has to be able to produce independently:
  `self_state`, `movement`, `combat`, `perception`, `memory`, `sound` and
  `world`. This is the practical checklist: if a target game cannot supply
  one of these channels honestly, the adapter must report the neutral
  "no information" encoding for it rather than substituting privileged
  data.
- `GameAdapter` — an `abc.ABC` with `reset(seed) -> observation`,
  `step(action) -> (observation, reward, done, info)`, and `close()`. This is
  the seam a future adapter implements. `GodotBatchClient` /
  `GodotVecEnv` (in `python/sandboxai/godot_env.py`) do **not** currently
  inherit from `GameAdapter` — they predate it and have a slightly different
  (vectorized, N-environments-at-once) surface for SB3 compatibility.
  Aligning them, or writing a thin per-environment `GameAdapter` wrapper
  around `GodotGymEnv`, is future work, not done in this milestone.

## The adapter boundary module (new)

`python/sandboxai/external_adapter.py` makes the boundary above executable
without implementing any integration. It contains:

- `EXTERNAL_CONTRACT_VERSION`, `EPISODE_STATES`
  (`idle`/`running`/`terminated`/`truncated`/`unavailable`) and
  `TimingContract` (`fixed`/`polled`/`event` modes, tick rate, maximum
  latency, an explicit `deterministic` flag that defaults to **false** for
  an external game).
- `AdapterCapabilities` — one boolean per `OBSERVATION_GROUPS` channel plus
  the declared `data_sources`. `ALLOWED_DATA_SOURCES` are
  `own_character_state`, `own_weapon_state`, `rendered_visibility`,
  `audible_events`, `own_memory`, `public_match_state`.
  `FORBIDDEN_DATA_SOURCES` are `server_authoritative_state`,
  `other_player_private_state`, `hidden_entity_positions`,
  `process_memory`, `network_packet_inspection` and `client_modification`;
  declaring one raises.
- `validate_observation` / `validate_action` — shape, range and NaN checks,
  plus the central rule: **a channel the adapter cannot supply must be the
  all-zero neutral encoding.** A non-zero value in an undeclared channel is
  an error, which is what stops a "sound" field from quietly carrying
  server-side positions.
- `MockExternalEnvironment` — a `GameAdapter`/`ExternalEnvironment`
  implementation with no network, no game and no external dependency. It
  exists so the contract tests are real tests.
- `AdapterContractChecker` — runs an adapter through reset/step/termination
  and returns a list of problems; `python/tests/test_external_adapter.py`
  asserts the mock returns zero, and that deliberately leaky or
  self-contradicting adapters do not.

Inspect the contract from the CLI with `sandboxai adapter-contract
--check-mock`.

Nothing in that module talks to Roblox, and the tests assert that no module
in the package contains client-modification or memory-reading technique
names.

## What a real implementation would need to do (not attempted here)

1. **Connect to a live Roblox session.** Roblox does not expose a documented,
   free, local, stdin/stdout-style headless simulation bridge equivalent to
   Godot's `--headless --script`. A realistic integration would likely need
   either a Roblox Studio plugin driving `RemoteEvent`s/`RemoteFunction`s to
   an out-of-process trainer, or Roblox's official Open Cloud APIs (subject
   to Roblox's own terms, rate limits, and possibly cost depending on usage)
   — evaluating exactly which mechanism is viable is unresearched and
   unimplemented here.
2. **Compute the same 84 fields from Roblox's game state**, using only
   information a Roblox player character could access — no server-only
   internals. Concretely, per channel:
   - *self state / movement*: own `CFrame`, `Humanoid` velocity, health,
     `FloorMaterial`/grounded state, jump state.
   - *combat*: weapon cooldown readiness, whether a target is within
     effective range.
   - *perception*: the adapter MUST do its own FOV + raycast occlusion test
     (`workspace:Raycast` from the camera to each candidate character) and
     only report characters that pass it. Reporting every character the
     server knows about would hand the policy information a player does not
     have and invalidate everything it learned about corner fights.
   - *memory*: the adapter keeps its own last-known-position/confidence
     store, decayed the same way `EnemyMemory` does, for characters that
     have left view.
   - *sound*: derived from the game's own audio events where available; if
     the target place emits nothing usable, report silence (all-zero sound
     fields) rather than leaking positions through a fake "sound".
   - *world*: nearest-cover distance/bearing from a short raycast fan; the
     corpse count from whatever death representation the place uses.
3. **Translate the 6-field `MultiDiscrete` action into Roblox input** (e.g.
   `Humanoid:Move()` for the two movement axes, camera rotation for the two
   look axes, a fire `RemoteEvent` for `shoot`, `Humanoid.Jump` for
   `jump`), respecting the same weapon-cooldown-style semantics so the
   policy's learned timing still applies.
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
