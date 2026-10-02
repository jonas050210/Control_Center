# TTK Testing reference and calibration boundary

## Purpose

SandboxAI is being reduced to a calibration-oriented representation of the
real Roblox experience **TTK Testing**. This document is the evidence boundary
for that work: a mechanic is either source-backed, awaiting player-visible
calibration, or excluded. It must never be quietly promoted from “plausible in
a tactical FPS” to “present in TTK Testing.”

The executable form of the same table is
[`python/sandboxai/ttk_testing.py`](../python/sandboxai/ttk_testing.py). Run:

```bash
sandboxai ttk-status
sandboxai ttk-status --json
```

The command is deliberately a status report, not a game connection. It does
not drive Roblox, inspect a client, read memory, or access any non-player
state.

## Evidence sources checked on 2026-10-01

1. [Official TTK Testing experience page](https://www.roblox.com/games/120189115846709/TTK-Testing)
   — current controls list **M1** fire, **M2** aim, **C** crouch, **Q/E**
   lean left/right, and weapon swapping via **number keys**. Its current
   official experience thumbnail, reviewed on 2026-10-01, is a normal
   first-person weapon/world view; it supports the existing no-Helmetcam
   presentation decision, but contains no HUD, timing, or numerical evidence.
2. [Official developer forum announcement](https://devforum.roblox.com/t/ttk-our-very-early-tactical-fps/4664539)
   — by **Sable Digital** (`PoptartNoahh` and `CanyonJack`), confirming the
   product direction around PvE, PvP, door-kicking, and planned modes
   (*Survival*, *Missions*, *Quick Play*, *Ground War*). It does not publish
   map scripts, AI rules, timing, weapon values, or physics.
3. [Official developer video: “TTK - wound painting/bleeding”](https://www.youtube.com/watch?v=fOpQt7dD4Ro)
   — confirms the wound-painting/bleeding feature family. Its title does not
   establish damage-over-time, healing, fatality, or any numerical rule.
4. [Official TTK Testing website & updates](https://www.ttktesting.com/updates)
   — confirms the live experience metadata (`TTK Testing [MAP VOTING]`,
   `universeId=10090256806`, `placeId=120189115846709`, 8-player server size),
   the **Gunsmith Update**, and **Transparent Optics** (clearer sight
   visibility while aiming).

TTK Testing is updated frequently. An older store snapshot that happened to
name an input or weapon is not proof for the current build; current official
sources and player-visible evidence win.

## Implementation matrix

| Status | Mechanic | Allowed project behavior |
| --- | --- | --- |
| Verified | Fire, aim, crouch, left/right lean | Retain/introduce the player controls, but do not assign unmeasured speed, accuracy, hitbox, or camera modifiers. |
| Verified | Weapon swap through number keys | Make switching an explicit action. **Never** auto-switch merely because a magazine is empty, the current weapon is cooling down, or an enemy is visible. |
| Verified | Wound painting / bleeding | A visible wound/bleeding event may be represented or logged. No damage-over-time, healing, or death rule may be invented. |
| Verified product direction | PvE, PvP, door kicking | Treat only as high-level direction, not a license to invent missions, maps, door timings, or AI behavior. |
| Needs calibration | Weapon roster/slots, ammo, reload behavior/timing, recoil/bloom/falloff, damage/TTK, fire modes | Do not claim any value or behavior as TTK Testing until it is visible in a current source or measured manually. |
| Needs calibration | Walk/sprint/jump/gravity, ADS/crouch/lean movement and collision | Do not reuse generic SandboxAI numbers as Roblox facts. Measure player-visible outcomes first. |
| Excluded by project decision | Helmet-camera presentation | The project uses normal first-person presentation with its regular GUI. Do not add a Helmetcam mode. |
| Excluded | Automatic weapon switch; sidearm-finish/pressure-reload/faster-reload drills | These are not TTK Testing mechanics for this project and must not be reintroduced under another name. |

## Cleanup already applied

The following invented evaluation drills were removed from
`ScenarioLibrary`; they are not supported TTK Testing content:

- `rifle_lane_drill`
- `shotgun_breach_drill`
- `sidearm_finish_drill`
- `smg_tracking_drill`

The removal is enforced by `python/tests/test_weapon_profiles_static.py`, so a
future change cannot silently restore one of these drills.

## Screenshot calibration protocol

Screenshots are sufficient for several important checks. For each image, note
the current build label if it is visible and what the image itself proves.

| Desired screenshot | Can establish | Cannot establish alone |
| --- | --- | --- |
| Normal first-person HUD during play | GUI layout, no Helmetcam target, current weapon/ammo displays | Recoil curve or movement speed |
| Before/after number-key weapon selection | Visible weapon slots and manual switching | A hidden auto-switch rule under every empty-magazine case |
| Empty-magazine HUD | Dry-fire/reload prompt/weapon state visible at that instant | Reload duration or cancellation timing |
| Crouched and leaned views near cover | The available stance/lean presentation and visible HUD changes | Exact collision/hitbox/speed effects |
| Hit/wound frame | Visible wound-painting/bleeding presentation | Damage-over-time, healing, or health calculations |

Temporal mechanics need repeated manual observations with timestamps. Keep
those as consented, player-visible annotations in the existing `ttk-report`
JSONL workflow; do not infer them from a single still image. The workflow
rejects data fields that suggest private game state, packet capture, process
memory, injected hooks, exploits, or aimbots.

## Calibration rules

1. Preserve the source URL/build context with every conclusion.
2. Treat player-visible screenshots and ordinary manual play as the only
   calibration sources unless an official developer source is available.
3. If evidence conflicts or is incomplete, mark the mechanic
   `calibration_required`; do not tune a “best guess.”
4. Record uncertainty rather than overwriting it: measurements can tune a
   local approximation, but they do not reveal server-authoritative internals.
5. Do not connect, automate, modify, inspect, or exploit the Roblox client.
