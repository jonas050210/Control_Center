# Security Policy

## Supported versions

SandboxAI is a research project developed on `main`. Security fixes land on
`main` and in the next tagged release; older tags are not patched.

| Version | Supported |
| --- | --- |
| `main` / latest tag | ✅ |
| older tags | ❌ |

## Threat model

SandboxAI is a **local, offline** tool. It opens no network sockets, exposes
no server, and requires no cloud service or API key. The realistic risk
surface is therefore small and mostly about what the tool does with local
input:

| Surface | Risk | Mitigation in tree |
| --- | --- | --- |
| Godot child process (`subprocess.Popen` in `adapter.py`, `godot_env.py`, `wsl.py`) | Executing an attacker-chosen binary via `--godot-executable`, `GODOT_PATH` or `.sandboxai/settings.json` | Argument lists only, never `shell=True`; the resolved path is validated by `runtime_validation.py` before use |
| JSON-lines bridge (stdin/stdout) | Malformed or hostile engine output | Strict line parsing, shape checks against `contract.py`, bounded reads |
| Run artifacts (`training/**`, JSONL datasets, replays) | Untrusted third-party artifact loaded into a run | Plain JSON/JSONL only — **never `pickle`**, by design (`league.py`, `replay.py`) |
| GDScript constant parsing (`weapons.py`) | Expression evaluation on a `.gd` file | Restricted AST evaluator: literals and arithmetic over already-known constants only, no names, no calls |
| CI workflows | Supply-chain via actions/dependencies | Actions pinned, `pip-audit` job, Dependabot for `pip` and `github-actions` |

Out of scope: anything that assumes SandboxAI is exposed to a network, and
anything that treats the Godot simulator as a sandbox for untrusted code —
it is not, and GDScript in this repository is trusted first-party code.

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

Use GitHub's private reporting: **Security → Report a vulnerability** on
<https://github.com/jonas050210/SandboxAI>. If private reporting is
unavailable, open an issue titled "security contact request" with no details
and wait to be contacted.

Include: affected file/command, reproduction steps (seed + command line if
the simulator is involved), impact, and the version or commit.

Expect an acknowledgement within 7 days and an assessment within 30 days.
This is a volunteer research project — there is no bug bounty.

## Scope reminder

SandboxAI is not a cheat, exploit, live-game bot, memory reader or client
modifier, and it has no working connection to any third-party game. Reports
asking for such capability are not security reports and will be closed.
