"""The RLBot v5 bot: drives a car in the real Rocket League.

Two brains are available: your own trained checkpoint (``--checkpoint``) or,
with ``--teacher``, the downloaded Nexto network (Grand-Champion level).

Started by RLBotServer through ``bot.toml``; the checkpoint comes from
``--checkpoint`` or the ``ROCKETAI_CHECKPOINT`` environment variable.
Like in training, it decides every 8 ticks (15 times per second) and holds
the controls in between.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from rlbot import flat
from rlbot.managers import Bot
from rlgym.rocket_league.obs_builders import DefaultObs

from ..config import OBS_PADDING, OBS_SIZE, TICK_SKIP
from ..env import lookup_table
from ..model import load_policy
from ..rlbot_convert import PacketConverter, controls_from_action

AGENT_ID = "rocketai/policy"


class RocketAIBot(Bot):
    def __init__(self, checkpoint: Path | None, deterministic: bool = True, teacher: bool = False):
        super().__init__(AGENT_ID)
        self.checkpoint = checkpoint
        self.deterministic = deterministic
        self.teacher_mode = teacher
        self.controls = flat.ControllerState()
        self.last_decision = -(10**9)

    def initialize(self) -> None:
        self.table = lookup_table()
        self.converter = PacketConverter(list(self.field_info.boost_pads))
        if self.teacher_mode:
            from ..teacher import NextoTeacher

            self.teacher = NextoTeacher(progress=self.logger.info)
            self.logger.info("RocketAI: Lehrer (Nexto) geladen (Spieler %s)", self.index)
            return
        self.policy = load_policy(self.checkpoint)
        self.obs_builder = DefaultObs(zero_padding=OBS_PADDING)
        self.logger.info(
            "RocketAI: %s geladen (Spieler %s, Team %s)",
            self.checkpoint.name,
            self.index,
            self.team,
        )

    def get_output(self, packet: flat.GamePacket) -> flat.ControllerState:
        frame = packet.match_info.frame_num
        # Same rhythm as training; a frame counter reset (new match) also triggers.
        if 0 <= frame - self.last_decision < TICK_SKIP:
            return self.controls
        self.last_decision = frame
        if self.index >= len(packet.players):
            return self.controls
        state = self.converter.convert(packet)
        me = self.converter.agent_id(self.index)
        if self.teacher_mode:
            action = self.teacher.act_state(state, [me])[me]
            self.controls = flat.ControllerState(**controls_from_action(self.table[action]))
            return self.controls
        obs = self.obs_builder.build_obs([me], state, {})[me]
        if obs.shape[0] != OBS_SIZE:  # more than 3 cars per team: not supported
            return self.controls
        actions, _, _ = self.policy.act(
            obs[None].astype(np.float32), deterministic=self.deterministic
        )
        self.controls = flat.ControllerState(**controls_from_action(self.table[int(actions[0])]))
        return self.controls


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RocketAI RLBot bot")
    parser.add_argument("--checkpoint", default=os.environ.get("ROCKETAI_CHECKPOINT", ""))
    parser.add_argument(
        "--sample", action="store_true", help="Aktionen würfeln statt immer die beste"
    )
    parser.add_argument(
        "--teacher", action="store_true", help="Nexto fahren lassen (Lehrer, offline)"
    )
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if not args.teacher and (checkpoint is None or not checkpoint.is_file()):
        print(f"RocketAI: Checkpoint nicht gefunden: {checkpoint!s}", file=sys.stderr)
        raise SystemExit(2)
    RocketAIBot(checkpoint, deterministic=not args.sample, teacher=args.teacher).run()


if __name__ == "__main__":
    main()
