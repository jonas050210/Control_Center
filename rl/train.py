"""CLI Alias for PPO Training in SandboxAI."""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.train_ppo import main

if __name__ == "__main__":
    main()
