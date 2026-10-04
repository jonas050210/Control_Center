"""The PPO update step."""

from __future__ import annotations

import numpy as np
import torch

from .model import ActorCritic
from .rollout import Batch


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    *,
    epochs: int,
    minibatch_size: int,
    clip_range: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    target_kl: float = 0.0,
    teacher_coef: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Clipped PPO over ``epochs`` passes; returns averaged diagnostics.

    With ``teacher_coef`` > 0 and teacher targets in the batch, every minibatch
    also learns from the teacher: the student's distribution is pulled towards
    the teacher's answer for the situations that were labelled (cross entropy,
    i.e. the KL divergence up to a constant).
    """
    rng = rng or np.random.default_rng()
    model.train()
    obs = torch.as_tensor(batch.obs)
    actions = torch.as_tensor(batch.actions)
    old_log_probs = torch.as_tensor(batch.log_probs)
    returns = torch.as_tensor(batch.returns)
    advantages = torch.as_tensor(batch.advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    teacher = None
    if teacher_coef and batch.teacher_probs is not None:
        teacher = (
            torch.as_tensor(batch.teacher_probs, dtype=torch.float32),
            torch.as_tensor(batch.teacher_mask, dtype=torch.bool),
        )

    n = len(batch)
    size = max(1, min(minibatch_size, n))
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "clip_fraction": 0.0,
        "teacher_loss": 0.0,
    }
    updates = 0
    stopped_early = False
    for _ in range(epochs):
        if stopped_early:
            break
        order = torch.as_tensor(rng.permutation(n))
        for start in range(0, n, size):
            index = order[start : start + size]
            logits = model.logits(obs[index])
            distribution = torch.distributions.Categorical(logits=logits)
            log_probs = distribution.log_prob(actions[index])
            entropy = distribution.entropy()
            values = model.value(obs[index])
            ratio = torch.exp(log_probs - old_log_probs[index])
            adv = advantages[index]
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, returns[index])
            entropy_mean = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
            teacher_loss = torch.zeros(())
            if teacher is not None:
                targets, mask = teacher
                if bool(mask[index].any()):
                    log_probs_teacher = torch.log_softmax(logits, dim=-1)
                    teacher_loss = (
                        -(targets[index][mask[index]] * log_probs_teacher[mask[index]])
                        .sum(dim=-1)
                        .mean()
                    )
                    loss = loss + teacher_coef * teacher_loss

            optimizer.zero_grad()
            loss.backward()
            # Clip separately: the critic's large value gradients must not
            # shrink the actor's update through a shared norm.
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = log_probs - old_log_probs[index]
                totals["kl"] += float(((ratio - 1) - log_ratio).mean())
                totals["clip_fraction"] += float(((ratio - 1).abs() > clip_range).float().mean())
            totals["teacher_loss"] += float(teacher_loss.detach())
            totals["policy_loss"] += policy_loss.item()
            totals["value_loss"] += value_loss.item()
            totals["entropy"] += entropy_mean.item()
            updates += 1
            if target_kl and totals["kl"] / updates > 1.5 * target_kl:
                stopped_early = True
                break
    model.eval()
    stats = {key: value / max(1, updates) for key, value in totals.items()}
    stats["updates"] = updates
    stats["stopped_early"] = stopped_early
    variance = float(np.var(batch.returns))
    stats["explained_variance"] = (
        float(1 - np.var(batch.returns - batch.values) / variance) if variance > 1e-8 else 0.0
    )
    return stats
