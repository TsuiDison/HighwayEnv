import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import copy
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# 1) Reward Wrapper (可按你需求改)
# ============================================================
class CustomRewardWrapper(gym.Wrapper):
    """
    你可以在这里做 reward shaping。
    注意：在 RL 里 reward shaping 可能改变最优策略，属于工程取舍。
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 对齐 PPO 版本的 reward shaping
        if info.get("crashed", False):
            reward -= 5.0

        if action == 3:  # FASTER
            reward += 0.5
        elif action == 4:  # SLOWER
            reward -= 0.05

        try:
            speed = self.env.unwrapped.vehicle.speed
            reward += speed / 15.0
        except AttributeError:
            pass

        return obs, reward, terminated, truncated, info


# ============================================================
# 2) Policy Network：输出 logits（更稳定），采样用 Categorical(logits=..)
# ============================================================
class PolicyNet(nn.Module):
    def __init__(self, obs_shape, act_dim):
        super().__init__()
        flat_dim = int(np.prod(obs_shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.logits = nn.Linear(256, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        return self.logits(x)


# ============================================================
# 3) 工具函数：KL(pi || pref)，以及 group advantage
# ============================================================
def categorical_kl_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """
    KL( P || Q ) where P and Q are categorical distributions given by logits.
    Returns KL per-sample (shape [B]).
    """
    logp = torch.log_softmax(logits_p, dim=-1)
    logq = torch.log_softmax(logits_q, dim=-1)
    p = torch.softmax(logits_p, dim=-1)
    kl = torch.sum(p * (logp - logq), dim=-1)
    return kl


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean()
    std = x.std() + eps
    return (x - mu) / std


# ============================================================
# 4) Rollout Storage：按“episode”为单位记录，便于做 GRPO 的组内相对优势
# ============================================================
@dataclass
class Episode:
    group_id: int
    seed_id: int               # 用于标记“同初始条件”
    obs: List[np.ndarray]
    actions: List[int]
    logp_old: List[float]
    rewards: List[float]
    infos: List[Dict[str, Any]]

    def total_return(self) -> float:
        return float(np.sum(self.rewards))

    def length(self) -> int:
        return len(self.rewards)


# ============================================================
# 5) GRPO Trainer（RL版本，工程可用）
#    核心点：
#    - 并行环境按 group_size 分组，每组共享 reset seed
#    - 每个 episode 得到一个 group-relative advantage（zscore）
#    - 用 PPO-clip 形式（可选） + reference KL 惩罚
# ============================================================
class GRPOTrainer:
    def __init__(
        self,
        envs: gym.vector.VectorEnv,
        policy: PolicyNet,
        device: str = "cpu",
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        kl_beta: float = 0.02,
        ent_coef: float = 0.0,        # GRPO 常用 KL，不一定需要 entropy；你也可设小一点如 0.001
        ref_update_interval: int = 1, # 每 N 次 update 同步一次 ref（1=每次都同步，越大约束越强）
        max_grad_norm: float = 1.0,
        group_size: int = 8,
    ):
        self.envs = envs
        self.policy = policy.to(device)
        self.device = device

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # PPO/GRPO 超参
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        # GRPO 的 reference policy（冻结）
        self.policy_ref = copy.deepcopy(self.policy).to(device)
        for p in self.policy_ref.parameters():
            p.requires_grad_(False)

        self.ref_update_interval = ref_update_interval
        self.group_size = group_size

        assert self.envs.num_envs % self.group_size == 0, "num_envs 必须能被 group_size 整除"
        self.num_groups = self.envs.num_envs // self.group_size

        self.update_count = 0

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        obs_np: shape [N, ...]
        return actions: [N], logp_old: [N]
        """
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        logits = self.policy(obs)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        logp = dist.log_prob(actions)
        return actions.cpu().numpy(), logp.cpu().numpy()

    def _sync_ref_if_needed(self):
        if (self.update_count % self.ref_update_interval) == 0:
            self.policy_ref.load_state_dict(self.policy.state_dict())

    def collect_episodes(
        self,
        steps_per_update: int,
        base_seed: int = 0,
    ) -> List[Episode]:
        """
        采样阶段（Rollout）：
        - 按 group 给环境 reset 相同 seed，近似“同 prompt 多样本”
        - 在 steps_per_update 的预算内，收集尽可能多的 episode
        """
        episodes: List[Episode] = []

        # 1) reset：给每个 group 一套 seed，组内所有 env 使用相同 seed
        #    seed_id 用来标记“同初始条件”
        seeds = []
        seed_ids = []
        for env_i in range(self.envs.num_envs):
            g = env_i // self.group_size
            seed_id = base_seed + g
            seeds.append(seed_id)
            seed_ids.append(seed_id)

        obs, infos = self.envs.reset(seed=seeds)

        # 2) 为每个 env 创建一个“当前 episode 缓存”
        current_eps: List[Episode] = []
        for env_i in range(self.envs.num_envs):
            g = env_i // self.group_size
            current_eps.append(
                Episode(
                    group_id=g,
                    seed_id=seed_ids[env_i],
                    obs=[],
                    actions=[],
                    logp_old=[],
                    rewards=[],
                    infos=[],
                )
            )

        # 3) 采样循环
        for _ in range(steps_per_update):
            actions, logp_old = self.act(obs)

            next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)
            done = np.logical_or(terminated, truncated)

            # 写入 transition 到对应 env 的 current episode
            for env_i in range(self.envs.num_envs):
                info_i = self._extract_info(infos, env_i, done[env_i])
                ep = current_eps[env_i]
                ep.obs.append(obs[env_i])
                ep.actions.append(int(actions[env_i]))
                ep.logp_old.append(float(logp_old[env_i]))
                ep.rewards.append(float(rewards[env_i]))
                ep.infos.append(info_i)

            obs = next_obs

            # 4) 如果 done，则该 env 的 episode 完成：保存并重置该 env 的 episode 容器
            #    注意：VectorEnv 通常会自动 reset done 的 env 并在 obs 中给新初始状态
            for env_i in range(self.envs.num_envs):
                if done[env_i]:
                    episodes.append(current_eps[env_i])

                    # 重开一个 episode 容器（seed_id 保持该 env 所属组一致）
                    g = env_i // self.group_size
                    current_eps[env_i] = Episode(
                        group_id=g,
                        seed_id=base_seed + g,
                        obs=[],
                        actions=[],
                        logp_old=[],
                        rewards=[],
                        infos=[],
                    )

        # 5) 把未结束的 episode 也收进来（部分轨迹）
        #    训练时依然可用，但 return 会更短；你也可选择丢弃短片段
        for env_i in range(self.envs.num_envs):
            if current_eps[env_i].length() > 0:
                episodes.append(current_eps[env_i])

        return episodes

    @staticmethod
    def _extract_info(infos: Any, env_i: int, done_i: bool) -> Dict[str, Any]:
        """
        Gymnasium VectorEnv 返回的 infos 可能是 dict-of-arrays；
        这里统一抽成单个 env 的 info dict。
        """
        if isinstance(infos, dict):
            info_i: Dict[str, Any] = {}
            for k, v in infos.items():
                if k in ("final_info", "final_observation"):
                    continue
                try:
                    info_i[k] = v[env_i]
                except Exception:
                    pass
            if done_i and "final_info" in infos:
                try:
                    final_info = infos["final_info"][env_i]
                    if final_info is not None:
                        info_i.update(final_info)
                except Exception:
                    pass
            return info_i
        try:
            return infos[env_i]
        except Exception:
            return {}

    def compute_group_advantages(self, episodes: List[Episode]) -> np.ndarray:
        """
        GRPO 核心：在 group 内对 episode return 做相对归一化，得到每个 episode 的 advantage。
        输出: adv_ep[i] 对应 episodes[i] 的优势（标量）
        """
        # 按 group 聚合 episode returns
        returns = np.array([ep.total_return() for ep in episodes], dtype=np.float32)
        group_ids = np.array([ep.group_id for ep in episodes], dtype=np.int32)

        adv_ep = np.zeros_like(returns)

        for g in range(self.num_groups):
            idx = np.where(group_ids == g)[0]
            if len(idx) <= 1:
                # 组里只有 1 条样本，没法做相对比较 -> advantage=0（或者用全局 baseline）
                adv_ep[idx] = 0.0
            else:
                adv_ep[idx] = zscore(returns[idx])

        return adv_ep

    def build_batch(self, episodes: List[Episode], adv_ep: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        把 episode 列表展开为 step-level batch。
        GRPO 在这里最常见做法：每个 step 继承它所在 episode 的 advantage（标量广播）
        """
        obs_list = []
        act_list = []
        logp_old_list = []
        adv_list = []

        for i, ep in enumerate(episodes):
            A = float(adv_ep[i])
            for t in range(ep.length()):
                obs_list.append(ep.obs[t])
                act_list.append(ep.actions[t])
                logp_old_list.append(ep.logp_old[t])
                adv_list.append(A)

        obs = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        act = torch.tensor(np.array(act_list), dtype=torch.int64, device=self.device)
        logp_old = torch.tensor(np.array(logp_old_list), dtype=torch.float32, device=self.device)
        adv = torch.tensor(np.array(adv_list), dtype=torch.float32, device=self.device)

        # Advantage 再做一次全局标准化（可选，但常用来稳一点）
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return {"obs": obs, "act": act, "logp_old": logp_old, "adv": adv}

    def update(self, batch: Dict[str, torch.Tensor], n_epochs: int = 10, minibatch_size: int = 4096) -> Dict[str, float]:
        """
        更新阶段：PPO-clip + reference KL penalty（GRPO风格稳定项）
        """
        self._sync_ref_if_needed()

        obs = batch["obs"]
        act = batch["act"]
        logp_old = batch["logp_old"]
        adv = batch["adv"]

        B = obs.shape[0]
        idxs = torch.arange(B, device=self.device)

        last_loss = 0.0
        last_kl = 0.0
        last_ent = 0.0

        for _ in range(n_epochs):
            perm = idxs[torch.randperm(B)]

            for start in range(0, B, minibatch_size):
                mb = perm[start:start + minibatch_size]
                mb_obs = obs[mb]
                mb_act = act[mb]
                mb_logp_old = logp_old[mb]
                mb_adv = adv[mb]

                # 当前策略分布
                logits = self.policy(mb_obs)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                # PPO ratio
                ratio = torch.exp(logp - mb_logp_old)

                # PPO clip objective（有的 GRPO 实现不用 clip；这里保留，更稳）
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # reference KL（GRPO 常用稳定项）
                with torch.no_grad():
                    logits_ref = self.policy_ref(mb_obs)
                kl = categorical_kl_from_logits(logits, logits_ref).mean()

                # 总 loss：最小化
                loss = policy_loss + self.kl_beta * kl - self.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                last_loss = float(loss.item())
                last_kl = float(kl.item())
                last_ent = float(entropy.item())

        self.update_count += 1
        return {"loss": last_loss, "kl": last_kl, "entropy": last_ent}

    @torch.no_grad()
    def evaluate(
        self,
        eval_env: gym.Env,
        episodes: int = 5,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        returns = []
        lengths = []
        speeds = []
        distances = []
        crashes = 0

        for _ in range(episodes):
            obs, _ = eval_env.reset()
            done = truncated = False
            total_reward = 0.0
            steps = 0
            total_distance = 0.0
            ep_speeds = []
            ep_crashed = False

            while not (done or truncated):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = self.policy(obs_t)
                dist = Categorical(logits=logits)
                action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()

                obs, reward, done, truncated, info = eval_env.step(int(action.item()))
                total_reward += float(reward)
                steps += 1
                try:
                    current_speed = eval_env.unwrapped.vehicle.speed
                    ep_speeds.append(current_speed)
                    total_distance += current_speed / 15.0
                except AttributeError:
                    ep_speeds.append(info.get("speed", 0))
                if info.get("crashed", False):
                    ep_crashed = True

            returns.append(total_reward)
            lengths.append(steps)
            speeds.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
            distances.append(total_distance)
            if ep_crashed:
                crashes += 1

        return {
            "eval_return_mean": float(np.mean(returns)),
            "eval_return_std": float(np.std(returns)),
            "eval_len_mean": float(np.mean(lengths)),
            "eval_speed_mean": float(np.mean(speeds)),
            "eval_distance_mean": float(np.mean(distances)),
            "eval_crash_rate": float(crashes / max(1, episodes)),
        }

    def train(
        self,
        total_updates: int = 200,
        steps_per_update: int = 256,
        n_epochs: int = 4,
        minibatch_size: int = 2048,
        seed: int = 0,
        log_interval: int = 10,
        eval_interval: int = 20,
        eval_episodes: int = 5,
        eval_env: gym.Env = None,
        writer: SummaryWriter = None,
        save_dir: str = "",
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        global_step = 0
        t0 = time.time()

        for u in range(1, total_updates + 1):
            # 采样：base_seed 随 update 改变，确保每次 update 的 group 初始条件不同（但组内一致）
            base_seed = seed + u * 1000
            episodes = self.collect_episodes(steps_per_update=steps_per_update, base_seed=base_seed)

            # GRPO：组内相对优势
            adv_ep = self.compute_group_advantages(episodes)

            # 展平 batch
            batch = self.build_batch(episodes, adv_ep)

            # 更新
            stats = self.update(batch, n_epochs=n_epochs, minibatch_size=minibatch_size)

            # 统计一些指标
            ep_returns = [ep.total_return() for ep in episodes]
            ep_lens = [ep.length() for ep in episodes]
            crash_rate = float(np.mean([1.0 if any(info.get("crashed", False) for info in ep.infos) else 0.0 for ep in episodes]))

            global_step += int(self.envs.num_envs * steps_per_update)

            if (u % log_interval) == 0:
                dt = time.time() - t0
                print(
                    f"[update {u}/{total_updates}] "
                    f"steps~{global_step} | "
                    f"ep_ret mean {np.mean(ep_returns):.2f} std {np.std(ep_returns):.2f} | "
                    f"ep_len {np.mean(ep_lens):.1f} | "
                    f"crash_rate {crash_rate:.3f} | "
                    f"loss {stats['loss']:.4f} | "
                    f"KL {stats['kl']:.4f} | "
                    f"ent {stats['entropy']:.4f} | "
                    f"time {dt:.1f}s"
                )
            if writer is not None:
                writer.add_scalar("train/ep_return_mean", float(np.mean(ep_returns)), u)
                writer.add_scalar("train/ep_len_mean", float(np.mean(ep_lens)), u)
                writer.add_scalar("train/crash_rate", crash_rate, u)
                writer.add_scalar("train/loss", stats["loss"], u)
                writer.add_scalar("train/kl", stats["kl"], u)
                writer.add_scalar("train/entropy", stats["entropy"], u)

            if eval_env is not None and (u % eval_interval) == 0:
                eval_stats = self.evaluate(eval_env, episodes=eval_episodes, deterministic=True)
                if writer is not None:
                    for k, v in eval_stats.items():
                        writer.add_scalar(f"eval/{k}", v, u)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    ckpt_path = os.path.join(save_dir, f"grpo_checkpoint_update_{u}.pth")
                    self.save(ckpt_path)

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)


# ============================================================
# 6) 构建 VectorEnv：按 group_size 组织并行环境
# ============================================================
def make_single_env(env_config: Dict[str, Any]):
    def thunk():
        env = gym.make("highway-fast-v0", render_mode=None, config=env_config)
        env = CustomRewardWrapper(env)
        return env
    return thunk


def main():
    # highway-env 的配置（你可按需求调）
    env_config = {
        "action": {"type": "DiscreteMetaAction"},
        "observation": {"type": "Kinematics"},
        "duration": 300,                # episode 最长步数由 env 控制（注意 highway-env 的 duration 是“步数”概念）
        "simulation_frequency": 15,
        "policy_frequency": 5,
        # 其它常见 config：你也可以加，比如：
        # "collision_reward": -1,
        # "high_speed_reward": 0.4,
        # "right_lane_reward": 0.1,
    }

    # ===== GRPO grouping 参数 =====
    num_envs = 128          # 并行环境总数
    group_size = 8          # 每组 8 个 env 共享相同 reset seed -> 一个“prompt”的多个样本
    assert num_envs % group_size == 0

    # gymnasium vector env
    env_fns = [make_single_env(env_config) for _ in range(num_envs)]
    envs = gym.vector.SyncVectorEnv(env_fns)

    # policy
    obs_shape = envs.single_observation_space.shape
    act_dim = envs.single_action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PolicyNet(obs_shape, act_dim)

    trainer = GRPOTrainer(
        envs=envs,
        policy=policy,
        device=device,
        lr=3e-4,
        clip_eps=0.2,       # PPO clip（可设 0 表示基本不用 clip，但建议保留）
        kl_beta=0.02,       # reference KL 强度（你可调：0.01~0.1）
        ent_coef=0.0,       # 如果你想更探索，可以设 0.001
        ref_update_interval=1,  # 每次 update 同步 ref（=让 ref 追随较快）；想更“拉住”可设 2~10
        max_grad_norm=1.0,
        group_size=group_size,
    )

    log_dir = "logs_GRPO_1228"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    eval_env = gym.make("highway-fast-v0", render_mode=None, config=env_config)
    eval_env = CustomRewardWrapper(eval_env)

    print(f"Device: {device}, num_envs={num_envs}, group_size={group_size}, groups={num_envs//group_size}")
    trainer.train(
        total_updates=200,
        steps_per_update=256,   # 每次 update 采样步数预算（越大越稳，但更耗时）
        n_epochs=4,
        minibatch_size=2048,
        seed=0,
        log_interval=10,
        eval_interval=20,
        eval_episodes=5,
        eval_env=eval_env,
        writer=writer,
        save_dir=log_dir,
    )

    trainer.save(os.path.join(log_dir, "grpo_highway_policy_final.pth"))
    eval_env.close()
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
