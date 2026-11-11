from collections.abc import Sequence

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.utils import configclass

from env.environment.isaac.direct_rl_env import CustomDirectRLEnv


@configclass
class IsaacRLEnvCfg(DirectRLEnvCfg):
    decimation: int = 3
    episode_length_s: float = 2000 * (1 / 60) * decimation
    observation_space: int = 0
    action_space: int = 0
    state_space: int = 0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=PhysxCfg(),
    )


class IsaacRLEnv(CustomDirectRLEnv):
    cfg: IsaacRLEnvCfg
    env_spacing: float = 4.0

    def __init__(self, cfg: IsaacRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.func = kwargs.get("func", None)
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        self.func["_setup_scene"](self)
        _clone_environments(self.scene)
        self.func["_post_setup_scene"](self)

    def _get_observations(self):
        pass

    def _get_dones(self):
        pass

    def _get_states(self):
        pass

    def _get_rewards(self):
        pass

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)
        self.func["_reset_idx"](self, env_ids)

    def _apply_action(self):
        pass

    def _pre_physics_step(self, actions, env_ids):
        pass


def _clone_environments(scene) -> None:
    """Independent copies of ``env_0`` at the grid's origins, then collisions filtered between environments.

    This is Isaac Lab 2.3's ``InteractiveScene.clone_environments(copy_from_source=True)``. Isaac Lab 3.0's scene
    clones only the assets declared in its configuration; RoboDojo builds ``env_0`` itself, so it is copied here with
    the same function 3.0's scene uses (``cloner.usd_replicate``). PhysX then parses every copy, where 2.3 replicated
    the parsed ``env_0``: slower to start, the same physics.
    """
    if hasattr(scene, "clone_environments"):          # Isaac Lab 2.x
        scene.clone_environments(copy_from_source=True)
        return
    import numpy as np
    from isaaclab import cloner

    positions, _ = cloner.grid_transforms(scene.num_envs, scene.cfg.env_spacing)
    cloner.usd_replicate(scene.stage, [scene.env_prim_paths[0]], [scene._env_fmt], np.arange(scene.num_envs),
                         positions=positions)
    if scene.cfg.filter_collisions:
        scene.filter_collisions(scene._global_prim_paths)
