
import ray
from ray import air,tune
from ray.tune import register_env

from MACA.env.cannon_reconn_hierarical_py37 import CannonReconnHieraricalEnv
from RL.callbacks.cr_callback import CRCallback
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig



def main():
    # test env
    cr_env = CannonReconnHieraricalEnv(None)

    # register env
    register_env(
        'cr_env_hier',
        lambda config: CannonReconnHieraricalEnv(config)
    )  
    
    # 这是一个PvE的settings，侦察机共享一个策略，攻击机共享一个一个策略
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if int(agent_id)-1 < cr_env.args.env.n_ally_reconn:
            return '0'
        else:
            return '1'  
        
    config = (
        PPOConfig()
        .environment("cr_env_hier")
        .framework("torch")
        .callbacks(CRCallback)
        .resources(num_gpus=1)
        .rollouts(num_rollout_workers = 32, num_envs_per_worker=4)
        .training(
            train_batch_size = 15000,
            sgd_minibatch_size = 512,
            entropy_coeff = 0.0,
            lr= 1e-5
            )
        .multi_agent(
            policies = {
                str(i): PolicySpec(
                    observation_space=cr_env.observation_space[i],
                    action_space=cr_env.action_space[i]
                    ) for i in range(2)
            },
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    tune.Tuner(
        "PPO",
        param_space = config, 
        run_config = air.RunConfig(
            storage_path ='~/ray_results',
            stop={"episodes_total": 100000},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=50,
            ),
            verbose=1,
        ),
    ).fit()

if __name__ == '__main__':
    main()