from argparse import ArgumentParser
import pickle

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from agent import MineRLAgent, ENV_KWARGS
from lib.gymnasium_compat import reset_env, step_env


def main(model, weights):
    env = HumanSurvival(**ENV_KWARGS).make()
    print("---Loading model---")
    with open(model, "rb") as f:
        agent_parameters = pickle.load(f)
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    print("---Launching MineRL enviroment (be patient)---")
    obs, _ = reset_env(env)

    while True:
        minerl_action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = step_env(env, minerl_action)
        env.render()
        if terminated or truncated:
            agent.reset()
            obs, _ = reset_env(env)


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")

    args = parser.parse_args()

    main(args.model, args.weights)
