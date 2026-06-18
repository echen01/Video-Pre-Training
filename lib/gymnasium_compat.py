from __future__ import annotations


def reset_env(env, **kwargs):
    result = env.reset(**kwargs)
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        return result

    observation, reward, done, info = result
    return observation, reward, done, False, info

