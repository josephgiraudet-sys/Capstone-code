from env import PrisonersDilemmaEnv 

def run_episode(opponent_name: str, actions: list[int], seed: int = 42) -> None:
    env = PrisonersDilemmaEnv(
        opponent=opponent_name, 
        p_noise=0.0,
        n_rounds=len(actions),
        render_mode=None,
    )
    obs, info = env.reset(seed=seed)
    print(f"\n=== Opponent: {opponent_name} ===")
    print(f"Initial observation: {obs}, info: {info}")

    total_reward = 0.0

    for step_idx, action in enumerate(actions, start=1):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {step_idx}")
        print(f"Agent intended action:   {info['intended_agent_action']}")
        print(f"Opponent intended action:{info['intended_opp_action']}")
        print(f"Agent executed action:   {info['executed_agent_action']}")
        print(f"Opponent executed action:{info['executed_opp_action']}")
        print(f"Reward:                  {reward}")
        print(f"Next obs:                {obs}")
        print(f"Terminated:              {terminated}")
        print(f"Truncated:               {truncated}")

        if terminated or truncated:
            print("Episode terminated.")
            break

    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    test_actions = [
        0,
        0,
        1,
        0,
        1,
    ]

    for opponent in ["allc", "alld", "tft", "grudger", "random"]:
        run_episode(opponent_name=opponent, actions=test_actions)
