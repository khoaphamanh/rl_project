"""
Entry point.

The Config is built here and handed to the agent, so every hyperparameter
lives in one place and nothing below imports Config on its own.

Run with:
    python main.py
"""

from agents.ppo import PPOAgent
from config.config import Config


def main():
    config = Config()

    agent = PPOAgent(config)

    print(f"env         {config.name_env}")
    print(f"encoder     {config.recurrent_model}  hidden_size {config.hidden_size}")
    print(f"seed        {agent.seed}")
    print(
        f"sampling    W={config.n_workers} x T={config.worker_steps} "
        f"-> batch_size {config.batch_size}"
    )

    buf, stats = agent.sample()

    print("\nBUFFER")
    for key, tensor in buf.items():
        print(f"  {key:<10} {str(tuple(tensor.shape)):<20} {tensor.dtype}")

    print("\nROLLOUT")
    print(f"  episodes finished {stats['episodes']}")
    print(f"  mean return       {stats['return_mean']:.3f}")
    print(f"  mean length       {stats['length_mean']:.1f}")
    print(f"  total reward      {buf['rewards'].sum():.3f}")

    for w in range(config.n_workers):
        ended = buf["dones"][w].nonzero().flatten().tolist()
        print(f"  worker {w} episodes ended at steps {ended}")

    agent.close()


if __name__ == "__main__":
    main()
