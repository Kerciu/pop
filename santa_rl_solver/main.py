import sys

from src.agent import PilotAgent
from src.router import SantaRouter
from src.utils import parse_input


def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py [train/solve] [input_file]")
        return

    mode = sys.argv[1]
    input_file = sys.argv[2]

    data = parse_input(input_file)
    agent = PilotAgent(data["acc_ranges"])

    if mode == "train":
        print("Training RL Agent...")
        agent.train(timesteps=200000)

    elif mode == "solve":
        print("Solving...")
        agent.load()
        router = SantaRouter(data, agent)
        count, commands = router.solve()

        output_file = input_file.replace(".in", ".out").replace("inputs", "outputs")
        with open(output_file, "w") as f:
            f.write(f"{count}\n")
            f.write("\n".join(commands))
        print(f"Solution saved to {output_file} with {count} actions.")


if __name__ == "__main__":
    main()
