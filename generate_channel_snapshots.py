import subprocess
import os

def run_experiment(nodes, rounds=2000, seed=30):
    prefix = f"generated/channels/{seed}/channel_spatial_{nodes}"
    command = [
        "uv", "run", "python", "examples/channel/spatial.py",
        "--num-nodes", str(nodes),
        "--rounds", str(rounds),
        "--no-gif",
        "--seed", str(seed),
        "--viz-prefix", prefix
    ]
    
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Error running for {nodes} nodes")

if __name__ == "__main__":
    node_counts = [1000, 5000, 10000, 20000]
    seeds = [1]
    os.makedirs("generated/channels", exist_ok=True)
    
    for nodes in node_counts:
        for seed in seeds:
            run_experiment(nodes, seed=seed)
