import subprocess
import sys


def run_script(script_path, args=None):
    if args is None:
        args = []
    cmd = ["uv", "run", "python", script_path, *args]
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {script_path}")
        print(f"Error:\n{e.stderr}")
        return False, e.stderr
    else:
        print(f"SUCCESS: {script_path}")
        return True, result.stdout


def main():
    scripts_to_test = [
        # Channel examples
        (
            "examples/channel/small.py",
            [
                "--rows",
                "5",
                "--cols",
                "5",
                "--rounds",
                "5",
                "--device",
                "cpu",
                "--no-viz",
                "--no-gif",
            ],
        ),
        (
            "examples/channel/large.py",
            [
                "--rows",
                "10",
                "--cols",
                "10",
                "--rounds",
                "5",
                "--device",
                "cpu",
                "--no-viz",
                "--no-gif",
            ],
        ),
        # Gradient examples
        (
            "examples/gradients/fixed.py",
            ["--rounds", "5", "--device", "cpu", "--no-viz", "--no-gif"],
        ),
        (
            "examples/gradients/local.py",
            ["--rows", "3", "--cols", "3", "--device", "cpu"],
        ),
        (
            "examples/gradients/moving_source.py",
            ["--rounds", "5", "--device", "cpu", "--no-viz", "--no-gif"],
        ),
        (
            "examples/gradients/moving_nodes.py",
            [
                "--rounds",
                "5",
                "--num-nodes",
                "10",
                "--device",
                "cpu",
                "--no-viz",
                "--no-gif",
            ],
        ),
        (
            "examples/gradients/large.py",
            [
                "--rows",
                "10",
                "--cols",
                "10",
                "--rounds",
                "5",
                "--device",
                "cpu",
                "--no-viz",
                "--no-gif",
            ],
        ),
        (
            "examples/gradients/learnable.py",
            [
                "--rows",
                "3",
                "--cols",
                "3",
                "--epochs",
                "1",
                "--device",
                "cpu",
                "--no-viz",
                "--no-gif",
            ],
        ),
        (
            "examples/gradients/attention.py",
            [
                "--rows",
                "3",
                "--cols",
                "3",
                "--epochs",
                "1",
                "--device",
                "cpu",
                "--no-viz",
                "--no-gif",
            ],
        ),
        (
            "examples/gradients/moving_nodes_learnable.py",
            [
                "--rounds",
                "5",
                "--epochs",
                "1",
                "--num-nodes",
                "15",
                "--target",
                "5",
                "--device",
                "cpu",
                "--no-viz",
                "--no-gif",
            ],
        ),
    ]

    results = []
    for script, args in scripts_to_test:
        success, _ = run_script(script, args)
        results.append((script, success))

    print("\n" + "=" * 30)
    print("SUMMARY")
    print("=" * 30)
    all_success = True
    for script, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{script:40} : {status}")
        if not success:
            all_success = False

    if not all_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
