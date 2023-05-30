import subprocess


def main():
    # Define different input values
    save_dir = "./results/rademacher_iqpe_reup_parity"
    seed = 0
    input_sets = [
        {
            "m": 2,
            "num_qubits": 3,
            "num_reups": 1,
            "num_layers": 1,
            "omega": 0.0,
            "num_sigma_samples": 10,
            "num_data_samples": 5,
            "num_epochs": 300,
            "opt_stop": 1e-16,
            "stagnation_threshold": 0.01,
            "stagnation_count": 100,
            "seed": seed,
            "save_dir": save_dir,
        },
    ]

    for i, input_set in enumerate(input_sets):
        # Generate the command-line arguments for the input set
        cmd_args = " ".join([f"--{key} {value}" for key, value in input_set.items()])

        # Run the script with the input set
        script = "rademacher_iqpe_reup_parity.py"
        subprocess.run(f"python {script} {cmd_args}", shell=True, check=True)


if __name__ == "__main__":
    main()
