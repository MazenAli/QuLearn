import subprocess


def main():
    # Define different input values
    save_dir = "./results/fat_iqpe_reup_parity"
    seed = 0
    input_sets = [
        {
            "dmin": 2,
            "dmax": 2,
            "dstep": 1,
            "gamma": 0.1,
            "gamma_fac": 2.0,
            "num_qubits": 3,
            "num_reups": 1,
            "num_layers": 1,
            "omega": 0.0,
            "Sb": 10,
            "Sr": 3,
            "num_epochs": 500,
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
        script = "fat_iqpe_reup_parity.py"
        subprocess.run(f"python {script} {cmd_args}", shell=True, check=True)


if __name__ == "__main__":
    main()
