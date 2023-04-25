import subprocess


def main():
    # Define different input values
    save_dir = "./results/torch_iqpe_reup_parity"
    seed = 0
    input_sets = [
        {
            "Nmin": 1,
            "Nmax": 2,
            "Nstep": 1,
            "num_qubits": 3,
            "num_reups": 1,
            "num_layers": 1,
            "omega": 0.0,
            "num_samples": 2,
            "opt_steps": 300,
            "opt_stop": 1e-16,
            "early_stop": "",
            "seed": seed,
            "save_dir": save_dir,
        },
        {
            "Nmin": 1,
            "Nmax": 2,
            "Nstep": 1,
            "num_qubits": 3,
            "num_reups": 1,
            "num_layers": 1,
            "omega": 0.0,
            "num_samples": 2,
            "opt_steps": 300,
            "opt_stop": 1e-16,
            "early_stop": "",
            "seed": seed,
            "save_dir": save_dir,
            "cuda": "",
        },
    ]

    for i, input_set in enumerate(input_sets):
        # Generate the command-line arguments for the input set
        cmd_args = " ".join([f"--{key} {value}" for key, value in input_set.items()])

        # Run the script with the input set
        script = "capacity_torch_iqpe_reup_parity.py"
        subprocess.run(f"python {script} {cmd_args}", shell=True, check=True)


if __name__ == "__main__":
    main()
