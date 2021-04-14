# Imports
import argparse
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from termcolor import cprint
# -------

# Constants
plots_dir = "plots"

# Parse arguments
parser = argparse.ArgumentParser(description='Test inference speed on dl1 machine')
parser.add_argument("run_id", type=str, help="the id of the run, eg '3.2' for run3.2")
parser.add_argument("i_file", type=int, help="the id of file to do inference on")

args = parser.parse_args()
run_id = args.run_id
i_file = args.i_file

n_threads_list = [1,2,3]

# Save the run name
run_name = f"run{run_id}"

# Make sure saved_models folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

cprint("Starting plotting of inference test for Pi 3B...", "yellow")

fig, ax = plt.subplots(1,1)

for n_threads in n_threads_list:
    with open(f'{plots_dir}/model_{run_name}_file_{i_file}_threads_{n_threads}_inference_test.npy', 'rb') as f:
        batch_sizes = np.load(f)
        times_mean = np.load(f)
        times_std = np.load(f)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(batch_sizes, times_mean, "*", color="mediumorchid", label=f"data, {n_threads} threads")

    ax.set(title='Amount of time for one inference as a function of events per inference')
    ax.set(xlabel=r"Events per inference $N_{events, inf}$")
    ax.set(ylabel=r"Time per inference $t_{inf}$ (s)")
    plt.tight_layout()
    # plt.xlabel("Batch size")
    # plt.ylabel("Time")



#plt.legend(loc="upper left")
plt.savefig(f"{plots_dir}/model_{run_name}_file_{i_file}_inference_plot_raspberryPi.eps", format="eps")



#plt.savefig(f"{plots_dir}/model_{run_name}_file_{i_file}_threads_{n_threads}_inference_plot.png")

cprint("Inference plot for Pi 3B done!", "green")




