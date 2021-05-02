# Imports
import argparse
import os
import time
import numpy as np
from constants import models_path
from toolbox import load_file
from termcolor import cprint

from pycoral.utils import edgetpu
from pycoral.adapters import common
# -------

# Constants
plots_dir = "plots"

# Parse arguments
parser = argparse.ArgumentParser(description='Test inference speed on dl1 machine')
parser.add_argument("run_id", type=str, help="the id of the run, eg '3.2' for run3.2")
parser.add_argument("i_file", type=int, help="the id of file to do inference on")
parser.add_argument("n_events_to_load", type=int, help="amount of events to load")
parser.add_argument("--noquant", dest="noquant", action="store_true", help="load model without quantization")

args = parser.parse_args()
run_id = args.run_id
i_file = args.i_file
n_events_to_load = args.n_events_to_load
noquant = args.noquant

print(noquant)

# Save the run name
run_name = f"run{run_id}"

# Make sure saved_models folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

cprint("Starting inference test for Coral...", "yellow")

# Initialize the TF interpreter
if noquant:
    quant_sting = ""
else:
    quant_sting= "_quantized_edgetpu"
path_to_model = f'{models_path}/model.{run_name}{quant_sting}.tflite'
interpreter = edgetpu.make_interpreter(path_to_model)
interpreter.allocate_tensors()

# Get input tensor.
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
print("Model loaded, input_shape:",input_shape)
print("Input details:", input_details)

# Load test file data
    # Load first file
data = load_file(i_file, n_events_to_load)

# Amount of times to do 1-inferences:
times = []

print("n_events_to_load/10.0", n_events_to_load/10.0)
print("data.shape", data.shape)

N = n_events_to_load
print("This time, N = ", N)
# Make pedictions and time it
for i in range(N):
    if i % 100 == 0:
        print(f"On step {i}/{N}...")
    data_tmp = data[i, :, :, :]
    #data_tmp = data_tmp[np.newaxis, :, :, :]

    #data_tmp = np.array(data_tmp, dtype=np.float32)
    data_tmp = np.array(data_tmp, dtype=np.int8)

    t0 = time.time()

    common.set_input(interpreter, data_tmp)
    interpreter.invoke()

    t = time.time() - t0

    if i != 0:
        times.append(t)

# print(times)

mean = np.mean(times)
median = np.median(times)
std = np.std(times)

print("Mean time:", f"{(mean*1000):.2f}", "ms")
print("Median time:", f"{(mean*1000):.2f}", "ms")
print("Standard deviation:", f"{(std*1000):.2f}", "ms")

with open(f'{plots_dir}/model_{run_name}_file_{i_file}_iterations_{n_events_to_load}_inference_test.npy', 'wb') as f:
    np.save(f, times)
    np.save(f, mean)
    np.save(f, std)

cprint("Inference test for Coral done!", "green")
