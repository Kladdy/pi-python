# Imports
import argparse
import os
import time
import numpy as np
from constants import models_path
from toolbox import load_file_with_labels, get_pred_angle_diff_data, calculate_percentage_interval
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

cprint("Starting performance test for Coral...", "yellow")

# Initialize the TF interpreter
if noquant:
    quant_sting = ""
else:
    quant_sting= "_quantized_REPR_DATA_edgetpu"
path_to_model = f'{models_path}/model.{run_name}{quant_sting}.tflite'
interpreter = edgetpu.make_interpreter(path_to_model)
interpreter.allocate_tensors()

# Get input tensor.
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
print("Model loaded, input_shape:",input_shape)
print("Input details:", input_details)

# Load test file data
data, nu_direction = load_file_with_labels(i_file, n_events_to_load, start_index = 0)
predictions_cartesian = np.zeros((n_events_to_load, 3))

print("data.shape", data.shape)

N = n_events_to_load
print("This time, N = ", N)
# Make pedictions and time it
for i in range(N):
    if i % 100 == 0 or i + 1 == N:
        print(f"On step {i+1}/{N}...")
    data_tmp = data[i, :, :, :]
    #data_tmp = data_tmp[np.newaxis, :, :, :]

    #data_tmp = np.array(data_tmp, dtype=np.float32)
    data_tmp = np.array(data_tmp, dtype=np.int8)

    common.set_input(interpreter, data_tmp)
    interpreter.invoke()

    # Get the results and store it
    #res = common.output_tensor(interpreter, 0)[0]
    res = interpreter.get_tensor(3)

    predictions_cartesian[i,:] = res

# print(times)

angle_differences_deg = get_pred_angle_diff_data(nu_direction, predictions_cartesian)

angle_68 = calculate_percentage_interval(angle_differences_deg)
print("68 pecent angle:", angle_68)

# with open(f'{plots_dir}/model_{run_name}_file_{i_file}_iterations_{n_events_to_load}_performance_test.npy', 'wb') as f:
#     np.save(f, predictions_cartesian)

cprint("Performance test for Coral done!", "green")
