# Imports
import os
import numpy as np
import time
from constants import datapath, data_filename, label_filename
import pickle
from radiotools import stats
from radiotools import helper as hp
# -------

# Loading data and label files
def load_file(i_file, n_events, norm=1e-6):
    # Load 500 MHz filter
    filt = np.load("bandpass_filters/500MHz_filter.npy")

    t0 = time.time()
    print(f"loading file {i_file}", flush=True)
    data = np.load(os.path.join(datapath, f"{data_filename}{i_file:04d}.npy"), allow_pickle=True, mmap_mode="r")
    
    data = data[0:n_events, :, :]
    data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
    data = data[:, :, :, np.newaxis]

    # check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]

    data /= norm

    return data

# Loading data and label files
def load_file_with_labels(i_file, n_events, start_index = 0, norm=1e-6):
    # Load 500 MHz filter
    filt = np.load("bandpass_filters/500MHz_filter.npy")

    t0 = time.time()
    print(f"loading file {i_file}", flush=True)
    data = np.load(os.path.join(datapath, f"{data_filename}{i_file:04d}.npy"), allow_pickle=True, mmap_mode="r")
    
    data = data[start_index:(n_events+start_index), :, :]
    data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
    data = data[:, :, :, np.newaxis]

    labels_tmp = np.load(os.path.join(datapath, f"{label_filename}{i_file:04d}.npy"), allow_pickle=True)
    print(f"finished loading file {i_file} in {time.time() - t0}s")
    nu_zenith = np.array(labels_tmp.item()["nu_zenith"])
    nu_azimuth = np.array(labels_tmp.item()["nu_azimuth"])
    nu_direction = hp.spherical_to_cartesian(nu_zenith, nu_azimuth)

    nu_direction = nu_direction[start_index:(n_events+start_index),:]

    # check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]
    nu_direction = nu_direction[idx]
    data /= norm

    return data, nu_direction


def get_pred_angle_diff_data(true_directions_cartesian, predicted_directions_cartesian):

    # Only pick first 100000 data
    # N = 100000
    # nu_direction_predict = nu_direction_predict[:N]
    # nu_direction = nu_direction[:N]

    angle_difference_data = np.array([hp.get_angle(predicted_directions_cartesian[i], true_directions_cartesian[i]) for i in range(len(true_directions_cartesian))]) * 180 / np.pi

    return angle_difference_data

def calculate_percentage_interval(angle_difference_data, percentage=0.68):
    # Redefine N
    N = angle_difference_data.size
    weights = np.ones(N)

    angle = stats.quantile_1d(angle_difference_data, weights, percentage)

    # OLD METHOD -------------------------------
    # Calculate Rayleigh fit
    # loc, scale = stats.rayleigh.fit(angle)
    # xl = np.linspace(angle.min(), angle.max(), 100) # linspace for plotting

    # Calculate 68 %
    #index_at_68 = int(0.68 * N)
    #angle_68 = np.sort(angle_difference_data)[index_at_68]
    # ------------------------------------------

    return angle