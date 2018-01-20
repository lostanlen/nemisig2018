from joblib import Memory
import math
import music21 as m21
import numpy as np
import os
from scipy.fftpack import fft, ifft


def get_composers():
    return ["Haydn", "Mozart"]

def get_data_dir():
    return "/scratch/vl1019/nemisig2018_data"

def get_dataset_name():
    return "nemisig2018"



def concatenate_layers(Sx, depth):
    layers = []
    for m in range(depth+1):
        layers.append(Sx[m].flatten())
    return np.concatenate(layers)


def frequential_filterbank(dim, J_fr, xi=0.4, sigma=0.16):
    N = 2**J_fr
    filterbank = np.zeros((N, 1, 2*(J_fr-2)+1))
    for j in range(J_fr-2):
        xi_j = xi * 2**(-j)
        sigma_j = sigma * 2**(-j)
        center = xi_j * N
        den = 2 * sigma_j * sigma_j * N * N
        psi = morlet(center, den, N, n_periods=4)
        filterbank[:, 0, j] = psi
    for j in range(J_fr-2, 2*(J_fr-2)):
        psi = filterbank[:, 0, j - (J_fr-2)]
        rev_psi = np.concatenate((psi[0:1], psi[1:][::-1]))
        filterbank[:, 0, j] = rev_psi
    sigma_phi = 2.0 * sigma * 2**(-(J_fr-2))
    center_phi = 0.0
    den_phi = sigma_phi * sigma_phi * N * N
    phi = gabor(center_phi, den_phi, N)
    rev_phi = np.concatenate((phi[0:1], phi[1:][::-1]))
    phi = phi + rev_phi
    phi[0] = 1.0
    filterbank[:, 0, -1] = phi
    for m in range(dim):
        filterbank = np.expand_dims(filterbank, axis=2)
    return filterbank


def gabor(center, den, N):
    omegas = np.array(range(N))
    return gauss(omegas - center, den)


def gauss(omega, den):
    return np.exp(- omega*omega / den)


def get_pianoroll_part(part, J_fr, J_tm, quantization):
    # Get the measure offsets
    measure_offset = {}
    for el in part.recurse(classFilter=('Measure')):
        measure_offset[el.measureNumber] = el.offset
    # Get the duration of the part
    N = 2**J_tm
    # Get the pitch and offset+duration
    piano_roll_part = np.zeros((2**J_fr, N))
    for this_note in part.recurse(classFilter=('Note')):
        note_start = get_start_time(this_note,measure_offset,quantization)
        note_end = get_end_time(this_note,measure_offset,quantization)
        piano_roll_part[this_note.midi,note_start:note_end] = 1
    return piano_roll_part


def get_start_time(el,measure_offset,quantization):
    if (el.offset is not None) and (el.measureNumber in measure_offset):
        return int(math.ceil((measure_offset[el.measureNumber] + el.offset)*quantization))
    # Else, no time defined for this element and the function returns None


def get_end_time(el,measure_offset,quantization):
    if (el.offset is not None) and (el.measureNumber in measure_offset):
        return int(math.ceil((measure_offset[el.measureNumber] + el.offset + el.duration.quarterLength)*quantization))
    # Else, no time defined for this element and the function returns None


def is_even(n):
    return (n%2 == 0)


def load_composer(composer_dir, J_fr, J_tm, quantization):
    pianorolls = []
    for (_, _, filenames) in os.walk(composer_dir):
        for filename in filenames:
            path = os.path.join(composer_dir, filename)
            score = m21.converter.parse(path)
            pianoroll = score_to_pianoroll(score, J_fr, J_tm, quantization)
            pianorolls.append(pianoroll)
    return pianorolls


def morlet(center, den, N, n_periods):
    half_N = N >> 1
    p_start = - ((n_periods-1) >> 1) - is_even(n_periods)
    p_stop = ((n_periods-1) >> 1) + 1
    omega_start = p_start * N
    omega_stop = p_stop * N
    omegas = np.array(range(omega_start, omega_stop))
    gauss_center = gauss(omegas - center, den)
    corrective_gaussians = np.zeros((N*n_periods, n_periods))
    for p in range(n_periods):
        offset = (p_start + p) * N
        corrective_gaussians[:, p] = gauss(omegas - offset, den)
    p_range = range(p_start, p_stop)
    b = np.array([gauss(p*N - center, den) for p in p_range])
    A = np.array([gauss((q-p)*N, den)
                 for p in range(n_periods)
                 for q in range(n_periods)]).reshape(n_periods, n_periods)
    corrective_factors = np.linalg.solve(A, b)
    y = gauss_center - np.dot(corrective_gaussians, corrective_factors)
    y = np.fft.fftshift(y)
    y = np.reshape(y, (n_periods, N))
    y = np.sum(y, axis=0)
    return y


def scatter(U, filterbank, dim):
    U_ft = fft(U, axis=dim)
    U_ft = np.expand_dims(U_ft, axis=-1)
    Y_ft = U_ft * filterbank
    Y = ifft(Y_ft, axis=dim)
    return Y


def score_to_pianoroll(score, J_fr, J_tm, quantization):
    pianoroll_parts = []
    n_parts = len(score.parts)
    for part_id in range(n_parts):
        part = score.parts[part_id]
        pianoroll_part = get_pianoroll_part(part, J_fr, J_tm, quantization)
        pianoroll_parts.append(pianoroll_part)
    mtrack_pianoroll = np.stack(pianoroll_parts, 2)
    pianoroll = mtrack_pianoroll.max(axis=2)
    return pianoroll


def setup_temporal_scattering(J_tm, depth):
    filterbanks = []
    for m in range(depth):
        filterbank = temporal_filterbank(m, J_tm)
        filterbanks.append(filterbank)
    return filterbanks


def setup_timefrequency_scattering(J_tm, J_fr, depth):
    filterbanks_tm = []
    filterbanks_fr = []
    for m in range(depth):
        filterbank_tm = temporal_filterbank(2*m, J_tm)
        filterbank_fr = frequential_filterbank(2*m+1, J_fr)
        filterbanks_tm.append(filterbank_tm)
        filterbanks_fr.append(filterbank_fr)
    return (filterbanks_tm, filterbanks_fr)


def temporal_filterbank(dim, J_tm, xi=0.4, sigma=0.16):
    N = 2**J_tm
    filterbank = np.zeros((1, N, J_tm-2))
    for j in range(J_tm-2):
        xi_j = xi * 2**(-j)
        sigma_j = sigma * 2**(-j)
        center = xi_j * N
        den = 2 * sigma_j * sigma_j * N * N
        psi = morlet(center, den, N, n_periods=4)
        filterbank[0, :, j] = psi
    for m in range(dim):
        filterbank = np.expand_dims(filterbank, axis=2)
    return filterbank


def temporal_scattering(pianoroll, filterbanks, nonlinearity):
    depth = len(filterbanks)
    Us = [pianoroll]
    Ss = []
    for m in range(depth):
        U = Us[m]
        S = np.sum(U, axis=(0, 1))
        filterbank = filterbanks[m]
        Y = scatter(U, filterbank, 1)
        if nonlinearity == "abs":
            U = np.abs(Y)
        else:
            raise NotImplementedError
        Us.append(U)
        Ss.append(S)
    S = np.sum(U, axis=(0, 1))
    Ss.append(S)
    return Ss


def timefrequency_scattering(pianoroll, filterbanks, nonlinearity):
    filterbanks_tm = filterbanks[0]
    filterbanks_fr = filterbanks[1]
    depth = len(filterbanks_tm)
    Us = [pianoroll]
    Ss = []
    for m in range(depth):
        U = Us[m]
        S = np.sum(U, axis=(0,1))
        filterbank_tm = filterbanks_tm[m]
        filterbank_fr = filterbanks_fr[m]
        Y_tm = scatter(U, filterbank_tm, 1)
        Y_fr = scatter(Y_tm, filterbank_fr, 0)
        if nonlinearity == "abs":
            U = np.abs(Y_fr)
        else:
            raise NotImplementedError
        Us.append(U)
        Ss.append(S)
    S = np.sum(U, axis=(0, 1))
    Ss.append(S)
    return Ss
