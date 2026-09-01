import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
    

def main():
    debug = False
    use_night = False

    if use_night:
        sample_file = 'MSF_10pm_1.txt'
        sample_rate = 250_000
    else:
        sample_file = 'MSF_9am_3.txt'
        sample_rate = 240_000

    # read samples from file
    samples = np.loadtxt(sample_file)
    num_samples = len(samples)
    print(f'Read {num_samples:,} samples from {sample_file} at {sample_rate/1000}ks/s ({1000*num_samples/sample_rate:0.3f}ms)')

    # trim to specific range of samples (use '0' for all samples)
    sample_start_ms = 0
    sample_end_ms = 0     # '0' for all samples
    assert(sample_end_ms >= sample_start_ms)
    first_index = int(sample_start_ms/1000 * sample_rate)
    if (sample_end_ms > 0):
        last_index = int(sample_end_ms/1000 * sample_rate)
    else:
        last_index = num_samples
    if (sample_start_ms > 0 or sample_end_ms > 0):
        samples = samples[first_index: last_index]
        num_samples = len(samples)
        print(f'using samples {sample_start_ms}ms - {sample_end_ms})')

    # decimate if required
    decimation = 1
    if decimation > 1:
        samples = samples[0::decimation]
        num_samples = len(samples)
        sample_rate /= decimation


    # remove bias
    samples -= np.mean(samples)

    # create timebase
    sample_period = 1/sample_rate
    T = num_samples * sample_period
    t = np.linspace(sample_period, T, num_samples)
    t_ms = 1000 * t

    # calculate sliding integral of energy in a period
    energy = np.power(samples, 2)
    bin_ms = 2.5
    binsz = int(bin_ms/1000 * sample_rate)

    power = np.zeros_like(t)
    s = sum(energy[0: binsz])
    for i in range(binsz, num_samples - 1):
        power[i] = s
        s -= energy[i - binsz]
        s += energy[i + 1]

    if debug:
        plt.figure()
        plt.plot(t_ms, samples)
        plt.plot(t_ms, power / 200e3)
        plt.title(f'sliding integral of energy ({bin_ms}ms)')
        plt.xlabel('time (ms)')


    # find first peak (will be bin_ms after the start of the first spike)
    spike_window_ms = 12.5
    spike_window = int(spike_window_ms / 1000 * sample_rate)
    peak_index = 0
    peak = 0
    for i in range(0, spike_window):
        if power[i] > peak:
            peak = power[i]
            peak_index = i

    print(f'first spike {peak_index * sample_period * 1000:0.3f}ms')

    #* calculate energy at 60KHz within a given window after each spike, using the Geortzel filter
    data_start_ms = 6     # 5 is a good conservative choice
    data_stop_ms = 7.25
    data_start = int(data_start_ms / 1000 * sample_rate)
    data_stop = int(data_stop_ms / 1000 * sample_rate)

    # calculate Gortzel params (one-time work)
    N = data_stop - data_start      # number of samples in the window
    kc = N * 60_000 / sample_rate   # frequency bin number for 60kHz
    wI = np.cos(2*np.pi * kc / N)
    wQ = np.sin(2*np.pi * kc / N)

    data = []
    i = peak_index
    spike_num = 1
    data_plot = np.zeros_like(t)
    while i + data_stop < num_samples:

        signal_extract = samples[i + data_start: i + data_stop]
        signal_extract = signal_extract * np.hanning(N)

        if spike_num == 6 or spike_num == 8 or spike_num == 59:
            plt.figure()
            plt.magnitude_spectrum(signal_extract, sample_rate)
            plt.title(f'spike {spike_num}')
            plt.grid(linestyle=':')
            plt.ylim(0,100)


        #signal_extract = signal_extract * np.hanning(N)

        # ----- Goertzel filter -----
        # calculate v(n) using the IIR
        v1 = v2 = 0
        for n in range(N):
            v = 2 * wI * v1 - v2 + signal_extract[n]
            v2 = v1
            v1 = v
        v = v1
        v1 = v2

        # calculate y(n) using the FIR
        yI = v - wI * v1
        yQ = wQ * v1
        # ---------------------------

        x = yI ** 2 + yQ ** 2       # power in 60kHz bin

        data.append(x)
        data_plot[i + data_start: i + data_stop] = x

        # find next peak
        peak_search_start_ms = 9
        peak_search_stop_ms = 11
        peak_search_start = int(peak_search_start_ms / 1000 * sample_rate)
        peak_search_stop = int(peak_search_stop_ms / 1000 * sample_rate)
        peak = 0
        peak_index = 0
        for j in range(i + peak_search_start, i + peak_search_stop):
            if power[j] > peak:
                peak = power[j]
                peak_index = j
        i = peak_index
        spike_num += 1

    plt.figure()
    plt.plot(t_ms, samples)
    plt.plot(t_ms, data_plot/2e5)
    plt.title(sample_file)
    plt.xlabel('time (ms)')

    plt.show()



if __name__ == '__main__':
    main()