import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
    
def main():
    sample_file = 'energy_6pm_1.txt'
    sample_rate = 25_000
    debug = False

    #* read samples from file
    samples = np.loadtxt(sample_file)
    num_samples = len(samples)
    print(f'Read {num_samples:,} samples from {sample_file} at {sample_rate/1000}ks/s ({1000*num_samples/sample_rate:0.3f}ms)')

    #* trim to specific range of samples if required
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


    #* create timebase
    sample_period = 1/sample_rate
    T = num_samples * sample_period
    t = np.linspace(sample_period, T, num_samples)
    t_ms = 1000 * t

    #* find first peak (will be bin_ms after the start of the first spike)
    spike_window_ms = 12.5
    spike_window = int(spike_window_ms / 1000 * sample_rate)
    peak_index = 0
    peak = 0
    for i in range(0, spike_window):
        if samples[i] > peak:
            peak = samples[i]
            peak_index = i

    print(f'first spike {peak_index * sample_period * 1000:0.3f}ms')

    # locate 'main' spikes
    spike_plot = np.zeros_like(t)

    i = peak_index

    #* locate main spikes
    peak_search_start_ms = 9
    peak_search_stop_ms = 11
    peak_search_start = int(peak_search_start_ms / 1000 * sample_rate)
    peak_search_stop = int(peak_search_stop_ms / 1000 * sample_rate)

    spikes = []
    while (i + peak_search_stop) < num_samples:
        spikes.append(i)
        spike_plot[i] = 1

        # find next peak
        peak = 0
        peak_index = 0
        for j in range(i + peak_search_start, i + peak_search_stop):
            if samples[j] > peak:
                peak = samples[j]
                peak_index = j
        i = peak_index

    #* calculate min spike period
    spike_period = min(np.diff(spikes))
    spike_period_ms = spike_period * sample_period * 1000
    print(f'min spike period {spike_period_ms:.3f}ms ({1000/spike_period_ms:.3f}Hz)')

    #* mean energy in samples below threshold
    centile = 75
    threshold = np.quantile(samples, centile/100)
    mask = np.zeros_like(t)
    for i in range(num_samples):
        if samples[i] < threshold:
            mask[i] = samples[i]

    if debug:
        plt.figure()
        plt.plot(t_ms, samples)
        plt.plot(t_ms, mask)
        plt.title(sample_file)
        plt.xlabel('time (ms)')

    #* take mean of samples below threshold in each spike period
    mean_plot = np.zeros_like(t)
    means = []
    for spike_index in spikes:
        if spike_index + spike_period <= num_samples:
            total = 0
            num = 0
            for i in range(0, spike_period):
                if samples[i + spike_index] < threshold:
                    total += samples[i + spike_index]
                    num += 1
            means.append(total / num)
            mean_plot[spike_index:] = means[-1]

    #* rebase to median detector output
    #mid = 0.5* (max(mean_plot[spikes[0]:]) + min(mean_plot[spikes[0]:]))
    mid = np.mean(means)
    mean_plot[:spikes[0]] = mid
    mean_plot -= mid
    means -= mid

    #* integrate means over previous 100ms bit period
    det_plot = np.zeros_like(t)
    det = []
    win_len = int(100 / spike_period_ms)
    for i in range(len(means)):
        det.append(sum(means[max(0, i - win_len): i]) / win_len)
        det_plot[spikes[0] + i * spike_period:] = det[-1]


    plt.figure()
    plt.plot(t_ms, mean_plot, label=f'10ms mean < {centile}th centile')
    plt.plot(t_ms, det_plot, label='trailing 100ms integral')
    plt.title(sample_file)
    plt.xlabel('time (ms)')
    plt.grid(linestyle=':')
    plt.legend()

    plt.show()



if __name__ == '__main__':
    main()