import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def find_spikes(samples, samples_per_sec, spike_centile=0.99, spike_width_ms=2, debug=False):
    # estimate level of spikes (top centile of samples)
    abs_samples = np.abs(samples)
    spike_level = np.quantile(abs_samples, spike_centile)
    if debug:
        print(f'{spike_centile*100}th centile {spike_level:.3f}')

    # find spikes
    spike_index = []
    i = 0
    while i < len(samples):
        while i < len(samples) and abs_samples[i] < spike_level:
            i += 1
        if i < len(samples):
            spike_index.append(i)                           # found the start of a spike
            i += int(samples_per_sec * spike_width_ms/1000) # skip over spike oscillations
    if debug:
        print(len(spike_index), 'probable spikes')

    # find most popular interval between spikes
    spike_interval = np.argmax(np.bincount(np.diff(spike_index)))

    plt.figure()
    plt.plot(np.bincount(np.diff(spike_index)))

    if debug:
        print(f'most popular interval {spike_interval} ({samples_per_sec/spike_interval:0.3f}Hz)')

    # find most popular phase of spikes
    offset = [spike % spike_interval for spike in spike_index]
    spike_offset = np.argmax(np.bincount(offset))
    if debug:
        print(f'most popular offset {spike_offset}')
    return [spike_interval, spike_offset]
    

def main():
    #sample_file = 'MSF_9am_3.txt'
    #sample_rate = 240_000
    sample_file = 'MSF_10pm_1.txt'
    sample_rate = 250_000
    debug=True

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
    if (first_index > 0 or last_index > 0):
        samples = samples[first_index: last_index]
        num_samples = len(samples)
        print(f'using samples {sample_start_ms}ms - {sample_end_ms}ms)')

    # remove bias
    samples -= np.mean(samples)

    # create timebase
    sample_period = 1/sample_rate
    T = num_samples * sample_period
    t = np.linspace(sample_period, T, num_samples)
    t_ms = 1000 * t

    # find spikes
    spike_interval, spike_offset = find_spikes(samples, sample_rate, debug=debug)
    spike_index = range(spike_offset, num_samples, spike_interval)

    # create time mask for signal
    # note: using a rectangular mask like this adds 100Hz artifacts to the signal - a smoother window would reduce that
    spike_duration_ms = 5.0     # how long to wait before enabling the mask (normally 3 - 7ms)
    pre_spike_ms = 0.1          # how soon before the next spike to disable the mask (normally 0.1ms)
    mask = np.zeros_like(t)
    mask_first_index = int(spike_duration_ms/1000 * sample_rate)
    mask_last_index = spike_interval - int(pre_spike_ms/1000 * sample_rate)
    for index in spike_index:
        mask[index + mask_first_index: index + mask_last_index] = 1

    if (debug):
        plt.figure()
        plt.plot(t_ms, samples, label='signal')
        plt.plot(t_ms, -200 + 400*mask, label='mask')
        plt.xlabel('time ms')
        plt.legend()
        plt.title('Spike mask')

    masked_signal = samples * mask
    if debug:
        plt.figure()
        plt.plot(t_ms, masked_signal)
        plt.xlabel('time ms')
        plt.title('Sampled signal')

    plt.show()
    return

    if debug:
        # show spectrum of masked signal
        # note: the spikes at 60.100, 60.200 etc are artifacts due to the abrupt edges of the mask (see above)
        plt.figure()
        plt.magnitude_spectrum(masked_signal, Fs=sample_rate/1000, scale='dB')
        plt.xlabel('Freq (kHz)')
        #plt.ylim(-80,-20)
        if (sample_start_ms > 0 or sample_end_ms > 0):
            plt.title(f'Signal with spikes masked ({sample_start_ms} - {sample_end_ms}ms)')
        else:
            plt.title(f'Signal with spikes masked')
        plt.grid(linestyle=':')

    # try a sliding sum of the absolute signal
    abs_masked_signal = np.abs(masked_signal)
    detector_output = np.zeros_like(t)

    integration_time_ms = 100
    integration_num_samples = int(integration_time_ms/1000 * sample_rate)
    print(integration_num_samples)
    for i in range(num_samples - integration_num_samples):
        result = sum(abs_masked_signal[i: i + integration_num_samples: 10])
        detector_output[i] = result

    plt.figure()
    plt.plot(t_ms, masked_signal)
    plt.plot(t_ms, detector_output)
    plt.title(f'Sliding {integration_time_ms}ms sum of signal envelope')

    plt.show()



if __name__ == '__main__':
    main()