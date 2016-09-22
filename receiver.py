import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

class Receiver(object):
    def __init__(self, sample_rate, carrier_freq, samples_per_bit):
        self.sample_rate = sample_rate
        self.carrier_freq = carrier_freq
        self.samples_per_bit = samples_per_bit
    
    def record(self, duration) -> np.array:
        recording = sd.rec(duration * self.sample_rate, samplerate=self.sample_rate, channels=1)
        sd.wait()
        return np.squeeze(recording)

    def find_bounds(self, wave):
        wave_no_start = wave[20000:]
        threshold = np.max(wave_no_start) / 2
        signal_indices = np.squeeze(np.where(abs(wave_no_start) > threshold))
        bounds = signal_indices[0] + 20000, signal_indices[-1] + 20000
        return bounds

    def high_pass(self, wave) -> np.array:
        cutoff_freq = 0.9 * self.carrier_freq
        kernel_indices = np.arange(-41, 42)
        kernel = -(cutoff_freq / np.pi) * np.sinc(cutoff_freq * kernel_indices / np.pi)
        kernel[kernel_indices==0] = 1 + kernel[kernel_indices==0]
        return np.convolve(wave, kernel)

    def demodulate(self, wave) -> np.array:
        transform = np.cos(self.carrier_freq * np.arange(len(wave)))
        return np.multiply(transform, wave)

    def low_pass(self, wave, cutoff_freq=0.05) -> np.array:
        kernel_indices = np.arange(-41, 42)
        kernel = (cutoff_freq / np.pi) * np.sinc(cutoff_freq * kernel_indices / np.pi)
        return np.convolve(wave, kernel)

    def wave_to_bits(self, wave, bounds) -> np.array:
        start, stop = bounds
        first_pass = wave[start:stop + 1]
        indices = np.arange(0, len(first_pass), self.samples_per_bit)
        chunks = np.split(first_pass, indices)
        bits = np.array([int(np.mean(x) > 0) for x in chunks if len(x) > 0])
        return bits

    def process_bits(self, bits):
        is_flipped = np.mean(bits[:8]) < 0.5
        if is_flipped:
            bits = 1 - bits
        first_index = np.squeeze(np.where(bits[3:] == 0))[0]
        data = bits[first_index + 3:]
        chunks = np.split(data, np.arange(0, len(data), 8))
        found = False
        index = -1
        
        while not found:
            is_last = np.sum(chunks[index]) == len(chunks[index]) # Checking if last chunk is all 1s
            if is_last:
                found = True
            else:
                index -= 1

        data = data[:8 * (len(chunks) + index - 1)] # chop off end byte and anything after
        return data

    def bits_to_string(self, bits: np.array) -> str:
        bit_string = ''.join([str(x) for x in bits.tolist()])
        n = int('0b{}'.format(bit_string[1:]), 2)
        string = n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()
        return string

    def receive_audio(self, duration) -> np.array:
        recording = self.record(duration)
        filtered_wave = self.high_pass(recording)
        # self.plot_waveform(filtered_wave)
        demodulated_wave = self.demodulate(filtered_wave)
        reconstructed_signal = self.low_pass(demodulated_wave)
        bounds = self.find_bounds(reconstructed_signal)
        bits = self.wave_to_bits(reconstructed_signal, bounds)
        processed_bits = self.process_bits(bits)
        self.plot_waveform(reconstructed_signal)
        string = self.bits_to_string(processed_bits)
        return string

    def plot_waveform(self, wave):
        plt.plot(wave)
        plt.show()


if __name__ == '__main__':
    sample_rate = 44000
    carrier_freq = 2000 * np.pi / sample_rate
    samples_per_bit = 1000
    receiver = Receiver(sample_rate, carrier_freq, samples_per_bit)
    print(receiver.receive_audio(10))
    # demodulated = receiver.low_pass(receiver.demodulate(recording))
    # receiver.plot_waveform(receiver.low_pass(recording))