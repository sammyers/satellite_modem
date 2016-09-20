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
        return recording

    def high_pass(self, wave, cutoff_freq) -> np.array:
        pass

    def demodulate(self, wave) -> np.array:
        pass

    def low_pass(self, wave, cutoff_freq) -> np.array:
        pass

    def wave_to_bits(self, wave) -> np.array:
        pass

    def bits_to_string(self, bits: np.array) -> str:
        bit_string = ''.join([str(x) for x in bits.tolist()])
        n = int('0b{}'.format(bit_string[1:]), 2)
        string = n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()
        return string

    def receive_audio(self, duration) -> np.array:
        recording = self.record(duration)
        filtered_wave = self.high_pass(recording)
        demodulated_wave = self.demodulate(filtered_wave)
        reconstructed_signal = self.low_pass(demodulatd_wave, cutoff_freq)
        bits = self.wave_to_bits(wave)
        string = self.bits_to_string(bits)
        return string

    def plot_waveform(self, wave):
        plt.plot(wave)
        plt.show()


if __name__ == '__main__':
    sample_rate = 44000
    carrier_freq = 2000 * np.pi / sample_rate
    samples_per_bit = 1000
    receiver = Receiver(sample_rate, carrier_freq, samples_per_bit)
    # duration = 10
    # string = receiver.recieve_audio(duration)
    bits = np.tile(np.array([0, 1, 1, 0, 0, 0, 0, 1]), 2)
    recording = receiver.record(10)
    receiver.plot_waveform(recording)