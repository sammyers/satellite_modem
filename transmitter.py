import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

class Transmitter(object):
    def __init__(self, sample_rate, carrier_freq, samples_per_bit):
        self.sample_rate = sample_rate
        self.carrier_freq = carrier_freq
        self.samples_per_bit = samples_per_bit

    def string_to_bits(self, string) -> np.array:
        bits = bin(int.from_bytes(string.encode(), 'big'))
        actual_bits = '0{}'.format(bits[2:])
        return np.array([int(x) for x in actual_bits])

    def bits_to_wave(self, bits) -> np.array:
        padded_bits = np.concatenate([np.ones(8), bits, np.ones(8)])
        padded_bits[padded_bits==0] = -1
        return np.repeat(padded_bits, self.samples_per_bit)

    def modulate(self, wave) -> np.array:
        carrier_wave = np.cos(self.carrier_freq * np.arange(len(wave)))
        return np.multiply(carrier_wave, wave)

    def play(self, wave):
        sd.play(wave, self.sample_rate)
        sd.wait()

    def transmit_audio(self, string):
        bits = self.string_to_bits(string)
        wave = self.bits_to_wave(bits)
        shifted_wave = self.modulate(wave)
        self.play(shifted_wave)

    def plot_waveform(self, wave):
        plt.plot(wave)
        plt.show()

    def plot_fft(self, wave):
        n = len(wave)
        freq_range = np.linspace(-np.pi, np.pi * (n - 1) / n, n)
        fft = np.fft.fft(wave)
        plt.plot(freq_range, np.fft.fftshift(np.abs(fft)))
        plt.show()

if __name__ == '__main__':
    sample_rate = 44000
    carrier_freq = 2000 * np.pi / sample_rate
    samples_per_bit = 1000
    transmitter = Transmitter(sample_rate, carrier_freq, samples_per_bit)
    signal = transmitter.bits_to_wave(transmitter.string_to_bits('asdfaeirauwenfajsdijr'))
    string = 'this is a test'
    print(string)
    # print(transmitter.string_to_bits(string))
    transmitter.transmit_audio(string)

