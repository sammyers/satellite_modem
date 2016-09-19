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
		bits[bits==0] = -1
		return np.repeat(bits, self.samples_per_bit)

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

if __name__ == '__main__':
	sample_rate = 44000
	carrier_freq = 2000 * np.pi / sample_rate
	samples_per_bit = 1000
	transmitter = Transmitter(sample_rate, carrier_freq, samples_per_bit)
	transmitter.transmit_audio('asdfaeirauwenfajsdijr')
