import numpy as np
import sounddevice as sd

class Receiver(object):
	def __init__(self, sample_rate, carrier_freq):
		self.sample_rate = sample_rate
		self.carrier_freq = carrier_freq
	
	def record(self, duration) -> np.array:
		recording = sd.rec(duration * self.sample_rate, samplerate=self.sample_rate, channels=1)
		return recording

	def high_pass(self, wave, cutoff_freq) -> np.array:
		pass

	def demodulate(self, wave) -> np.array:
		pass

	def low_pass(self, wave, cutoff_freq) -> np.array:
		pass

	def wave_to_bits(self, wave) -> np.array:
		pass

	def bits_to_char(self, bits) -> str:
		pass

	def bits_to_string(self, bits) -> str:
		pass

	def receive_audio(self, duration) -> np.array:
		recording = self.record(duration)
		filtered_wave = self.high_pass(recording)
		demodulated_wave = self.demodulate(filtered_wave)
		reconstructed_signal = self.low_pass(demodulatd_wave, cutoff_freq)
		bits = self.wave_to_bits(wave)
		string = self.bits_to_string(bits)
		return string


if __name__ == '__main__':
	receiver = Receiver()
	duration = 10
	string = receiver.recieve_audio(duration)