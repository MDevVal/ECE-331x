import adi
import numpy as np
import matplotlib.pyplot as plt
import time

sdr = adi.Pluto("ip:192.168.2.1")
sdr.rx_lo = int(433.9e6)
sdr.sample_rate = int(1e6)
sdr.rx_rf_bandwidth = int(1e6)
sdr.rx_buffer_size = 1024 * 64
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = 50

threshold = 90
max_wait_time = 60

print("Waiting for signal transmission...")
start_time = time.time()
signal_detected = False

while not signal_detected and (time.time() - start_time) < max_wait_time:
    iq_data = sdr.rx()
    magnitude = np.abs(iq_data)
    if np.max(magnitude) > threshold:
        signal_detected = True
        print("Signal detected!")
    else:
        time.sleep(0.1)

if not signal_detected:
    print("Signal not detected within the maximum wait time.")
    exit()

N = len(iq_data)
t = np.arange(N) / sdr.sample_rate

IQ_FFT = np.fft.fftshift(np.fft.fft(iq_data))
freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / sdr.sample_rate))
peak_freq_idx = np.argmax(np.abs(IQ_FFT))
freq_offset = freqs[peak_freq_idx]
print(f"Estimated frequency offset: {freq_offset} Hz")

correction_signal = np.exp(-1j * 2 * np.pi * freq_offset * (t * 2))
iq_data_corrected = iq_data * correction_signal

magnitude_raw = np.abs(iq_data)
phase_raw = np.angle(iq_data)
magnitude_corrected = np.abs(iq_data_corrected)
phase_corrected = np.angle(iq_data_corrected)

fig, axs = plt.subplots(3, 2, figsize=(15, 12))

# Magnitude
axs[0, 0].plot(t, magnitude_raw)
axs[0, 0].set_title("Magnitude - Before Correction")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Magnitude")

axs[0, 1].plot(t, magnitude_corrected)
axs[0, 1].set_title("Magnitude - After Correction")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Magnitude")

# Phase
axs[1, 0].plot(t, phase_raw)
axs[1, 0].set_title("Phase - Before Correction")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Phase (radians)")

axs[1, 1].plot(t, phase_corrected)
axs[1, 1].set_title("Phase - After Correction")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Phase (radians)")

# Constellation
axs[2, 0].scatter(np.real(iq_data), np.imag(iq_data), s=10)
axs[2, 0].set_title("Constellation - Before Correction")
axs[2, 0].set_xlabel("In-phase")
axs[2, 0].set_ylabel("Quadrature")
axs[2, 0].grid(True)
axs[2, 0].axis("equal")

axs[2, 1].scatter(np.real(iq_data_corrected), np.imag(iq_data_corrected), s=10)
axs[2, 1].set_title("Constellation - After Correction")
axs[2, 1].set_xlabel("In-phase")
axs[2, 1].set_ylabel("Quadrature")
axs[2, 1].grid(True)
axs[2, 1].axis("equal")

plt.tight_layout()
plt.savefig("figure1.png")
plt.close()
