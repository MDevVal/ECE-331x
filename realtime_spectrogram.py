import numpy as np
import adi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# SDR parameters
sample_rate = 1e6  # Hz
center_freq = 100e6  # Hz
fft_size = 1024  # FFT size
num_rows = 10  # Number of rows in the spectrogram display
num_ffts_per_update = 10  # Number of FFTs to average per update

sdr = adi.Pluto("ip:192.168.2.1")
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = 70.0  # dB
sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = fft_size * num_ffts_per_update  # Number of samples per read

spectrogram_data = np.zeros((num_rows, fft_size))

freqs = np.linspace(-sample_rate / 2, sample_rate / 2, fft_size) / 1e6  # Convert to MHz

fig, ax = plt.subplots()
im = ax.imshow(
    spectrogram_data,
    aspect="auto",
    extent=[freqs[0], freqs[-1], 0, num_rows * (fft_size / sample_rate)],
    origin="lower",
    cmap="viridis",
)
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("Time")
ax.set_title("Live Spectrogram")
plt.tight_layout()


def init():
    """Initialize the image data."""
    im.set_data(spectrogram_data)
    return [im]


def update(frame):
    x = sdr.rx()

    psd_accum = np.zeros(fft_size)

    for i in range(num_ffts_per_update):
        segment = x[i * fft_size : (i + 1) * fft_size]

        # Compute the FFT and shift zero frequency component to center
        X = np.fft.fftshift(np.fft.fft(segment, n=fft_size))

        psd = np.abs(X) ** 2

        psd_accum += psd

    psd_avg = psd_accum / num_ffts_per_update

    psd_db = 10 * np.log10(psd_avg)

    # Anti-artificating: Replace NaN and -Inf values with the minimum finite value
    psd_db = np.nan_to_num(
        psd_db,
        nan=np.min(psd_db[np.isfinite(psd_db)]),
        neginf=np.min(psd_db[np.isfinite(psd_db)]),
    )

    spectrogram_data[:-1] = spectrogram_data[1:]
    spectrogram_data[-1] = psd_db

    im.set_data(spectrogram_data)

    im.set_clim(np.min(spectrogram_data), np.max(spectrogram_data))

    return [im]


ani = FuncAnimation(fig, update, init_func=init, blit=False)

plt.show()
