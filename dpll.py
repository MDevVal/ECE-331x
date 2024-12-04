import numpy as np
import matplotlib.pyplot as plt
import adi
import scipy.signal as signal
import sys


def check_pluto_connection():
    try:
        sdr = adi.Pluto("ip:192.168.2.1")
        print("PlutoSDR connected successfully.")
        return sdr
    except Exception as e:
        print(f"Error connecting to PlutoSDR: {e}")
        sys.exit(1)


def configure_sdr(sdr):
    try:
        sdr.rx_rf_bandwidth = int(1e6)
        sdr.rx_lo = int(2.426e9)
        sdr.rx_sample_rate = int(2.5e6)
        sdr.rx_buffer_size = int(1024 * 1024)
        print("PlutoSDR configured successfully.")
    except Exception as e:
        print(f"Error configuring PlutoSDR: {e}")
        sys.exit(1)


def capture_samples(sdr, num_samples):
    try:
        print(f"Capturing {num_samples} samples...")
        samples = sdr.rx()
        if samples.size == 0:
            print("No samples captured. Exiting.")
            sys.exit(1)
        print("Capture complete.")
        return samples
    except Exception as e:
        print(f"Error capturing samples: {e}")
        sys.exit(1)


def process_and_plot(samples, sdr):
    samples = samples / np.max(np.abs(samples))
    samp_rate = sdr.rx_sample_rate
    nyq_rate = samp_rate / 2
    cutoff_freq = 100e3
    b, a = signal.butter(5, cutoff_freq / nyq_rate, btype="low")
    filtered_samples = signal.lfilter(b, a, samples)

    symbol_rate = 100e3
    samples_per_symbol = int(samp_rate / symbol_rate)
    symbols_original = filtered_samples[::samples_per_symbol]

    N = len(filtered_samples)
    phase = 0.0
    freq = 0.0
    alpha = 0.132
    beta = 0.00932
    out = np.zeros(N, dtype=np.complex64)
    error_log = []

    for i in range(N):
        out[i] = filtered_samples[i] * np.exp(-1j * phase)
        error = np.real(out[i]) * np.imag(out[i])
        error_log.append(error)
        freq += beta * error
        phase += freq + (alpha * error)
        if phase >= 2 * np.pi:
            phase -= 2 * np.pi
        elif phase < 0:
            phase += 2 * np.pi

    symbols_corrected = out[::samples_per_symbol]

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    axs[0].plot(
        symbols_original.real, symbols_original.imag, ".", markersize=2, alpha=0.5
    )
    axs[0].set_title("Original BPSK Constellation")
    axs[0].set_xlabel("In-phase")
    axs[0].set_ylabel("Quadrature")
    axs[0].grid(True)
    axs[0].axhline(0, color="gray", linewidth=0.5)
    axs[0].axvline(0, color="gray", linewidth=0.5)
    axs[0].axis("equal")

    axs[1].plot(
        symbols_corrected.real,
        symbols_corrected.imag,
        ".",
        markersize=2,
        alpha=0.5,
        color="orange",
    )
    axs[1].set_title("Corrected BPSK Constellation (After DPLL)")
    axs[1].set_xlabel("In-phase")
    axs[1].set_ylabel("Quadrature")
    axs[1].grid(True)
    axs[1].axhline(0, color="gray", linewidth=0.5)
    axs[1].axvline(0, color="gray", linewidth=0.5)
    axs[1].axis("equal")

    axs[2].plot(error_log, ".", markersize=1, color="green")
    axs[2].set_title("Phase Error Over Time")
    axs[2].set_xlabel("Sample Index")
    axs[2].set_ylabel("Phase Error")
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("DPLL_Figure.png")
    plt.show()


def main():
    sdr = check_pluto_connection()
    configure_sdr(sdr)
    num_samples = 2**20
    samples = capture_samples(sdr, num_samples)
    process_and_plot(samples, sdr)


if __name__ == "__main__":
    main()
