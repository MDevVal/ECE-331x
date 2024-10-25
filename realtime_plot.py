import numpy as np
import adi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Adjust sample_count to change data buffer/frame width
sample_count = 25

fig, ax = plt.subplots()
(line_i,) = ax.plot([], [], label="I (In-phase)")
(line_q,) = ax.plot([], [], label="Q (Quadrature)")
ax.set_xlim(0, sample_count)
ax.set_ylim(-50, 50)
ax.set_ylabel("Amplitude")
ax.set_title("Real-Time Pluto SDR Incoming Data")
ax.legend()
ax.grid(True)


def init():
    line_i.set_data([], [])
    line_q.set_data([], [])
    return line_i, line_q


def update(frame):
    samples = sdr.rx()
    i_data = np.real(samples)[:sample_count]
    q_data = np.imag(samples)[:sample_count]

    line_i.set_data(range(len(i_data)), i_data)
    line_q.set_data(range(len(q_data)), q_data)
    return line_i, line_q


if __name__ == "__main__":
    sdr = adi.Pluto("ip:192.168.2.1")
    sdr.rx_rf_bandwidth = 4000000
    ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=100)
    plt.show()
