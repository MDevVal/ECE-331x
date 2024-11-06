import adi
import numpy as np
import time
import sys
from collections import deque


def lfsr_digest8(message, gen, key):
    """
    Computes an 8-bit checksum using a Linear Feedback Shift Register (LFSR).

    Args:
        message (list of int): The message bytes to compute the checksum for.
        gen (int): The generator polynomial for the LFSR.
        key (int): The initial key value for the LFSR.

    Returns:
        int: The computed 8-bit checksum.
    """
    checksum = 0
    for byte in message:
        for bit in range(7, -1, -1):
            if (byte >> bit) & 1:
                checksum ^= key
            if key & 1:
                key = (key >> 1) ^ gen
            else:
                key >>= 1
            key &= 0xFF
    return checksum & 0xFF


def decode_message(message):
    """
    Decodes a 4-byte message and extracts sensor data if the checksum is valid.

    Args:
        message (list of int): The 4-byte message to decode.

    Returns:
        dict or None: Decoded sensor data if valid, else None.
    """
    if len(message) != 4 or all(b == 0 for b in message[:4]):
        return None

    gen = 0x98
    key = 0xF1
    checksum = lfsr_digest8(message[:3], gen, key)

    if checksum != message[3]:
        return None

    sensor_id = message[0]
    battery_ok = (message[1] & 0x80) >> 7
    button = (message[1] & 0x40) >> 6
    channel = ((message[1] & 0x30) >> 4) + 1
    temp_raw = ((message[1] & 0x0F) << 8) | message[2]
    temp_c = temp_raw * 0.1

    return {
        "sensor_id": sensor_id,
        "battery_ok": battery_ok,
        "button": button,
        "channel": channel,
        "temperature_C": temp_c,
    }


# Modulation parameters
SHORT_WIDTH = 2000  # microseconds
LONG_WIDTH = 4000  # microseconds
RESET_LIMIT = 10000  # microseconds


def duration_to_bit(duration):
    """
    Maps signal duration to a binary bit based on modulation parameters.

    Args:
        duration (float): Duration of the low signal in microseconds.

    Returns:
        str or None: '0' or '1' if duration matches SHORT_WIDTH or LONG_WIDTH respectively, else None.
    """
    if abs(duration - SHORT_WIDTH) < 500:
        return "0"
    elif abs(duration - LONG_WIDTH) < 500:
        return "1"
    return None


def process_buffer(buffer, sample_rate):
    """
    Processes the sample buffer to extract valid frames based on signal durations.

    Args:
        buffer (np.ndarray): The complex sample buffer.
        sample_rate (int): The sample rate in samples per second.

    Returns:
        tuple: A tuple containing a list of potential frames and the updated buffer.
    """
    if not buffer.size:
        return [], buffer

    magnitude = np.abs(buffer)
    threshold = np.mean(magnitude) + np.std(magnitude)
    binary_signal = magnitude > threshold

    diff_signal = np.diff(binary_signal.astype(int))
    rising_edges = np.where(diff_signal == 1)[0]
    falling_edges = np.where(diff_signal == -1)[0]

    if not rising_edges.size or not falling_edges.size:
        return [], buffer

    if falling_edges[0] < rising_edges[0]:
        falling_edges = falling_edges[1:]
    if rising_edges.size > falling_edges.size:
        rising_edges = rising_edges[:-1]

    low_durations = (rising_edges[1:] - falling_edges[:-1]) / sample_rate * 1e6

    frames = []
    current_bits = []
    consumed_samples = 0

    for i, duration in enumerate(low_durations):
        if duration > RESET_LIMIT:
            if current_bits:
                frames.append("".join(current_bits))
                current_bits = []
                consumed_samples = falling_edges[i]
        else:
            bit = duration_to_bit(duration)
            if bit:
                current_bits.append(bit)

    if current_bits:
        frames.append("".join(current_bits))
        consumed_samples = rising_edges[-1] if rising_edges.size else consumed_samples

    potential_frames = [
        [
            int(frame_bits[j : j + 8], 2)
            for j in range(0, len(frame_bits), 8)
            if len(frame_bits[j : j + 8]) == 8
        ]
        for frame_bits in frames
    ]

    potential_frames = [frame for frame in potential_frames if frame]

    if consumed_samples > 0:
        buffer = buffer[consumed_samples:]
    else:
        max_bits = 200
        max_samples = int((max_bits * LONG_WIDTH * 1e-6) * sample_rate)
        buffer = (
            buffer[-max_samples:]
            if buffer.size > max_samples
            else np.array([], dtype=complex)
        )

    return potential_frames, buffer


def main():
    try:
        sdr = adi.Pluto("ip:192.168.2.1")
    except Exception as e:
        print(f"Failed to connect to PlutoSDR: {e}")
        sys.exit(1)

    sample_rate = 1_000_000  # 1 MSPS
    sdr.sample_rate = sample_rate
    sdr.rx_lo = 433_920_000  # 433.92 MHz
    sdr.rx_rf_bandwidth = 1_000_000  # 1 MHz
    sdr.gain_control_mode_chan0 = "slow_attack"
    sdr.rx_enabled_channels = [0]
    sdr.rx_buffer_size = 1024 * 1024

    buffer = np.array([], dtype=complex)
    frame_cache = deque(maxlen=1000)

    print("Starting continuous capture and frame detection...")

    try:
        while True:
            try:
                data = sdr.rx()
                if data is not None and data.size > 0:
                    buffer = np.concatenate((buffer, data))
                else:
                    time.sleep(0.1)
                    continue

                potential_frames, buffer = process_buffer(buffer, sample_rate)

                for frame in potential_frames:
                    frame_tuple = tuple(frame)

                    decoded_messages = []
                    unique_messages = set()

                    for i in range(0, len(frame) - 3, 4):
                        message = frame[i : i + 4]
                        message_tuple = tuple(message)
                        if message_tuple in unique_messages:
                            continue
                        unique_messages.add(message_tuple)

                        decoded = decode_message(message)
                        if decoded:
                            decoded_messages.append(decoded)

                    if decoded_messages:
                        if frame_tuple in frame_cache:
                            continue
                        frame_cache.append(frame_tuple)

                        print(f"\nValid Frame {len(frame_cache)}: {frame}")
                        for idx, data in enumerate(decoded_messages, 1):
                            print(f"  Message {idx}:")
                            print(f"    Sensor ID        : {data['sensor_id']}")
                            print(
                                f"    Battery OK       : {'Yes' if data['battery_ok'] else 'No'}"
                            )
                            print(
                                f"    Button Pressed   : {'Yes' if data['button'] else 'No'}"
                            )
                            print(f"    Channel          : {data['channel']}")
                            print(
                                f"    Temperature      : {data['temperature_C']:.1f} °C"
                            )
            except KeyboardInterrupt:
                print("\nInterrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"Error during reception: {e}")
                time.sleep(1)
    finally:
        del sdr
        print("SDR device deleted.")


if __name__ == "__main__":
    main()
