import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

def generate_harmonic_signal_with_noise(t, frequency, amplitude, phase, noise_level):
    signal = amplitude * np.cos(2 * np.pi * frequency * t + phase)
    noise = np.random.normal(0, noise_level, len(t))
    return signal + noise

def analyze_signal(signal, sampling_rate):
    spectrum = fft(signal)
    amplitudes = np.abs(spectrum)
    phases = np.angle(spectrum)

    main_frequency_index = np.argmax(amplitudes)
    main_frequency = main_frequency_index * (sampling_rate / len(signal))
    main_amplitude = amplitudes[main_frequency_index]
    main_phase = phases[main_frequency_index]

    return main_amplitude, main_frequency, main_phase

def main():
    # 采样率和时间数组的长度
    N = 10  # 2的N次方
    sampling_rate = 2**N  # 采样率为2的10次方

    t = np.arange(0, 1, 1 / sampling_rate)[:2**N]  # 时间数组长度为2的N次方

    frequencies_list = [[10, 20], [30, 40]]
    amplitudes_list = [[1, 0.8], [0.5, 0.7]]
    phases_list = [[np.pi / 4, np.pi / 2], [np.pi, np.pi / 3]]

    plt.figure(figsize=(12, 8))

    E1_matrix = np.zeros((len(frequencies_list), 1))
    Phase_matrix = np.zeros((len(frequencies_list), 1))

    for i in range(len(frequencies_list)):
        frequencies = frequencies_list[i]
        amplitudes = amplitudes_list[i]
        phases = phases_list[i]

        example_signal = generate_harmonic_signal_with_noise(t, frequencies[0], amplitudes[0], phases[0], noise_level=0.1) + \
                         generate_harmonic_signal_with_noise(t, frequencies[1], amplitudes[1], phases[1], noise_level=0.1)

        E1, _, main_phase = analyze_signal(example_signal, sampling_rate)
        E1_matrix[i, 0] = E1  # 将主振幅添加到矩阵中
        Phase_matrix[i, 0] = main_phase  # 将主相位添加到矩阵中

        print(f"Example {i + 1}: E1 = {E1:.2f}, Main Phase = {main_phase:.2f}")

        plt.subplot(len(frequencies_list), 2, 2 * i + 1)
        plt.plot(range(len(example_signal)), example_signal)
        plt.title(f'Example Signal {i + 1}')
        plt.xlabel('Sample Number')
        plt.ylabel('Amplitude')

        plt.subplot(len(frequencies_list), 2, 2 * i + 2)
        frequencies = np.fft.fftfreq(len(example_signal), 1 / sampling_rate)
        plt.plot(frequencies[:len(frequencies) // 2], np.abs(np.fft.fft(example_signal))[:len(frequencies) // 2])
        plt.title(f'Frequency Spectrum {i + 1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

    # 输出矩阵 S
    print("Matrix S:")
    print(E1_matrix)

    # 输出矩阵 T
    print("Matrix T:")
    print(Phase_matrix)

    for i in range(len(frequencies_list)):
        print(f"Example {i + 1}: E1 = {E1_matrix[i, 0]:.2f}, Main Phase = {Phase_matrix[i, 0]:.2f}")

    return E1_matrix,Phase_matrix

if __name__ == "__main__":
    main()


