import numpy as np
import matplotlib.pyplot as plt

# 时长为1秒
t = 1
# 采样率为60hz
fs = 60
t_split = np.arange(0, t * fs)

# 1hz与25hz叠加的正弦信号
x_1hz = t_split * 1 * np.pi * 2 / fs
x_25hz = t_split * 25 * np.pi * 2 / fs
signal_sin_1hz = np.sin(x_1hz)
signal_sin_25hz = np.sin(x_25hz)

signal_sin = signal_sin_1hz + 0.25 * signal_sin_25hz

# 滤波器参数
pass_band_edge = 10 # Hz
stop_band_edge = 22 # Hz
fc = (pass_band_edge + stop_band_edge) / 2 # Hz
M = 17 # 窗内项数17

# TODO: 补全这部分代码
# 通带边缘频率为10Hz，
# 阻带边缘频率为22Hz，
# 阻带衰减为44dB，窗内项数为13的汉宁窗函数
# 构建低通滤波器
# 函数需要返回滤波后的信号
def filter_fir(input, clip=1):
    def hi(n):
        # 理想数字滤波器 h(n) = sin(w_c*n) / (n*pi)
        a = 2 * fc / fs
        return a * np.sinc(a * n)
    # 汉宁窗
    ha = np.hanning(M)
    # 单位冲激响应
    h = [hi(n - (M-1)/2) * ha[n] for n in range(M)]
    # 卷积，计算输出 y = x * h
    L = len(input)
    y = np.zeros(L+M)
    for n in range(len(y)):
        for k in range(max(0, n-L+1), min(M, n+1)):
            y[n] += h[k] * input[n-k]
    if clip:
        y = y[:L]
    return y

# TODO: 首先正向对信号滤波(此时输出信号有一定相移)
# 将输出信号反向，再次用该滤波器进行滤波
# 再将输出信号反向
# 函数需要返回零相位滤波后的信号
def filter_zero_phase(input):
    y1 = filter_fir(input, clip=0)
    x1 = np.flip(y1)
    y2 = filter_fir(x1, clip=0)
    return np.flip(y2)[M:M+len(input)]

if __name__ == "__main__":
    delay_filtered_signal = filter_fir(signal_sin)
    zerophase_filtered_signal = filter_zero_phase(signal_sin)

    plt.plot(t_split, signal_sin, label='origin')
    plt.plot(t_split, delay_filtered_signal, label='fir')
    plt.plot(t_split, zerophase_filtered_signal, label='zero phase')
    plt.legend()
    plt.show()
