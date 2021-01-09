import numpy as np
import numpy.fft as fft
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile
import os

audio_paths = [
    '稻香.m4a',
    '告白气球.m4a',
    '七里香.m4a',
    '青花瓷.m4a',
    '夜曲.m4a',
]

# 单帧时长
SECONEDS_PER_FRAME = 1 # s

MAX_TIME_DURATION = 240000
MAX_FREQ = 3400
SAMPLE_RATE = 8000
DURATION_PER_FRAME = SECONEDS_PER_FRAME * 10000
N = MAX_TIME_DURATION // DURATION_PER_FRAME + 1
MAX_TIME_DURATION = DURATION_PER_FRAME * N
save_dir = f"frame_{SECONEDS_PER_FRAME}s"
os.makedirs(save_dir, exist_ok=True)

def split(audio, n):
    return np.split(audio, n)


def concat(audios):
    return np.concatenate(audios)


def myfft(audio):
    sp = fft.fft(audio)
    freqs = fft.fftfreq(sp.shape[-1], d=1/SAMPLE_RATE)
    clamped_sp = np.where(np.abs(freqs) < MAX_FREQ, sp, np.zeros(sp.shape, dtype=complex))
    return sp, freqs, clamped_sp


def preprocess(audio_file):
    """
    预处理: 
        采样频率8000
        过滤掉3400以上的频率
    """
    _y, sr = librosa.load(path=audio_file, sr=SAMPLE_RATE)
    time = _y.shape[0]

    if _y.shape[0] < MAX_TIME_DURATION:
        y = np.concatenate((_y, np.zeros(MAX_TIME_DURATION-_y.shape[0])))
    else:
        y = _y[:MAX_TIME_DURATION]
    plt.figure()
    librosa.display.waveplot(_y, sr=sr, linewidth=0.5, color='blue')
    plt.savefig(os.path.join(save_dir, audio_file.split('.')[0] + "_time.png"))

    sp, freqs, clamped_sp = myfft(y)
    plt.figure()
    plt.plot(freqs, np.abs(sp), linewidth=0.6, color='red')
    plt.savefig(os.path.join(save_dir, audio_file.split('.')[0] + "_freq_init.pdf"))

    plt.figure()
    plt.plot(freqs, np.abs(clamped_sp), linewidth=0.6, color='red')
    plt.savefig(os.path.join(save_dir, audio_file.split('.')[0] + "_freq_clamped.pdf"))

    plt.close()

    # 分帧处理
    ys = split(y, N)
    sp_split = []
    for my in ys:
        sp_split.append(myfft(my)[-1])

    return clamped_sp, y, sp_split, time

def save_audio(audio, name):
    plt.figure()
    librosa.display.waveplot(np.real(audio), sr=SAMPLE_RATE, linewidth=0.5, color='blue')
    plt.savefig(os.path.join(save_dir, name + "_time.png"))
    soundfile.write(os.path.join(save_dir, name + ".wav"), np.real(audio), samplerate=SAMPLE_RATE)
    plt.close()


def encode(sps):
    """
    调制过程，4段音频调制到同一个频谱
    保持频谱共轭对称的性质
    """
    number = len(sps)
    total_sp = number*DURATION_PER_FRAME
    half_sample_rate = DURATION_PER_FRAME // 2
    joined_frames = []
    for j in range(N):
        joined_sp = np.zeros(total_sp, dtype=complex)
        for i in range(number):
            joined_sp[i*half_sample_rate:(i+1)*half_sample_rate] = sps[i][j][0:half_sample_rate]
            joined_sp[(total_sp-(i+1)*half_sample_rate):(total_sp-i*half_sample_rate)] = sps[i][j][DURATION_PER_FRAME-half_sample_rate:DURATION_PER_FRAME]
        
        freqs = fft.fftfreq(joined_sp.shape[-1], d=1/(SAMPLE_RATE*number))
        plt.figure()
        plt.plot(freqs, np.abs(joined_sp), linewidth=0.6, color='red')
        plt.savefig(os.path.join(save_dir, f"joined_freq_frame_{j}.pdf"))
        plt.close()

        joined_audio = fft.ifft(joined_sp)
        joined_frames.append(joined_audio)
    save_audio(concat(joined_frames), 'joined')


def decode(number, times):
    y, sr = librosa.load(path=os.path.join(save_dir, "joined.wav"), sr=SAMPLE_RATE)
    ys = split(y, N)
    sp_split = []
    for _y in ys:
        sp_split.append(myfft(_y)[0])
    total_sp = number*DURATION_PER_FRAME
    half_sample_rate = DURATION_PER_FRAME // 2
    recover_audios = []
    for i in range(number):
        audio_frame = []
        sp = np.zeros(DURATION_PER_FRAME, dtype=complex)
        for j in range(N):
            sp[0:half_sample_rate] = sp_split[j][i*half_sample_rate:(i+1)*half_sample_rate]
            sp[DURATION_PER_FRAME-half_sample_rate:DURATION_PER_FRAME] = sp_split[j][(total_sp-(i+1)*half_sample_rate):(total_sp-i*half_sample_rate)]
            decode_audio = fft.ifft(sp)
            audio_frame.append(decode_audio)
        concat_audio = concat(audio_frame)
        recover_audios.append(concat_audio)
        save_audio(concat_audio[:times[i]], audio_paths[i].split('.')[0] + '_decode')
        
        decode_sp = fft.fft(concat_audio[:times[i]])
        decode_freqs = fft.fftfreq(decode_sp.shape[-1], d=1/SAMPLE_RATE)
        plt.figure()
        plt.plot(decode_freqs, np.abs(decode_sp), linewidth=0.6, color='red')
        plt.savefig(os.path.join(save_dir, f"{audio_paths[i].split('.')[0]}_freq_recover.pdf"))
        plt.close()

    return recover_audios


def plot():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    Ns = [1, 2, 5, 10, 20, 30]
    results = []
    for n in Ns:
        mses = []
        with open(os.path.join(f"frame_{n}s", "mse.log"), 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                mses.append(float(line.split()[-1]))
        results.append(mses)
    #print(results)
    results = np.array(results).T / 10
    plt.figure()
    for i, mse in enumerate(results):
        plt.plot(Ns, mse, label=audio_paths[i])
    plt.xlabel('Frame Duration (s)')
    plt.ylabel('MSE Score')
    plt.legend()
    plt.savefig('mse_summary.pdf')
   

if __name__ == "__main__":
    sps = []
    init_audios = []
    split_sps = []
    times = []
    for audio in audio_paths:
        sp, y, sp_split, time = preprocess(audio)
        sps.append(sp)
        init_audios.append(y)
        split_sps.append(sp_split)
        times.append(time)
    encode(split_sps)
    recover_audios = decode(len(sps), times)
    print(f"MSE Score: ", file=open(os.path.join(save_dir, "mse.log"), 'w'))
    # 计算MSE
    for i, (init_audio, recover_audio) in enumerate(zip(init_audios, recover_audios)):
        print(f"{audio_paths[i]}: ", np.linalg.norm(init_audio - recover_audio, ord=2).mean(), file=open(os.path.join(save_dir, "mse.log"), 'a'))
    # plot()
