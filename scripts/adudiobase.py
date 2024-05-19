import pyaudio
import numpy as np
from pyAudioAnalysis import ShortTermFeatures

def is_speech_detected(signal, sampling_rate):
    if len(signal) == 0 or sampling_rate <= 0:
        print("Ошибка: Недопустимый сигнал или частота дискретизации")
        return False

    print("Длина сигнала:", len(signal))
    print("Частота дискретизации:", sampling_rate)

    # Извлечение признаков
    window = 2  
    step = 1
    st_feats, feature_names = ShortTermFeatures.feature_extraction(signal, sampling_rate, window, step)
    print("Извлеченные признаки:", st_feats)
    print("Количество признаков:", len(st_feats))

    if len(st_feats) == 0:
        print("Ошибка: Нет извлеченных признаков")
        return False

    # Проверка наличия 'speech' в списке названий признаков
    if 'speech' in feature_names:
        return True
    else:
        return False

def callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    if is_speech_detected(audio_data, 16000):
        print("Обнаружен речевой сигнал")
    else:
        print("Речевой сигнал не обнаружен")
    return (in_data, pyaudio.paContinue)

if __name__ == "__main__":
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, output=True, stream_callback=callback)

    stream.start_stream()

    try:
        while stream.is_active():
            pass
    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    p.terminate()
