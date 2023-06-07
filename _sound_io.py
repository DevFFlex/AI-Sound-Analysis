import pyaudio
import wave
import numpy
import matplotlib.pyplot as plt 
import io
import numpy as np
from _data_manager import DataManager
import os
import sounddevice as sd
import soundfile as sf
import time

class SoundOption:
    def __init__(self) -> None:
        self.FREQ_MIN   = None
        self.FREQ_MAX   = None
        self.FREQ_SPACE = None
        self.AMPLITUDE_FILTER = None

class SoundIO:
    def __init__(self):

        self.soundOption = SoundOption()
        self.soundOption.FREQ_MIN = 0
        self.soundOption.FREQ_MAX = 3000
        self.soundOption.FREQ_SPACE = 50
        self.soundOption.AMPLITUDE_FILTER = 0

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 2

        self.FRAME_LIST = []

        self.SOUND_PATH = os.path.join(os.getcwd(),"audio_output/output.wav")
        if not os.path.isdir(os.path.join(os.getcwd(),"audio_output")):
            os.mkdir("audio_output")

        print(type(self.list_audio_devices()))
        print(self.list_audio_devices())

        for x in self.list_audio_devices():
            print(f"index {x['index']}\tname  {x['name']}")

    def list_audio_devices(self):
        audio = pyaudio.PyAudio()
        device_info = audio.get_host_api_info_by_index(0)
        device_count = device_info.get('deviceCount')
        
        devices = []
        for i in range(device_count):
            device = audio.get_device_info_by_host_api_device_index(0, i)
            devices.append(device)
        
        return devices
    
    def playSound(self):
        data, fs = sf.read(self.SOUND_PATH, dtype='float32')
        sd.play(data, fs)
        sd.wait()
    
    def getOutputDuratoion(self):
        wav_file = wave.open(self.SOUND_PATH, 'r')
        signal = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)

        signal_length = len(signal)
        sample_rate = wav_file.getframerate()
        duration = signal_length / sample_rate

        return duration

    def __plotGraph(self,x,y,x_name,y_name):
        plt.plot(y, x)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.show()


    def __Record(self,device_index = 1):
        
        self.AUDIO = pyaudio.PyAudio()

        stream = self.AUDIO.open(format=self.FORMAT, channels=self.CHANNELS,
                            rate=self.RATE, input=True,
                            frames_per_buffer=self.CHUNK,
                            input_device_index=device_index)

        print("Recording...")

        self.FRAME_LIST.clear()
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            self.FRAME_LIST.append(data)

        print("Finished recording.")

        stream.stop_stream()
        stream.close()
        self.AUDIO.terminate()

        self.__Save()


    def __Save(self):
        wave_file = wave.open(self.SOUND_PATH, 'wb')
        wave_file.setnchannels(self.CHANNELS)
        wave_file.setsampwidth(self.AUDIO.get_sample_size(self.FORMAT))
        wave_file.setframerate(self.RATE)
        wave_file.writeframes(b''.join(self.FRAME_LIST))
        wave_file.close()
        
    def __GetFFT(self,frames = None):
        if frames == None:
            frames = self.FRAME_LIST

        audio_data = io.BytesIO()
        for frame in frames:
            audio_data.write(frame)
        audio_data.seek(0)

        signal = np.frombuffer(audio_data.read(), dtype=np.int16)
        signal = signal / 2.0**15
        signal = signal[0:]

        amplitude = np.fft.fft(signal)
        frequency = np.fft.fftfreq(len(signal), d=1/self.RATE)

        # fft_spectrum_abs = np.abs(amplitude)

        return (frequency,amplitude)

    def __Filter(self,frequency,amplitude):

        FREQ_MIN = self.soundOption.FREQ_MIN
        FREQ_MAX = self.soundOption.FREQ_MAX
        FREQ_SPACE = self.soundOption.FREQ_SPACE

        amplitude_abs = np.abs(amplitude)

        frequency_list = []
        amplitude_list = []

        for i,f in enumerate(amplitude_abs):
            if i % FREQ_SPACE == FREQ_MIN and np.round(frequency[i]) < FREQ_MAX and i != 0:
                freq = np.round(frequency[i],1)
                amp = np.round(f)
                frequency_list.append(freq)
                amplitude_list.append(amp)


                # print(f"freq {freq}\tampli = {amp}")

            if i > 40000:
                break

        frequency_np = np.array(frequency_list)
        amplitude_np = np.array(amplitude_list)

        # return [frequency_np,amplitude_np]
        return [frequency_np,amplitude_np]


    
    def GetFFTWithSoundFile(self,filename = None):
        if filename == None:
            return
        
        wav_file = wave.open(filename, 'r')

        signal = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)

        fft_spectrum = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal), d=1/wav_file.getframerate())
        # frequencies = frequencies_float.astype(int)

        wav_file.close()

        return [frequencies,fft_spectrum]

    def genSound(self,freq,durationIn):
        p = pyaudio.PyAudio()

        volume = 0.5  # range [0.0, 1.0]
        fs = 44100  # sampling rate, Hz, must be integer
        duration = durationIn  # in seconds, may be float
        f = freq  # sine frequency, Hz, may be float

        samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

        output_bytes = (volume * samples).tobytes()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=fs,
                        output=True)

        start_time = time.time()
        stream.write(output_bytes)
        print("Played sound for {:.2f} seconds".format(time.time() - start_time))

        stream.stop_stream()
        stream.close()

        p.terminate()

        print("finish")

    def process(self,device_index = 1):
        self.__Record(device_index=device_index)
        frequency_numpy,amplitude_numpy = self.__GetFFT()

        frequencyOutput,amplitudeOutput = self.__Filter(frequency_numpy,amplitude_numpy)


        return [frequencyOutput,amplitudeOutput]