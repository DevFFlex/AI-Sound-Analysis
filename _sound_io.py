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

class SoundIO:
    def __init__(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 2

        self.SOUND_PATH = os.path.join(os.getcwd(),"tone/output.wav")

        if not os.path.isdir(os.path.join(os.getcwd(),"tone")):
            os.mkdir("tone")

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


    def record(self,device_index = 1):
        audio = pyaudio.PyAudio()

        # for element in self.list_audio_devices():
        #     print(type(element))
    
        stream = audio.open(format=self.FORMAT, channels=self.CHANNELS,
                            rate=self.RATE, input=True,
                            frames_per_buffer=self.CHUNK,
                            input_device_index=device_index)

        print("Recording...")

        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        print("Finished recording.")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        wave_file = wave.open(self.SOUND_PATH, 'wb')
        wave_file.setnchannels(self.CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(self.FORMAT))
        wave_file.setframerate(self.RATE)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()

        return frames

    
    def GetFFT(self,frames):
        audio_data = io.BytesIO()
        for frame in frames:
            audio_data.write(frame)
        audio_data.seek(0)

        signal = np.frombuffer(audio_data.read(), dtype=np.int16)
        
        signal = signal / 2.0**15

        signal = signal[0:]

        fft_spectrum = np.fft.fft(signal)

        frequencies = np.fft.fftfreq(len(signal), d=1/self.RATE)


        fft_spectrum_abs = np.abs(fft_spectrum)

        freqlist = []
        fft_spectrum_list = []

        for i,f in enumerate(fft_spectrum_abs):
            if i % 50 == 0 and np.round(frequencies[i]) < 3000 and i != 0:
                freq = np.round(frequencies[i],1)
                amp = np.round(f)
                # print('frequency = {} Hz with amplitude {} '.format(freq,amp))
                freqlist.append(freq)
                fft_spectrum_list.append(amp)

            if i > 40000:
                break

        return [freqlist,fft_spectrum_list]
    

    def GetFFTWithSoundFile(self,filename = None,display = False):
        if filename == None:
            return
        
        wav_file = wave.open(filename, 'r')

        signal = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)

        # print(wav_file.getsampwidth())

        fft_result = np.fft.fft(signal)

        frequencies_float = np.fft.fftfreq(len(signal), d=1/wav_file.getframerate())
        frequencies_int = frequencies_float.astype(int)

        # print(frequencies_int.dtype)
        if display:
            plt.plot(frequencies_int, np.abs(fft_result))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.show()

        wav_file.close()

        return [frequencies_float,fft_result]


    def process(self,device_index = 1):
        frames = self.record(device_index=device_index)
        freq,amp = self.GetFFT(frames)

        freq_numpy = numpy.array(freq)
        ampl_numpy = numpy.array(amp)

        


        return [freq_numpy,ampl_numpy]
    


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