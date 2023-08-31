import wave
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
import tkinter as tk
import glob

def read_wav(file, as_float=False):
    sampwidth_types = {
        1: np.uint8,
        2: np.int16,
        4: np.int32
    }
    
    with wave.open(file, 'rb') as wav:
        params = wav.getparams()
        print('params of wav:', params)
        data = wav.readframes(params.nframes)
        if params.sampwidth in sampwidth_types:
            data = np.frombuffer(data, dtype=sampwidth_types[params.sampwidth])
        else:
            raise RuntimeError("Couldn't process file {}: unsupported sample width {}"
                               .format(file, params.sampwidth))
        data = np.reshape(data, (params.nframes, params.nchannels))
        if as_float:
            data = (data - np.mean(data)) / (np.std(data) + 1e-15)
            
    return params.framerate, data


def resample(audio, sample_rate, new_sample_rate):
    duration = audio.shape[1] / float(sample_rate)
    x_old = np.linspace(0, duration, audio.shape[1])
    x_new = np.linspace(0, duration, int(duration*new_sample_rate))
    data = np.array([np.interp(x_new, x_old, channel) for channel in audio])
    return data


class AudioSource:
    def __init__(self, source, channels=2, samplerate=None):
        self.samplerate = samplerate
        samplerate, audio = read_wav(source, as_float=True)
        audio = audio.T
        if audio.shape[0] != channels:
            raise RuntimeError("Audio has unsupported number of channels - {} (expected {})"
                               .format(audio.shape[0], channels))
        if self.samplerate:
            if self.samplerate != samplerate:
                audio = resample(audio, samplerate, self.samplerate)
        else:
            self.samplerate = samplerate
            
        self.audio = audio
        
        
    def duration(self):
        return self.audio.shape[1] / self.samplerate
    
        
    def chunks(self):            
        num_chunks = (self.audio.shape[1] + self.samplerate - 1) // self.samplerate
        
        for i in range(num_chunks):
            chunk = np.zeros((self.audio.shape[0], self.samplerate), dtype=self.audio.dtype)
            start, end = i*self.samplerate, min((i+1)*self.samplerate, self.audio.shape[1]-1)
            chunk[:, :end-start] = self.audio[:, start:end]
            yield chunk
            
            
def inf(input_wav, compiled_model, channels, sample_rate):
    output_tensor = compiled_model.outputs[0]
    infer_request = compiled_model.create_infer_request()
    audio = AudioSource(input_wav, channels=channels, samplerate=sample_rate)
    Y = []
    for chunk in audio.chunks():
        chunk = np.array([chunk])
        output = infer_request.infer({input_tensor_name: chunk})[output_tensor]
        Y.extend(output[0])
    Y = np.array(Y)
    X = np.arange(Y.shape[0])
    fig, axs = plt.subplots(3, 1)
    fig.suptitle(input_wav.split('\\')[-1])
    for i, t in zip(range(3), ['VAD', 'SNR', 'C50']):
        axs[i].plot(X, Y[:,i])
        axs[i].set_title(t)
        
    plt.subplots_adjust(hspace=0.5)
    plt.show()
            
            
if __name__ == '__main__':
    onnx_path = './best.onnx'
    device = 'CPU'
    # input_wav = './test_data/home.wav'
    input_path = './test_data/'
    core = Core()
    model = core.read_model(onnx_path)
    
    input_tensor_name = model.inputs[0].get_any_name()
    batch_size, channels, length = model.inputs[0].shape
    compiled_model = core.compile_model(model, device)
    # inf(input_wav, compiled_model, channels, length)
    
    samples = glob.glob(f"{input_path}*.wav")
    window = tk.Tk()    
    window.geometry("400x300")
    
    listbox = tk.Listbox(window)
    listbox.pack(fill=tk.BOTH, expand=1)    
    
    for name in samples:
        listbox.insert(tk.END, name.split('\\')[-1])

    listbox.bind("<Double-Button-1>", lambda x:\
        inf(samples[listbox.curselection()[0]], compiled_model, channels, length))
    listbox.pack(side=tk.LEFT, fill=tk.BOTH)
    window.mainloop()