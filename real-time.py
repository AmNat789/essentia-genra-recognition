import json

from essentia.streaming import (
    VectorInput,
    FrameCutter,
    TensorflowInputMusiCNN,
    VectorRealToTensor,
    TensorToPool,
    TensorflowPredict,
    PoolToTensor,
    TensorToVectorReal
)
from essentia import Pool, run, reset
import numpy as np
from scipy.special import softmax
import soundcard as sc

with open('models/msd-musicnn-1.json', 'r') as json_file:
    metadata = json.load(json_file)

model_file = 'models/msd-musicnn-1.pb'
input_layer = metadata['schema']['inputs'][0]['name']
output_layer = metadata['schema']['outputs'][0]['name']
classes = metadata['classes']
n_classes = len(classes)

# Analysis parameters.
sample_rate = 16000
frame_size = 512
hop_size = 256
n_bands = 96
patch_size = 64
display_size = 10

buffer_size = patch_size * hop_size

buffer = np.zeros(buffer_size, dtype='float32')
vimp = VectorInput(buffer)
fc = FrameCutter(frameSize=frame_size, hopSize=hop_size)
tim = TensorflowInputMusiCNN()
vtt = VectorRealToTensor(shape=[1, 1, patch_size, n_bands],
                         lastPatchMode='discard')
ttp = TensorToPool(namespace=input_layer)
tfp = TensorflowPredict(graphFilename=model_file,
                        inputs=[input_layer],
                        outputs=[output_layer])
ptt = PoolToTensor(namespace=output_layer)
ttv = TensorToVectorReal()
pool = Pool()

vimp.data >> fc.signal
fc.frame >> tim.frame
tim.bands >> vtt.frame
tim.bands >> (pool, 'melbands')
vtt.tensor >> ttp.tensor
ttp.pool >> tfp.poolIn
tfp.poolOut >> ptt.pool
ptt.tensor >> ttv.tensor
ttv.frame >> (pool, output_layer)


def callback(data):
    buffer[:] = data.flatten()

    # Generate predictions.
    reset(vimp)
    run(vimp)

    # Update the mel-spectrograms and activations buffers.
    mel_buffer[:] = np.roll(mel_buffer, -patch_size)
    mel_buffer[:, -patch_size:] = pool['melbands'][-patch_size:, :].T

    act_buffer[:] = np.roll(act_buffer, -1)
    act_buffer[:, -1] = softmax(20 * pool[output_layer][-1, :].T)


    output = act_buffer[:, -1]
    genra = metadata["classes"][np.argmax(output)]
    print("Genra :", genra)




mel_buffer = np.zeros([n_bands, patch_size * display_size])
act_buffer = np.zeros([n_classes, display_size])

pool.clear()

# f, ax = plt.subplots(1, 2, figsize=[9.6, 7])
# f.canvas.draw()
#
# ax[0].set_title('Mel Spectrogram')
# img_mel = ax[0].imshow(mel_buffer, aspect='auto',
#                        origin='lower', vmin=0, vmax=6)
# ax[0].set_xticks([])
#
# ax[1].set_title('Activations')
# img_act = ax[1].matshow(act_buffer, aspect='0.5', vmin=0, vmax=1)
# ax[1].set_xticks([])
# ax[1].yaxis.set_ticks_position('right')
# plt.yticks(np.arange(n_classes), classes, fontsize=6)

# Capture and process the speakers loopback.
with sc.all_microphones(include_loopback=True)[0].recorder(samplerate=sample_rate) as mic:
    while True:
        callback(mic.record(numframes=buffer_size).mean(axis=1))

# with sc.all_microphones(include_loopback=True)[0].recorder(samplerate=sample_rate) as mic:
#     callback(mic.record(numframes=buffer_size).mean(axis=1))