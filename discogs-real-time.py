import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy

import SendData
import numpy as np
import config
import AudioPlayer
import argparse

from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
from labels.discogs400 import labels

from pydub import AudioSegment


send_data = SendData.SendData()


buffer = []

def get_genra(data):
    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb",
                                                     output="PartitionedCall:1")
    embeddings = embedding_model(data)
    model = TensorflowPredict2D(graphFilename="models/genre_discogs400-discogs-effnet-1.pb",
                                input="serving_default_model_Placeholder", output="PartitionedCall:0")
    predictions = model(embeddings)

    combined_list = [(num, label) for num, label in zip(predictions[0], labels)]
    sorted_combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)

    return sorted_combined_list

def audio_data_cb(audio_buffer: np.ndarray, frequency_bins: np.ndarray, frequency: np.ndarray) -> None:
    global buffer
    buffer.append(audio_buffer)

    if(len(buffer) > 200):
        flat_buffer = numpy.array(buffer).flatten()
        # audio_data_bytes = flat_buffer.astype(np.int16).tobytes()
        audio_segment = AudioSegment(data=flat_buffer, frame_rate=48000, sample_width=2, channels=1)
        resampled_audio = audio_segment.set_frame_rate(16000)
        resampled_audio_data = np.array(resampled_audio.get_array_of_samples())

        genra = get_genra(resampled_audio_data)
        print(genra)
        # print(len(flat_buffer))
        # print(len(resampled_audio_data))

        buffer = []



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("song")
    args = parser.parse_args()

    #initialize the audio player
    player = AudioPlayer.AudioPlayer(args.song)

    #start the audio player with callback function as input
    player.open_stream(audio_data_cb)