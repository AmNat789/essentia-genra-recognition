import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
from labels.discogs400 import labels

if __name__ == "__main__":
    audio = MonoLoader(filename="songs/Turkish March - Mozart .wav", sampleRate=16000, resampleQuality=4)()
    # print(numpy.array(audio)[0: 100])
    print(numpy.array(audio).shape)

    # embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    # embeddings = embedding_model(audio)
    #
    # model = TensorflowPredict2D(graphFilename="models/genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
    # predictions = model(embeddings)
    #
    # combined_list = [(num, label) for num, label in zip(predictions[0], labels)]
    # sorted_combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)
    # combined = [f"{label}:{num}" for num, label in sorted_combined_list]
    #
    # print(combined)