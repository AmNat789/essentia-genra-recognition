from essentia.standard import MonoLoader, TensorflowPredictMusiCNN
import numpy as np

if __name__ == "__main__":
    audio = MonoLoader(filename="songs/Turkish March - Mozart .wav", sampleRate=16000, resampleQuality=4)()
    model = TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
    embeddings = model(audio)

    prediction_labels = [
        "rock", "pop", "alternative", "indie", "electronic", "female vocalists",
        "dance", "00s", "alternative rock", "jazz", "beautiful", "metal",
        "chillout", "male vocalists", "classic rock", "soul", "indie rock",
        "Mellow", "electronica", "80s", "folk", "90s", "chill", "instrumental",
        "punk", "oldies", "blues", "hard rock", "ambient", "acoustic",
        "experimental", "female vocalist", "guitar", "Hip-Hop", "70s",
        "party", "country", "easy listening", "sexy", "catchy", "funk",
        "electro", "heavy metal", "Progressive rock", "60s", "rnb",
        "indie pop", "sad", "House", "happy"
    ]

    combined_list = [(num, label) for num, label in zip(embeddings[0], prediction_labels)]
    sorted_combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)
    combined = [f"{label}:{num}" for num, label in sorted_combined_list]

    print(combined)