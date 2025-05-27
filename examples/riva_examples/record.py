import pyaudio
import wave

# Paramètres d'enregistrement
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10
OUTPUT_FILENAME = "output.wav"

audio = pyaudio.PyAudio()

# Démarrer l'enregistrement
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("Enregistrement en cours...")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

# Arrêter l'enregistrement
print("Enregistrement terminé.")

stream.stop_stream()
stream.close()
audio.terminate()

# Sauvegarder le fichier
wf = wave.open(OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
