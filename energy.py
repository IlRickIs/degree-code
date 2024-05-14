import numpy as np
import librosa
import matplotlib.pyplot as plt

# Caricamento del file audio
filename = 'D:/laurea-codici/EMOVO/f2/dis-f2-b1.wav'  
y, sr = librosa.load(filename, sr=None)  # Carica il file audio con la frequenza di campionamento originale

# Calcolo della FFT del segnale
Y = np.fft.fft(y)

# Calcolo del modulo quadrato (densità spettrale di potenza)
P = np.abs(Y)**2

# Energia totale del segnale (dividendo per il numero di campioni per normalizzare)
E = np.sum(P) / len(P)

print(f"Energia totale del segnale: {E}")

# Plot del segnale e del suo spettro
plt.figure(figsize=(12, 6))

# Segnale audio nel dominio del tempo
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, len(y) / sr, len(y)), y)
plt.title('Segnale Audio')
plt.xlabel('Tempo [s]')
plt.ylabel('Ampiezza')

# Spettro del segnale audio
plt.subplot(2, 1, 2)
freqs = np.fft.fftfreq(len(P), 1 / sr)
plt.plot(freqs[:len(P) // 2], P[:len(P) // 2])
plt.title('Spettro del Segnale Audio')
plt.xlabel('Frequenza [Hz]')
plt.ylabel('Densità Spettrale di Potenza')

plt.tight_layout()
plt.show()
