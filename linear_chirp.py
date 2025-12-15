import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, convolve

# working with [us,MHz]
res = 1e4
T = 10
t = np.arange(0, T + 1/res, 1/res)
B = 2
f_0 = 1
phi = np.pi/1.7

s_linchirp = chirp(t, f0=f_0, t1=T, f1=f_0 + B, method='linear')
matched_filter = np.conj(s_linchirp[::-1])
# matched_filter = np.conj(s_linchirp)

SNR_dB = -13
signal_power = np.mean(np.abs(s_linchirp)**2)
noise_power = signal_power / (10**(SNR_dB/10))
noise = np.sqrt(noise_power) * np.random.randn(len(s_linchirp))
rx = (s_linchirp + noise) * np.exp(1j*phi)

y = convolve(rx, matched_filter, mode='same')
# y = np.correlate(rx, matched_filter, mode='same')

plt.subplot(2,1,1)
plt.plot(t, rx)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Received Signal with Noise')

plt.subplot(2,1,2)
plt.plot(t, np.abs(y))
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Pulse Compressed Output')

plt.show()