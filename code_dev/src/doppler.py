#! /usr/bin/env python
import sys

import numpy as np
import scipy as sp
import scipy.fftpack
import scipy.io.wavfile

import matplotlib.pyplot as plt
import scipy.signal as sps

import dbv

# load input data from wav file
if (len(sys.argv) < 2):
    wav_file = 'C:\Users\mut\Desktop\doppler_test.wav'
#    wav_file = 'exit17.wav' # if no wav specified
else:
    wav_file = sys.argv[1]  # get wav filename from command line

(fs, y) = scipy.io.wavfile.read(wav_file)   # fs = sample rate, y = sample data
y_t = np.arange(0, len(y)) * (1. / fs)
# constants
c = 3e8  # c = 299792458 (m/s)

# RADAR parameters
tp = 0.250          # pulse time (s)
fc = 2.59e9         # center frequency (Hz)
ns = int(fs * tp)   # number of samples per pulse

# invert samples and normalize to 16 bit
s = -y[:, 1] / 32768.0  # y[,1] is actual data channel
npulse = int(round(s.size / ns))   # number of pulses

s = sps.medfilt(s, 19)
#s = sps.savgol_filter(s, 19, 2)
#Savitzky-Golay library function sounds nice but is not available
#  in the stone age version of scipy I am using
#  http://docs.scipy.org/doc/scipy/reference/generated/\
#        scipy.signal.savgol_filter.html

# reshape for doppler vs time
sif = np.reshape(s[0:npulse * ns], (npulse, ns))
tif = np.reshape(y_t[0:npulse * ns], (npulse, ns))

print sif.shape, tif.shape

# subtract average DC from data
sif = sif - np.mean(s)

# ifft of each pulse of samples
zpad = 8 * ns / 2
v_ifft = scipy.fftpack.ifft(sif, zpad, 1)
v = dbv.dbv(v_ifft[:, 0:zpad / 2])    # log scale of data (dB)
mmax = np.amax(np.amax(v))

# velocity range
delta_f = np.linspace(0, fs / 2.0, zpad / 2)
wl = c / fc
velocity = delta_f * wl / 2.0

# time range
time = np.linspace(1, tp * npulse, npulse)


# Plotting
fig = plt.figure()
ax1 = fig.add_subplot(211)
#ax1.plot(x, y)
ax2 = fig.add_subplot(212)
#ax2.plot(y, x)
#plt.show()

# plot velocity vs time
# need to finish plotting properly
#im = mpl.imshow(v - mmax, extent=(0,5,0,5))
#subset = sif[0:10, :]
window_start = 3
window_width = 8  # use 8 ``pulse-widths'' for 2 second window
#for i in range(window_start, window_start + window_width):
#    ax1.plot(tif[i, :], sif[i, :], label=i)
#    print i, tif[i, :].shape, tif[i, :].ndim
# convert in two steps
# (a) extract the selected set of data
sub_s = sif[window_start:window_start + window_width, :]
sub_t = tif[window_start:window_start + window_width, :]
# (b) convert the 2d array back into a 1d array (like the original input)
sub_s = np.reshape(sub_s, (1, -1)).ravel()
sub_t = np.reshape(sub_t, (1, -1)).ravel()

ax1.plot(sub_t, sub_s)

# Get frequency spectrum and plot
Fs = 44100.  # TODO: replace hardcoded sample rate
n = len(sub_s)  # length of the signal
k = np.arange(n)
T = n / Fs
frq = k / T  # two sides frequency range
frq = frq[range(n / 2)]  # one side frequency range

Y = sp.fft(sub_s) / n  # fft computing and normalization
Y = Y[range(n / 2)]

ax2.plot(frq, np.abs(Y))
ax2.set_xlabel('Freq (Hz)')
ax2.set_ylabel('|Y(freq)|')
x0, x1 = ax2.get_xlim()
ax2.set_xlim([0, 100])
print n, T, k[:10]
for i in dir(ax2):
    if "max" in i.lower():
        print i
plt.savefig("scratch.pdf")


def windowing_test():
    from numpy.fft import fft, fftshift
    window = np.hamming(51)
    plt.plot(window)
    plt.title("Hamming window")
    plt.ylabel("Amplitude")
    plt.xlabel("Sample")
    plt.show()

    plt.figure()
    A = fft(window, 2048) / 25.5
    mag = np.abs(fftshift(A))
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(mag)
    response = np.clip(response, -100, 100)
    plt.plot(freq, response)
    plt.title("Frequency response of Hamming window")
    plt.ylabel("Magnitude [dB]")
    plt.xlabel("Normalized frequency [cycles per sample]")
    plt.axis('tight')
    plt.show()


def plotSpectrum(y, Fs):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(y)  # length of the signal
    k = np.arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    frq = frq[range(n / 2)]  # one side frequency range

    Y = sp.fft(y) / n  # fft computing and normalization
    Y = Y[range(n / 2)]

    #pylab.plot(frq, abs(Y), 'r')  # plotting the spectrum
    #pylab.xlabel('Freq (Hz)')
    #pylab.ylabel('|Y(freq)|')
