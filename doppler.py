# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 13:25:15 2014

@author: mut
"""

import scipy
import scipy.fftpack
import scipy.io.wavfile
import numpy
import matplotlib.pyplot as mpl
import sys
import dbv

#load input data from wav file
if (len(sys.argv) < 2):
    wav_file = 'exit17.wav' # if no wav specified
else:
    wav_file = sys.argv[1]  # get wav filename from command line
(Fs, Y) = scipy.io.wavfile.read(wav_file)   # Fs = sample rate, Y = sample data

#constants
c = 3e8 # c = 299792458 (speed of light)

#RADAR parameters
Tp = 0.250  # pulse time (seconds)
fc = 2.59e9 # center frequency (Hz)
ns = int(Fs * Tp)    # number of samples per pulse

# invert samples and normalize to 16 bit
s = -Y[:, 1] / 32768.0  # Y[,1] is actual data channel
np = int(round(s.size / ns))   # number of pulses

# reshape for doppler vs time
# theres probably a built in way to do this without loop
sif = numpy.empty((np, ns))    # shaped empty numpy ndarray
for i in xrange(np):
    for j in xrange(ns):
        sif[i, j] = s[i*ns+j]

# subtract average DC from data
sif = sif - numpy.mean(s)

# ifft of each pulse of samples
zpad = 8*ns/2
v_ifft = scipy.fftpack.ifft(sif, zpad, 1)

# log scale of data (dB)
v = dbv.dbv(v_ifft[:, 0:zpad/2])
print v[0,0]
print v[5,5]
mmax = numpy.amax(numpy.amax(v))

# velocity range
delta_f = numpy.linspace(0, Fs/2, zpad/2)
wl = c / fc
velocity = delta_f * wl / 2

# time range
time = numpy.linspace(1, Tp*np, np)

# plot velocity vs time
# need to figure out how to plot properly
im = mpl.imshow(v - mmax, extent=(0,5,0,5))
mpl.show()