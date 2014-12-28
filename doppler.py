# TODO: PLOT DATA/COMPLETE PLOTTING (matplotlib)
import scipy
import scipy.fftpack
import scipy.io.wavfile
import numpy
import matplotlib.pyplot as mpl
import sys
import dbv

# load input data from wav file
if (len(sys.argv) < 2):
    wav_file = 'exit17.wav' # if no wav specified
else:
    wav_file = sys.argv[1]  # get wav filename from command line
    
(fs, y) = scipy.io.wavfile.read(wav_file)   # fs = sample rate, y = sample data

# constants
c = 3e8 # c = 299792458 (m/s)

# RADAR parameters
tp = 0.250  # pulse time (s)
fc = 2.59e9 # center frequency (Hz)
ns = int(fs * tp)    # number of samples per pulse

# invert samples and normalize to 16 bit
s = -y[:, 1] / 32768.0  # y[,1] is actual data channel
np = int(round(s.size / ns))   # number of pulses

# reshape for doppler vs time
sif = numpy.reshape(s[0:np*ns], (np, ns))

# subtract average DC from data
sif = sif - numpy.mean(s)

# ifft of each pulse of samples
zpad = 8*ns/2
v_ifft = scipy.fftpack.ifft(sif, zpad, 1)
v = dbv.dbv(v_ifft[:, 0:zpad/2])    # log scale of data (dB)
mmax = numpy.amax(numpy.amax(v))

# velocity range
delta_f = numpy.linspace(0, fs/2, zpad/2)
wl = c / fc
velocity = delta_f * wl / 2

# time range
time = numpy.linspace(1, tp*np, np)

# plot velocity vs time
# need to finish plotting properly
im = mpl.imshow(v - mmax, extent=(0,5,0,5))
mpl.show()