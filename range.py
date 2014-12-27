# TODO: PLOT DATA/COMPLETE PLOTTING (matplotlib)
import scipy.fftpack
import scipy.io.wavfile
import numpy
import matplotlib.pyplot as mpl
import sys
import dbv

# load input data from wav file
if (len(sys.argv) < 2):
    wav_file = 'running_outside_20ms.wav' # if no wav specified
else:
    wav_file = sys.argv[1]  # get wav filename from command line
(fs, y) = scipy.io.wavfile.read(wav_file)   # Fs = sample rate, Y = sample data

# constants
c = 3e8 # c = 299792458 (m/s)

# RADAR parameters
tp = 20e-3  # pulse time (seconds)
ns = int(fs * tp)    # number of samples per pulse
fstart = 2.26e9 # LFM start frequency (Hz)
fstop = 2.59e9  # LFM stop frequency (Hz)
bw = fstop - fstart
f = numpy.linspace(fstart, fstop, ns/2)

# range resolution
rr = c / (2 * bw)
max_range = rr * ns / 2

# invert channels and normalize to 16bit
trig = -y[:, 0] / 32768.0
s = -y[:, 1] / 32768.0

# parse data using trig channel
count = 0
thresh = 0
start = trig > thresh
sif = numpy.array([])
time = numpy.array([])
for i in xrange(99, start.size - ns + 1):
    if (start[i] == 1) and (numpy.mean(start[i-11:i]) == 0):
        count += 1
        sif = numpy.append(sif, s[i:i+ns])
        time = numpy.append(time, i * 1 / fs)
sif = numpy.reshape(sif, (count, ns))

# subtract out average from data
ave = numpy.mean(sif, axis = 0)
for i in xrange(count):
    sif[i, :] = sif[i, :] - ave

# ift and plot data
zpad = 8 * ns / 2
v_ifft = scipy.fftpack.ifft(sif, zpad, 1)
v = dbv.dbv(v_ifft[:, 0:zpad/2])
mmax = numpy.amax(numpy.amax(v))

# plot RTI

# pulse canceler RTI
sif2 = sif[1:count+1, :] - sif[0:count, :]
v_ifft = scipy.fftpack.ifft(sif2, zpad, 1)
v = dbv.dbv(v_ifft[:, 0:zpad/2])
mmax = numpy.amax(numpy.amax(v))

# plot pulse canceler 