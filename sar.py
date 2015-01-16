# TODO: PLOT DATA/COMPLETE PLOTTING (matplotlib)
# TODO: finish rest of SAR RMA_IFP and plotting
import scipy
import scipy.fftpack
import scipy.io.wavfile
import numpy
import matplotlib.pyplot as mpl
import sys
import dbv

# load input data from wav file
if (len(sys.argv) < 2):
    wav_file = 'towardswarehouse.wav' # if no wav specified
else:
    wav_file = sys.argv[1]  # get wav filename from command line
(fs, y) = scipy.io.wavfile.read(wav_file)   # fs = sample rate, y = sample data

# constants
c = 3e8 # c = 299792458 (m/s)

# RADAR parameters
tp = 20e-3  # pulse time (seconds)
trp = 0.25  # min range profile duration (s)
ns = int(fs * tp)    # number of samples per pulse
nrp = int(fs * trp)  # min num of samples between range profiles
fstart = 2.26e9 # LFM start frequency (Hz)
fstop = 2.59e9  # LFM stop frequency (Hz)
bw = fstop - fstart
f = numpy.linspace(fstart, fstop, ns/2)

# invert channels and normalize to 16bit
trig = -y[:, 0] / 32768.0   # trigger channel
s = -y[:, 1] / 32768.0  # data channel

# parse data by position (silence between recorded data)
rpstart = numpy.absolute(trig) > numpy.mean(numpy.absolute(trig))
count_pos = 0
rp = numpy.array([])
rptrig = numpy.array([])

for i in xrange(nrp, rpstart.size - nrp):
    if (rpstart[i] == 1) and (numpy.sum(rpstart[i-nrp:i]) == 0):
        count_pos += 1
        rp = numpy.append(rp, s[i:i+nrp])
        rptrig = numpy.append(rptrig, trig[i:i+nrp])
        
rp = numpy.reshape(rp, (count_pos, nrp))
rptrig = numpy.reshape(rptrig, (count_pos, nrp))

# parse data by pulse
count_pul = 0
thresh = 0.08   # ?
sif = numpy.array([])

for i in xrange(0, count_pos):
    sif_temp = numpy.zeros(ns)
    start = rptrig[i, :] > thresh
    count_pul = 0
    
    for j in xrange(11, nrp - 2*ns):
        trigmax = numpy.argmax(rptrig[i, j:j+2*ns+1])
        if (numpy.mean(start[j-10:j-1]) == 0) and (trigmax == 0):
            count_pul += 1
            sif_temp = rp[i, j:j+ns] + sif_temp
            
    q = scipy.fftpack.ifft(sif_temp / count_pul)
    q_fft = scipy.fftpack.fft(q[ns/2:ns])
    sif = numpy.append(sif, q_fft)
    
sif = numpy.reshape(sif, (count_pos, ns/2))
sif[numpy.nonzero(numpy.isnan(sif))] = 1e-30    # set NaN values to "zero"

# subtract out mean from each range profile
for i in xrange(count_pos):
    sif[i, :] = sif[i, :] - numpy.mean(sif, axis = 0)
    
sif_int = sif

# more radar parameters
fc = (fstop - fstart) / 2.0 + fstart  # center radar freq (Hz)
cr = bw / tp # chirp rate (Hz/s)
rs = 0  # distance to cal target (m)
xa = 0  # beginning of new aperture length (m)
delta_x = 0.508 # antenna spacing, 2 inches (m)
l = delta_x * count_pos # aperture length (m)
xa = numpy.linspace(-l/2.0, l/2.0, count_pos)   # cross range position of radar on aperture l (m)
ya = rs # apparently very important
za = 0
t = numpy.linspace(0, tp, ns/2)
kr = numpy.linspace(((4.0*numpy.pi/c) * (fc - bw/2.0)), ((4.0*numpy.pi/c) * (fc + bw/2.0)), ns/2)

# apply hanning window to data
N = int(ns / 2)
H = numpy.empty(N)
sif_h = numpy.empty((count_pos, N), dtype = complex)
for i in xrange(N):
    H[i] = 0.5 + 0.5*numpy.cos(2.0*numpy.pi*((i+1) - N/2.0)/N)
    
for i in xrange(count_pos):
    sif_h[i, :] = numpy.multiply(sif[i, :], H)
    
sif = sif_h

# along track FFT (in the slow time domain)
zpad = 2048
szeros = numpy.zeros((zpad, N), dtype = complex)
index = int(round((zpad - count_pos) / 2.0))
for i in xrange(N):
    szeros[index:index+count_pos, i] = sif[:, i]

sif = szeros

# TODO: finish rest of SAR RMA_IFP and plotting