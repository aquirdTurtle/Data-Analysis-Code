import serial #for pySerial
import math
import time
import random

freqstartoffset = 512
freqstopoffset = 1024
gainoffset = 1536
loadoffset = 2048
moveoffset = 2560

def getFTW(freq):
    return int(round(freq*2**28/(800.0/3.0)))

def writestartfreq(freq,channel):
    addr = freqstartoffset + channel
    FTW = getFTW(freq*2.0)
    
    print 'FTW: {0}'.format(FTW)
    
    addr_hi = int(math.floor(addr / 256))
    addr_lo = addr - addr_hi * 256
    
    FTW3 = int(math.floor(FTW / 256 / 256 / 256))
    FTW = FTW - FTW3 * 256 * 256 * 256
    FTW2 = int(math.floor(FTW / 256 / 256))
    FTW = FTW - FTW2 * 256 * 256
    FTW1 = int(math.floor(FTW / 256))
    FTW = FTW - FTW1 * 256
    FTW0 = int(math.floor(FTW))
    
    print 'hi {0}  -  lo {1}   --   FTW {2} {3} {4} {5}'.format(addr_hi, addr_lo,FTW3,FTW2,FTW1,FTW0)
    return bytearray([161,addr_hi, addr_lo,FTW3,FTW2,FTW1,FTW0])

def writestopfreq(freq,channel):
    addr = freqstopoffset + channel
    FTW = getFTW(freq*2.0)
    
    print 'FTW: {0}'.format(FTW)
    
    addr_hi = int(math.floor(addr / 256))
    addr_lo = addr - addr_hi * 256
    
    FTW3 = int(math.floor(FTW / 256 / 256 / 256))
    FTW = FTW - FTW3 * 256 * 256 * 256
    FTW2 = int(math.floor(FTW / 256 / 256))
    FTW = FTW - FTW2 * 256 * 256
    FTW1 = int(math.floor(FTW / 256))
    FTW = FTW - FTW1 * 256
    FTW0 = int(math.floor(FTW))
    
    print 'hi {0}  -  lo {1}   --   FTW {2} {3} {4} {5}'.format(addr_hi, addr_lo,FTW3,FTW2,FTW1,FTW0)
    return bytearray([161,addr_hi, addr_lo,FTW3,FTW2,FTW1,FTW0])

def writegain(gain,channel):
    addr = gainoffset + channel
    
    addr_hi = int(math.floor(addr / 256.0))
    addr_lo = addr - addr_hi * 256
    
    GW3 = 0
    GW2 = 0
    GW1 = int(math.floor(gain / 256.0))
    gain = gain - GW1 * 256
    GW0 = int(math.floor(gain))
    
    print 'hi {0}  -  lo {1}   --   gain {2} {3} {4} {5}'.format(addr_hi, addr_lo,GW3,GW2,GW1,GW0)
    return bytearray([161,addr_hi, addr_lo,GW3,GW2,GW1,GW0])

def writeloadphase(phase,channel):
    addr = loadoffset + channel
    
    addr_hi = int(math.floor(addr / 256.0))
    addr_lo = addr - addr_hi * 256
    
    GW3 = 0
    GW2 = 0
    GW1 = int(math.floor(phase / 256.0))
    phase = phase - GW1 * 256
    GW0 = int(math.floor(phase))
    
    print 'hi {0}  -  lo {1}   --   phaseload {2} {3} {4} {5}'.format(addr_hi, addr_lo,GW3,GW2,GW1,GW0)
    return bytearray([161,addr_hi, addr_lo,GW3,GW2,GW1,GW0])

def writemovephase(phase,channel):
    addr = moveoffset + channel
    
    addr_hi = int(math.floor(addr / 256.0))
    addr_lo = addr - addr_hi * 256
    
    GW3 = 0
    GW2 = 0
    GW1 = int(math.floor(phase / 256.0))
    phase = phase - GW1 * 256
    GW0 = int(math.floor(phase))
    
    print 'hi {0}  -  lo {1}   --   phasemove {2} {3} {4} {5}'.format(addr_hi, addr_lo,GW3,GW2,GW1,GW0)
    return bytearray([161,addr_hi, addr_lo,GW3,GW2,GW1,GW0])

def writestart():
    print 'writing start'
    return bytearray([161,0,0,0,0,0,1])
    
def writestop():
    print 'writing stop'
    return bytearray([161,0,1,0,0,0,1])

def writeonoff(onoff):
    onoff3 = int(math.floor(onoff / 256 / 256 / 256))
    onoff = onoff - onoff3 * 256 * 256 * 256
    onoff2 = int(math.floor(onoff / 256 / 256))
    onoff = onoff - onoff2 * 256 * 256
    onoff1 = int(math.floor(onoff / 256))
    onoff = onoff - onoff1 * 256
    onoff0 = int(math.floor(onoff))
    
    print 'hi {0}  -  lo {1}   --   onoff {2} {3} {4} {5}'.format(0, 3,onoff3,onoff2,onoff1,onoff0)
    return bytearray([161,0, 3,onoff3,onoff2,onoff1,onoff0])

with open('test_me_out.txt', 'w') as f:
    with serial.Serial('COM10', 77700) as ser: #open serial port
            print ('serial port = ' + ser.name)     #print the port used
            initialized = False
            while (True):
                if not initialized:
                    #turning all channels on
                    ser.write(writeonoff(2**32-1))
                    
                    #resetting gain, start freq, stop freq
                    for i in range(32):
                        ser.write(writegain(0,i))
                        ser.write(writestartfreq(0,i))
                        ser.write(writestopfreq(0,i))
                    
                    #setting gain, start freq, stop freq
                    for i in range(1):
                        ser.write(writegain(65535,i))
                        ser.write(writestartfreq(93.5 + (9.5/19) * i,i))
                        ser.write(writestopfreq(93.5 + (9.5/19) * i,i))
                    
                    #setting phases for loading
                    for i in range(4):
                        #~ ser.write(writeloadphase((4096/(i+1))%4096 ,5*i))
			#~ ser.write(writeloadphase(((4096/(i+1)) + (819)*i)%4096,(5*i)+1))
			#~ ser.write(writeloadphase(((4096/(i+1)) + (1638)*i)%4096,(5*i)+2))
			#~ ser.write(writeloadphase(((4096/(i+1)) + (2458)*i)%4096,(5*i)+3))
			#~ ser.write(writeloadphase(((4096/(i+1)) + (3277)*i)%4096,(5*i)+4))
			ser.write(writeloadphase(0 ,5*i))
			ser.write(writeloadphase((0 + 819 * i) ,(5*i)+1))
			ser.write(writeloadphase((0 + 1638 * i) ,(5*i)+2))
			ser.write(writeloadphase((0 + 2458 * i) ,(5*i)+3))
			ser.write(writeloadphase((0 + 3277 * i) ,(5*i)+4))
			

		    for i in range(32):
                        ser.write(writegain(65535,i))
		    
		   # ser.write(writegain(63535,0))
		   # ser.write(writegain(62535,1))
		    #~ ser.write(writegain(65535,2))
		    #~ ser.write(writegain(65535,3))
		    #~ ser.write(writegain(65535,4))
		    #~ ser.write(writegain(60535,5))
		    #~ ser.write(writegain(60535,6))
		    #~ ser.write(writegain(60535,7))
		    #~ ser.write(writegain(60535,8))
		   # ser.write(writegain(63535,9))
		    #~ ser.write(writegain(60535,10))
		    #~ ser.write(writegain(60535,11))
		    #~ ser.write(writegain(60535,12))
		    #ser.write(writegain(63535,13))
		    #~ ser.write(writegain(60535,14))
		    #ser.write(writegain(64535,15))
		    #ser.write(writegain(63535,16))
		    #ser.write(writegain(63535,17))
		    #ser.write(writegain(63535,18))
		    #ser.write(writegain(64535,19))
		    #~ ser.write(writegain(64535,20))
		    
		    #~ for i in range(32):
                        #~ ser.write(writegain(65535,i))
		    
		    
		    
                    #setting phases for after the move
                    for i in range(20):
                        ser.write(writemovephase(0,i))
                    
                    #increment for linear ramp
                    #1 LSB is ~8MHz per sec (800 / 96 MHz per sec)
                    ser.write([161,0,2,0,0,0,1])
                    
                    
                #this shows two sine waves with offset phases
                    #~ ser.write(writegain(65535,0))
                    #~ ser.write(writestartfreq(16.0,0))
                    #~ ser.write(writestopfreq(30.0,0))
                    
                    #~ ser.write(writegain(65535,1))
                    #~ ser.write(writestartfreq(32.0000005,1))
                    #~ ser.write(writestopfreq(30.0,1))
                    
                    #~ ser.write(writeloadphase(0,0))
                    #~ ser.write(writeloadphase(0,1))
                #this shows two sine waves with offset phases

                    #trigger loading with loading input (5V level)
                    #trigger move with move input (5V level)
                    
                    initialized = True
                if (ser.in_waiting>0):
                    values = ser.read(ser.in_waiting)
                    for i in values:
                        print ('{0} \n'.format(ord(i)))
                        f.write('{0} \n'.format(ord(i)))