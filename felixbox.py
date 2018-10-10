#for pySerial
import serial 
import math
import time
import random


freqstartoffset = 512
freqstopoffset = 1024
gainoffset = 1536
loadoffset = 2048
moveoffset = 2560
startMsg = bytearray([161,0,0,0,0,0,1])
stopMsg = bytearray([161,0,1,0,0,0,1])


def makeAddr(offset, channel):
    addr = offset + channel
    addr_hi = int(math.floor(addr / 256))
    addr_lo = addr - addr_hi * 256
    return addr_lo, addr_hi

def getFTW(freq):
    return int(round(freq*2**28/(800.0/3.0)))


def writeStartFreq(ser, freq, channel):
    FTW = getFTW(freq*2.0)
    #print( 'FTW: {0}'.format(FTW))

    addr_lo, addr_hi = makeAddr(freqstartoffset, channel)
    
    FTW3 = int(math.floor(FTW / 256 / 256 / 256))
    FTW = FTW - FTW3 * 256 * 256 * 256
    FTW2 = int(math.floor(FTW / 256 / 256))
    FTW = FTW - FTW2 * 256 * 256
    FTW1 = int(math.floor(FTW / 256))
    FTW = FTW - FTW1 * 256
    FTW0 = int(math.floor(FTW))
    
    #print( 'hi {0}  -  lo {1}   --   FTW {2} {3} {4} {5}'.format(addr_hi, addr_lo,FTW3,FTW2,FTW1,FTW0))
    serWrite(ser, bytearray([161,addr_hi, addr_lo,FTW3,FTW2,FTW1,FTW0]))


def writeStopFreq(ser, freq,channel):
    FTW = getFTW(freq*2.0)
    #print( 'FTW: {0}'.format(FTW))

    addr_lo, addr_hi = makeAddr(freqstopoffset, channel)    
    
    FTW3 = int(math.floor(FTW / 256 / 256 / 256))
    FTW = FTW - FTW3 * 256 * 256 * 256
    FTW2 = int(math.floor(FTW / 256 / 256))
    FTW = FTW - FTW2 * 256 * 256
    FTW1 = int(math.floor(FTW / 256))
    FTW = FTW - FTW1 * 256
    FTW0 = int(math.floor(FTW))
    
    #print ('hi {0}  -  lo {1}   --   FTW {2} {3} {4} {5}'.format(addr_hi, addr_lo,FTW3,FTW2,FTW1,FTW0))
    serWrite(ser, bytearray([161,addr_hi, addr_lo,FTW3,FTW2,FTW1,FTW0]))


def writeGain(ser, gain,channel): 
    addr_lo, addr_hi = makeAddr(gainoffset, channel)

    GW2, GW3 = 0, 0
    GW1 = int(math.floor(gain / 256.0))
    gain = gain - GW1 * 256
    GW0 = int(math.floor(gain))
    
    #print('hi {0}  -  lo {1}   --   gain {2} {3} {4} {5}'.format(addr_hi, addr_lo,GW3,GW2,GW1,GW0))
    serWrite(ser, bytearray([161,addr_hi, addr_lo,GW3,GW2,GW1,GW0]))


def writeLoadPhaseMsg(ser, phase,channel):
    addr_lo, addr_hi = makeAddr(loadoffset, channel)
    
    GW2, GW3 = 0, 0
    GW1 = int(math.floor(phase / 256.0))
    phase = phase - GW1 * 256
    GW0 = int(math.floor(phase))
    
    #print('hi {0}  -  lo {1}   --   phaseload {2} {3} {4} {5}'.format(addr_hi, addr_lo,GW3,GW2,GW1,GW0))
    serWrite(ser, bytearray([161,addr_hi, addr_lo,GW3,GW2,GW1,GW0]))


def writeMovePhase(ser, phase,channel):
    addr_lo, addr_hi = makeAddr(moveoffset, channel)
    
    addr = moveoffset + channel
    addr_hi = int(math.floor(addr / 256.0))
    addr_lo = addr - addr_hi * 256
    
    GW3 = 0
    GW2 = 0
    GW1 = int(math.floor(phase / 256.0))
    phase = phase - GW1 * 256
    GW0 = int(math.floor(phase))
    
    #print('hi {0}  -  lo {1}   --   phasemove {2} {3} {4} {5}'.format(addr_hi, addr_lo,GW3,GW2,GW1,GW0))
    serWrite(ser, bytearray([161,addr_hi, addr_lo,GW3,GW2,GW1,GW0]))


def writeOnOff(ser, onoff):
    onoff3 = int(math.floor(onoff / 256 / 256 / 256))
    onoff = onoff - onoff3 * 256 * 256 * 256
    onoff2 = int(math.floor(onoff / 256 / 256))
    onoff = onoff - onoff2 * 256 * 256
    onoff1 = int(math.floor(onoff / 256))
    onoff = onoff - onoff1 * 256
    onoff0 = int(math.floor(onoff))
    
    #print('hi {0}  -  lo {1}   --   onoff {2} {3} {4} {5}'.format(0, 3,onoff3,onoff2,onoff1,onoff0))
    serWrite(ser, bytearray([161,0, 3,onoff3,onoff2,onoff1,onoff0]))

    
    

def serWrite(serObj, msg):
    print('.',end='')
    if serObj.write(msg) != len(msg):
        raise RuntimeError("Serial write failed! Didn't write all of msg!")

with serial.Serial('COM3') as ser:
    print('serial port = ' + ser.name)
    print('read:',ser.read_all())
    writeOnOff(ser, 2**32-1)
    for i in range(32):
        writeGain(ser, 65535,i)
        writeStartFreq(ser, 31.0+2.0*i*0,i)
        writeStopFreq(ser, 31.0+2*i*0,i)
    serWrite(ser, [161,0,2,0,0,0,1])
    serWrite(ser, startMsg)
    print('waiting:',ser.in_waiting)
    print('read:',ser.read_all())
    time.sleep(2)