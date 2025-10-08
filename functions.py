# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:06:06 2025

@author: Agrima Agarwal
"""
import scipy as sp
import numpy as np
import scipy.ndimage as scnd
import pandas as pd

class THzData:
    def __init__(self,raw_waveform,reference,baseline=None):
        self.baseline = baseline
        if type(raw_waveform)==type(''):#if input is a file
            matrix = np.array(readMatrix(raw_waveform))
            time, raw_waveform,_,_ = read_Menlo_pressure(matrix)
        else:
            assert time is not None
        if type(reference)== type(''):#if input is a file
            reference = np.array(pd.read_csv(reference, sep='\t',header=None))[:-1, 1]
            self.reference = reference[:1495]
        if type(baseline)== type(''):#if input is a file
            baseline = np.array(pd.read_csv(baseline, sep='\t',header=None))[:-1, 1]
            self.baseline = baseline[:1495]
            
        exclude = 10 #excldue the first exclude measurements
        self.time = time[:1495]
        self.raw_waveform = raw_waveform[:1495,exclude:]

        self.make_impulses()

    def make_impulses(self):
        t = self.time
        wf=self.raw_waveform

        N = 1
        tn = np.linspace(t[0], t[-1], (len(t)) * N)
        reference = self.reference
        baseline = self.baseline
        b1 = 300*N
        if baseline is not None:  
            shift1 = b1-np.argmin(baseline[:len(baseline)//2])
            baseline = scnd.shift(baseline,  shift1.item(), mode='nearest')
         
        b2 = np.argmin(reference[:len(reference)//2])
        shift1 = b1-b2
        fac1 = (np.max(baseline[:len(baseline)//2]) - np.min(baseline[:len(baseline)//2])) / (np.max(reference[:len(reference)//2]) - np.min(reference[:len(reference)//2]))
        reference = scnd.shift(reference,  shift1.item(), mode='nearest')*fac1
        
        
        data_str = []
        Wf = np.zeros((len(tn), wf.shape[1]))
        # for all samples 
        for i in range(wf.shape[1]):
            # align the signal w.r.t their lowest peaks to remove shift due to
            # mechanical translation
            # Wf[:, i] = np.interp(tn, t, wf[:, i])
            Wf[:, i] =  wf[:, i]
            b3 = np.argmin(Wf[:len(Wf)//2, i])
            shift = b1-b3
            fac = (np.max(reference[:len(reference)//2]) - np.min(reference[:len(reference)//2])) / (np.max(Wf[:len(Wf)//2, i]) - np.min(Wf[:len(Wf)//2, i]))
            sample = scnd.shift(Wf[:, i],  shift.item(), mode='nearest') * fac
            if baseline is not None:
                data_str.append(process(sample-baseline,reference-baseline,tn,0,0,0,0,0))
            else:
                data_str.append(process(sample,reference,tn,0,0,0,0,0))
            

        output_data = MENLO_main_processing_better(data_str,window_para=200,zerofill=0,mid_2nd_pulse=1050*N,window_size=400*N)
        imp1 = np.zeros(( len(output_data[0].M_cut_sample),len(output_data)),dtype=complex)
        #  for all samples, store impulse function in imp
        for i in range(len(output_data)):
            imp1[:, i] = output_data[i].M_cut_sample
            
        imp = np.zeros(( len(output_data[0].imp),len(output_data)))
        P2P_sample = np.zeros((len(output_data),1))
        P2P_imp = np.zeros((len(output_data),1))
        sample_td = np.zeros(( len(output_data[0].sample_td),len(output_data)))
        amp_map = np.zeros(( len(output_data[0].M_cut_sample),len(output_data)))
        phase_map = np.zeros(( len(output_data[0].M_cut_sample),len(output_data)))
       
        #  for all 233 samples, store impulse function in imp
        for i in range(len(output_data)):
            imp[:, i] = output_data[i].imp
            sample_td[:,i] = output_data[i].sample_td
            P2P_sample[i] = np.ptp(output_data[i].sample_td)
            P2P_imp[i] = np.ptp(output_data[i].imp)
            amp_map[:,i] = np.abs(output_data[i].M_cut_sample)
            phase_map[:,i] = np.angle(output_data[i].M_cut_sample)
            
            
        # P2P_norm = P2P/max(P2P)     
        # self.P2P_norm = P2P_norm
        self.impulses = imp
        self.impulses_freq = imp1
        self.output_data = output_data
        self.reference = reference
        self.time = tn
        self.P2P_sample = P2P_sample[:,0]
        self.P2P_imp = P2P_imp[:,0]
        self.P2P_ref = np.ptp(reference)
        self.phase_map = phase_map
        self.amp_map = amp_map
        self.sample_td = sample_td
       
        
        

#%%

def read_Menlo_pressure(x):
    wf = x[2:-1, 1:-1]
    P1 = x[0, 1:-1]
    P2 = x[1, 1:-1]
    t = x[2:-1, 0]
    return t, wf, P1, P2


def readMatrix(filNam):
    with open(filNam, 'r') as matrixFile:
        matrix = []
        lines = matrixFile.readlines()
        for line in lines:
            items = line.split()
            numbers = [float(item) for item in items]
            matrix.append(numbers)
        return matrix
    

class process:
    def __init__(self, sample_td, reference_td, time, freq, M_cut_sample, sample_fd,reference_fd,imp):
        self.sample_td = sample_td
        self.reference_td = reference_td
        self.time = time
        self.freq = freq
        self.M_cut_sample = M_cut_sample
        self.sample_fd = sample_fd
        self.reference_fd = sample_fd
        self.imp = imp


def MENLO_main_processing_better(data_str, window_para, zerofill,mid_2nd_pulse,window_size):
    if zerofill is None:
        zerofill = 0
    if window_para is None:
        window_para = 100
    data_str = MENLO_HP_filter(data_str)
    data_str =  MENLO_tuckey_window(data_str, window_para, mid_2nd_pulse,window_size)
    data_str = MENLO_FFT(data_str)
    data_str = MENLO_impulse_direct_freq(data_str, zerofill)
    output_data = data_str
    return output_data


def MENLO_HP_filter(data_str):
    signal_length = len(data_str[0].time)
    deltaT = (data_str[0].time)[1] - (data_str[0].time)[0]
    MaxFreq = 1 / deltaT
    DeltaFreq = MaxFreq / (signal_length - 1)
    freq = np.arange(0, MaxFreq + DeltaFreq, DeltaFreq)
    HP_filter = np.exp(-((freq - 0.1) ** 2) / 0.02 ** 2)
    f_ind = np.where(freq > 0.1)
    HP_filter[f_ind] = 1
    output_datas = data_str
    m = len(data_str)
    for i in range(m):
        sample_fd = np.fft.rfft(data_str[i].sample_td)
        filter_use = HP_filter[:len(sample_fd)]
        sample_fd = sample_fd * filter_use
        new_sample_td = np.fft.irfft(sample_fd, signal_length)
        reference_fd = np.fft.rfft(data_str[i].reference_td)
        reference_fd = reference_fd * filter_use
        new_reference_td = np.fft.irfft(reference_fd, signal_length)
        output_datas[i].sample_td = new_sample_td
        output_datas[i].reference_td = new_reference_td
        output_datas[i].time = data_str[i].time
    return output_datas


def MENLO_tuckey_window(data_str, window_para, mid_2nd_pulse,window_size):
    output_datas = data_str
    if window_para != 0:
        normal_pos = len(output_datas[0].reference_td)//2+(np.argmin(output_datas[0].reference_td[len(output_datas[0].reference_td)//2:])+np.argmax(output_datas[0].reference_td[len(output_datas[0].reference_td)//2:]))//2
        # normal_pos = (np.argmin(output_datas[0].reference_td)+np.argmax(output_datas[0].reference_td))//2
        mid_2nd_pulse = normal_pos
        window_ftn1 = sp.signal.windows.tukey(window_size, 0.85)
        m = len(data_str)
        for i in range(m):
            output_datas[i].sample_td = data_str[i].sample_td[mid_2nd_pulse-(window_size//2):mid_2nd_pulse+(window_size//2)]
            output_datas[i].reference_td = data_str[i].reference_td[mid_2nd_pulse-(window_size//2):mid_2nd_pulse+(window_size//2)]
            # normal_pos = (np.argmin(output_datas[0].reference_td)+np.argmax(output_datas[0].reference_td))//2
            normal_pos = len(output_datas[0].reference_td)//2+(np.argmin(output_datas[0].reference_td[len(output_datas[0].reference_td)//2:])+np.argmax(output_datas[0].reference_td[len(output_datas[0].reference_td)//2:]))//2
            window_ftn = scnd.shift(window_ftn1,  normal_pos.item()-window_size//2, mode='nearest')
            output_datas[i].sample_td = window_ftn * output_datas[i].sample_td
            output_datas[i].reference_td = window_ftn * output_datas[i].reference_td
            output_datas[i].time = data_str[i].time[:window_size]
    else:
        m, n = np.shape(data_str)    
    return output_datas


def MENLO_FFT(data_str):
    fft_length = len(data_str[0].sample_td)
    # fft_length = 3613
    signal_length = len(data_str[0].time)
    # signal_length = fft_length
    deltaT = (data_str[0].time)[1] - (data_str[0].time)[0]
    MaxFreq = 1 / deltaT
    DeltaFreq = MaxFreq / (signal_length - 1)
    freq = np.arange(0, MaxFreq + DeltaFreq, DeltaFreq)
    output_datas = data_str
    m = len(data_str)
    for i in range(m):
        sample_fd = np.fft.rfft(data_str[i].sample_td,fft_length)
        # sample_fd1 = np.fft.rfft(data_str[i].sample_td)
        reference_fd = np.fft.rfft(data_str[i].reference_td,fft_length)
        output_datas[i].sample_fd = sample_fd
        output_datas[i].reference_fd = reference_fd
        output_datas[i].freq = freq[:len(sample_fd)]
    return output_datas
    

def MENLO_impulse_direct_freq(data_str, zerofill):
    if zerofill > len(data_str[0].sample_td):
        fft_length = zerofill
    else:
        fft_length = len(data_str[0].sample_td)
    # fft_length = 3613
    freq = data_str[0].freq
    output_datas = data_str
    m = len(data_str)
    # freq_low=0.1
    # freq_up=0.6
    freq_low=0.1
    freq_up=1.0
    L=1/(freq_low*np.pi)
    H=1/(freq_up*np.pi)
    sample_fd = data_str[0].sample_fd
    tmax=1/(freq[1]-freq[0]);
    Dbl_filter=(np.exp(-(H*np.pi)**2*freq**2)-np.exp(-(L*np.pi)**2*freq**2))*np.exp(1j*tmax*freq*np.pi);

    for i in range(m):
        sample_fd = data_str[i].sample_fd
        sample_fd_mag = np.abs(sample_fd)
        sample_fd_phase = np.angle(sample_fd)
        reference_fd = data_str[i].reference_fd
        reference_fd_mag = np.abs(reference_fd)
        reference_fd_phase = np.angle(reference_fd)
        M_mag = sample_fd_mag/reference_fd_mag
        M_phase=np.exp(1j*(sample_fd_phase-reference_fd_phase))
        M=M_mag*M_phase;
        impulse_fd=M*Dbl_filter;
        output_datas[i].sample_fd = sample_fd[(freq > freq_low) & (freq<freq_up)]
        output_datas[i].Dbl_filter = Dbl_filter
        output_datas[i].imp=np.fft.irfft(impulse_fd,fft_length)
        output_datas[i].M_cut_sample=M[(freq > freq_low) & (freq<freq_up)]
        output_datas[i].limited_freq=freq[(freq > freq_low) & (freq<freq_up)]
    return output_datas

    

        
        
