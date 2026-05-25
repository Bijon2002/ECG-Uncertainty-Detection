import numpy as np
import pandas as pd
import wfdb
from scipy import signal
from sklearn.preprocessing import StandardScaler
import neurokit2 as nk

class ECGPreprocessor:
    def __init__(self, sampling_rate=360, target_length=180):
        self.sampling_rate = sampling_rate
        self.target_length = target_length  # 0.5 seconds at 360Hz
        
    def bandpass_filter(self, ecg_signal, lowcut=0.5, highcut=40):
        """Apply bandpass filter to remove noise"""
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, ecg_signal)
        return filtered_signal
    
    def detect_r_peaks(self, ecg_signal):
        """Detect R-peaks using NeuroKit2"""
        _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=self.sampling_rate)
        return rpeaks['ECG_R_Peaks']
    
    def extract_beats(self, ecg_signal, r_peaks):
        """Extract individual heartbeats around R-peaks"""
        beats = []
        half_window = self.target_length // 2
        
        for r_peak in r_peaks:
            start = r_peak - half_window
            end = r_peak + half_window
            
            # Check bounds
            if start >= 0 and end < len(ecg_signal):
                beat = ecg_signal[start:end]
                if len(beat) == self.target_length:
                    beats.append(beat)
        
        return np.array(beats)
    
    def normalize_beats(self, beats):
        """Normalize beats using StandardScaler"""
        scaler = StandardScaler()
        beats_normalized = []
        
        for beat in beats:
            beat_norm = scaler.fit_transform(beat.reshape(-1, 1)).flatten()
            beats_normalized.append(beat_norm)
            
        return np.array(beats_normalized)