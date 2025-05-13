import numpy as np
from scipy import signal
import mne
from typing import Dict, List, Tuple, Optional, Union


class EEGProcessor:
    """
    Class for processing and analyzing EEG signals for the Adaptive Neural Stimulation System.
    Provides methods for filtering, feature extraction, and analysis of EEG data.
    """
    
    def __init__(self, sampling_rate: float = 250.0, n_channels: int = 32):
        """
        Initialize the EEG processor.
        
        Args:
            sampling_rate: The sampling rate of the EEG data in Hz
            n_channels: Number of EEG channels
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.filters = {
            'notch': {'freq': 50.0, 'quality_factor': 30.0},  # Default notch filter at 50Hz (can be changed to 60Hz)
            'bandpass': {'low': 0.5, 'high': 45.0, 'order': 4}  # Default bandpass filter from 0.5Hz to 45Hz
        }
        
    def apply_filters(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Apply filters to the EEG data.
        
        Args:
            eeg_data: Raw EEG data with shape (n_channels, n_samples)
            
        Returns:
            Filtered EEG data with the same shape as input
        """
        # Make a copy to avoid modifying the original data
        filtered_data = eeg_data.copy()
        
        # Apply notch filter (for power line interference)
        notch_freq = self.filters['notch']['freq']
        quality_factor = self.filters['notch']['quality_factor']
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, self.sampling_rate)
        
        # Apply bandpass filter
        low_freq = self.filters['bandpass']['low']
        high_freq = self.filters['bandpass']['high']
        order = self.filters['bandpass']['order']
        nyquist = 0.5 * self.sampling_rate
        low = low_freq / nyquist
        high = high_freq / nyquist
        b_bandpass, a_bandpass = signal.butter(order, [low, high], btype='band')
        
        # Apply filters to each channel
        for ch in range(filtered_data.shape[0]):
            # Apply notch filter
            filtered_data[ch, :] = signal.filtfilt(b_notch, a_notch, filtered_data[ch, :])
            # Apply bandpass filter
            filtered_data[ch, :] = signal.filtfilt(b_bandpass, a_bandpass, filtered_data[ch, :])
            
        return filtered_data
    
    def extract_bands(self, eeg_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract standard frequency bands from EEG data.
        
        Args:
            eeg_data: Filtered EEG data with shape (n_channels, n_samples)
            
        Returns:
            Dictionary containing the bandpassed EEG data for each frequency band
        """
        bands = {
            'delta': (0.5, 4.0),   # Delta band: 0.5-4 Hz
            'theta': (4.0, 8.0),   # Theta band: 4-8 Hz
            'alpha': (8.0, 13.0),  # Alpha band: 8-13 Hz
            'beta': (13.0, 30.0),  # Beta band: 13-30 Hz
            'gamma': (30.0, 45.0)  # Gamma band: 30-45 Hz
        }
        
        band_data = {}
        nyquist = 0.5 * self.sampling_rate
        
        for band_name, (low_freq, high_freq) in bands.items():
            low = low_freq / nyquist
            high = high_freq / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            
            band_data[band_name] = np.zeros_like(eeg_data)
            for ch in range(eeg_data.shape[0]):
                band_data[band_name][ch, :] = signal.filtfilt(b, a, eeg_data[ch, :])
                
        return band_data
    
    def compute_psd(self, eeg_data: np.ndarray, window_length: float = 4.0, 
                   overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Power Spectral Density (PSD) of the EEG data.
        
        Args:
            eeg_data: EEG data with shape (n_channels, n_samples)
            window_length: Length of the window for PSD computation in seconds
            overlap: Overlap between consecutive windows (0.0 to 1.0)
            
        Returns:
            Tuple containing frequencies and PSD values
        """
        n_samples = eeg_data.shape[1]
        window_samples = int(window_length * self.sampling_rate)
        overlap_samples = int(window_samples * overlap)
        
        # Calculate frequencies
        freqs, _ = signal.welch(eeg_data[0, :], fs=self.sampling_rate, 
                              nperseg=window_samples, noverlap=overlap_samples)
        
        # Initialize PSD array for all channels
        psd = np.zeros((self.n_channels, len(freqs)))
        
        # Compute PSD for each channel
        for ch in range(eeg_data.shape[0]):
            _, psd[ch, :] = signal.welch(eeg_data[ch, :], fs=self.sampling_rate, 
                                       nperseg=window_samples, noverlap=overlap_samples)
            
        return freqs, psd
    
    def compute_band_power(self, eeg_data: np.ndarray, band: Tuple[float, float] = None) -> np.ndarray:
        """
        Compute power in a specific frequency band for each channel.
        
        Args:
            eeg_data: EEG data with shape (n_channels, n_samples)
            band: Tuple of (low_freq, high_freq) for the band of interest
                  If None, returns the total power
        
        Returns:
            Array of power values for each channel
        """
        freqs, psd = self.compute_psd(eeg_data)
        
        if band is None:
            # Compute total power
            return np.sum(psd, axis=1)
        else:
            # Find indices corresponding to the specified frequency band
            low_freq, high_freq = band
            idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            
            # Compute power in the specified band
            return np.sum(psd[:, idx_band], axis=1)
    
    def compute_band_ratios(self, eeg_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute common band ratios which are useful for various analyses.
        
        Args:
            eeg_data: EEG data with shape (n_channels, n_samples)
        
        Returns:
            Dictionary containing the band ratios for each channel
        """
        # Extract bands
        band_data = self.extract_bands(eeg_data)
        
        # Calculate power for each band
        band_power = {}
        for band_name, band_signal in band_data.items():
            band_power[band_name] = self.compute_band_power(band_signal)
            
        # Calculate ratios
        ratios = {
            'theta_alpha': band_power['theta'] / band_power['alpha'],
            'alpha_beta': band_power['alpha'] / band_power['beta'],
            'theta_beta': band_power['theta'] / band_power['beta'],
            'delta_beta': band_power['delta'] / band_power['beta'],
            'gamma_theta': band_power['gamma'] / band_power['theta']
        }
        
        return ratios
    
    def compute_coherence(self, eeg_data: np.ndarray, ch1: int, ch2: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute coherence between two EEG channels across frequencies.
        
        Args:
            eeg_data: EEG data with shape (n_channels, n_samples)
            ch1: Index of first channel
            ch2: Index of second channel
            
        Returns:
            Tuple of (frequencies, coherence values)
        """
        f, Cxy = signal.coherence(eeg_data[ch1, :], eeg_data[ch2, :], fs=self.sampling_rate)
        return f, Cxy
    
    def compute_phase_locking_value(self, eeg_data: np.ndarray, band: Tuple[float, float],
                                   ch1: int, ch2: int) -> float:
        """
        Compute Phase Locking Value (PLV) between two channels in a specific frequency band.
        
        Args:
            eeg_data: EEG data with shape (n_channels, n_samples)
            band: Tuple of (low_freq, high_freq) for the band of interest
            ch1: Index of first channel
            ch2: Index of second channel
            
        Returns:
            Phase Locking Value (between 0 and 1)
        """
        # Extract band-specific signals
        nyquist = 0.5 * self.sampling_rate
        low_freq, high_freq = band
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Filter signals to get band-specific data
        signal1 = signal.filtfilt(b, a, eeg_data[ch1, :])
        signal2 = signal.filtfilt(b, a, eeg_data[ch2, :])
        
        # Compute analytic signal (using Hilbert transform)
        analytic_signal1 = signal.hilbert(signal1)
        analytic_signal2 = signal.hilbert(signal2)
        
        # Get instantaneous phase
        phase1 = np.angle(analytic_signal1)
        phase2 = np.angle(analytic_signal2)
        
        # Compute phase difference
        phase_diff = phase1 - phase2
        
        # Compute PLV
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return plv
    
    def detect_artifacts(self, eeg_data: np.ndarray, threshold: float = 100.0) -> np.ndarray:
        """
        Detect artifacts in EEG data based on amplitude threshold.
        
        Args:
            eeg_data: EEG data with shape (n_channels, n_samples)
            threshold: Amplitude threshold for artifact detection
            
        Returns:
            Boolean mask indicating artifacts (True where artifacts are detected)
        """
        # Initialize artifact mask
        n_samples = eeg_data.shape[1]
        artifact_mask = np.zeros(n_samples, dtype=bool)
        
        # Check each channel for threshold crossings
        for ch in range(eeg_data.shape[0]):
            # Mark samples that exceed the threshold
            artifact_mask = np.logical_or(artifact_mask, np.abs(eeg_data[ch, :]) > threshold)
            
        return artifact_mask
    
    def create_mne_raw(self, eeg_data: np.ndarray, ch_names: List[str] = None, 
                      ch_types: List[str] = None) -> mne.io.RawArray:
        """
        Create an MNE Raw object from the EEG data for advanced processing.
        
        Args:
            eeg_data: EEG data with shape (n_channels, n_samples)
            ch_names: List of channel names (defaults to ['CH1', 'CH2', ...])
            ch_types: List of channel types (defaults to all 'eeg')
            
        Returns:
            MNE Raw object for advanced processing
        """
        # Create default channel names if not provided
        if ch_names is None:
            ch_names = [f'CH{i+1}' for i in range(eeg_data.shape[0])]
            
        # Create default channel types if not provided
        if ch_types is None:
            ch_types = ['eeg'] * eeg_data.shape[0]
            
        # Create info structure
        info = mne.create_info(ch_names=ch_names, sfreq=self.sampling_rate, ch_types=ch_types)
        
        # Create Raw object
        raw = mne.io.RawArray(eeg_data, info)
        
        return raw
    
    def process_batch(self, eeg_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process a batch of EEG data and extract key features.
        
        Args:
            eeg_data: EEG data with shape (n_channels, n_samples)
            
        Returns:
            Dictionary of extracted features
        """
        results = {}
        
        # Apply filters
        filtered_data = self.apply_filters(eeg_data)
        results['filtered_data'] = filtered_data
        
        # Extract frequency bands
        band_data = self.extract_bands(filtered_data)
        results['band_data'] = band_data
        
        # Compute PSD
        freqs, psd = self.compute_psd(filtered_data)
        results['freqs'] = freqs
        results['psd'] = psd
        
        # Compute band power
        band_power = {}
        band_definitions = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 45.0)
        }
        
        for band_name, band_range in band_definitions.items():
            band_power[band_name] = self.compute_band_power(filtered_data, band=band_range)
            
        results['band_power'] = band_power
        
        # Compute band ratios
        results['band_ratios'] = self.compute_band_ratios(filtered_data)
        
        # Detect artifacts
        results['artifacts'] = self.detect_artifacts(filtered_data)
        
        return results