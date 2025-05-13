"""
Brainwave Synchronization Module

This module implements algorithms for synchronizing neural stimulation with the user's 
real-time brainwave patterns, specifically focusing on phase alignment between
stimulation signals and ongoing neural oscillations.

Key features:
- Real-time phase extraction from EEG signals
- Dynamically adjusted stimulation timing based on neural phase
- Support for multiple frequency bands (theta, alpha, beta, gamma)
- Adaptive phase offset based on neural circuit targets
"""

import numpy as np
from scipy import signal

class BrainwaveSynchronization:
    """
    Class for implementing phase-locked neural stimulation synchronized with ongoing
    brain rhythms measured through EEG.
    """
    
    def __init__(self, sampling_rate=1000, target_band='alpha', n_channels=8):
        """
        Initialize the brainwave synchronization module.
        
        Args:
            sampling_rate (int): EEG sampling rate in Hz
            target_band (str): Target frequency band for synchronization 
                               ('theta', 'alpha', 'beta', 'gamma')
            n_channels (int): Number of EEG channels
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        
        # Define frequency bands
        self.bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # Set target band
        self.set_target_band(target_band)
        
        # Initialize buffers
        self.buffer_duration = 2  # seconds
        self.buffer_size = int(self.buffer_duration * self.sampling_rate)
        self.eeg_buffer = np.zeros((self.n_channels, self.buffer_size))
        
        # Phase alignment settings
        self.phase_offset = 0  # degrees
        self.last_phases = np.zeros(n_channels)
        
    def set_target_band(self, band_name):
        """
        Set the target frequency band for phase synchronization.
        
        Args:
            band_name (str): Name of the frequency band 
                            ('theta', 'alpha', 'beta', 'gamma')
        
        Raises:
            ValueError: If the band name is not recognized
        """
        if band_name not in self.bands:
            raise ValueError(f"Unknown band: {band_name}. " 
                            f"Available bands: {list(self.bands.keys())}")
        
        self.target_band = band_name
        self.band_range = self.bands[band_name]
        
        # Design bandpass filter for the target band
        nyquist = self.sampling_rate / 2
        low = self.band_range[0] / nyquist
        high = self.band_range[1] / nyquist
        self.filter_b, self.filter_a = signal.butter(4, [low, high], btype='bandpass')
    
    def update_buffer(self, new_eeg_data):
        """
        Update the EEG buffer with new data.
        
        Args:
            new_eeg_data (numpy.ndarray): New EEG data with shape (n_channels, n_samples)
        """
        n_samples = new_eeg_data.shape[1]
        
        # Shift the buffer and add new data
        self.eeg_buffer = np.roll(self.eeg_buffer, -n_samples, axis=1)
        self.eeg_buffer[:, -n_samples:] = new_eeg_data
    
    def extract_phase(self, channel_idx=None):
        """
        Extract the instantaneous phase from the EEG buffer for specific channels.
        
        Args:
            channel_idx (int or list, optional): Index or indices of channels to process.
                                                If None, all channels are processed.
        
        Returns:
            numpy.ndarray: Instantaneous phase in degrees for each channel
        """
        if channel_idx is None:
            channels_to_process = range(self.n_channels)
        elif isinstance(channel_idx, int):
            channels_to_process = [channel_idx]
        else:
            channels_to_process = channel_idx
        
        phases = np.zeros(len(channels_to_process))
        
        for i, channel in enumerate(channels_to_process):
            # Apply bandpass filter
            filtered = signal.filtfilt(self.filter_b, self.filter_a, 
                                      self.eeg_buffer[channel, :])
            
            # Perform Hilbert transform to get analytic signal
            analytic_signal = signal.hilbert(filtered)
            
            # Extract instantaneous phase
            phase = np.angle(analytic_signal[-1])  # Get phase of the most recent sample
            phase_deg = np.degrees(phase) % 360  # Convert to degrees (0-360)
            
            phases[i] = phase_deg
        
        self.last_phases[channels_to_process] = phases
        return phases
    
    def get_optimal_stimulation_timings(self, target_phase, window_size=50, channel_weights=None):
        """
        Calculate the optimal timing for delivering stimulation to achieve
        the target phase alignment.
        
        Args:
            target_phase (float): Target phase in degrees (0-360)
            window_size (int): Number of future samples to consider
            channel_weights (numpy.ndarray, optional): Weights for each channel for
                                                    weighted phase calculation
        
        Returns:
            int: Sample offset for optimal stimulation timing
            float: Confidence score for the timing estimate (0-1)
        """
        if channel_weights is None:
            channel_weights = np.ones(self.n_channels) / self.n_channels
            
        # Apply phase offset to target phase
        target_phase = (target_phase + self.phase_offset) % 360
        
        # Extract phases for all channels
        current_phases = self.extract_phase()
        
        # Estimate phase progression rate (degrees per sample)
        # For each frequency band, use the center frequency
        center_freq = sum(self.band_range) / 2
        phase_step = center_freq * 360 / self.sampling_rate
        
        # Calculate predicted phase differences over the window
        phase_differences = np.zeros(window_size)
        confidences = np.zeros(window_size)
        
        for i in range(window_size):
            # Predict phases after i samples
            predicted_phases = (current_phases + i * phase_step) % 360
            
            # Calculate circular distance to target phase
            phase_distances = np.minimum(
                np.abs(predicted_phases - target_phase),
                360 - np.abs(predicted_phases - target_phase)
            )
            
            # Weighted average phase difference
            phase_differences[i] = np.sum(phase_distances * channel_weights)
            
            # Confidence based on phase coherence
            phase_vectors = np.exp(1j * np.radians(predicted_phases))
            mean_vector = np.sum(phase_vectors * channel_weights[:, np.newaxis], axis=0)
            coherence = np.abs(mean_vector) / np.sum(channel_weights)
            confidences[i] = coherence
        
        # Find the time point with minimum phase difference and high confidence
        combined_score = phase_differences * (1 - confidences)
        optimal_sample = np.argmin(combined_score)
        
        # Calculate final confidence score (0-1)
        confidence_score = 1 - (combined_score[optimal_sample] / 360)
        
        return optimal_sample, confidence_score
    
    def set_phase_offset(self, offset_degrees):
        """
        Set the phase offset for stimulation.
        
        Args:
            offset_degrees (float): Phase offset in degrees (0-360)
        """
        self.phase_offset = offset_degrees % 360
