"""
Neural Plasticity Window Detector

This module implements algorithms for detecting the optimal time windows for neural
plasticity, enabling precise timing of stimulation to maximize therapeutic effects.

The module analyzes various biomarkers including:
- EEG patterns associated with heightened plasticity
- Heart rate variability as an indicator of parasympathetic activity
- Circadian rhythm information
- Sleep stage and post-sleep timing
"""

import numpy as np
from scipy import signal
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class PlasticityWindowDetector:
    """
    Class for detecting and predicting optimal time windows for 
    neuroplasticity-enhancing stimulation.
    """
    
    def __init__(self, sampling_rate=250, history_duration=14):
        """
        Initialize the plasticity window detector.
        
        Args:
            sampling_rate (int): Biosignal sampling rate in Hz
            history_duration (int): Duration in days for historical data analysis
        """
        self.sampling_rate = sampling_rate
        self.history_duration = history_duration
        
        # Initialize feature extractors
        self._init_eeg_features()
        self._init_physio_features()
        self._init_temporal_features()
        
        # Initialize model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Storage for historical data
        self.history = pd.DataFrame(columns=[
            'timestamp', 'alpha_theta_ratio', 'beta_gamma_ratio',
            'hrv_hf', 'circadian_phase', 'time_since_sleep',
            'plasticity_score'
        ])
        
        # Current state
        self.current_state = {
            'alpha_theta_ratio': None,
            'beta_gamma_ratio': None,
            'hrv_hf': None,
            'circadian_phase': None,
            'time_since_sleep': None
        }
        
        # Default thresholds for high plasticity
        self.thresholds = {
            'plasticity_score': 0.7,  # 0-1 scale
            'confidence': 0.6         # 0-1 scale
        }
    
    def _init_eeg_features(self):
        """Initialize EEG feature extraction parameters."""
        # Frequency bands
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 80)
        }
        
        # Design filters
        self.filters = {}
        nyquist = self.sampling_rate / 2
        
        for band, (low, high) in self.bands.items():
            low_norm = low / nyquist
            high_norm = high / nyquist
            self.filters[band] = signal.butter(4, [low_norm, high_norm], btype='bandpass')
    
    def _init_physio_features(self):
        """Initialize physiological signal feature extraction parameters."""
        # HRV parameters
        self.hrv_window = 5 * 60  # 5 minutes in seconds
        self.rr_segment_size = 300  # Number of R-R intervals to analyze
        
        # GSR parameters
        self.gsr_smoothing_window = 5 * self.sampling_rate  # 5 seconds
    
    def _init_temporal_features(self):
        """Initialize temporal feature extraction parameters."""
        # Circadian parameters
        self.circadian_period = 24  # hours
        self.typical_sleep_duration = 8  # hours
        
        # Default sleep window if no sleep data available
        self.default_sleep_window = (22, 6)  # 10 PM to 6 AM
    
    def update_user_profile(self, sleep_schedule=None, chronotype=None):
        """
        Update user profile with personalized temporal information.
        
        Args:
            sleep_schedule (tuple, optional): Typical sleep start and end times (hour, hour)
            chronotype (str, optional): User chronotype ('early', 'intermediate', 'late')
        """
        if sleep_schedule is not None:
            self.default_sleep_window = sleep_schedule
        
        if chronotype is not None:
            # Adjust plasticity prediction based on chronotype
            if chronotype == 'early':
                # Early chronotypes have earlier plasticity peaks
                self.chronotype_offset = -2  # hours
            elif chronotype == 'late':
                # Late chronotypes have later plasticity peaks
                self.chronotype_offset = 2  # hours
            else:
                # Intermediate chronotype
                self.chronotype_offset = 0  # hours
    
    def extract_eeg_features(self, eeg_data):
        """
        Extract EEG features relevant for plasticity detection.
        
        Args:
            eeg_data (numpy.ndarray): EEG data with shape (n_channels, n_samples)
        
        Returns:
            dict: Dictionary of extracted features
        """
        n_channels = eeg_data.shape[0]
        features = {}
        
        # Calculate band powers for each channel
        band_powers = {band: np.zeros(n_channels) for band in self.bands}
        
        for ch in range(n_channels):
            for band, (b, a) in self.filters.items():
                # Apply bandpass filter
                filtered = signal.filtfilt(b, a, eeg_data[ch, :])
                
                # Calculate power
                band_powers[band][ch] = np.mean(filtered**2)
        
        # Average across channels
        avg_powers = {band: np.mean(powers) for band, powers in band_powers.items()}
        
        # Calculate plasticity-relevant ratios
        features['alpha_theta_ratio'] = avg_powers['alpha'] / avg_powers['theta']
        features['beta_gamma_ratio'] = avg_powers['beta'] / avg_powers['gamma']
        
        return features
    
    def extract_hrv_features(self, rr_intervals):
        """
        Extract heart rate variability features relevant for plasticity detection.
        
        Args:
            rr_intervals (numpy.ndarray): Array of R-R intervals in milliseconds
        
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        # Ensure enough data points
        if len(rr_intervals) < 3:
            return {'hrv_hf': None}
        
        # Calculate time domain HRV metrics
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
        
        # Calculate frequency domain HRV metrics
        # Resample RR intervals to regular time series
        rr_times = np.cumsum(rr_intervals) / 1000  # Convert to seconds
        rr_times = rr_times - rr_times[0]  # Start at 0
        
        # Create evenly sampled time series (4 Hz)
        fs = 4
        interpolated_time = np.arange(0, rr_times[-1], 1/fs)
        interpolated_rr = np.interp(interpolated_time, rr_times, rr_intervals)
        
        # Calculate power spectral density
        frequencies, psd = signal.welch(interpolated_rr, fs=fs, nperseg=len(interpolated_time)//2)
        
        # Extract high frequency power (parasympathetic activity indicator)
        hf_band = (0.15, 0.4)  # High frequency band (Hz)
        hf_mask = (frequencies >= hf_band[0]) & (frequencies <= hf_band[1])
        hf_power = np.trapz(psd[hf_mask], frequencies[hf_mask])
        
        features['hrv_hf'] = hf_power
        
        return features
    
    def calculate_circadian_phase(self, timestamp, sleep_data=None):
        """
        Calculate the current circadian phase based on timestamp and sleep history.
        
        Args:
            timestamp (datetime): Current timestamp
            sleep_data (dict, optional): Recent sleep data including:
                - last_sleep_start: datetime of last sleep onset
                - last_sleep_end: datetime of last wake time
                - sleep_quality: float between 0-1
        
        Returns:
            dict: Dictionary of circadian features
        """
        features = {}
        
        # Default calculation based on time of day
        hour = timestamp.hour + timestamp.minute / 60
        
        # Normalize to 0-1 scale where 0 is midnight
        circadian_phase = hour / 24
        
        # Adjust based on sleep data if available
        if sleep_data is not None:
            time_since_sleep = (timestamp - sleep_data['last_sleep_end']).total_seconds() / 3600
            features['time_since_sleep'] = time_since_sleep
            
            # Adjust phase based on actual sleep timing
            sleep_mid_point = sleep_data['last_sleep_start'] + \
                             (sleep_data['last_sleep_end'] - sleep_data['last_sleep_start']) / 2
            target_mid_point = timestamp.replace(hour=3, minute=0, second=0)  # 3 AM reference
            
            # Calculate phase shift based on difference from expected sleep midpoint
            phase_shift = (sleep_mid_point - target_mid_point).total_seconds() / (3600 * 24)
            circadian_phase = (circadian_phase - phase_shift) % 1
        else:
            # Estimate time since sleep based on default sleep window
            sleep_end_hour = self.default_sleep_window[1]
            current_hour = hour
            
            if current_hour >= sleep_end_hour:
                time_since_sleep = current_hour - sleep_end_hour
            else:
                time_since_sleep = current_hour + 24 - sleep_end_hour
                
            features['time_since_sleep'] = time_since_sleep
        
        features['circadian_phase'] = circadian_phase
        
        return features
    
    def update_features(self, eeg_data=None, rr_intervals=None, timestamp=None, sleep_data=None):
        """
        Update current state with new biosignal data.
        
        Args:
            eeg_data (numpy.ndarray, optional): EEG data
            rr_intervals (numpy.ndarray, optional): R-R intervals
            timestamp (datetime, optional): Current timestamp
            sleep_data (dict, optional): Sleep data
            
        Returns:
            dict: Updated feature state
        """
        # Set default timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update EEG features
        if eeg_data is not None:
            eeg_features = self.extract_eeg_features(eeg_data)
            self.current_state.update(eeg_features)
        
        # Update HRV features
        if rr_intervals is not None:
            hrv_features = self.extract_hrv_features(rr_intervals)
            self.current_state.update(hrv_features)
        
        # Update temporal features
        temporal_features = self.calculate_circadian_phase(timestamp, sleep_data)
        self.current_state.update(temporal_features)
        
        # Add timestamp
        self.current_state['timestamp'] = timestamp
        
        return self.current_state.copy()
    
    def predict_plasticity(self, features=None):
        """
        Predict current plasticity level based on features.
        
        Args:
            features (dict, optional): Feature dictionary. If None, use current state.
            
        Returns:
            float: Plasticity score (0-1)
            float: Confidence score (0-1)
        """
        if features is None:
            features = self.current_state
            
        # Simple heuristic model if not enough training data
        if len(self.history) < 10:
            score = self._heuristic_plasticity_score(features)
            confidence = 0.5  # Low confidence due to limited data
            return score, confidence
        
        # Use trained model if available
        try:
            # Prepare feature vector
            X = np.array([[
                features['alpha_theta_ratio'],
                features['beta_gamma_ratio'],
                features['hrv_hf'],
                features['circadian_phase'],
                features['time_since_sleep']
            ]])
            
            # Make prediction
            score = float(self.model.predict(X)[0])
            
            # Calculate confidence based on prediction variance
            predictions = []
            for estimator in self.model.estimators_:
                predictions.append(estimator.predict(X)[0])
            
            prediction_std = np.std(predictions)
            confidence = 1 - min(prediction_std / 0.5, 1.0)  # Scale to 0-1
            
            return score, confidence
            
        except Exception as e:
            print(f"Model prediction error: {e}")
            # Fallback to heuristic
            score = self._heuristic_plasticity_score(features)
            confidence = 0.4  # Lower confidence due to model failure
            return score, confidence
    
    def _heuristic_plasticity_score(self, features):
        """
        Calculate plasticity score using heuristic rules when model is unavailable.
        
        Args:
            features (dict): Feature dictionary
            
        Returns:
            float: Plasticity score (0-1)
        """
        score = 0.5  # Default score
        
        # Adjust based on alpha/theta ratio (higher ratio -> higher plasticity)
        if features['alpha_theta_ratio'] is not None:
            # Optimal range is typically 0.8-1.2
            alpha_theta_score = 1 - min(abs(features['alpha_theta_ratio'] - 1.0) / 0.5, 1.0)
            score += 0.2 * (alpha_theta_score - 0.5)
        
        # Adjust based on HRV (higher HF power -> higher plasticity)
        if features['hrv_hf'] is not None:
            # Normalize HRV-HF to 0-1 range (typical range: 100-1000 ms²)
            hrv_norm = min(features['hrv_hf'] / 1000, 1.0)
            score += 0.15 * (hrv_norm - 0.5)
        
        # Adjust based on time since sleep (peak at 60-90 minutes post-waking)
        if features['time_since_sleep'] is not None:
            time_score = 1 - min(abs(features['time_since_sleep'] - 1.25) / 5, 1.0)
            score += 0.25 * (time_score - 0.5)
        
        # Clamp to 0-1 range
        return max(0, min(1, score))
    
    def is_high_plasticity_window(self, threshold=None, confidence_threshold=None):
        """
        Determine if current state represents a high plasticity window.
        
        Args:
            threshold (float, optional): Plasticity score threshold (0-1)
            confidence_threshold (float, optional): Confidence threshold (0-1)
            
        Returns:
            bool: True if high plasticity window, False otherwise
            dict: Additional information including scores and confidence
        """
        if threshold is None:
            threshold = self.thresholds['plasticity_score']
            
        if confidence_threshold is None:
            confidence_threshold = self.thresholds['confidence']
            
        # Get current prediction
        score, confidence = self.predict_plasticity()
        
        # Determine if high plasticity window
        is_high = score >= threshold and confidence >= confidence_threshold
        
        return is_high, {
            'plasticity_score': score,
            'confidence': confidence,
            'threshold': threshold,
            'confidence_threshold': confidence_threshold
        }
    
    def predict_upcoming_windows(self, hours_ahead=24, interval_minutes=30):
        """
        Predict upcoming plasticity windows.
        
        Args:
            hours_ahead (int): Number of hours to predict ahead
            interval_minutes (int): Time interval between predictions in minutes
            
        Returns:
            list: List of dicts with predicted windows
        """
        results = []
        base_time = datetime.now()
        
        for minute in range(0, hours_ahead * 60, interval_minutes):
            future_time = base_time + timedelta(minutes=minute)
            
            # Create feature set based on expected future state
            future_features = self.current_state.copy()
            future_features.update(self.calculate_circadian_phase(future_time))
            
            # Make prediction
            score, confidence = self.predict_plasticity(future_features)
            
            # Add to results if above minimum confidence
            if confidence >= 0.4:
                results.append({
                    'timestamp': future_time,
                    'plasticity_score': score,
                    'confidence': confidence,
                    'is_high_plasticity': score >= self.thresholds['plasticity_score']
                })
        
        return results
    
    def train_model(self):
        """
        Train the plasticity prediction model using historical data.
        
        Returns:
            float: Model training score (R²)
        """
        if len(self.history) < 10:
            print("Insufficient training data, need at least 10 samples")
            return 0
        
        # Prepare training data
        X = self.history[['alpha_theta_ratio', 'beta_gamma_ratio', 'hrv_hf', 
                        'circadian_phase', 'time_since_sleep']].values
        y = self.history['plasticity_score'].values
        
        # Fit model
        self.model.fit(X, y)
        
        # Return score
        return self.model.score(X, y)
    
    def add_to_history(self, features, plasticity_score=None):
        """
        Add current feature set to history.
        
        Args:
            features (dict): Feature dictionary
            plasticity_score (float, optional): Known plasticity score if available
            
        Returns:
            int: New history length
        """
        # If plasticity score not provided, use predicted score
        if plasticity_score is None:
            plasticity_score, _ = self.predict_plasticity(features)
        
        # Create new row
        new_row = pd.DataFrame({
            'timestamp': [features['timestamp']],
            'alpha_theta_ratio': [features['alpha_theta_ratio']],
            'beta_gamma_ratio': [features['beta_gamma_ratio']],
            'hrv_hf': [features['hrv_hf']],
            'circadian_phase': [features['circadian_phase']],
            'time_since_sleep': [features['time_since_sleep']],
            'plasticity_score': [plasticity_score]
        })
        
        # Add to history
        self.history = pd.concat([self.history, new_row], ignore_index=True)
        
        # Trim history if needed
        history_limit = timedelta(days=self.history_duration)
        cutoff = datetime.now() - history_limit
        self.history = self.history[self.history['timestamp'] > cutoff]
        
        return len(self.history)
