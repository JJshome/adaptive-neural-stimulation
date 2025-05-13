import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import threading
import time
from collections import deque
from scipy import signal
import datetime

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class FeedbackMetric:
    """Base class for neural feedback metrics."""
    
    def __init__(self, name: str, target_range: Tuple[float, float],
                weight: float = 1.0, time_window: float = 5.0):
        """
        Initialize the feedback metric.
        
        Args:
            name: Unique name of the metric
            target_range: Tuple of (min_value, max_value) for optimal range
            weight: Weight of this metric in the overall feedback score (0-1)
            time_window: Time window in seconds for metric calculation
        """
        self.name = name
        self.target_range = target_range
        self.weight = weight
        self.time_window = time_window
        self.current_value = 0.0
        self.history = deque(maxlen=100)  # Store last 100 values
        self.timestamps = deque(maxlen=100)  # Timestamps for history values
        self.is_within_target = False
        
    def update(self, data: np.ndarray, sampling_rate: float) -> float:
        """
        Update the metric with new data.
        
        Args:
            data: New neural data
            sampling_rate: Sampling rate of the data in Hz
            
        Returns:
            Current metric value
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement update method")
        
    def calculate_score(self) -> float:
        """
        Calculate a normalized score (0-1) based on current value and target range.
        Higher score means closer to target range.
        
        Returns:
            Normalized score (0-1)
        """
        min_target, max_target = self.target_range
        
        # If within target range, score is 1.0
        if min_target <= self.current_value <= max_target:
            self.is_within_target = True
            return 1.0
            
        self.is_within_target = False
        
        # Calculate how far outside the range we are
        if self.current_value < min_target:
            # Below target range
            # Normalize distance from 0 to min_target
            distance = (min_target - self.current_value) / min_target if min_target > 0 else min_target - self.current_value
            return max(0, 1.0 - distance)
        else:
            # Above target range
            # Normalize distance from max_target to max_target*2
            range_size = max_target - min_target
            distance = (self.current_value - max_target) / range_size if range_size > 0 else self.current_value - max_target
            return max(0, 1.0 - distance)
            
    def log_value(self):
        """Log the current value and timestamp to history."""
        self.history.append(self.current_value)
        self.timestamps.append(datetime.datetime.now())
        
    def get_trend(self, window: int = 10) -> float:
        """
        Calculate the trend over the last window values.
        Positive value indicates increasing trend, negative indicates decreasing.
        
        Args:
            window: Number of recent values to consider
            
        Returns:
            Trend coefficient
        """
        if len(self.history) < window:
            return 0.0
            
        # Get last window values
        recent_values = list(self.history)[-window:]
        x = np.arange(len(recent_values))
        
        # Linear regression
        if np.std(recent_values) > 0:  # Avoid division by zero
            # Calculate correlation coefficient
            correlation = np.corrcoef(x, recent_values)[0, 1]
            return correlation
        else:
            return 0.0
            
    def __str__(self) -> str:
        """String representation of the metric."""
        return (f"{self.name}: {self.current_value:.3f} "
              f"(target: {self.target_range[0]:.1f}-{self.target_range[1]:.1f}, "
              f"{'within target' if self.is_within_target else 'outside target'})")


class BandPowerMetric(FeedbackMetric):
    """Metric for monitoring power in a specific frequency band."""
    
    def __init__(self, name: str, band: Tuple[float, float],
                target_range: Tuple[float, float], channels: List[int] = None,
                weight: float = 1.0, time_window: float = 5.0,
                relative_to_total: bool = False):
        """
        Initialize the band power metric.
        
        Args:
            name: Unique name of the metric
            band: Tuple of (low_freq, high_freq) in Hz
            target_range: Tuple of (min_value, max_value) for optimal range
            channels: List of channel indices to include (None for all)
            weight: Weight of this metric in the overall feedback score (0-1)
            time_window: Time window in seconds for metric calculation
            relative_to_total: Whether to calculate relative power (band/total)
        """
        super().__init__(name, target_range, weight, time_window)
        self.band = band
        self.channels = channels
        self.relative_to_total = relative_to_total
        
    def update(self, data: np.ndarray, sampling_rate: float) -> float:
        """
        Update the metric with new EEG data.
        
        Args:
            data: EEG data with shape (n_channels, n_samples)
            sampling_rate: Sampling rate of the data in Hz
            
        Returns:
            Current band power value
        """
        # Select channels if specified
        if self.channels is not None:
            channel_data = data[self.channels, :]
        else:
            channel_data = data
            
        # Calculate number of samples for window
        window_samples = int(self.time_window * sampling_rate)
        
        # Use the most recent window_samples
        if channel_data.shape[1] > window_samples:
            channel_data = channel_data[:, -window_samples:]
            
        # Calculate power spectrum
        fft_size = min(4096, int(2 ** np.ceil(np.log2(channel_data.shape[1]))))
        frequencies, psd = signal.welch(channel_data, sampling_rate, nperseg=fft_size)
        
        # Find indices for the specified frequency band
        band_indices = np.logical_and(frequencies >= self.band[0], frequencies <= self.band[1])
        
        # Calculate band power (average across channels and frequencies)
        band_power = np.mean(np.mean(psd[:, band_indices], axis=0))
        
        if self.relative_to_total:
            # Calculate total power and then relative power
            total_power = np.mean(np.mean(psd, axis=0))
            if total_power > 0:  # Avoid division by zero
                band_power = band_power / total_power
            else:
                band_power = 0.0
                
        # Update current value and history
        self.current_value = band_power
        self.log_value()
        
        return self.current_value


class BandRatioMetric(FeedbackMetric):
    """Metric for monitoring the ratio between powers in two frequency bands."""
    
    def __init__(self, name: str, band1: Tuple[float, float], band2: Tuple[float, float],
                target_range: Tuple[float, float], channels: List[int] = None,
                weight: float = 1.0, time_window: float = 5.0):
        """
        Initialize the band ratio metric.
        
        Args:
            name: Unique name of the metric
            band1: Tuple of (low_freq, high_freq) for numerator band
            band2: Tuple of (low_freq, high_freq) for denominator band
            target_range: Tuple of (min_value, max_value) for optimal range
            channels: List of channel indices to include (None for all)
            weight: Weight of this metric in the overall feedback score (0-1)
            time_window: Time window in seconds for metric calculation
        """
        super().__init__(name, target_range, weight, time_window)
        self.band1 = band1  # Numerator band
        self.band2 = band2  # Denominator band
        self.channels = channels
        
    def update(self, data: np.ndarray, sampling_rate: float) -> float:
        """
        Update the metric with new EEG data.
        
        Args:
            data: EEG data with shape (n_channels, n_samples)
            sampling_rate: Sampling rate of the data in Hz
            
        Returns:
            Current band ratio value
        """
        # Select channels if specified
        if self.channels is not None:
            channel_data = data[self.channels, :]
        else:
            channel_data = data
            
        # Calculate number of samples for window
        window_samples = int(self.time_window * sampling_rate)
        
        # Use the most recent window_samples
        if channel_data.shape[1] > window_samples:
            channel_data = channel_data[:, -window_samples:]
            
        # Calculate power spectrum
        fft_size = min(4096, int(2 ** np.ceil(np.log2(channel_data.shape[1]))))
        frequencies, psd = signal.welch(channel_data, sampling_rate, nperseg=fft_size)
        
        # Find indices for the two frequency bands
        band1_indices = np.logical_and(frequencies >= self.band1[0], frequencies <= self.band1[1])
        band2_indices = np.logical_and(frequencies >= self.band2[0], frequencies <= self.band2[1])
        
        # Calculate band powers (average across channels and frequencies)
        band1_power = np.mean(np.mean(psd[:, band1_indices], axis=0))
        band2_power = np.mean(np.mean(psd[:, band2_indices], axis=0))
        
        # Calculate ratio (avoid division by zero)
        if band2_power > 0:
            ratio = band1_power / band2_power
        else:
            ratio = 0.0
            
        # Update current value and history
        self.current_value = ratio
        self.log_value()
        
        return self.current_value


class PhaseLockingMetric(FeedbackMetric):
    """Metric for monitoring phase locking value between brain regions."""
    
    def __init__(self, name: str, channel_pair: Tuple[int, int], band: Tuple[float, float],
                target_range: Tuple[float, float], weight: float = 1.0, time_window: float = 5.0):
        """
        Initialize the phase locking metric.
        
        Args:
            name: Unique name of the metric
            channel_pair: Tuple of (channel1, channel2) indices
            band: Tuple of (low_freq, high_freq) in Hz for the band to analyze
            target_range: Tuple of (min_value, max_value) for optimal range
            weight: Weight of this metric in the overall feedback score (0-1)
            time_window: Time window in seconds for metric calculation
        """
        super().__init__(name, target_range, weight, time_window)
        self.channel_pair = channel_pair
        self.band = band
        
    def update(self, data: np.ndarray, sampling_rate: float) -> float:
        """
        Update the metric with new EEG data.
        
        Args:
            data: EEG data with shape (n_channels, n_samples)
            sampling_rate: Sampling rate of the data in Hz
            
        Returns:
            Current phase locking value
        """
        ch1, ch2 = self.channel_pair
        
        # Check if channels are valid
        if ch1 >= data.shape[0] or ch2 >= data.shape[0]:
            logger.error(f"Invalid channel indices: {ch1}, {ch2}. Data has {data.shape[0]} channels.")
            return 0.0
            
        # Calculate number of samples for window
        window_samples = int(self.time_window * sampling_rate)
        
        # Use the most recent window_samples
        if data.shape[1] > window_samples:
            signal1 = data[ch1, -window_samples:]
            signal2 = data[ch2, -window_samples:]
        else:
            signal1 = data[ch1, :]
            signal2 = data[ch2, :]
            
        # Filter signals to the specified frequency band
        nyquist = 0.5 * sampling_rate
        low = self.band[0] / nyquist
        high = self.band[1] / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        signal1_filtered = signal.filtfilt(b, a, signal1)
        signal2_filtered = signal.filtfilt(b, a, signal2)
        
        # Compute analytic signal (using Hilbert transform)
        analytic_signal1 = signal.hilbert(signal1_filtered)
        analytic_signal2 = signal.hilbert(signal2_filtered)
        
        # Get instantaneous phase
        phase1 = np.angle(analytic_signal1)
        phase2 = np.angle(analytic_signal2)
        
        # Compute phase difference
        phase_diff = phase1 - phase2
        
        # Compute PLV
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        # Update current value and history
        self.current_value = plv
        self.log_value()
        
        return self.current_value


class CoherenceMetric(FeedbackMetric):
    """Metric for monitoring coherence between brain regions."""
    
    def __init__(self, name: str, channel_pair: Tuple[int, int], band: Tuple[float, float],
                target_range: Tuple[float, float], weight: float = 1.0, time_window: float = 5.0):
        """
        Initialize the coherence metric.
        
        Args:
            name: Unique name of the metric
            channel_pair: Tuple of (channel1, channel2) indices
            band: Tuple of (low_freq, high_freq) in Hz for the band to analyze
            target_range: Tuple of (min_value, max_value) for optimal range
            weight: Weight of this metric in the overall feedback score (0-1)
            time_window: Time window in seconds for metric calculation
        """
        super().__init__(name, target_range, weight, time_window)
        self.channel_pair = channel_pair
        self.band = band
        
    def update(self, data: np.ndarray, sampling_rate: float) -> float:
        """
        Update the metric with new EEG data.
        
        Args:
            data: EEG data with shape (n_channels, n_samples)
            sampling_rate: Sampling rate of the data in Hz
            
        Returns:
            Current coherence value
        """
        ch1, ch2 = self.channel_pair
        
        # Check if channels are valid
        if ch1 >= data.shape[0] or ch2 >= data.shape[0]:
            logger.error(f"Invalid channel indices: {ch1}, {ch2}. Data has {data.shape[0]} channels.")
            return 0.0
            
        # Calculate number of samples for window
        window_samples = int(self.time_window * sampling_rate)
        
        # Use the most recent window_samples
        if data.shape[1] > window_samples:
            signal1 = data[ch1, -window_samples:]
            signal2 = data[ch2, -window_samples:]
        else:
            signal1 = data[ch1, :]
            signal2 = data[ch2, :]
            
        # Compute coherence
        fft_size = min(1024, int(2 ** np.ceil(np.log2(len(signal1)))))
        f, Cxy = signal.coherence(signal1, signal2, fs=sampling_rate, nperseg=fft_size)
        
        # Find indices for the specified frequency band
        band_indices = np.logical_and(f >= self.band[0], f <= self.band[1])
        
        # Calculate average coherence in the band
        coherence = np.mean(Cxy[band_indices])
        
        # Update current value and history
        self.current_value = coherence
        self.log_value()
        
        return self.current_value


class EMGActivityMetric(FeedbackMetric):
    """Metric for monitoring muscle activity from EMG data."""
    
    def __init__(self, name: str, target_range: Tuple[float, float],
                channels: List[int] = None, weight: float = 1.0,
                time_window: float = 2.0, activity_threshold: float = 10.0):
        """
        Initialize the EMG activity metric.
        
        Args:
            name: Unique name of the metric
            target_range: Tuple of (min_value, max_value) for optimal range
            channels: List of channel indices to include (None for all)
            weight: Weight of this metric in the overall feedback score (0-1)
            time_window: Time window in seconds for metric calculation
            activity_threshold: Threshold for detecting muscle activation
        """
        super().__init__(name, target_range, weight, time_window)
        self.channels = channels
        self.activity_threshold = activity_threshold
        self.activation_count = 0
        
    def update(self, data: np.ndarray, sampling_rate: float) -> float:
        """
        Update the metric with new EMG data.
        
        Args:
            data: EMG data with shape (n_channels, n_samples)
            sampling_rate: Sampling rate of the data in Hz
            
        Returns:
            Current EMG activity level
        """
        # Select channels if specified
        if self.channels is not None:
            channel_data = data[self.channels, :]
        else:
            channel_data = data
            
        # Calculate number of samples for window
        window_samples = int(self.time_window * sampling_rate)
        
        # Use the most recent window_samples
        if channel_data.shape[1] > window_samples:
            channel_data = channel_data[:, -window_samples:]
            
        # Rectify the signal
        rectified = np.abs(channel_data)
        
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(np.power(rectified, 2), axis=1))
        
        # Calculate mean activity across channels
        activity = np.mean(rms)
        
        # Count activations (when activity exceeds threshold)
        if activity > self.activity_threshold:
            self.activation_count += 1
        
        # Update current value and history
        self.current_value = activity
        self.log_value()
        
        return self.current_value


class NeuralFeedbackSystem:
    """
    System for real-time neural feedback monitoring and stimulation adaptation.
    """
    
    def __init__(self, sampling_rate: float = 250.0):
        """
        Initialize the neural feedback system.
        
        Args:
            sampling_rate: Sampling rate of the input data in Hz
        """
        self.sampling_rate = sampling_rate
        self.metrics = {}  # Dict of metrics by name
        self.feedback_thread = None
        self.is_running = False
        self.data_buffer = None
        self.buffer_lock = threading.Lock()
        self.update_interval = 0.5  # seconds
        self.last_update_time = 0.0
        self.overall_score = 0.0
        self.score_history = deque(maxlen=100)
        self.score_timestamps = deque(maxlen=100)
        self.adaptation_callbacks = []
        
    def add_metric(self, metric: FeedbackMetric) -> None:
        """
        Add a feedback metric to the system.
        
        Args:
            metric: FeedbackMetric instance
        """
        self.metrics[metric.name] = metric
        logger.info(f"Added metric '{metric.name}' to feedback system")
        
    def remove_metric(self, metric_name: str) -> bool:
        """
        Remove a feedback metric from the system.
        
        Args:
            metric_name: Name of the metric to remove
            
        Returns:
            True if metric was removed, False if not found
        """
        if metric_name in self.metrics:
            del self.metrics[metric_name]
            logger.info(f"Removed metric '{metric_name}' from feedback system")
            return True
        else:
            logger.warning(f"Metric '{metric_name}' not found in feedback system")
            return False
            
    def start(self) -> bool:
        """
        Start the feedback system.
        
        Returns:
            True if system started successfully, False if already running
        """
        if self.is_running:
            logger.warning("Feedback system is already running")
            return False
            
        # Initialize buffer if needed
        if self.data_buffer is None:
            # Create a small initial buffer (will grow as data comes in)
            self.data_buffer = np.zeros((1, int(self.sampling_rate * 0.1)))
            
        self.is_running = True
        self.feedback_thread = threading.Thread(
            target=self._feedback_loop,
            daemon=True
        )
        self.feedback_thread.start()
        
        logger.info("Neural feedback system started")
        return True
        
    def stop(self) -> bool:
        """
        Stop the feedback system.
        
        Returns:
            True if system stopped successfully, False if not running
        """
        if not self.is_running:
            logger.warning("Feedback system is not running")
            return False
            
        self.is_running = False
        if self.feedback_thread and self.feedback_thread.is_alive():
            self.feedback_thread.join(timeout=2.0)
            
        logger.info("Neural feedback system stopped")
        return True
        
    def update(self, data: np.ndarray) -> float:
        """
        Update the system with new neural data.
        
        Args:
            data: Neural data with shape (n_channels, n_samples)
            
        Returns:
            Current overall feedback score (0-1)
        """
        current_time = time.time()
        
        # Update data buffer
        with self.buffer_lock:
            if self.data_buffer is None:
                self.data_buffer = data
            else:
                # Check if channel dimensions match
                if self.data_buffer.shape[0] != data.shape[0]:
                    logger.error(f"Data channel mismatch: {self.data_buffer.shape[0]} vs {data.shape[0]}")
                    return self.overall_score
                    
                # Append new data to buffer
                self.data_buffer = np.hstack((self.data_buffer, data))
                
                # Limit buffer size to 30 seconds
                max_samples = int(30 * self.sampling_rate)
                if self.data_buffer.shape[1] > max_samples:
                    self.data_buffer = self.data_buffer[:, -max_samples:]
        
        # Only update metrics if enough time has passed
        if current_time - self.last_update_time >= self.update_interval:
            self._update_metrics()
            self.last_update_time = current_time
            
        return self.overall_score
        
    def _update_metrics(self) -> None:
        """Update all metrics with current data and calculate overall score."""
        if not self.metrics:
            logger.warning("No metrics defined in feedback system")
            return
            
        # Make a copy of the data buffer to avoid threading issues
        with self.buffer_lock:
            if self.data_buffer is None or self.data_buffer.size == 0:
                return
                
            data_copy = self.data_buffer.copy()
            
        # Update each metric
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for name, metric in self.metrics.items():
            # Update metric with current data
            metric.update(data_copy, self.sampling_rate)
            
            # Calculate weighted score
            score = metric.calculate_score()
            weighted_score = score * metric.weight
            
            total_weighted_score += weighted_score
            total_weight += metric.weight
            
            logger.debug(f"Metric '{name}' updated: value={metric.current_value:.3f}, score={score:.3f}")
            
        # Calculate overall score
        if total_weight > 0:
            self.overall_score = total_weighted_score / total_weight
        else:
            self.overall_score = 0.0
            
        # Log overall score and timestamp
        self.score_history.append(self.overall_score)
        self.score_timestamps.append(datetime.datetime.now())
        
        logger.debug(f"Overall feedback score: {self.overall_score:.3f}")
        
        # Call adaptation callbacks
        self._trigger_adaptations()
        
    def _feedback_loop(self) -> None:
        """Background loop for continuous feedback processing."""
        logger.info("Feedback processing loop started")
        
        while self.is_running:
            try:
                # Update metrics with current data
                self._update_metrics()
                
                # Sleep for update interval
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in feedback loop: {str(e)}")
                
        logger.info("Feedback processing loop stopped")
        
    def add_adaptation_callback(self, callback: callable, threshold: float = 0.0,
                              direction: str = 'below', metric_name: str = None) -> None:
        """
        Add a callback function to be triggered when adaptation is needed.
        
        Args:
            callback: Function to call when adaptation is needed
            threshold: Score threshold for triggering adaptation
            direction: 'below', 'above', or 'change' to specify when to trigger
            metric_name: Name of specific metric to monitor (None for overall score)
        """
        if direction not in ['below', 'above', 'change']:
            raise ValueError("Direction must be 'below', 'above', or 'change'")
            
        self.adaptation_callbacks.append({
            'callback': callback,
            'threshold': threshold,
            'direction': direction,
            'metric_name': metric_name,
            'last_value': None
        })
        
        logger.info(f"Added adaptation callback for {'overall score' if metric_name is None else f'metric {metric_name}'}")
        
    def _trigger_adaptations(self) -> None:
        """Check conditions and trigger adaptation callbacks as needed."""
        for callback_info in self.adaptation_callbacks:
            # Get the value to check (specific metric or overall score)
            if callback_info['metric_name'] is not None:
                if callback_info['metric_name'] in self.metrics:
                    current_value = self.metrics[callback_info['metric_name']].current_value
                else:
                    logger.warning(f"Metric '{callback_info['metric_name']}' not found for adaptation callback")
                    continue
            else:
                current_value = self.overall_score
                
            # Check if adaptation should be triggered
            trigger = False
            
            if callback_info['direction'] == 'below':
                trigger = current_value < callback_info['threshold']
            elif callback_info['direction'] == 'above':
                trigger = current_value > callback_info['threshold']
            elif callback_info['direction'] == 'change':
                # Trigger on significant change from last value
                if callback_info['last_value'] is not None:
                    change = abs(current_value - callback_info['last_value'])
                    trigger = change > callback_info['threshold']
                    
            # Update last value
            callback_info['last_value'] = current_value
            
            # Trigger callback if needed
            if trigger:
                try:
                    # Call with current value and all metrics data
                    metrics_data = {name: metric.current_value for name, metric in self.metrics.items()}
                    callback_info['callback'](current_value, metrics_data)
                except Exception as e:
                    logger.error(f"Error in adaptation callback: {str(e)}")
                    
    def get_metric_status(self) -> Dict[str, Dict]:
        """
        Get status information for all metrics.
        
        Returns:
            Dictionary mapping metric names to status dictionaries
        """
        status = {}
        for name, metric in self.metrics.items():
            status[name] = {
                'current_value': metric.current_value,
                'target_range': metric.target_range,
                'is_within_target': metric.is_within_target,
                'score': metric.calculate_score(),
                'weight': metric.weight,
                'trend': metric.get_trend()
            }
        return status
        
    def get_overall_status(self) -> Dict:
        """
        Get overall status of the feedback system.
        
        Returns:
            Dictionary with system status information
        """
        metrics_within_target = sum(1 for metric in self.metrics.values() if metric.is_within_target)
        total_metrics = len(self.metrics)
        
        # Calculate trend of overall score
        score_trend = 0.0
        if len(self.score_history) >= 10:
            recent_scores = list(self.score_history)[-10:]
            x = np.arange(len(recent_scores))
            if np.std(recent_scores) > 0:
                score_trend = np.corrcoef(x, recent_scores)[0, 1]
                
        return {
            'overall_score': self.overall_score,
            'metrics_count': total_metrics,
            'metrics_within_target': metrics_within_target,
            'score_trend': score_trend,
            'is_running': self.is_running
        }


class AdaptiveStimulationController:
    """
    Controller for adapting stimulation parameters based on neural feedback.
    """
    
    def __init__(self, feedback_system: NeuralFeedbackSystem):
        """
        Initialize the adaptive stimulation controller.
        
        Args:
            feedback_system: NeuralFeedbackSystem instance
        """
        self.feedback_system = feedback_system
        self.stimulation_devices = {}  # Dict of stimulation device objects by ID
        self.current_parameters = {}  # Current stimulation parameters by device ID
        self.adaptation_rules = []  # List of adaptation rules
        self.parameter_limits = {
            'amplitude': (0.1, 5.0),  # mA
            'frequency': (0.1, 100.0),  # Hz
            'phase': (0.0, 360.0),  # degrees
        }
        self.adaptation_history = deque(maxlen=100)  # History of parameter adaptations
        
        # Register callbacks with feedback system
        self._register_callbacks()
        
    def add_stimulation_device(self, device_id: str, device_obj: Any) -> None:
        """
        Add a stimulation device to the controller.
        
        Args:
            device_id: Unique identifier for the device
            device_obj: Device object with set_parameters method
        """
        if not hasattr(device_obj, 'set_parameters'):
            raise ValueError("Device object must have set_parameters method")
            
        self.stimulation_devices[device_id] = device_obj
        self.current_parameters[device_id] = {}
        
        logger.info(f"Added stimulation device '{device_id}' to controller")
        
    def remove_stimulation_device(self, device_id: str) -> bool:
        """
        Remove a stimulation device from the controller.
        
        Args:
            device_id: ID of the device to remove
            
        Returns:
            True if device was removed, False if not found
        """
        if device_id in self.stimulation_devices:
            del self.stimulation_devices[device_id]
            
            if device_id in self.current_parameters:
                del self.current_parameters[device_id]
                
            logger.info(f"Removed stimulation device '{device_id}' from controller")
            return True
        else:
            logger.warning(f"Device '{device_id}' not found in controller")
            return False
            
    def set_initial_parameters(self, device_id: str, parameters: Dict) -> bool:
        """
        Set initial stimulation parameters for a device.
        
        Args:
            device_id: ID of the stimulation device
            parameters: Dictionary of parameter values
            
        Returns:
            True if parameters set successfully, False otherwise
        """
        if device_id not in self.stimulation_devices:
            logger.error(f"Device '{device_id}' not found in controller")
            return False
            
        # Apply parameter limits
        for param, value in parameters.items():
            if param in self.parameter_limits:
                min_val, max_val = self.parameter_limits[param]
                parameters[param] = max(min_val, min(max_val, value))
                
        # Set parameters on device
        device = self.stimulation_devices[device_id]
        success = device.set_parameters(**parameters)
        
        if success:
            # Store current parameters
            self.current_parameters[device_id] = parameters.copy()
            logger.info(f"Set initial parameters for device '{device_id}': {parameters}")
            
        return success
        
    def add_adaptation_rule(self, rule: Dict) -> None:
        """
        Add a rule for parameter adaptation.
        
        Args:
            rule: Dictionary defining the adaptation rule with fields:
                - metric_name: Name of the metric to monitor
                - condition: 'above', 'below', or 'change'
                - threshold: Threshold value for the condition
                - parameter: Parameter to adapt ('amplitude', 'frequency', etc.)
                - device_id: ID of the device to adapt
                - adjustment: Absolute or relative adjustment value
                - adjustment_type: 'absolute' or 'relative'
                - max_adjustment: Maximum total adjustment allowed
        """
        required_fields = ['metric_name', 'condition', 'threshold', 'parameter', 
                         'device_id', 'adjustment', 'adjustment_type']
                         
        # Check required fields
        for field in required_fields:
            if field not in rule:
                raise ValueError(f"Missing required field '{field}' in adaptation rule")
                
        # Validate condition
        if rule['condition'] not in ['above', 'below', 'change']:
            raise ValueError("Condition must be 'above', 'below', or 'change'")
            
        # Validate adjustment type
        if rule['adjustment_type'] not in ['absolute', 'relative']:
            raise ValueError("Adjustment type must be 'absolute' or 'relative'")
            
        # Add default max_adjustment if not specified
        if 'max_adjustment' not in rule:
            rule['max_adjustment'] = float('inf')
            
        # Add tracking fields
        rule['total_adjustment'] = 0.0
        rule['last_value'] = None
        
        # Add rule
        self.adaptation_rules.append(rule)
        logger.info(f"Added adaptation rule: {rule}")
        
    def _register_callbacks(self) -> None:
        """Register callbacks with the feedback system."""
        # Register global callback for all metrics
        self.feedback_system.add_adaptation_callback(
            callback=self._adapt_parameters,
            threshold=0.0,  # Will be checked in the callback
            direction='change',  # Trigger on any change
            metric_name=None  # Monitor all metrics
        )
        
    def _adapt_parameters(self, current_value: float, metrics_data: Dict) -> None:
        """
        Callback function for adapting parameters based on neural feedback.
        
        Args:
            current_value: Current overall score or metric value
            metrics_data: Dictionary mapping metric names to current values
        """
        # Apply each adaptation rule
        for rule in self.adaptation_rules:
            try:
                self._apply_rule(rule, metrics_data)
            except Exception as e:
                logger.error(f"Error applying adaptation rule: {str(e)}")
                
    def _apply_rule(self, rule: Dict, metrics_data: Dict) -> None:
        """
        Apply a single adaptation rule.
        
        Args:
            rule: Adaptation rule dictionary
            metrics_data: Dictionary mapping metric names to current values
        """
        # Get current metric value
        metric_name = rule['metric_name']
        if metric_name not in metrics_data:
            logger.warning(f"Metric '{metric_name}' not found for adaptation rule")
            return
            
        current_value = metrics_data[metric_name]
        
        # Skip if device not in controller
        device_id = rule['device_id']
        if device_id not in self.stimulation_devices:
            logger.warning(f"Device '{device_id}' not found for adaptation rule")
            return
            
        # Check if parameter exists for the device
        parameter = rule['parameter']
        if parameter not in self.current_parameters.get(device_id, {}):
            logger.warning(f"Parameter '{parameter}' not found for device '{device_id}'")
            return
            
        # Check condition
        trigger = False
        
        if rule['condition'] == 'below':
            trigger = current_value < rule['threshold']
        elif rule['condition'] == 'above':
            trigger = current_value > rule['threshold']
        elif rule['condition'] == 'change':
            # Trigger on significant change from last value
            if rule['last_value'] is not None:
                change = abs(current_value - rule['last_value'])
                trigger = change > rule['threshold']
                
        # Update last value
        rule['last_value'] = current_value
        
        # Apply adaptation if triggered
        if trigger:
            self._adjust_parameter(rule)
            
    def _adjust_parameter(self, rule: Dict) -> None:
        """
        Adjust a stimulation parameter according to a rule.
        
        Args:
            rule: Adaptation rule dictionary
        """
        device_id = rule['device_id']
        parameter = rule['parameter']
        adjustment = rule['adjustment']
        adjustment_type = rule['adjustment_type']
        
        # Get current parameter value
        current_value = self.current_parameters[device_id][parameter]
        
        # Calculate new value
        if adjustment_type == 'absolute':
            new_value = current_value + adjustment
        else:  # relative
            new_value = current_value * (1.0 + adjustment)
            
        # Apply limits
        if parameter in self.parameter_limits:
            min_val, max_val = self.parameter_limits[parameter]
            new_value = max(min_val, min(max_val, new_value))
            
        # Check max_adjustment constraint
        actual_adjustment = new_value - current_value
        potential_total = abs(rule['total_adjustment'] + actual_adjustment)
        
        if potential_total > rule['max_adjustment']:
            # Scale down adjustment to stay within limit
            if actual_adjustment > 0:
                max_allowed = rule['max_adjustment'] - rule['total_adjustment']
                new_value = current_value + max_allowed
            else:
                max_allowed = rule['max_adjustment'] + rule['total_adjustment']
                new_value = current_value - max_allowed
                
        # Apply adjustment
        actual_adjustment = new_value - current_value
        rule['total_adjustment'] += actual_adjustment
        
        # Update current parameters
        self.current_parameters[device_id][parameter] = new_value
        
        # Apply to device
        device = self.stimulation_devices[device_id]
        params = {parameter: new_value}
        success = device.set_parameters(**params)
        
        if success:
            # Log adaptation
            adaptation_info = {
                'timestamp': datetime.datetime.now(),
                'device_id': device_id,
                'parameter': parameter,
                'old_value': current_value,
                'new_value': new_value,
                'adjustment': actual_adjustment,
                'rule': rule['metric_name'] + ' ' + rule['condition'],
                'metric_value': rule['last_value']
            }
            
            self.adaptation_history.append(adaptation_info)
            
            logger.info(f"Adapted parameter '{parameter}' for device '{device_id}': "
                      f"{current_value:.3f} -> {new_value:.3f} ({actual_adjustment:+.3f})")
