import numpy as np
import scipy.signal as signal
from typing import Dict, List, Tuple, Optional, Union
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class SpatiotemporalPatternGenerator:
    """
    Class for generating complex spatiotemporal stimulation patterns
    that follow natural neural activation pathways.
    """
    
    def __init__(self, num_channels: int = 8, sampling_rate: float = 1000.0):
        """
        Initialize the spatiotemporal pattern generator.
        
        Args:
            num_channels: Number of stimulation channels
            sampling_rate: Sampling rate in Hz for temporal pattern generation
        """
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        
        # Default waveform parameters
        self.default_parameters = {
            'frequency': 10.0,  # Hz
            'amplitude': 1.0,   # Normalized amplitude (0-1)
            'phase': 0.0,       # Degrees (0-360)
            'duration': 1.0,    # Seconds
            'waveform': 'sine'  # Waveform type
        }
        
        # Available waveform types
        self.waveform_functions = {
            'sine': self._generate_sine,
            'square': self._generate_square,
            'triangle': self._generate_triangle,
            'sawtooth': self._generate_sawtooth,
            'gaussian_pulse': self._generate_gaussian_pulse,
            'chirp': self._generate_chirp,
            'gamma': self._generate_gamma,
            'burst': self._generate_burst
        }
        
        # Predefined neural pathways
        self.neural_pathways = {
            'motor': [0, 1, 2, 3, 4],           # Primary motor to spinal pathway
            'sensory': [5, 4, 3, 2, 1],         # Peripheral to sensory cortex pathway
            'cognitive': [6, 2, 7, 1, 5],       # Prefrontal to hippocampal pathway
            'language': [7, 6, 4, 5, 3],        # Broca to Wernicke pathway
            'visual': [7, 5, 3, 1, 0],          # Visual processing pathway
            'auditory': [6, 4, 2, 0, 3],        # Auditory processing pathway
            'attention': [7, 6, 3, 1, 4, 5],    # Attention network pathway
            'memory': [6, 7, 3, 5, 1, 2],       # Memory consolidation pathway
            'executive': [7, 6, 5, 3, 1, 0],    # Executive function pathway
            'emotion': [5, 4, 6, 7, 2, 3],      # Limbic system pathway
        }
        
    def _generate_sine(self, t: np.ndarray, frequency: float = 10.0, 
                     amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
        """
        Generate a sine wave.
        
        Args:
            t: Time array
            frequency: Frequency in Hz
            amplitude: Amplitude (0-1)
            phase: Phase offset in degrees
            
        Returns:
            Numpy array containing the sine wave
        """
        phase_rad = np.deg2rad(phase)
        return amplitude * np.sin(2 * np.pi * frequency * t + phase_rad)
    
    def _generate_square(self, t: np.ndarray, frequency: float = 10.0, 
                       amplitude: float = 1.0, phase: float = 0.0, 
                       duty: float = 0.5) -> np.ndarray:
        """
        Generate a square wave.
        
        Args:
            t: Time array
            frequency: Frequency in Hz
            amplitude: Amplitude (0-1)
            phase: Phase offset in degrees
            duty: Duty cycle (0-1)
            
        Returns:
            Numpy array containing the square wave
        """
        phase_rad = np.deg2rad(phase)
        return amplitude * signal.square(2 * np.pi * frequency * t + phase_rad, duty=duty)
    
    def _generate_triangle(self, t: np.ndarray, frequency: float = 10.0, 
                         amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
        """
        Generate a triangle wave.
        
        Args:
            t: Time array
            frequency: Frequency in Hz
            amplitude: Amplitude (0-1)
            phase: Phase offset in degrees
            
        Returns:
            Numpy array containing the triangle wave
        """
        phase_rad = np.deg2rad(phase)
        return amplitude * signal.sawtooth(2 * np.pi * frequency * t + phase_rad, width=0.5)
    
    def _generate_sawtooth(self, t: np.ndarray, frequency: float = 10.0, 
                         amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
        """
        Generate a sawtooth wave.
        
        Args:
            t: Time array
            frequency: Frequency in Hz
            amplitude: Amplitude (0-1)
            phase: Phase offset in degrees
            
        Returns:
            Numpy array containing the sawtooth wave
        """
        phase_rad = np.deg2rad(phase)
        return amplitude * signal.sawtooth(2 * np.pi * frequency * t + phase_rad)
    
    def _generate_gaussian_pulse(self, t: np.ndarray, frequency: float = 10.0, 
                               amplitude: float = 1.0, phase: float = 0.0, 
                               std: float = 0.1) -> np.ndarray:
        """
        Generate a Gaussian pulse train.
        
        Args:
            t: Time array
            frequency: Pulse repetition frequency in Hz
            amplitude: Amplitude (0-1)
            phase: Phase offset in degrees
            std: Standard deviation of each pulse (seconds)
            
        Returns:
            Numpy array containing the Gaussian pulse train
        """
        phase_sec = phase / 360.0 / frequency  # Convert phase from degrees to seconds
        
        # Period in seconds
        period = 1.0 / frequency
        
        # Initialize output array
        output = np.zeros_like(t)
        
        # Generate pulses for each period
        for i in np.arange(t[0] - period, t[-1] + period, period):
            pulse_center = i + phase_sec
            gaussian = np.exp(-0.5 * ((t - pulse_center) / std) ** 2)
            output += gaussian
            
        # Normalize and scale
        if np.max(output) > 0:
            output = output / np.max(output) * amplitude
            
        return output
    
    def _generate_chirp(self, t: np.ndarray, f0: float = 1.0, f1: float = 20.0, 
                      amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
        """
        Generate a chirp signal (frequency sweep).
        
        Args:
            t: Time array
            f0: Starting frequency in Hz
            f1: Ending frequency in Hz
            amplitude: Amplitude (0-1)
            phase: Initial phase offset in degrees
            
        Returns:
            Numpy array containing the chirp signal
        """
        phase_rad = np.deg2rad(phase)
        t_max = t[-1] - t[0]
        return amplitude * signal.chirp(t, f0=f0, f1=f1, t1=t_max, phi=phase_rad)
    
    def _generate_gamma(self, t: np.ndarray, frequency: float = 10.0, 
                      amplitude: float = 1.0, phase: float = 0.0, 
                      shape: float = 2.0) -> np.ndarray:
        """
        Generate a gamma waveform (similar to neural PSPs).
        
        Args:
            t: Time array
            frequency: Base frequency in Hz
            amplitude: Amplitude (0-1)
            phase: Phase offset in degrees
            shape: Shape parameter 
            
        Returns:
            Numpy array containing the gamma waveform
        """
        # Convert phase from degrees to time offset
        phase_sec = phase / 360.0 / frequency
        
        # Period in seconds
        period = 1.0 / frequency
        
        # Initialize output array
        output = np.zeros_like(t)
        
        # Generate gamma pulses for each period
        for i in np.arange(t[0] - period, t[-1] + period, period):
            pulse_center = i + phase_sec
            # Gamma function approximates post-synaptic potential shape
            t_rel = (t - pulse_center) * 20  # Scale to appropriate time constant
            mask = t_rel > 0  # Only consider positive time values
            gamma = np.zeros_like(t_rel)
            gamma[mask] = t_rel[mask] ** shape * np.exp(-t_rel[mask])
            
            # Normalize each pulse to have maximum of 1
            if np.max(gamma) > 0:
                gamma = gamma / np.max(gamma)
                
            output += gamma
            
        # Normalize and scale
        if np.max(output) > 0:
            output = output / np.max(output) * amplitude
            
        return output
    
    def _generate_burst(self, t: np.ndarray, frequency: float = 10.0, 
                      amplitude: float = 1.0, phase: float = 0.0,
                      burst_freq: float = 100.0, burst_count: int = 3) -> np.ndarray:
        """
        Generate a burst pattern (groups of high-frequency pulses).
        
        Args:
            t: Time array
            frequency: Burst repetition frequency in Hz
            amplitude: Amplitude (0-1)
            phase: Phase offset in degrees
            burst_freq: Frequency of pulses within each burst
            burst_count: Number of pulses per burst
            
        Returns:
            Numpy array containing the burst pattern
        """
        phase_rad = np.deg2rad(phase)
        
        # Period of burst repetition
        burst_period = 1.0 / frequency
        
        # Period of individual pulses in burst
        pulse_period = 1.0 / burst_freq
        
        # Duration of each burst
        burst_duration = burst_count * pulse_period
        
        # Duty cycle for burst envelope
        burst_duty = burst_duration / burst_period
        
        # Generate burst envelope
        burst_envelope = 0.5 * (signal.square(2 * np.pi * frequency * t + phase_rad, 
                                           duty=burst_duty) + 1)
        
        # Generate high-frequency carrier
        carrier = np.sin(2 * np.pi * burst_freq * t)
        
        # Combine to get bursts
        return amplitude * burst_envelope * carrier
    
    def generate_waveform(self, waveform_type: str = 'sine', duration: float = 1.0, 
                        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a waveform of the specified type.
        
        Args:
            waveform_type: Type of waveform to generate
            duration: Duration of the waveform in seconds
            **kwargs: Additional parameters for specific waveform types
            
        Returns:
            Tuple of (time_array, signal_array)
        """
        # Check if waveform type is valid
        if waveform_type not in self.waveform_functions:
            raise ValueError(f"Waveform type '{waveform_type}' not recognized. "
                           f"Available types: {list(self.waveform_functions.keys())}")
        
        # Create time array
        t = np.arange(0, duration, 1.0/self.sampling_rate)
        
        # Get waveform generation function
        waveform_func = self.waveform_functions[waveform_type]
        
        # Generate waveform using provided parameters
        wave = waveform_func(t, **kwargs)
        
        return t, wave
    
    def create_sequential_pattern(self, channel_sequence: List[int], 
                                base_waveform: str = 'sine',
                                propagation_speed: float = 20.0,  # m/s
                                distance_between_channels: float = 0.02,  # 2cm
                                duration: float = 1.0,
                                **waveform_params) -> Dict[int, np.ndarray]:
        """
        Create a sequential stimulation pattern that propagates along the specified 
        channel sequence.
        
        Args:
            channel_sequence: List of channel indices defining the propagation path
            base_waveform: Base waveform type to use
            propagation_speed: Neural propagation speed in m/s
            distance_between_channels: Approximate distance between channels in meters
            duration: Total duration of the pattern in seconds
            **waveform_params: Additional parameters for the base waveform
            
        Returns:
            Dictionary mapping channel indices to stimulation waveforms
        """
        # Create time array
        t = np.arange(0, duration, 1.0/self.sampling_rate)
        
        # Calculate delay between consecutive channels
        delay = distance_between_channels / propagation_speed  # seconds
        
        # Initialize output dictionary
        patterns = {}
        
        # Generate waveform for each channel in sequence with appropriate delay
        for i, channel in enumerate(channel_sequence):
            # Calculate phase offset in degrees
            phase_offset = (i * delay * 360.0 * waveform_params.get('frequency', 
                                                                 self.default_parameters['frequency']))
            
            # Update parameters with phase offset
            params = waveform_params.copy()
            params['phase'] = params.get('phase', 0.0) + phase_offset
            
            # Generate waveform
            _, wave = self.generate_waveform(
                waveform_type=base_waveform,
                duration=duration,
                **params
            )
            
            # Store waveform for this channel
            patterns[channel] = wave
            
        # Fill in zeros for any channels not in the sequence
        for channel in range(self.num_channels):
            if channel not in channel_sequence:
                patterns[channel] = np.zeros_like(t)
                
        return patterns
    
    def create_pathway_pattern(self, pathway_name: str, **kwargs) -> Dict[int, np.ndarray]:
        """
        Create a sequential stimulation pattern following a predefined neural pathway.
        
        Args:
            pathway_name: Name of predefined neural pathway
            **kwargs: Additional parameters passed to create_sequential_pattern
            
        Returns:
            Dictionary mapping channel indices to stimulation waveforms
        """
        # Check if pathway exists
        if pathway_name not in self.neural_pathways:
            raise ValueError(f"Neural pathway '{pathway_name}' not recognized. "
                           f"Available pathways: {list(self.neural_pathways.keys())}")
        
        # Get channel sequence for pathway
        channel_sequence = self.neural_pathways[pathway_name]
        
        # Create sequential pattern based on pathway
        return self.create_sequential_pattern(channel_sequence, **kwargs)
    
    def create_phase_gradient_pattern(self, channel_layout: List[Tuple[float, float]], 
                                    propagation_direction: Tuple[float, float] = (1.0, 0.0),
                                    wave_speed: float = 10.0,  # m/s
                                    base_waveform: str = 'sine',
                                    duration: float = 1.0,
                                    **waveform_params) -> Dict[int, np.ndarray]:
        """
        Create a pattern with a continuous phase gradient in a specific direction.
        
        Args:
            channel_layout: List of (x, y) coordinates for each channel
            propagation_direction: Direction vector (dx, dy) for the wave
            wave_speed: Speed of the phase wave in m/s
            base_waveform: Base waveform type to use
            duration: Total duration of the pattern in seconds
            **waveform_params: Additional parameters for the base waveform
            
        Returns:
            Dictionary mapping channel indices to stimulation waveforms
        """
        # Create time array
        t = np.arange(0, duration, 1.0/self.sampling_rate)
        
        # Normalize direction vector
        direction = np.array(propagation_direction)
        direction = direction / np.linalg.norm(direction)
        
        # Get base frequency from parameters or use default
        frequency = waveform_params.get('frequency', self.default_parameters['frequency'])
        
        # Wavelength in meters
        wavelength = wave_speed / frequency
        
        # Initialize output dictionary
        patterns = {}
        
        # Generate waveform for each channel based on its position
        for i, (x, y) in enumerate(channel_layout):
            # Project position onto direction vector to get distance along wave path
            position = np.array([x, y])
            distance_along_path = np.dot(position, direction)
            
            # Calculate phase offset based on position
            # One complete wavelength (360 degrees) corresponds to the wavelength in meters
            phase_offset = (distance_along_path % wavelength) / wavelength * 360.0
            
            # Update parameters with phase offset
            params = waveform_params.copy()
            params['phase'] = params.get('phase', 0.0) + phase_offset
            
            # Generate waveform
            _, wave = self.generate_waveform(
                waveform_type=base_waveform,
                duration=duration,
                **params
            )
            
            # Store waveform for this channel
            patterns[i] = wave
            
        # Fill in zeros for any channels not in the layout
        for channel in range(self.num_channels):
            if channel not in patterns:
                patterns[channel] = np.zeros_like(t)
                
        return patterns
    
    def create_interference_pattern(self, focus_point: Tuple[float, float], 
                                  channel_layout: List[Tuple[float, float]],
                                  phase_alignment: str = 'constructive',
                                  base_waveform: str = 'sine',
                                  duration: float = 1.0,
                                  **waveform_params) -> Dict[int, np.ndarray]:
        """
        Create a pattern where phases are aligned to create constructive or destructive
        interference at a specific focus point.
        
        Args:
            focus_point: (x, y) coordinates of the desired interference focus
            channel_layout: List of (x, y) coordinates for each channel
            phase_alignment: 'constructive' or 'destructive'
            base_waveform: Base waveform type to use
            duration: Total duration of the pattern in seconds
            **waveform_params: Additional parameters for the base waveform
            
        Returns:
            Dictionary mapping channel indices to stimulation waveforms
        """
        # Create time array
        t = np.arange(0, duration, 1.0/self.sampling_rate)
        
        # Get base frequency from parameters or use default
        frequency = waveform_params.get('frequency', self.default_parameters['frequency'])
        
        # Determine phase shift for interference type
        phase_shift = 0.0 if phase_alignment == 'constructive' else 180.0
        
        # Initialize output dictionary
        patterns = {}
        
        # Generate waveform for each channel based on its distance from focus point
        for i, (x, y) in enumerate(channel_layout):
            # Calculate Euclidean distance from channel to focus point
            channel_pos = np.array([x, y])
            focus_pos = np.array(focus_point)
            distance = np.linalg.norm(channel_pos - focus_pos)
            
            # Calculate phase offset based on distance
            # Adjust phases so they align at the focus point
            # Assuming a nominal propagation speed of 10 m/s for demonstration
            propagation_speed = 10.0  # m/s
            wavelength = propagation_speed / frequency
            phase_offset = (distance % wavelength) / wavelength * 360.0
            
            # For constructive interference, we want all waves to arrive in phase
            # For destructive, we want them to arrive out of phase
            # So we apply the negative phase offset to make them align at focus
            # (plus an additional 180° for destructive interference)
            params = waveform_params.copy()
            params['phase'] = params.get('phase', 0.0) - phase_offset + phase_shift
            
            # Generate waveform
            _, wave = self.generate_waveform(
                waveform_type=base_waveform,
                duration=duration,
                **params
            )
            
            # Store waveform for this channel
            patterns[i] = wave
            
        # Fill in zeros for any channels not in the layout
        for channel in range(self.num_channels):
            if channel not in patterns:
                patterns[channel] = np.zeros_like(t)
                
        return patterns
    
    def create_spatiotemporal_envelope(self, patterns: Dict[int, np.ndarray], 
                                     envelope_type: str = 'gaussian',
                                     center_time: float = None,
                                     width: float = 0.2) -> Dict[int, np.ndarray]:
        """
        Apply a spatiotemporal envelope to modulate the amplitude of patterns over time.
        
        Args:
            patterns: Dictionary mapping channel indices to stimulation waveforms
            envelope_type: Type of envelope ('gaussian', 'linear', 'quadratic')
            center_time: Center time of the envelope (None for middle)
            width: Width parameter of the envelope
            
        Returns:
            Dictionary mapping channel indices to modulated waveforms
        """
        # Get sample count from first pattern
        first_channel = next(iter(patterns.values()))
        n_samples = len(first_channel)
        t = np.linspace(0, 1, n_samples)  # Normalized time
        
        # Set default center time to middle
        if center_time is None:
            center_time = 0.5
        else:
            # Normalize center time to 0-1 range
            center_time = center_time / (n_samples / self.sampling_rate)
            
        # Generate envelope
        if envelope_type == 'gaussian':
            envelope = np.exp(-((t - center_time) ** 2) / (2 * width ** 2))
        elif envelope_type == 'linear':
            # Linear ramp up and down
            envelope = 1.0 - np.minimum(
                np.abs(t - center_time) / width,
                np.ones_like(t)
            )
            envelope = np.maximum(0, envelope)  # Clip negative values to 0
        elif envelope_type == 'quadratic':
            # Smoother quadratic envelope
            envelope = 1.0 - np.minimum(
                (np.abs(t - center_time) / width) ** 2,
                np.ones_like(t)
            )
            envelope = np.maximum(0, envelope)  # Clip negative values to 0
        else:
            raise ValueError(f"Envelope type '{envelope_type}' not recognized. "
                           f"Available types: ['gaussian', 'linear', 'quadratic']")
            
        # Apply envelope to all patterns
        modulated_patterns = {}
        for channel, pattern in patterns.items():
            modulated_patterns[channel] = pattern * envelope
            
        return modulated_patterns
    
    def blend_patterns(self, patterns1: Dict[int, np.ndarray], 
                     patterns2: Dict[int, np.ndarray],
                     blend_ratio: float = 0.5) -> Dict[int, np.ndarray]:
        """
        Blend two sets of patterns with the specified ratio.
        
        Args:
            patterns1: First set of patterns
            patterns2: Second set of patterns
            blend_ratio: Blending ratio (0 = only patterns1, 1 = only patterns2)
            
        Returns:
            Dictionary mapping channel indices to blended waveforms
        """
        # Check patterns have same channels and lengths
        if set(patterns1.keys()) != set(patterns2.keys()):
            raise ValueError("Pattern sets must have the same channel indices")
            
        first_channel1 = next(iter(patterns1.values()))
        first_channel2 = next(iter(patterns2.values()))
        if len(first_channel1) != len(first_channel2):
            raise ValueError("Pattern sets must have the same length")
            
        # Initialize output dictionary
        blended_patterns = {}
        
        # Blend patterns for each channel
        for channel in patterns1.keys():
            blended_patterns[channel] = (
                (1 - blend_ratio) * patterns1[channel] + 
                blend_ratio * patterns2[channel]
            )
            
        return blended_patterns
    
    def apply_adaptive_intensity(self, patterns: Dict[int, np.ndarray], 
                               intensity_map: Dict[int, float]) -> Dict[int, np.ndarray]:
        """
        Apply channel-specific intensity scaling based on neural feedback.
        
        Args:
            patterns: Dictionary mapping channel indices to stimulation waveforms
            intensity_map: Dictionary mapping channel indices to intensity scale factors
            
        Returns:
            Dictionary mapping channel indices to intensity-adjusted waveforms
        """
        # Initialize output dictionary
        adjusted_patterns = {}
        
        # Apply intensity scaling to each channel
        for channel, pattern in patterns.items():
            # Get intensity scale factor for this channel (default to 1.0)
            scale_factor = intensity_map.get(channel, 1.0)
            
            # Apply scaling
            adjusted_patterns[channel] = pattern * scale_factor
            
        return adjusted_patterns
    
    def visualize_pattern(self, patterns: Dict[int, np.ndarray], 
                        sampling_rate: float = None,
                        title: str = "Spatiotemporal Stimulation Pattern") -> None:
        """
        Visualize the generated stimulation patterns.
        
        Args:
            patterns: Dictionary mapping channel indices to stimulation waveforms
            sampling_rate: Sampling rate in Hz (default is self.sampling_rate)
            title: Title for the visualization
            
        Returns:
            None (displays plot)
        """
        try:
            import matplotlib.pyplot as plt
            
            # Use default sampling rate if not specified
            if sampling_rate is None:
                sampling_rate = self.sampling_rate
                
            # Get number of samples from first pattern
            first_channel = next(iter(patterns.values()))
            n_samples = len(first_channel)
            
            # Create time array
            t = np.arange(n_samples) / sampling_rate
            
            # Get all channels sorted by index
            channels = sorted(patterns.keys())
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot each channel
            for i, channel in enumerate(channels):
                # Offset pattern for visualization
                offset = i * 2  # Vertical spacing between channels
                plt.plot(t, patterns[channel] + offset, label=f"Channel {channel}")
                
            # Add labels and title
            plt.xlabel("Time (s)")
            plt.ylabel("Channel")
            plt.title(title)
            plt.yticks([i * 2 for i in range(len(channels))], 
                     [f"Ch {ch}" for ch in channels])
            plt.grid(True, alpha=0.3)
            
            # Show plot
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available. Cannot visualize pattern.")
            
    def export_patterns(self, patterns: Dict[int, np.ndarray], 
                      filename: str, 
                      sampling_rate: float = None) -> None:
        """
        Export generated patterns to a numpy file.
        
        Args:
            patterns: Dictionary mapping channel indices to stimulation waveforms
            filename: Output filename (.npy)
            sampling_rate: Sampling rate in Hz to save as metadata
            
        Returns:
            None (saves file)
        """
        # Use default sampling rate if not specified
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
            
        # Convert patterns to a structured array
        max_channel = max(patterns.keys())
        n_samples = len(next(iter(patterns.values())))
        
        # Create a 2D array (channels x samples)
        pattern_array = np.zeros((max_channel + 1, n_samples))
        
        # Fill in patterns
        for channel, waveform in patterns.items():
            pattern_array[channel, :] = waveform
            
        # Save with metadata
        np.savez(filename, 
               patterns=pattern_array, 
               sampling_rate=sampling_rate,
               channels=list(patterns.keys()))
        
        logger.info(f"Patterns exported to {filename}")
