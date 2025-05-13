#!/usr/bin/env python3
"""
Main application script for the Adaptive Phase-Differential Brain Nerve Stimulation System.
This script provides a unified interface for controlling the system components.
"""

import os
import sys
import time
import logging
import argparse
import json
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('adaptive_neural_stimulation.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import system components
try:
    from src.core.system_controller import SystemController
    from src.data_processing.eeg_processor import EEGProcessor
    from src.data_processing.neural_feedback import NeuralFeedbackSystem, AdaptiveStimulationController
    from src.data_processing.neural_feedback import BandPowerMetric, BandRatioMetric, PhaseLockingMetric
    from src.hardware_interface.stimulation_device import StimulationDeviceManager, WirelessStimulationDevice
    from src.algorithms.brainwave_sync import BrainwaveSynchronizer
    from src.algorithms.plasticity_window import PlasticityWindowDetector
    from src.algorithms.spatiotemporal_patterns import SpatiotemporalPatternGenerator
    from src.protocols.treatment_protocols import ProtocolLibrary, StimulationProtocol
    from src.visualization.stimulation_visualizer import StimulationVisualizer
except ImportError as e:
    logger.error(f"Failed to import system components: {e}")
    sys.exit(1)


class AdaptiveNeuralStimulationApp:
    """
    Main application class for the Adaptive Neural Stimulation System.
    Provides a unified interface for controlling all system components.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the application.
        
        Args:
            config_path: Path to configuration file
        """
        self.is_running = False
        self.config = self._load_config(config_path)
        
        # Initialize components
        self._init_components()
        
        # Paths for data storage
        self.data_dir = self.config.get('data_dir', 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info("Adaptive Neural Stimulation Application initialized")
        
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'sampling_rate': 250.0,
            'eeg_channels': 32,
            'emg_channels': 4,
            'data_dir': 'data',
            'protocols_dir': 'protocols',
            'visualization_theme': 'light',
            'log_level': 'INFO',
            'device_scan_interval': 5.0,
            'feedback_update_interval': 0.5,
            'safety_limits': {
                'max_amplitude': 5.0,  # mA
                'max_frequency': 100.0,  # Hz
                'max_daily_sessions': 3
            }
        }
        
        # Load config from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    logger.info(f"Loaded configuration from {config_path}")
                    
                    # Merge with default config
                    config = {**default_config, **loaded_config}
                    return config
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                
        logger.info("Using default configuration")
        return default_config
        
    def _init_components(self):
        """Initialize all system components."""
        sampling_rate = self.config.get('sampling_rate', 250.0)
        
        # Initialize core components
        self.system_controller = SystemController()
        
        # Initialize data processing components
        self.eeg_processor = EEGProcessor(
            sampling_rate=sampling_rate,
            n_channels=self.config.get('eeg_channels', 32)
        )
        
        # Initialize neural feedback system
        self.feedback_system = NeuralFeedbackSystem(sampling_rate=sampling_rate)
        self.feedback_system.update_interval = self.config.get('feedback_update_interval', 0.5)
        
        # Initialize default metrics
        self._init_default_metrics()
        
        # Initialize hardware interface components
        self.device_manager = StimulationDeviceManager()
        
        # Initialize stimulation controller
        self.stimulation_controller = AdaptiveStimulationController(self.feedback_system)
        
        # Initialize algorithm components
        self.brainwave_synchronizer = BrainwaveSynchronizer(sampling_rate=sampling_rate)
        self.plasticity_detector = PlasticityWindowDetector()
        self.pattern_generator = SpatiotemporalPatternGenerator(
            num_channels=8,
            sampling_rate=sampling_rate
        )
        
        # Initialize protocol library
        protocols_dir = self.config.get('protocols_dir', 'protocols')
        os.makedirs(protocols_dir, exist_ok=True)
        self.protocol_library = ProtocolLibrary(protocols_dir)
        
        # Initialize visualization components
        theme = self.config.get('visualization_theme', 'light')
        self.visualizer = StimulationVisualizer(theme=theme)
        
        # Start device discovery thread
        self.device_scan_thread = threading.Thread(
            target=self._scan_for_devices,
            daemon=True
        )
        self.device_scan_should_run = True
        
    def _init_default_metrics(self):
        """Initialize default feedback metrics."""
        # Add common EEG band power metrics
        self.feedback_system.add_metric(BandPowerMetric(
            name="alpha_power",
            band=(8.0, 13.0),
            target_range=(50.0, 200.0),
            weight=1.0
        ))
        
        self.feedback_system.add_metric(BandPowerMetric(
            name="theta_power",
            band=(4.0, 8.0),
            target_range=(30.0, 150.0),
            weight=1.0
        ))
        
        self.feedback_system.add_metric(BandPowerMetric(
            name="beta_power",
            band=(13.0, 30.0),
            target_range=(20.0, 100.0),
            weight=1.0
        ))
        
        # Add band ratio metric (theta/beta)
        self.feedback_system.add_metric(BandRatioMetric(
            name="theta_beta_ratio",
            band1=(4.0, 8.0),  # theta
            band2=(13.0, 30.0),  # beta
            target_range=(0.5, 3.0),
            weight=1.0
        ))
        
        # Add interhemispheric coherence metric (if enough channels available)
        if self.config.get('eeg_channels', 32) >= 8:
            self.feedback_system.add_metric(PhaseLockingMetric(
                name="frontal_coherence",
                channel_pair=(3, 4),  # Example channels (F3, F4)
                band=(8.0, 13.0),  # Alpha band
                target_range=(0.3, 0.8),
                weight=1.0
            ))
            
    def _scan_for_devices(self):
        """Background thread for scanning and connecting to stimulation devices."""
        logger.info("Device scanning thread started")
        scan_interval = self.config.get('device_scan_interval', 5.0)
        
        while self.device_scan_should_run:
            try:
                # Simulate device discovery (would be replaced with actual discovery code)
                devices = self._discover_devices()
                
                # Connect to new devices
                for device_info in devices:
                    if device_info['id'] not in self.device_manager.devices:
                        # Create device object
                        device = WirelessStimulationDevice(
                            device_id=device_info['id'],
                            max_current=device_info.get('max_current', 5.0),
                            bluetooth_address=device_info.get('address')
                        )
                        
                        # Add to manager
                        self.device_manager.add_device(device)
                        
                        # Add to stimulation controller
                        self.stimulation_controller.add_stimulation_device(
                            device_id=device_info['id'],
                            device_obj=device
                        )
                        
                        logger.info(f"Connected to device: {device_info['id']}")
                        
            except Exception as e:
                logger.error(f"Error in device scanning: {e}")
                
            # Sleep for scan interval
            time.sleep(scan_interval)
            
        logger.info("Device scanning thread stopped")
        
    def _discover_devices(self):
        """
        Discover available stimulation devices.
        This is a placeholder for actual device discovery code.
        
        Returns:
            List of device information dictionaries
        """
        # In a real implementation, this would scan for Bluetooth/USB devices
        # For demonstration, return some simulated devices
        
        # Check if devices already created
        if hasattr(self, '_simulated_devices'):
            return self._simulated_devices
            
        # Create simulated devices
        self._simulated_devices = [
            {
                'id': 'ANS-001',
                'type': 'wireless',
                'address': '00:11:22:33:44:01',
                'max_current': 5.0,
                'channels': 2
            },
            {
                'id': 'ANS-002',
                'type': 'wireless',
                'address': '00:11:22:33:44:02',
                'max_current': 5.0,
                'channels': 2
            }
        ]
        
        return self._simulated_devices
        
    def start(self):
        """Start the application and all components."""
        if self.is_running:
            logger.warning("Application already running")
            return False
            
        logger.info("Starting Adaptive Neural Stimulation System")
        
        # Start device scanning thread
        self.device_scan_thread.start()
        
        # Start neural feedback system
        self.feedback_system.start()
        
        self.is_running = True
        return True
        
    def stop(self):
        """Stop the application and all components."""
        if not self.is_running:
            logger.warning("Application not running")
            return False
            
        logger.info("Stopping Adaptive Neural Stimulation System")
        
        # Stop device scanning
        self.device_scan_should_run = False
        if self.device_scan_thread.is_alive():
            self.device_scan_thread.join(timeout=2.0)
            
        # Stop neural feedback system
        self.feedback_system.stop()
        
        # Stop all stimulation
        self.device_manager.stop_all_stimulation()
        
        self.is_running = False
        return True
        
    def load_protocol(self, protocol_id: str) -> StimulationProtocol:
        """
        Load a stimulation protocol.
        
        Args:
            protocol_id: ID of the protocol to load
            
        Returns:
            Loaded protocol or None if not found
        """
        protocol = self.protocol_library.get_protocol(protocol_id)
        if protocol:
            logger.info(f"Loaded protocol: {protocol.name}")
            return protocol
        else:
            logger.error(f"Protocol not found: {protocol_id}")
            return None
            
    def start_protocol(self, protocol: StimulationProtocol) -> bool:
        """
        Start executing a stimulation protocol.
        
        Args:
            protocol: Protocol to execute
            
        Returns:
            True if protocol started successfully, False otherwise
        """
        if not self.is_running:
            logger.error("Cannot start protocol: System not running")
            return False
            
        if not protocol:
            logger.error("Invalid protocol")
            return False
            
        # Validate protocol
        valid, errors = protocol.validate()
        if not valid:
            for error in errors:
                logger.error(f"Protocol validation error: {error}")
            return False
            
        logger.info(f"Starting protocol: {protocol.name}")
        
        # Configure stimulation parameters based on protocol
        try:
            self._configure_stimulation_from_protocol(protocol)
            return True
        except Exception as e:
            logger.error(f"Failed to start protocol: {e}")
            return False
            
    def _configure_stimulation_from_protocol(self, protocol: StimulationProtocol):
        """
        Configure stimulation parameters from a protocol.
        
        Args:
            protocol: Protocol with stimulation parameters
        """
        # Get available devices
        devices = list(self.device_manager.devices.values())
        if not devices:
            raise ValueError("No stimulation devices available")
            
        # Get the first step with stimulation parameters
        stimulation_step = None
        for step in protocol.steps:
            if step.get('type') in ['stimulation', 'combined'] and 'parameters' in step:
                stimulation_step = step
                break
                
        if not stimulation_step:
            raise ValueError("Protocol contains no stimulation steps")
            
        # Extract parameters
        params = stimulation_step['parameters']
        
        # Configure each device
        for i, device in enumerate(devices):
            # Set basic parameters
            device_params = {
                'amplitude': params.get('amplitude', 1.0),
                'frequency': params.get('frequency', 10.0),
                'waveform': params.get('waveform', 'sine'),
            }
            
            # Set phase for each device (with appropriate offset)
            if i > 0 and 'phase_difference' in params:
                device_params['phase'] = (params.get('phase', 0.0) + 
                                       i * params['phase_difference']) % 360.0
            else:
                device_params['phase'] = params.get('phase', 0.0)
                
            # Apply parameters
            device.set_parameters(**device_params)
            
            # Add to stimulation controller's current parameters
            self.stimulation_controller.current_parameters[device.device_id] = device_params
            
        logger.info(f"Configured stimulation parameters from protocol: {protocol.name}")
        
    def process_eeg_data(self, eeg_data: np.ndarray) -> dict:
        """
        Process incoming EEG data.
        
        Args:
            eeg_data: EEG data array with shape (n_channels, n_samples)
            
        Returns:
            Dictionary with processing results
        """
        if not self.is_running:
            logger.warning("Cannot process data: System not running")
            return {}
            
        # Check data shape
        if len(eeg_data.shape) != 2:
            logger.error(f"Invalid EEG data shape: {eeg_data.shape}")
            return {}
            
        # Process EEG data
        try:
            # Apply filters and extract features
            filtered_data = self.eeg_processor.apply_filters(eeg_data)
            bands = self.eeg_processor.extract_bands(filtered_data)
            
            # Update feedback system
            self.feedback_system.update(filtered_data)
            
            # Check for plasticity window
            is_plasticity_window = self.plasticity_detector.detect_window(filtered_data, 
                                                              self.eeg_processor.sampling_rate)
                                                              
            # Generate results
            results = {
                'filtered_data': filtered_data,
                'band_data': bands,
                'feedback_score': self.feedback_system.overall_score,
                'metrics': self.feedback_system.get_metric_status(),
                'is_plasticity_window': is_plasticity_window
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing EEG data: {e}")
            return {}
            
    def generate_stimulation_pattern(self, pattern_type: str, **kwargs) -> dict:
        """
        Generate a stimulation pattern.
        
        Args:
            pattern_type: Type of pattern to generate
            **kwargs: Pattern-specific parameters
            
        Returns:
            Dictionary with pattern data
        """
        try:
            if pattern_type == 'sequential':
                # Generate sequential pattern
                channel_sequence = kwargs.get('channel_sequence', [0, 1, 2, 3, 4])
                patterns = self.pattern_generator.create_sequential_pattern(
                    channel_sequence=channel_sequence,
                    base_waveform=kwargs.get('waveform', 'sine'),
                    propagation_speed=kwargs.get('speed', 20.0),
                    duration=kwargs.get('duration', 1.0),
                    frequency=kwargs.get('frequency', 10.0),
                    amplitude=kwargs.get('amplitude', 1.0)
                )
                return {'pattern_type': 'sequential', 'patterns': patterns}
                
            elif pattern_type == 'pathway':
                # Generate pattern for predefined neural pathway
                pathway_name = kwargs.get('pathway', 'motor')
                patterns = self.pattern_generator.create_pathway_pattern(
                    pathway_name=pathway_name,
                    base_waveform=kwargs.get('waveform', 'sine'),
                    duration=kwargs.get('duration', 1.0),
                    frequency=kwargs.get('frequency', 10.0),
                    amplitude=kwargs.get('amplitude', 1.0)
                )
                return {'pattern_type': 'pathway', 'patterns': patterns}
                
            elif pattern_type == 'phase_gradient':
                # Generate pattern with continuous phase gradient
                channel_layout = kwargs.get('layout', [(0, 0), (1, 0), (0, 1), (1, 1)])
                patterns = self.pattern_generator.create_phase_gradient_pattern(
                    channel_layout=channel_layout,
                    propagation_direction=kwargs.get('direction', (1.0, 0.0)),
                    wave_speed=kwargs.get('speed', 10.0),
                    base_waveform=kwargs.get('waveform', 'sine'),
                    duration=kwargs.get('duration', 1.0),
                    frequency=kwargs.get('frequency', 10.0),
                    amplitude=kwargs.get('amplitude', 1.0)
                )
                return {'pattern_type': 'phase_gradient', 'patterns': patterns}
                
            else:
                logger.error(f"Unknown pattern type: {pattern_type}")
                return {}
                
        except Exception as e:
            logger.error(f"Error generating stimulation pattern: {e}")
            return {}
            
    def apply_stimulation_pattern(self, pattern_data: dict) -> bool:
        """
        Apply a stimulation pattern to connected devices.
        
        Args:
            pattern_data: Pattern data from generate_stimulation_pattern
            
        Returns:
            True if pattern applied successfully, False otherwise
        """
        if not self.is_running:
            logger.warning("Cannot apply pattern: System not running")
            return False
            
        if not pattern_data or 'patterns' not in pattern_data:
            logger.error("Invalid pattern data")
            return False
            
        # Get available devices
        devices = list(self.device_manager.devices.values())
        if not devices:
            logger.error("No stimulation devices available")
            return False
            
        try:
            patterns = pattern_data['patterns']
            
            # Map pattern channels to available devices
            for i, (channel, pattern) in enumerate(patterns.items()):
                if i >= len(devices):
                    break
                    
                device = devices[i]
                
                # Calculate parameters from the pattern
                amplitude = np.max(np.abs(pattern))
                
                # Set parameters on device
                device.set_parameters(
                    amplitude=amplitude,
                    waveform='custom_pattern',
                    pattern_data=pattern
                )
                
            logger.info(f"Applied {pattern_data['pattern_type']} stimulation pattern to {min(len(patterns), len(devices))} devices")
            return True
            
        except Exception as e:
            logger.error(f"Error applying stimulation pattern: {e}")
            return False
            
    def get_system_status(self) -> dict:
        """
        Get current status of the entire system.
        
        Returns:
            Dictionary with system status information
        """
        # Collect status from all components
        status = {
            'is_running': self.is_running,
            'devices': {
                'count': len(self.device_manager.devices),
                'active': len(self.device_manager.active_devices),
                'devices': {device_id: device.get_status() 
                         for device_id, device in self.device_manager.devices.items()}
            },
            'feedback': {
                'running': self.feedback_system.is_running,
                'overall_score': self.feedback_system.overall_score,
                'metrics': self.feedback_system.get_metric_status()
            },
            'protocols': {
                'count': len(self.protocol_library.protocols),
                'summary': self.protocol_library.get_protocol_summary()
            }
        }
        
        # Add plasticity window status if detector is active
        if hasattr(self, 'plasticity_detector'):
            status['plasticity'] = {
                'is_window_open': self.plasticity_detector.is_window_active,
                'window_quality': self.plasticity_detector.window_quality,
                'time_since_detection': self.plasticity_detector.time_since_detection
            }
            
        return status
        
    def visualize(self, data_type: str, data: dict) -> str:
        """
        Generate visualization for data.
        
        Args:
            data_type: Type of data to visualize
            data: Data to visualize
            
        Returns:
            Base64-encoded image data
        """
        try:
            if data_type == 'eeg':
                # Visualize EEG data
                if 'data' not in data:
                    logger.error("Missing 'data' field for EEG visualization")
                    return ""
                    
                return self.visualizer.visualize_eeg_data(
                    eeg_data=data['data'],
                    sampling_rate=data.get('sampling_rate', self.eeg_processor.sampling_rate),
                    channel_names=data.get('channel_names'),
                    time_window=data.get('time_window', 5.0),
                    title=data.get('title', 'EEG Signal'),
                    as_base64=True
                )
                
            elif data_type == 'spectral':
                # Visualize spectral data
                if 'freqs' not in data or 'psd' not in data:
                    logger.error("Missing 'freqs' or 'psd' fields for spectral visualization")
                    return ""
                    
                return self.visualizer.visualize_spectral_data(
                    freqs=data['freqs'],
                    psd=data['psd'],
                    channel_names=data.get('channel_names'),
                    freq_range=data.get('freq_range'),
                    title=data.get('title', 'Spectral Power'),
                    as_base64=True
                )
                
            elif data_type == 'stimulation':
                # Visualize stimulation pattern
                if 'patterns' not in data:
                    logger.error("Missing 'patterns' field for stimulation visualization")
                    return ""
                    
                return self.visualizer.visualize_stimulation_pattern(
                    patterns=data['patterns'],
                    time_axis=data.get('time_axis'),
                    title=data.get('title', 'Stimulation Pattern'),
                    channel_labels=data.get('channel_labels'),
                    as_base64=True
                )
                
            elif data_type == 'feedback':
                # Visualize feedback metrics
                if 'metrics' not in data:
                    logger.error("Missing 'metrics' field for feedback visualization")
                    return ""
                    
                # Convert metrics to time series data
                metrics_data = {}
                timestamps = []
                
                for metric_name, metric_data in data['metrics'].items():
                    if 'history' in metric_data and 'timestamps' in metric_data:
                        metrics_data[metric_name] = metric_data['history']
                        
                        if not timestamps and 'timestamps' in metric_data:
                            timestamps = metric_data['timestamps']
                            
                return self.visualizer.visualize_treatment_progress(
                    timestamps=timestamps,
                    metrics=metrics_data,
                    metric_labels=data.get('metric_labels'),
                    title=data.get('title', 'Feedback Metrics'),
                    as_base64=True
                )
                
            elif data_type == 'device_status':
                # Visualize device status
                return self.visualizer.visualize_device_status(
                    device_status=data.get('devices', {}),
                    as_base64=True
                )
                
            else:
                logger.error(f"Unknown visualization type: {data_type}")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return ""
            
    def save_session_data(self, session_data: dict) -> str:
        """
        Save session data to disk.
        
        Args:
            session_data: Session data dictionary
            
        Returns:
            Path to saved file
        """
        if not session_data:
            logger.error("No session data to save")
            return ""
            
        try:
            # Create session directory
            session_id = session_data.get('session_id', f"session_{int(time.time())}")
            session_dir = os.path.join(self.data_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            # Save session metadata
            metadata = {
                'session_id': session_id,
                'timestamp': time.time(),
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'user_id': session_data.get('user_id', 'unknown'),
                'protocol_id': session_data.get('protocol_id', 'unknown'),
                'total_duration': session_data.get('duration', 0.0),
                'device_ids': list(self.device_manager.devices.keys()),
                'parameters': session_data.get('parameters', {})
            }
            
            metadata_path = os.path.join(session_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Save EEG data if available
            if 'eeg_data' in session_data:
                eeg_path = os.path.join(session_dir, 'eeg_data.npz')
                np.savez_compressed(
                    eeg_path,
                    data=session_data['eeg_data'],
                    sampling_rate=self.eeg_processor.sampling_rate
                )
                
            # Save metrics data if available
            if 'metrics' in session_data:
                metrics_path = os.path.join(session_dir, 'metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(session_data['metrics'], f, indent=2)
                    
            # Save stimulation patterns if available
            if 'patterns' in session_data:
                patterns_path = os.path.join(session_dir, 'patterns.npz')
                np.savez_compressed(
                    patterns_path,
                    **session_data['patterns']
                )
                
            logger.info(f"Saved session data to {session_dir}")
            return session_dir
            
        except Exception as e:
            logger.error(f"Error saving session data: {e}")
            return ""


def main():
    """Main function for running the application."""
    parser = argparse.ArgumentParser(description='Adaptive Neural Stimulation System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--start', action='store_true', help='Start the system automatically')
    parser.add_argument('--simulate', action='store_true', help='Run in simulation mode')
    parser.add_argument('--protocol', type=str, help='Protocol ID to load and start')
    args = parser.parse_args()
    
    # Create application instance
    app = AdaptiveNeuralStimulationApp(config_path=args.config)
    
    # Start if requested
    if args.start:
        app.start()
        
        # Load and start protocol if specified
        if args.protocol:
            protocol = app.load_protocol(args.protocol)
            if protocol:
                app.start_protocol(protocol)
                
    # In a real application, we would integrate with a UI framework
    # or provide a REST API for control. For this example, we just sleep.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        app.stop()
        logger.info("Application stopped by user")


if __name__ == '__main__':
    main()
