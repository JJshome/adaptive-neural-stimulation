"""
System Controller

This module serves as the main controller for the Adaptive Phase-Differential Brain 
Nerve Stimulation System, coordinating between the different components and algorithms
to provide an integrated therapeutic system.
"""

import time
import logging
import threading
import numpy as np
import json
from datetime import datetime
from enum import Enum
from pathlib import Path

# Import core system modules
from .session_manager import SessionManager
from .config_handler import ConfigHandler
from .device_manager import DeviceManager
from .data_manager import DataManager

# Import algorithm modules
from ..algorithms.brainwave_sync import BrainwaveSynchronization
from ..algorithms.plasticity_window import PlasticityWindowDetector
from ..algorithms.spatiotemporal_pattern import SpatiotemporalPatternGenerator
from ..algorithms.adaptive_feedback import AdaptiveFeedbackController

# Import data processing modules
from ..data_processing.eeg_processor import EEGProcessor
from ..data_processing.biosignal_processor import BiosignalProcessor
from ..data_processing.neural_response import NeuralResponseAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('system_controller')

class SystemState(Enum):
    """Enumeration of possible system states."""
    IDLE = 0
    INITIALIZING = 1
    CALIBRATING = 2
    MONITORING = 3
    STIMULATING = 4
    ANALYZING = 5
    ERROR = 6
    SHUTDOWN = 7

class SystemController:
    """
    Main controller class for the Adaptive Phase-Differential Brain Nerve Stimulation System.
    
    This class coordinates between the different components and algorithms to provide
    an integrated therapeutic system. It manages the system state, handles user interactions,
    and orchestrates the data flow between various modules.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the system controller.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.state = SystemState.IDLE
        self.logger = logger
        
        # Load configuration
        self._load_configuration(config_path)
        
        # Initialize core components
        self._init_core_components()
        
        # Initialize algorithm modules
        self._init_algorithm_modules()
        
        # Initialize data processing modules
        self._init_data_processing()
        
        # System state variables
        self.is_running = False
        self.threads = {}
        self.last_error = None
        self.current_session = None
        
        self.logger.info("System controller initialized")
    
    def _load_configuration(self, config_path):
        """
        Load system configuration from file.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.logger.info("Loading configuration")
        self.state = SystemState.INITIALIZING
        
        # Set default config path if not provided
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "system_config.json"
        
        # Initialize configuration handler
        self.config_handler = ConfigHandler(config_path)
        self.config = self.config_handler.load_config()
        
        # Set up logging level based on configuration
        if 'logging' in self.config and 'level' in self.config['logging']:
            log_level = getattr(logging, self.config['logging']['level'].upper())
            self.logger.setLevel(log_level)
            self.logger.info(f"Log level set to {log_level}")
    
    def _init_core_components(self):
        """Initialize core system components."""
        self.logger.info("Initializing core components")
        
        # Initialize session manager
        self.session_manager = SessionManager()
        
        # Initialize device manager
        self.device_manager = DeviceManager(self.config.get('devices', {}))
        
        # Initialize data manager
        data_storage_path = self.config.get('data_storage', {}).get('path', 'data')
        self.data_manager = DataManager(data_storage_path)
    
    def _init_algorithm_modules(self):
        """Initialize algorithm modules."""
        self.logger.info("Initializing algorithm modules")
        
        # Extract algorithm configuration
        algo_config = self.config.get('algorithms', {})
        
        # Initialize brainwave synchronization module
        brainwave_config = algo_config.get('brainwave_sync', {})
        self.brainwave_sync = BrainwaveSynchronization(
            sampling_rate=brainwave_config.get('sampling_rate', 1000),
            target_band=brainwave_config.get('target_band', 'alpha'),
            n_channels=brainwave_config.get('n_channels', 8)
        )
        
        # Initialize plasticity window detector
        plasticity_config = algo_config.get('plasticity_window', {})
        self.plasticity_detector = PlasticityWindowDetector(
            sampling_rate=plasticity_config.get('sampling_rate', 250),
            history_duration=plasticity_config.get('history_duration', 14)
        )
        
        # Initialize spatiotemporal pattern generator
        pattern_config = algo_config.get('spatiotemporal_pattern', {})
        self.pattern_generator = SpatiotemporalPatternGenerator(
            n_channels=pattern_config.get('n_channels', 16),
            sampling_rate=pattern_config.get('sampling_rate', 1000)
        )
        
        # Initialize adaptive feedback controller
        feedback_config = algo_config.get('adaptive_feedback', {})
        self.feedback_controller = AdaptiveFeedbackController(
            update_rate=feedback_config.get('update_rate', 1.0),
            learning_rate=feedback_config.get('learning_rate', 0.1)
        )
    
    def _init_data_processing(self):
        """Initialize data processing modules."""
        self.logger.info("Initializing data processing modules")
        
        # Extract processing configuration
        processing_config = self.config.get('data_processing', {})
        
        # Initialize EEG processor
        eeg_config = processing_config.get('eeg', {})
        self.eeg_processor = EEGProcessor(
            sampling_rate=eeg_config.get('sampling_rate', 1000),
            n_channels=eeg_config.get('n_channels', 8),
            filter_settings=eeg_config.get('filters', None)
        )
        
        # Initialize biosignal processor
        biosignal_config = processing_config.get('biosignals', {})
        self.biosignal_processor = BiosignalProcessor(
            signals=biosignal_config.get('signals', ['ecg', 'gsr']),
            sampling_rates=biosignal_config.get('sampling_rates', {'ecg': 250, 'gsr': 100})
        )
        
        # Initialize neural response analyzer
        response_config = processing_config.get('neural_response', {})
        self.response_analyzer = NeuralResponseAnalyzer(
            metrics=response_config.get('metrics', ['amplitude', 'latency']),
            window_size=response_config.get('window_size', 0.5)
        )
    
    def start(self):
        """
        Start the system and transition to monitoring state.
        
        Returns:
            bool: True if system started successfully, False otherwise
        """
        if self.is_running:
            self.logger.warning("System is already running")
            return False
        
        self.logger.info("Starting system")
        self.state = SystemState.INITIALIZING
        
        try:
            # Connect to devices
            if not self.device_manager.connect_all():
                self.logger.error("Failed to connect to all devices")
                self.state = SystemState.ERROR
                self.last_error = "Device connection failure"
                return False
            
            # Start data acquisition threads
            self._start_data_acquisition()
            
            # Transition to monitoring state
            self.state = SystemState.MONITORING
            self.is_running = True
            
            # Start main control loop in a separate thread
            self.threads['control_loop'] = threading.Thread(
                target=self._control_loop,
                daemon=True
            )
            self.threads['control_loop'].start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting system: {str(e)}")
            self.state = SystemState.ERROR
            self.last_error = str(e)
            return False
    
    def stop(self):
        """
        Stop the system and return to idle state.
        
        Returns:
            bool: True if system stopped successfully, False otherwise
        """
        if not self.is_running:
            self.logger.warning("System is not running")
            return False
        
        self.logger.info("Stopping system")
        self.state = SystemState.SHUTDOWN
        
        try:
            # Set running flag to false to stop threads
            self.is_running = False
            
            # Wait for threads to finish
            for thread_name, thread in self.threads.items():
                if thread.is_alive():
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        self.logger.warning(f"Thread {thread_name} did not exit cleanly")
            
            # Disconnect from devices
            self.device_manager.disconnect_all()
            
            # End current session if active
            if self.current_session is not None:
                self.session_manager.end_session(self.current_session)
                self.current_session = None
            
            # Return to idle state
            self.state = SystemState.IDLE
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping system: {str(e)}")
            self.state = SystemState.ERROR
            self.last_error = str(e)
            return False
    
    def create_session(self, patient_id, protocol_id=None, notes=None):
        """
        Create a new therapy session.
        
        Args:
            patient_id (str): Unique patient identifier
            protocol_id (str, optional): Identifier for treatment protocol
            notes (str, optional): Session notes
            
        Returns:
            str: Session ID if successful, None otherwise
        """
        if self.current_session is not None:
            self.logger.warning("A session is already active")
            return None
        
        try:
            # Create new session
            session_data = {
                'patient_id': patient_id,
                'protocol_id': protocol_id,
                'notes': notes,
                'start_time': datetime.now().isoformat()
            }
            
            session_id = self.session_manager.create_session(session_data)
            self.current_session = session_id
            
            self.logger.info(f"Created new session with ID: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error creating session: {str(e)}")
            self.last_error = str(e)
            return None
    
    def end_session(self):
        """
        End the current therapy session.
        
        Returns:
            bool: True if session ended successfully, False otherwise
        """
        if self.current_session is None:
            self.logger.warning("No active session to end")
            return False
        
        try:
            # End current session
            self.session_manager.end_session(self.current_session)
            
            # Generate session summary
            summary = self._generate_session_summary()
            
            # Save summary to data manager
            summary_path = f"sessions/{self.current_session}/summary.json"
            self.data_manager.save_json(summary_path, summary)
            
            # Clear current session
            session_id = self.current_session
            self.current_session = None
            
            self.logger.info(f"Ended session with ID: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error ending session: {str(e)}")
            self.last_error = str(e)
            return False
    
    def _generate_session_summary(self):
        """
        Generate a summary of the current session.
        
        Returns:
            dict: Session summary data
        """
        # Get session details
        session_details = self.session_manager.get_session(self.current_session)
        
        # Create summary structure
        summary = {
            'session_id': self.current_session,
            'patient_id': session_details.get('patient_id'),
            'protocol_id': session_details.get('protocol_id'),
            'start_time': session_details.get('start_time'),
            'end_time': datetime.now().isoformat(),
            'stimulation_events': session_details.get('stimulation_events', []),
            'neural_responses': session_details.get('neural_responses', []),
            'metrics': {
                'total_stimulation_time': session_details.get('total_stimulation_time', 0),
                'average_response_amplitude': session_details.get('average_response_amplitude', 0),
                'plasticity_score': session_details.get('plasticity_score', 0)
            }
        }
        
        return summary
    
    def _start_data_acquisition(self):
        """Start data acquisition threads for each connected device."""
        self.logger.info("Starting data acquisition threads")
        
        # Get connected devices by type
        eeg_devices = self.device_manager.get_devices_by_type('eeg')
        biosignal_devices = self.device_manager.get_devices_by_type('biosignal')
        
        # Start EEG acquisition thread if EEG devices are connected
        if eeg_devices:
            self.threads['eeg_acquisition'] = threading.Thread(
                target=self._eeg_acquisition_loop,
                args=(eeg_devices[0],),  # Use the first EEG device
                daemon=True
            )
            self.threads['eeg_acquisition'].start()
        
        # Start biosignal acquisition threads for each biosignal device
        for i, device in enumerate(biosignal_devices):
            thread_name = f'biosignal_acquisition_{i}'
            self.threads[thread_name] = threading.Thread(
                target=self._biosignal_acquisition_loop,
                args=(device,),
                daemon=True
            )
            self.threads[thread_name].start()
    
    def _eeg_acquisition_loop(self, device):
        """
        Main loop for EEG data acquisition.
        
        Args:
            device: EEG device object
        """
        self.logger.info(f"Starting EEG acquisition from device: {device.name}")
        
        buffer_size = self.config.get('data_processing', {}).get('eeg', {}).get('buffer_size', 1000)
        eeg_buffer = np.zeros((device.n_channels, buffer_size))
        
        while self.is_running:
            try:
                # Get new EEG data from device
                new_data = device.get_data()
                
                if new_data is not None and new_data.size > 0:
                    # Update buffer
                    n_samples = new_data.shape[1]
                    eeg_buffer = np.roll(eeg_buffer, -n_samples, axis=1)
                    eeg_buffer[:, -n_samples:] = new_data
                    
                    # Process EEG data
                    processed_data = self.eeg_processor.process(eeg_buffer)
                    
                    # Update brainwave synchronization module
                    self.brainwave_sync.update_buffer(new_data)
                    
                    # Save data if session is active
                    if self.current_session is not None:
                        timestamp = datetime.now().isoformat()
                        data_path = f"sessions/{self.current_session}/eeg/{timestamp}.npy"
                        self.data_manager.save_array(data_path, new_data)
                
                # Small sleep to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                self.logger.error(f"Error in EEG acquisition: {str(e)}")
                time.sleep(1.0)  # Longer sleep on error
    
    def _biosignal_acquisition_loop(self, device):
        """
        Main loop for biosignal data acquisition.
        
        Args:
            device: Biosignal device object
        """
        self.logger.info(f"Starting biosignal acquisition from device: {device.name}")
        
        while self.is_running:
            try:
                # Get new biosignal data from device
                new_data = device.get_data()
                
                if new_data is not None and len(new_data) > 0:
                    # Process biosignal data
                    processed_data = self.biosignal_processor.process(
                        new_data, 
                        signal_type=device.signal_type
                    )
                    
                    # Update plasticity detector if ECG data available
                    if device.signal_type == 'ecg' and 'rr_intervals' in processed_data:
                        self.plasticity_detector.update_features(
                            rr_intervals=processed_data['rr_intervals'],
                            timestamp=datetime.now()
                        )
                    
                    # Save data if session is active
                    if self.current_session is not None:
                        timestamp = datetime.now().isoformat()
                        data_path = f"sessions/{self.current_session}/biosignals/{device.signal_type}/{timestamp}.json"
                        self.data_manager.save_json(data_path, processed_data)
                
                # Small sleep to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in biosignal acquisition: {str(e)}")
                time.sleep(1.0)  # Longer sleep on error
    
    def _control_loop(self):
        """Main control loop for system operation."""
        self.logger.info("Starting main control loop")
        
        control_interval = self.config.get('system', {}).get('control_interval', 0.1)
        
        while self.is_running:
            try:
                # Update system state based on current conditions
                self._update_system_state()
                
                # Perform state-specific actions
                if self.state == SystemState.MONITORING:
                    self._monitoring_actions()
                elif self.state == SystemState.STIMULATING:
                    self._stimulation_actions()
                elif self.state == SystemState.ANALYZING:
                    self._analysis_actions()
                elif self.state == SystemState.ERROR:
                    self._error_actions()
                
                # Sleep for the specified control interval
                time.sleep(control_interval)
                
            except Exception as e:
                self.logger.error(f"Error in control loop: {str(e)}")
                self.state = SystemState.ERROR
                self.last_error = str(e)
                time.sleep(1.0)  # Longer sleep on error
    
    def _update_system_state(self):
        """Update system state based on current conditions."""
        # Skip state update if in special states
        if self.state in [SystemState.INITIALIZING, SystemState.CALIBRATING, 
                        SystemState.SHUTDOWN, SystemState.ERROR]:
            return
        
        # Check for device errors
        if self.device_manager.check_errors():
            self.state = SystemState.ERROR
            self.last_error = "Device error detected"
            return
        
        # Check for active stimulation
        stimulation_devices = self.device_manager.get_devices_by_type('stimulator')
        if any(device.is_stimulating() for device in stimulation_devices):
            if self.state != SystemState.STIMULATING:
                self.state = SystemState.STIMULATING
            return
        
        # Default to monitoring state
        if self.state != SystemState.MONITORING:
            self.state = SystemState.MONITORING
    
    def _monitoring_actions(self):
        """Perform actions during monitoring state."""
        # Check if current time is in a high plasticity window
        is_high_plasticity, plasticity_info = self.plasticity_detector.is_high_plasticity_window()
        
        if is_high_plasticity and self.current_session is not None:
            # Transition to stimulation state if in high plasticity window and session active
            target_phase = self.config.get('algorithms', {}).get('brainwave_sync', {}).get('target_phase', 0)
            
            # Prepare stimulation pattern based on optimal timing
            self._prepare_stimulation(target_phase)
            
            # Log plasticity window detection
            self.logger.info(f"High plasticity window detected, score: {plasticity_info['plasticity_score']:.2f}")
            
            # Update session data
            self.session_manager.update_session(
                self.current_session,
                {'plasticity_events': plasticity_info}
            )
    
    def _prepare_stimulation(self, target_phase):
        """
        Prepare stimulation pattern based on brainwave phase and neural state.
        
        Args:
            target_phase (float): Target phase for stimulation in degrees
        """
        # Get optimal stimulation timing
        timing, confidence = self.brainwave_sync.get_optimal_stimulation_timings(target_phase)
        
        if confidence < 0.5:
            self.logger.warning(f"Low confidence in phase estimation: {confidence:.2f}")
            return
        
        # Get stimulation devices
        stimulation_devices = self.device_manager.get_devices_by_type('stimulator')
        if not stimulation_devices:
            self.logger.error("No stimulation devices available")
            return
        
        # Generate spatiotemporal stimulation pattern
        pattern = self.pattern_generator.generate_pattern()
        
        # Apply pattern to stimulation devices with appropriate timing
        delay_ms = int(timing * 1000 / self.brainwave_sync.sampling_rate)
        
        for i, device in enumerate(stimulation_devices):
            # Calculate device-specific parameters based on pattern
            params = {
                'amplitude': pattern['amplitudes'][i] if i < len(pattern['amplitudes']) else 0,
                'frequency': pattern['frequencies'][i] if i < len(pattern['frequencies']) else 0,
                'duration': pattern['durations'][i] if i < len(pattern['durations']) else 0,
                'delay': delay_ms + pattern['delays'][i] if i < len(pattern['delays']) else delay_ms
            }
            
            # Schedule stimulation on device
            device.schedule_stimulation(params)
            
            # Log stimulation event
            stimulation_event = {
                'device_id': device.id,
                'parameters': params,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update session data if active
            if self.current_session is not None:
                self.session_manager.add_event(
                    self.current_session,
                    'stimulation_events',
                    stimulation_event
                )
    
    def _stimulation_actions(self):
        """Perform actions during stimulation state."""
        # Monitor neural responses during stimulation
        eeg_devices = self.device_manager.get_devices_by_type('eeg')
        if not eeg_devices:
            return
        
        # Get latest EEG data
        eeg_data = eeg_devices[0].get_data()
        if eeg_data is None or eeg_data.size == 0:
            return
        
        # Analyze neural response to stimulation
        response = self.response_analyzer.analyze(eeg_data)
        
        # Update adaptive feedback controller with response data
        self.feedback_controller.update(response)
        
        # Get adaptive parameter adjustments
        adjustments = self.feedback_controller.get_adjustments()
        
        # Apply adjustments to ongoing or future stimulation
        if adjustments is not None:
            stimulation_devices = self.device_manager.get_devices_by_type('stimulator')
            for device in stimulation_devices:
                if device.is_stimulating():
                    device.adjust_parameters(adjustments)
        
        # Record neural response
        if self.current_session is not None and response is not None:
            response_event = {
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
            
            self.session_manager.add_event(
                self.current_session,
                'neural_responses',
                response_event
            )
    
    def _analysis_actions(self):
        """Perform actions during analysis state."""
        # Analyze accumulated data to update models and algorithms
        if self.current_session is not None:
            # Get session data
            session_data = self.session_manager.get_session(self.current_session)
            
            # Update plasticity model with observed outcomes
            if 'neural_responses' in session_data and 'stimulation_events' in session_data:
                # Pair stimulation events with corresponding neural responses
                paired_data = self._pair_stimulation_responses(
                    session_data['stimulation_events'],
                    session_data['neural_responses']
                )
                
                for pair in paired_data:
                    # Extract plasticity-relevant features
                    features = self._extract_plasticity_features(pair)
                    
                    # Calculate outcome score based on neural response
                    outcome_score = self._calculate_outcome_score(pair['response'])
                    
                    # Add data point to plasticity detector's history
                    self.plasticity_detector.add_to_history(features, outcome_score)
                
                # Retrain plasticity model
                self.plasticity_detector.train_model()
    
    def _pair_stimulation_responses(self, stimulation_events, neural_responses):
        """
        Pair stimulation events with corresponding neural responses.
        
        Args:
            stimulation_events (list): List of stimulation event dictionaries
            neural_responses (list): List of neural response dictionaries
            
        Returns:
            list: List of paired data dictionaries
        """
        paired_data = []
        
        for stim_event in stimulation_events:
            stim_time = datetime.fromisoformat(stim_event['timestamp'])
            
            # Find the closest neural response after the stimulation
            closest_response = None
            min_time_diff = float('inf')
            
            for response_event in neural_responses:
                response_time = datetime.fromisoformat(response_event['timestamp'])
                time_diff = (response_time - stim_time).total_seconds()
                
                # Only consider responses after stimulation within a reasonable time window (1 second)
                if 0 < time_diff < 1.0 and time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_response = response_event
            
            # If a matching response was found, add to paired data
            if closest_response is not None:
                paired_data.append({
                    'stimulation': stim_event,
                    'response': closest_response,
                    'time_diff': min_time_diff
                })
        
        return paired_data
    
    def _extract_plasticity_features(self, stim_response_pair):
        """
        Extract features relevant for plasticity prediction from a stimulation-response pair.
        
        Args:
            stim_response_pair (dict): Dictionary containing stimulation and response data
            
        Returns:
            dict: Dictionary of extracted features
        """
        # Extract timestamp
        timestamp = datetime.fromisoformat(stim_response_pair['stimulation']['timestamp'])
        
        # Get current EEG features
        eeg_devices = self.device_manager.get_devices_by_type('eeg')
        eeg_features = {}
        
        if eeg_devices:
            eeg_data = eeg_devices[0].get_data()
            if eeg_data is not None and eeg_data.size > 0:
                # Extract EEG features
                eeg_features = self.plasticity_detector.extract_eeg_features(eeg_data)
        
        # Get HRV features from the latest biosignal data
        biosignal_devices = self.device_manager.get_devices_by_type('biosignal')
        hrv_features = {'hrv_hf': None}
        
        for device in biosignal_devices:
            if device.signal_type == 'ecg':
                biosignal_data = device.get_data()
                if biosignal_data is not None and len(biosignal_data) > 0:
                    # Process ECG data to get RR intervals
                    processed_data = self.biosignal_processor.process(
                        biosignal_data, 
                        signal_type='ecg'
                    )
                    
                    if 'rr_intervals' in processed_data:
                        # Extract HRV features
                        hrv_features = self.plasticity_detector.extract_hrv_features(
                            processed_data['rr_intervals']
                        )
        
        # Get circadian phase and time since sleep
        circadian_features = self.plasticity_detector.calculate_circadian_phase(timestamp)
        
        # Combine all features
        features = {
            'timestamp': timestamp,
            'alpha_theta_ratio': eeg_features.get('alpha_theta_ratio', None),
            'beta_gamma_ratio': eeg_features.get('beta_gamma_ratio', None),
            'hrv_hf': hrv_features.get('hrv_hf', None),
            'circadian_phase': circadian_features.get('circadian_phase', None),
            'time_since_sleep': circadian_features.get('time_since_sleep', None)
        }
        
        return features
    
    def _calculate_outcome_score(self, response_event):
        """
        Calculate an outcome score based on neural response.
        
        Args:
            response_event (dict): Neural response event data
            
        Returns:
            float: Outcome score (0-1)
        """
        response = response_event['response']
        
        # Calculate score based on neural response metrics
        if 'amplitude' in response and 'latency' in response:
            # Normalize amplitude (higher is better)
            norm_amplitude = min(response['amplitude'] / 10.0, 1.0)
            
            # Normalize latency (lower is better)
            norm_latency = max(1.0 - response['latency'] / 0.5, 0.0)
            
            # Weighted combination
            score = 0.7 * norm_amplitude + 0.3 * norm_latency
            
            return score
        
        return 0.5  # Default score if metrics are missing
    
    def _error_actions(self):
        """Perform actions during error state."""
        self.logger.error(f"System in ERROR state: {self.last_error}")
        
        # Attempt to recover from error
        if self.device_manager.check_errors():
            # Try to reconnect problematic devices
            self.device_manager.reconnect_devices()
        
        # Check if error is resolved
        if not self.device_manager.check_errors():
            self.logger.info("System recovered from error state")
            self.state = SystemState.MONITORING
        else:
            # If error persists, sleep to prevent rapid retry loops
            time.sleep(5.0)
    
    def get_system_status(self):
        """
        Get current system status.
        
        Returns:
            dict: System status information
        """
        # Get device status
        device_status = self.device_manager.get_all_device_status()
        
        # Get session status
        if self.current_session is not None:
            session_status = {
                'session_id': self.current_session,
                'duration': self.session_manager.get_session_duration(self.current_session),
                'events': self.session_manager.get_session_event_counts(self.current_session)
            }
        else:
            session_status = None
        
        # Compile system status
        status = {
            'state': self.state.name,
            'is_running': self.is_running,
            'last_error': self.last_error,
            'devices': device_status,
            'session': session_status,
            'timestamp': datetime.now().isoformat()
        }
        
        return status
    
    def get_stimulation_predictions(self, hours_ahead=24):
        """
        Get predictions of upcoming optimal stimulation windows.
        
        Args:
            hours_ahead (int): Number of hours to predict ahead
            
        Returns:
            list: List of predicted stimulation windows
        """
        # Get plasticity window predictions
        plasticity_windows = self.plasticity_detector.predict_upcoming_windows(
            hours_ahead=hours_ahead,
            interval_minutes=30
        )
        
        # Convert to stimulation windows with additional information
        stimulation_windows = []
        
        for window in plasticity_windows:
            if window['is_high_plasticity']:
                stimulation_windows.append({
                    'start_time': window['timestamp'].isoformat(),
                    'duration_minutes': 30,
                    'plasticity_score': window['plasticity_score'],
                    'confidence': window['confidence'],
                    'recommended_parameters': self._get_recommended_parameters(window['plasticity_score'])
                })
        
        return stimulation_windows
    
    def _get_recommended_parameters(self, plasticity_score):
        """
        Get recommended stimulation parameters based on plasticity score.
        
        Args:
            plasticity_score (float): Plasticity score (0-1)
            
        Returns:
            dict: Recommended stimulation parameters
        """
        # Scale parameters based on plasticity score
        recommended = {
            'amplitude': min(2.0 + 3.0 * plasticity_score, 5.0),  # 2-5 mA
            'frequency': 10 + 20 * plasticity_score,  # 10-30 Hz
            'duration': 60 + 60 * plasticity_score,  # 60-120 seconds
            'electrodes': ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']  # Default electrode set
        }
        
        return recommended
