import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional, Union
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class StimulationDevice:
    """
    Base class for stimulation devices.
    Provides a common interface for different types of stimulation hardware.
    """
    
    def __init__(self, device_id: str, max_current: float = 5.0):
        """
        Initialize the stimulation device.
        
        Args:
            device_id: Unique identifier for the device
            max_current: Maximum allowed current in mA
        """
        self.device_id = device_id
        self.max_current = max_current
        self.current_amplitude = 0.0
        self.frequency = 0.0
        self.phase = 0.0
        self.waveform = 'sine'
        self.is_active = False
        self.is_connected = False
        self.battery_level = 100.0
        self.impedance = {"channel1": 0.0, "channel2": 0.0}
        
    def connect(self) -> bool:
        """
        Establish connection to the device.
        
        Returns:
            True if connection successful, False otherwise
        """
        # Placeholder for actual hardware connection code
        logger.info(f"Connecting to stimulation device {self.device_id}")
        self.is_connected = True
        return self.is_connected
    
    def disconnect(self) -> bool:
        """
        Disconnect from the device.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        # Placeholder for actual hardware disconnection code
        logger.info(f"Disconnecting from stimulation device {self.device_id}")
        self.is_connected = False
        return not self.is_connected
    
    def start_stimulation(self) -> bool:
        """
        Start stimulation with current parameters.
        
        Returns:
            True if stimulation started successfully, False otherwise
        """
        if not self.is_connected:
            logger.error("Cannot start stimulation: Device not connected")
            return False
        
        logger.info(f"Starting stimulation on device {self.device_id}: "
                  f"{self.current_amplitude:.1f}mA, {self.frequency:.1f}Hz, "
                  f"{self.phase:.1f}°, {self.waveform}")
        
        self.is_active = True
        return self.is_active
    
    def stop_stimulation(self) -> bool:
        """
        Stop ongoing stimulation.
        
        Returns:
            True if stimulation stopped successfully, False otherwise
        """
        logger.info(f"Stopping stimulation on device {self.device_id}")
        self.is_active = False
        return not self.is_active
    
    def set_parameters(self, amplitude: Optional[float] = None, 
                      frequency: Optional[float] = None,
                      phase: Optional[float] = None,
                      waveform: Optional[str] = None) -> bool:
        """
        Set stimulation parameters.
        
        Args:
            amplitude: Current amplitude in mA
            frequency: Stimulation frequency in Hz
            phase: Phase offset in degrees
            waveform: Waveform type (sine, square, triangle, etc.)
            
        Returns:
            True if parameters set successfully, False otherwise
        """
        if not self.is_connected:
            logger.error("Cannot set parameters: Device not connected")
            return False
        
        # Update amplitude if provided and within safe limits
        if amplitude is not None:
            if 0 <= amplitude <= self.max_current:
                self.current_amplitude = amplitude
            else:
                logger.error(f"Amplitude {amplitude}mA exceeds safe limit of {self.max_current}mA")
                return False
        
        # Update frequency if provided
        if frequency is not None:
            if 0.1 <= frequency <= 1000:
                self.frequency = frequency
            else:
                logger.error(f"Frequency {frequency}Hz outside supported range (0.1-1000Hz)")
                return False
        
        # Update phase if provided
        if phase is not None:
            # Normalize phase to 0-360 degrees
            self.phase = phase % 360.0
        
        # Update waveform if provided
        if waveform is not None:
            valid_waveforms = ['sine', 'square', 'triangle', 'sawtooth']
            if waveform.lower() in valid_waveforms:
                self.waveform = waveform.lower()
            else:
                logger.error(f"Waveform {waveform} not supported. Must be one of {valid_waveforms}")
                return False
        
        logger.info(f"Parameters set for device {self.device_id}: "
                  f"{self.current_amplitude:.1f}mA, {self.frequency:.1f}Hz, "
                  f"{self.phase:.1f}°, {self.waveform}")
        
        return True
    
    def get_status(self) -> Dict:
        """
        Get current device status.
        
        Returns:
            Dictionary containing device status information
        """
        return {
            "device_id": self.device_id,
            "connected": self.is_connected,
            "active": self.is_active,
            "current_amplitude": self.current_amplitude,
            "frequency": self.frequency,
            "phase": self.phase,
            "waveform": self.waveform,
            "battery_level": self.battery_level,
            "impedance": self.impedance
        }
    
    def check_impedance(self) -> Dict[str, float]:
        """
        Check electrode impedance.
        
        Returns:
            Dictionary of electrode impedances in kOhms
        """
        # Placeholder for actual impedance measurement code
        # In a real implementation, this would communicate with the device
        self.impedance = {"channel1": 2.5 + np.random.rand(), "channel2": 3.0 + np.random.rand()}
        logger.info(f"Device {self.device_id} impedance: {self.impedance}")
        return self.impedance
    
    def update_battery_level(self) -> float:
        """
        Update battery level information.
        
        Returns:
            Battery level percentage
        """
        # Placeholder for actual battery level polling
        # In a real implementation, this would communicate with the device
        if self.is_active:
            # Battery drains faster when stimulation is active
            self.battery_level = max(0, self.battery_level - np.random.uniform(0.05, 0.1))
        else:
            # Battery drains slower when idle
            self.battery_level = max(0, self.battery_level - np.random.uniform(0.01, 0.02))
            
        return self.battery_level
    
    def reset(self) -> bool:
        """
        Reset device to default state.
        
        Returns:
            True if reset successful, False otherwise
        """
        logger.info(f"Resetting device {self.device_id}")
        self.stop_stimulation()
        self.current_amplitude = 0.0
        self.frequency = 0.0
        self.phase = 0.0
        self.waveform = 'sine'
        return True


class WirelessStimulationDevice(StimulationDevice):
    """
    Class for wireless stimulation devices with Bluetooth connectivity.
    """
    
    def __init__(self, device_id: str, max_current: float = 5.0, 
                bluetooth_address: Optional[str] = None):
        """
        Initialize the wireless stimulation device.
        
        Args:
            device_id: Unique identifier for the device
            max_current: Maximum allowed current in mA
            bluetooth_address: Bluetooth MAC address of the device
        """
        super().__init__(device_id, max_current)
        self.bluetooth_address = bluetooth_address or f"00:11:22:33:44:{device_id[-2:]}"
        self.connection_strength = 100.0  # Signal strength as percentage
    
    def connect(self) -> bool:
        """
        Establish Bluetooth connection to the device.
        
        Returns:
            True if connection successful, False otherwise
        """
        # Placeholder for actual Bluetooth connection code
        logger.info(f"Connecting to wireless device {self.device_id} at {self.bluetooth_address}")
        
        # Simulate connection process
        time.sleep(0.5)  # Simulate connection delay
        
        self.is_connected = True
        self.connection_strength = 85.0 + np.random.uniform(0, 15)
        return self.is_connected
    
    def get_status(self) -> Dict:
        """
        Get current device status including wireless-specific information.
        
        Returns:
            Dictionary containing device status information
        """
        status = super().get_status()
        status.update({
            "bluetooth_address": self.bluetooth_address,
            "connection_strength": self.connection_strength
        })
        return status
    
    def check_connection_strength(self) -> float:
        """
        Check wireless connection signal strength.
        
        Returns:
            Signal strength as percentage
        """
        # Placeholder for actual signal strength measurement
        if self.is_connected:
            # Simulate some random fluctuation in signal strength
            self.connection_strength = min(100, max(0, self.connection_strength + 
                                                 np.random.uniform(-5, 5)))
        else:
            self.connection_strength = 0.0
            
        logger.info(f"Device {self.device_id} connection strength: {self.connection_strength:.1f}%")
        return self.connection_strength


class PhaseLockedDevice(WirelessStimulationDevice):
    """
    Class for stimulation devices that support phase-locked stimulation
    between multiple channels or devices.
    """
    
    def __init__(self, device_id: str, max_current: float = 5.0, 
                bluetooth_address: Optional[str] = None,
                num_channels: int = 2):
        """
        Initialize the phase-locked stimulation device.
        
        Args:
            device_id: Unique identifier for the device
            max_current: Maximum allowed current in mA
            bluetooth_address: Bluetooth MAC address of the device
            num_channels: Number of stimulation channels
        """
        super().__init__(device_id, max_current, bluetooth_address)
        self.num_channels = num_channels
        self.channel_phases = [0.0] * num_channels
        self.channel_amplitudes = [0.0] * num_channels
        self.phase_locked = False
        self.master_frequency = 0.0
    
    def set_channel_parameters(self, channel: int, amplitude: Optional[float] = None,
                              phase: Optional[float] = None) -> bool:
        """
        Set parameters for a specific channel.
        
        Args:
            channel: Channel index (0 to num_channels-1)
            amplitude: Current amplitude in mA
            phase: Phase offset in degrees
            
        Returns:
            True if parameters set successfully, False otherwise
        """
        if not 0 <= channel < self.num_channels:
            logger.error(f"Invalid channel index: {channel}. Must be between 0 and {self.num_channels-1}")
            return False
        
        # Update amplitude if provided and within safe limits
        if amplitude is not None:
            if 0 <= amplitude <= self.max_current:
                self.channel_amplitudes[channel] = amplitude
            else:
                logger.error(f"Amplitude {amplitude}mA exceeds safe limit of {self.max_current}mA")
                return False
        
        # Update phase if provided
        if phase is not None:
            # Normalize phase to 0-360 degrees
            self.channel_phases[channel] = phase % 360.0
        
        logger.info(f"Parameters set for device {self.device_id}, channel {channel}: "
                  f"amplitude={self.channel_amplitudes[channel]:.1f}mA, "
                  f"phase={self.channel_phases[channel]:.1f}°")
        
        return True
    
    def enable_phase_locking(self, frequency: float) -> bool:
        """
        Enable phase-locked stimulation across all channels.
        
        Args:
            frequency: Master frequency for phase-locked stimulation (Hz)
            
        Returns:
            True if phase-locking enabled successfully, False otherwise
        """
        if not 0.1 <= frequency <= 1000:
            logger.error(f"Frequency {frequency}Hz outside supported range (0.1-1000Hz)")
            return False
        
        self.master_frequency = frequency
        self.phase_locked = True
        
        logger.info(f"Phase-locked stimulation enabled on device {self.device_id} "
                  f"at {self.master_frequency:.1f}Hz")
        
        return True
    
    def disable_phase_locking(self) -> bool:
        """
        Disable phase-locked stimulation.
        
        Returns:
            True if phase-locking disabled successfully, False otherwise
        """
        self.phase_locked = False
        logger.info(f"Phase-locked stimulation disabled on device {self.device_id}")
        return True
    
    def get_status(self) -> Dict:
        """
        Get current device status including phase-locking information.
        
        Returns:
            Dictionary containing device status information
        """
        status = super().get_status()
        status.update({
            "num_channels": self.num_channels,
            "channel_phases": self.channel_phases,
            "channel_amplitudes": self.channel_amplitudes,
            "phase_locked": self.phase_locked,
            "master_frequency": self.master_frequency
        })
        return status


class StimulationDeviceManager:
    """
    Class for managing multiple stimulation devices.
    Provides methods for coordinating stimulation across devices.
    """
    
    def __init__(self):
        """
        Initialize the stimulation device manager.
        """
        self.devices = {}  # Maps device_id to device object
        self.active_devices = set()  # Set of active device_ids
        self.monitoring_thread = None
        self.is_monitoring = False
        self.lock = threading.Lock()
    
    def add_device(self, device: StimulationDevice) -> bool:
        """
        Add a device to the manager.
        
        Args:
            device: StimulationDevice object
            
        Returns:
            True if device added successfully, False otherwise
        """
        device_id = device.device_id
        
        if device_id in self.devices:
            logger.warning(f"Device {device_id} already exists in manager. Replacing.")
            
        self.devices[device_id] = device
        logger.info(f"Device {device_id} added to manager")
        return True
    
    def remove_device(self, device_id: str) -> bool:
        """
        Remove a device from the manager.
        
        Args:
            device_id: ID of the device to remove
            
        Returns:
            True if device removed successfully, False otherwise
        """
        if device_id not in self.devices:
            logger.warning(f"Device {device_id} not found in manager")
            return False
        
        # Stop stimulation and disconnect if active
        device = self.devices[device_id]
        if device.is_active:
            device.stop_stimulation()
        if device.is_connected:
            device.disconnect()
            
        # Remove device from collections
        if device_id in self.active_devices:
            self.active_devices.remove(device_id)
        del self.devices[device_id]
        
        logger.info(f"Device {device_id} removed from manager")
        return True
    
    def get_device(self, device_id: str) -> Optional[StimulationDevice]:
        """
        Get a device by ID.
        
        Args:
            device_id: ID of the device to retrieve
            
        Returns:
            StimulationDevice object or None if not found
        """
        return self.devices.get(device_id)
    
    def connect_all_devices(self) -> Dict[str, bool]:
        """
        Connect to all devices.
        
        Returns:
            Dictionary mapping device_id to connection success status
        """
        results = {}
        
        for device_id, device in self.devices.items():
            results[device_id] = device.connect()
            
        logger.info(f"Connected to {sum(results.values())}/{len(results)} devices")
        return results
    
    def disconnect_all_devices(self) -> Dict[str, bool]:
        """
        Disconnect from all devices.
        
        Returns:
            Dictionary mapping device_id to disconnection success status
        """
        results = {}
        
        # First stop all active stimulation
        self.stop_all_stimulation()
        
        # Then disconnect from each device
        for device_id, device in self.devices.items():
            results[device_id] = device.disconnect()
            
        logger.info(f"Disconnected from {sum(results.values())}/{len(results)} devices")
        return results
    
    def start_stimulation(self, device_id: str) -> bool:
        """
        Start stimulation on a specific device.
        
        Args:
            device_id: ID of the device to start stimulation on
            
        Returns:
            True if stimulation started successfully, False otherwise
        """
        device = self.get_device(device_id)
        if not device:
            logger.error(f"Device {device_id} not found")
            return False
        
        success = device.start_stimulation()
        if success:
            with self.lock:
                self.active_devices.add(device_id)
                
        return success
    
    def stop_stimulation(self, device_id: str) -> bool:
        """
        Stop stimulation on a specific device.
        
        Args:
            device_id: ID of the device to stop stimulation on
            
        Returns:
            True if stimulation stopped successfully, False otherwise
        """
        device = self.get_device(device_id)
        if not device:
            logger.error(f"Device {device_id} not found")
            return False
        
        success = device.stop_stimulation()
        if success:
            with self.lock:
                if device_id in self.active_devices:
                    self.active_devices.remove(device_id)
                    
        return success
    
    def start_all_stimulation(self) -> Dict[str, bool]:
        """
        Start stimulation on all connected devices.
        
        Returns:
            Dictionary mapping device_id to stimulation start success status
        """
        results = {}
        
        for device_id, device in self.devices.items():
            if device.is_connected:
                results[device_id] = self.start_stimulation(device_id)
            else:
                results[device_id] = False
                
        logger.info(f"Started stimulation on {sum(results.values())}/{len(results)} devices")
        return results
    
    def stop_all_stimulation(self) -> Dict[str, bool]:
        """
        Stop stimulation on all devices.
        
        Returns:
            Dictionary mapping device_id to stimulation stop success status
        """
        results = {}
        
        # Make a copy of active_devices to iterate over
        active_devices_copy = set(self.active_devices)
        
        for device_id in active_devices_copy:
            results[device_id] = self.stop_stimulation(device_id)
                
        logger.info(f"Stopped stimulation on {sum(results.values())}/{len(results)} devices")
        return results
    
    def set_parameters_all(self, amplitude: Optional[float] = None, 
                         frequency: Optional[float] = None,
                         phase: Optional[float] = None,
                         waveform: Optional[str] = None) -> Dict[str, bool]:
        """
        Set parameters on all connected devices.
        
        Args:
            amplitude: Current amplitude in mA
            frequency: Stimulation frequency in Hz
            phase: Phase offset in degrees
            waveform: Waveform type
            
        Returns:
            Dictionary mapping device_id to parameter set success status
        """
        results = {}
        
        for device_id, device in self.devices.items():
            if device.is_connected:
                results[device_id] = device.set_parameters(
                    amplitude=amplitude,
                    frequency=frequency,
                    phase=phase,
                    waveform=waveform
                )
            else:
                results[device_id] = False
                
        return results
    
    def create_phase_sequence(self, device_ids: List[str], base_phase: float = 0.0, 
                             step_size: float = 45.0) -> Dict[str, bool]:
        """
        Create a sequential phase pattern across multiple devices.
        
        Args:
            device_ids: List of device IDs to include in the sequence
            base_phase: Starting phase value in degrees
            step_size: Phase difference between consecutive devices in degrees
            
        Returns:
            Dictionary mapping device_id to success status
        """
        results = {}
        
        for i, device_id in enumerate(device_ids):
            device = self.get_device(device_id)
            if not device:
                logger.error(f"Device {device_id} not found")
                results[device_id] = False
                continue
                
            phase = (base_phase + i * step_size) % 360.0
            results[device_id] = device.set_parameters(phase=phase)
            
        return results
    
    def start_monitoring(self, interval: float = 5.0) -> bool:
        """
        Start a background thread to monitor device status.
        
        Args:
            interval: Monitoring interval in seconds
            
        Returns:
            True if monitoring started successfully, False otherwise
        """
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return False
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Device monitoring started with {interval}s interval")
        return True
    
    def stop_monitoring(self) -> bool:
        """
        Stop the monitoring thread.
        
        Returns:
            True if monitoring stopped successfully, False otherwise
        """
        if not self.is_monitoring:
            logger.warning("Monitoring not active")
            return False
        
        self.is_monitoring = False
        
        # Wait for thread to terminate
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
            
        logger.info("Device monitoring stopped")
        return True
    
    def _monitoring_loop(self, interval: float) -> None:
        """
        Background loop for device monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        while self.is_monitoring:
            try:
                # Check each device
                for device_id, device in self.devices.items():
                    if device.is_connected:
                        # Update battery level
                        battery_level = device.update_battery_level()
                        
                        # Log low battery warnings
                        if battery_level < 20:
                            logger.warning(f"Device {device_id} battery low: {battery_level:.1f}%")
                            
                        # Check impedance if device is active
                        if device.is_active:
                            impedance = device.check_impedance()
                            
                            # Log high impedance warnings
                            for channel, value in impedance.items():
                                if value > 10:  # kOhms
                                    logger.warning(f"Device {device_id}, {channel} impedance high: {value:.1f} kOhm")
                                    
                        # For wireless devices, check connection strength
                        if isinstance(device, WirelessStimulationDevice):
                            signal = device.check_connection_strength()
                            
                            # Log low signal warnings
                            if signal < 30:
                                logger.warning(f"Device {device_id} signal strength low: {signal:.1f}%")
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                
            # Sleep for the specified interval
            time.sleep(interval)
