"""
Neural Stimulation Simulation Application

This module provides a GUI-based simulation of the Adaptive Phase-Differential 
Brain Nerve Stimulation System, allowing users to visualize the effects of various 
stimulation parameters on neural circuit dynamics.

The simulation is based on biophysically realistic neural network models and 
provides an interactive environment for experimenting with the system.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import os
import time
from pathlib import Path

# Add parent directory to path to allow imports from src
import sys
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import relevant modules from src
from src.algorithms.brainwave_sync import BrainwaveSynchronization
from src.algorithms.plasticity_window import PlasticityWindowDetector
from src.data_processing.eeg_processor import EEGProcessor

class NeuralCircuitSimulator:
    """
    Simulates neural circuit dynamics and responses to stimulation.
    """
    
    def __init__(self, n_neurons=100, connection_probability=0.1, 
                simulation_dt=0.1, sampling_rate=1000):
        """
        Initialize the neural circuit simulator.
        
        Args:
            n_neurons (int): Number of neurons in the simulated circuit
            connection_probability (float): Probability of connection between neurons
            simulation_dt (float): Simulation time step in milliseconds
            sampling_rate (float): Sampling rate in Hz
        """
        self.n_neurons = n_neurons
        self.connection_probability = connection_probability
        self.dt = simulation_dt  # in ms
        self.sampling_rate = sampling_rate
        
        # Initialize neuron parameters
        self._init_neurons()
        
        # Initialize network connectivity
        self._init_connectivity()
        
        # Set up recording variables
        self.spike_history = []
        self.membrane_potential_history = []
        self.field_potential_history = []
        
        # Stimulation parameters
        self.stim_amplitude = 0.0  # mA
        self.stim_frequency = 0.0  # Hz
        self.stim_phase = 0.0  # degrees
        self.stim_duration = 0.0  # ms
        self.stim_active = False
        self.stim_start_time = 0.0  # ms
        
        # Current simulation time
        self.time = 0.0  # ms
        
    def _init_neurons(self):
        """Initialize neuron parameters."""
        # Neuron types (80% excitatory, 20% inhibitory)
        self.neuron_types = np.random.choice(
            ['excitatory', 'inhibitory'], 
            size=self.n_neurons,
            p=[0.8, 0.2]
        )
        
        # Initialize neuron state variables
        self.v = -65.0 * np.ones(self.n_neurons)  # Membrane potential (mV)
        self.u = -14.0 * np.ones(self.n_neurons)  # Recovery variable
        
        # Neuron parameters (Izhikevich model)
        self.a = np.zeros(self.n_neurons)
        self.b = np.zeros(self.n_neurons)
        self.c = np.zeros(self.n_neurons)
        self.d = np.zeros(self.n_neurons)
        
        # Set parameters based on neuron type
        for i in range(self.n_neurons):
            if self.neuron_types[i] == 'excitatory':
                # Regular spiking neurons
                self.a[i] = 0.02
                self.b[i] = 0.2
                self.c[i] = -65.0
                self.d[i] = 8.0
            else:
                # Fast spiking neurons
                self.a[i] = 0.1
                self.b[i] = 0.2
                self.c[i] = -65.0
                self.d[i] = 2.0
        
        # Add heterogeneity
        self.a += 0.008 * np.random.randn(self.n_neurons)
        self.b += 0.05 * np.random.randn(self.n_neurons)
        self.c += 15.0 * np.random.randn(self.n_neurons)
        self.d += 4.0 * np.random.randn(self.n_neurons)
        
        # Spatial positions (for visualization and stimulation)
        self.positions = np.random.rand(self.n_neurons, 3)  # 3D positions
        
        # Background noise level
        self.noise_level = 1.0
    
    def _init_connectivity(self):
        """Initialize neural network connectivity."""
        # Generate random connectivity matrix
        self.weights = np.random.rand(self.n_neurons, self.n_neurons) < self.connection_probability
        
        # Set weight values based on pre-synaptic neuron type
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if self.weights[i, j]:
                    if self.neuron_types[i] == 'excitatory':
                        # Excitatory synapse
                        self.weights[i, j] = 0.5 * np.random.rand()
                    else:
                        # Inhibitory synapse
                        self.weights[i, j] = -1.0 * np.random.rand()
        
        # No self-connections
        np.fill_diagonal(self.weights, 0)
        
        # Initialize synaptic variables
        self.synaptic_currents = np.zeros(self.n_neurons)
        self.synaptic_conductances = np.zeros((self.n_neurons, self.n_neurons))
        
        # Synaptic time constants (ms)
        self.tau_exc = 5.0  # Excitatory synaptic time constant
        self.tau_inh = 10.0  # Inhibitory synaptic time constant
    
    def set_stimulation_parameters(self, amplitude, frequency, phase, duration):
        """
        Set stimulation parameters.
        
        Args:
            amplitude (float): Stimulation amplitude in mA
            frequency (float): Stimulation frequency in Hz
            phase (float): Stimulation phase in degrees
            duration (float): Stimulation duration in ms
        """
        self.stim_amplitude = amplitude
        self.stim_frequency = frequency
        self.stim_phase = phase
        self.stim_duration = duration
    
    def start_stimulation(self):
        """Start stimulation at the current simulation time."""
        self.stim_active = True
        self.stim_start_time = self.time
    
    def stop_stimulation(self):
        """Stop ongoing stimulation."""
        self.stim_active = False
    
    def _calculate_stimulation_current(self):
        """
        Calculate the stimulation current for each neuron based on
        current simulation time and stimulation parameters.
        
        Returns:
            numpy.ndarray: Stimulation current for each neuron
        """
        if not self.stim_active:
            return np.zeros(self.n_neurons)
        
        # Check if stimulation duration has elapsed
        if self.time - self.stim_start_time > self.stim_duration:
            self.stim_active = False
            return np.zeros(self.n_neurons)
        
        # Calculate phase of stimulation
        time_s = self.time / 1000.0  # Convert to seconds
        stim_phase_rad = np.radians(self.stim_phase)
        
        # Sinusoidal stimulation
        stim_value = self.stim_amplitude * np.sin(
            2 * np.pi * self.stim_frequency * time_s + stim_phase_rad
        )
        
        # Calculate current for each neuron based on spatial position
        # (simulate electric field falloff with distance)
        stim_center = np.array([0.5, 0.5, 0.5])  # Center of stimulation
        distances = np.linalg.norm(self.positions - stim_center, axis=1)
        field_falloff = np.exp(-distances * 5)  # Exponential falloff with distance
        
        stim_currents = stim_value * field_falloff * 20.0  # Scale to appropriate range
        
        return stim_currents
    
    def _update_synaptic_currents(self, spikes):
        """
        Update synaptic currents based on spikes.
        
        Args:
            spikes (numpy.ndarray): Boolean array indicating which neurons spiked
        """
        # Decay existing conductances
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if self.weights[i, j] > 0:
                    # Excitatory synapse
                    self.synaptic_conductances[i, j] *= np.exp(-self.dt / self.tau_exc)
                elif self.weights[i, j] < 0:
                    # Inhibitory synapse
                    self.synaptic_conductances[i, j] *= np.exp(-self.dt / self.tau_inh)
        
        # Update conductances for neurons that just spiked
        for i in range(self.n_neurons):
            if spikes[i]:
                for j in range(self.n_neurons):
                    if self.weights[i, j] != 0:
                        self.synaptic_conductances[i, j] += np.abs(self.weights[i, j])
        
        # Calculate total synaptic current for each neuron
        self.synaptic_currents = np.zeros(self.n_neurons)
        for j in range(self.n_neurons):
            # Sum over all inputs to neuron j
            for i in range(self.n_neurons):
                if self.weights[i, j] != 0:
                    # Current depends on conductance and reversal potential
                    if self.weights[i, j] > 0:
                        # Excitatory current (reversal at 0 mV)
                        self.synaptic_currents[j] += self.synaptic_conductances[i, j] * (0 - self.v[j])
                    else:
                        # Inhibitory current (reversal at -80 mV)
                        self.synaptic_currents[j] += self.synaptic_conductances[i, j] * (-80 - self.v[j])
    
    def step(self, external_input=None):
        """
        Advance the simulation by one time step.
        
        Args:
            external_input (numpy.ndarray, optional): External current input to each neuron
            
        Returns:
            dict: Simulation state including spikes and field potential
        """
        # Initialize external input if not provided
        if external_input is None:
            external_input = np.zeros(self.n_neurons)
        
        # Add noise to external input
        noise = self.noise_level * np.random.randn(self.n_neurons)
        external_input = external_input + noise
        
        # Add stimulation current if active
        stim_current = self._calculate_stimulation_current()
        external_input = external_input + stim_current
        
        # Total input current: external + synaptic
        total_input = external_input + self.synaptic_currents
        
        # Initialize spike array
        spikes = np.zeros(self.n_neurons, dtype=bool)
        
        # Update each neuron (Izhikevich model)
        for i in range(self.n_neurons):
            # Update membrane potential
            self.v[i] += self.dt * (0.04 * self.v[i]**2 + 5 * self.v[i] + 140 - self.u[i] + total_input[i])
            
            # Update recovery variable
            self.u[i] += self.dt * self.a[i] * (self.b[i] * self.v[i] - self.u[i])
            
            # Check for spike
            if self.v[i] >= 30:
                spikes[i] = True
                self.v[i] = self.c[i]  # Reset membrane potential
                self.u[i] += self.d[i]  # Reset recovery variable
        
        # Update synaptic currents based on spikes
        self._update_synaptic_currents(spikes)
        
        # Calculate field potential (simplified as average membrane potential)
        field_potential = np.mean(self.v)
        
        # Record history
        self.spike_history.append(spikes)
        self.membrane_potential_history.append(self.v.copy())
        self.field_potential_history.append(field_potential)
        
        # Advance time
        self.time += self.dt
        
        return {
            'time': self.time,
            'spikes': spikes,
            'membrane_potentials': self.v.copy(),
            'field_potential': field_potential,
            'stimulation_active': self.stim_active
        }
    
    def run(self, duration, external_input=None):
        """
        Run simulation for specified duration.
        
        Args:
            duration (float): Simulation duration in ms
            external_input (callable or numpy.ndarray, optional): 
                External input function or array
                
        Returns:
            dict: Simulation results
        """
        # Calculate number of steps
        n_steps = int(duration / self.dt)
        
        # Initialize results
        times = np.zeros(n_steps)
        all_spikes = np.zeros((n_steps, self.n_neurons), dtype=bool)
        field_potentials = np.zeros(n_steps)
        
        # Run simulation
        for i in range(n_steps):
            # Determine external input for this step
            if external_input is None:
                ext_input = None
            elif callable(external_input):
                ext_input = external_input(self.time)
            else:
                ext_input = external_input
            
            # Step simulation
            result = self.step(ext_input)
            
            # Record results
            times[i] = result['time']
            all_spikes[i] = result['spikes']
            field_potentials[i] = result['field_potential']
        
        # Compile results
        results = {
            'times': times,
            'spikes': all_spikes,
            'field_potentials': field_potentials,
            'neuron_types': self.neuron_types,
            'positions': self.positions
        }
        
        return results
    
    def get_eeg_like_signal(self, duration, noise_level=0.1):
        """
        Generate a simulated EEG-like signal based on circuit activity.
        
        Args:
            duration (float): Signal duration in seconds
            noise_level (float): Level of noise to add to the signal
            
        Returns:
            numpy.ndarray: EEG-like signal
        """
        # Run simulation to generate field potentials
        n_steps = int(duration * 1000 / self.dt)  # Convert seconds to ms
        field_potentials = np.zeros(n_steps)
        
        for i in range(n_steps):
            result = self.step()
            field_potentials[i] = result['field_potential']
        
        # Resample to typical EEG sampling rate
        target_samples = int(duration * self.sampling_rate)
        indices = np.linspace(0, n_steps-1, target_samples, dtype=int)
        eeg_signal = field_potentials[indices]
        
        # Add noise
        eeg_signal += noise_level * np.random.randn(len(eeg_signal))
        
        # Scale to typical EEG range (microvolts)
        eeg_signal = eeg_signal * 50
        
        return eeg_signal
    
    def reset(self):
        """Reset the simulation to initial state."""
        self._init_neurons()
        self._init_connectivity()
        self.spike_history = []
        self.membrane_potential_history = []
        self.field_potential_history = []
        self.time = 0.0
        self.stim_active = False


class StimulationEffect:
    """
    Simulates the effects of phase-differential stimulation on neural plasticity.
    """
    
    def __init__(self, simulator, eeg_processor):
        """
        Initialize the stimulation effect simulator.
        
        Args:
            simulator (NeuralCircuitSimulator): Neural circuit simulator
            eeg_processor (EEGProcessor): EEG processing module
        """
        self.simulator = simulator
        self.eeg_processor = eeg_processor
        self.brainwave_sync = BrainwaveSynchronization(
            sampling_rate=simulator.sampling_rate,
            target_band='alpha',
            n_channels=1
        )
        
        # Initialize plasticity parameters
        self._init_plasticity_parameters()
    
    def _init_plasticity_parameters(self):
        """Initialize plasticity-related parameters."""
        # Synaptic weights before stimulation
        self.initial_weights = self.simulator.weights.copy()
        
        # Parameters for STDP
        self.stdp_window = 20.0  # ms
        self.stdp_rate = 0.01
        self.stdp_ratio = 1.2  # LTP/LTD ratio
        
        # Parameters for stimulation effects
        self.phase_sensitivity = {
            'alpha': {
                0: 1.5,     # In-phase: enhanced potentiation
                90: 1.0,    # 90 degrees: neutral
                180: 0.5,   # Anti-phase: reduced potentiation
                270: 1.0    # 270 degrees: neutral
            }
        }
    
    def apply_stdp(self, spikes_pre, spikes_post, delta_t):
        """
        Apply spike-timing dependent plasticity based on pre/post spike timing.
        
        Args:
            spikes_pre (numpy.ndarray): Pre-synaptic spikes
            spikes_post (numpy.ndarray): Post-synaptic spikes
            delta_t (float): Time difference in ms
            
        Returns:
            numpy.ndarray: Weight changes
        """
        weight_changes = np.zeros_like(self.simulator.weights)
        
        # Loop through all neuron pairs
        for i in range(self.simulator.n_neurons):
            for j in range(self.simulator.n_neurons):
                # Skip if no connection
                if self.simulator.weights[i, j] == 0:
                    continue
                
                # Apply STDP rule
                if spikes_pre[i] and spikes_post[j]:
                    # Calculate time difference (post - pre)
                    t_diff = delta_t
                    
                    # STDP function
                    if t_diff > 0:
                        # LTP: pre -> post (causal)
                        weight_change = self.stdp_rate * np.exp(-t_diff / self.stdp_window)
                    else:
                        # LTD: post -> pre (acausal)
                        weight_change = -self.stdp_rate / self.stdp_ratio * np.exp(t_diff / self.stdp_window)
                    
                    weight_changes[i, j] = weight_change
        
        return weight_changes
    
    def simulate_stimulation_effect(self, stim_params, duration=1000):
        """
        Simulate the effect of phase-differential stimulation on neural plasticity.
        
        Args:
            stim_params (dict): Stimulation parameters
            duration (float): Simulation duration in ms
            
        Returns:
            dict: Simulation results including plasticity metrics
        """
        # Extract stimulation parameters
        amplitude = stim_params.get('amplitude', 1.0)
        frequency = stim_params.get('frequency', 10.0)
        phase = stim_params.get('phase', 0.0)
        stim_duration = stim_params.get('duration', 500.0)
        
        # Set stimulation parameters in simulator
        self.simulator.set_stimulation_parameters(amplitude, frequency, phase, stim_duration)
        
        # Get initial state of the network
        initial_weights = self.simulator.weights.copy()
        
        # Generate EEG-like signals before stimulation
        pre_stim_eeg = self.simulator.get_eeg_like_signal(0.5)  # 0.5 seconds
        
        # Extract dominant frequency
        pre_stim_features = self.eeg_processor.extract_features(pre_stim_eeg.reshape(1, -1))
        dominant_freq = pre_stim_features.get('dominant_frequency', 10.0)
        dominant_band = self._get_frequency_band(dominant_freq)
        
        # Calculate phase sensitivity factor
        closest_phase = self._find_closest_phase(phase)
        phase_factor = self.phase_sensitivity.get(dominant_band, {}).get(closest_phase, 1.0)
        
        # Start stimulation
        self.simulator.start_stimulation()
        
        # Run simulation
        results = self.simulator.run(duration)
        
        # Calculate plasticity metrics
        weight_changes = self.simulator.weights - initial_weights
        
        # Scale by phase factor
        weight_changes *= phase_factor
        
        # Analyze changes
        avg_change = np.mean(np.abs(weight_changes))
        potentiation = np.sum(weight_changes > 0) / np.sum(initial_weights != 0)
        depression = np.sum(weight_changes < 0) / np.sum(initial_weights != 0)
        
        # Generate EEG-like signals after stimulation
        post_stim_eeg = self.simulator.get_eeg_like_signal(0.5)  # 0.5 seconds
        
        # Calculate changes in EEG features
        post_stim_features = self.eeg_processor.extract_features(post_stim_eeg.reshape(1, -1))
        
        # Compile plasticity results
        plasticity_results = {
            'average_weight_change': float(avg_change),
            'potentiation_ratio': float(potentiation),
            'depression_ratio': float(depression),
            'phase_factor': phase_factor,
            'dominant_frequency_pre': pre_stim_features.get('dominant_frequency'),
            'dominant_frequency_post': post_stim_features.get('dominant_frequency'),
            'band_power_pre': pre_stim_features.get('band_powers'),
            'band_power_post': post_stim_features.get('band_powers')
        }
        
        # Add plasticity results to simulation results
        results['plasticity'] = plasticity_results
        
        return results
    
    def _get_frequency_band(self, frequency):
        """
        Get the frequency band name for a given frequency.
        
        Args:
            frequency (float): Frequency in Hz
            
        Returns:
            str: Band name ('delta', 'theta', 'alpha', 'beta', 'gamma')
        """
        if frequency < 4:
            return 'delta'
        elif frequency < 8:
            return 'theta'
        elif frequency < 13:
            return 'alpha'
        elif frequency < 30:
            return 'beta'
        else:
            return 'gamma'
    
    def _find_closest_phase(self, phase):
        """
        Find the closest phase value from the phase sensitivity dictionary.
        
        Args:
            phase (float): Phase in degrees
            
        Returns:
            float: Closest phase from the sensitivity dictionary
        """
        # Normalize phase to 0-360
        phase = phase % 360
        
        # Find closest key in phase sensitivity dictionary
        available_phases = list(self.phase_sensitivity.get('alpha', {}).keys())
        closest_phase = min(available_phases, key=lambda x: abs(x - phase))
        
        return closest_phase


class SimulationGUI:
    """
    GUI for the Neural Stimulation Simulation Application.
    """
    
    def __init__(self, root):
        """
        Initialize the simulation GUI.
        
        Args:
            root (tkinter.Tk): Root Tkinter window
        """
        self.root = root
        self.root.title("Adaptive Neural Stimulation Simulator")
        self.root.geometry("1200x800")
        
        # Initialize the simulation components
        self._init_simulator()
        
        # Set up the GUI components
        self._setup_gui()
        
        # Animation parameters
        self.is_animating = False
        self.animation = None
    
    def _init_simulator(self):
        """Initialize simulation components."""
        # Neural circuit simulator
        self.simulator = NeuralCircuitSimulator(
            n_neurons=100,
            connection_probability=0.1,
            simulation_dt=0.1,
            sampling_rate=1000
        )
        
        # EEG processor
        self.eeg_processor = EEGProcessor(
            sampling_rate=1000,
            n_channels=1
        )
        
        # Stimulation effect simulator
        self.stim_effect = StimulationEffect(
            simulator=self.simulator,
            eeg_processor=self.eeg_processor
        )
        
        # Current stimulation parameters
        self.current_params = {
            'amplitude': 1.0,
            'frequency': 10.0,
            'phase': 0.0,
            'duration': 500.0
        }
    
    def _setup_gui(self):
        """Set up the GUI components."""
        # Main container frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel on the left
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Stimulation Parameters")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Visualization panel on the right
        self.vis_frame = ttk.Frame(self.main_frame)
        self.vis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Set up control panel
        self._setup_control_panel()
        
        # Set up visualization panel
        self._setup_visualization_panel()
    
    def _setup_control_panel(self):
        """Set up the control panel."""
        # Stimulation amplitude
        ttk.Label(self.control_frame, text="Amplitude (mA):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.amplitude_var = tk.DoubleVar(value=self.current_params['amplitude'])
        amplitude_slider = ttk.Scale(
            self.control_frame,
            from_=0.0,
            to=5.0,
            orient=tk.HORIZONTAL,
            variable=self.amplitude_var,
            length=200
        )
        amplitude_slider.grid(row=0, column=1, padx=5, pady=5)
        amplitude_slider.bind("<ButtonRelease-1>", self._update_parameters)
        
        # Stimulation frequency
        ttk.Label(self.control_frame, text="Frequency (Hz):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.frequency_var = tk.DoubleVar(value=self.current_params['frequency'])
        frequency_slider = ttk.Scale(
            self.control_frame,
            from_=1.0,
            to=100.0,
            orient=tk.HORIZONTAL,
            variable=self.frequency_var,
            length=200
        )
        frequency_slider.grid(row=1, column=1, padx=5, pady=5)
        frequency_slider.bind("<ButtonRelease-1>", self._update_parameters)
        
        # Stimulation phase
        ttk.Label(self.control_frame, text="Phase (degrees):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.phase_var = tk.DoubleVar(value=self.current_params['phase'])
        phase_slider = ttk.Scale(
            self.control_frame,
            from_=0.0,
            to=360.0,
            orient=tk.HORIZONTAL,
            variable=self.phase_var,
            length=200
        )
        phase_slider.grid(row=2, column=1, padx=5, pady=5)
        phase_slider.bind("<ButtonRelease-1>", self._update_parameters)
        
        # Stimulation duration
        ttk.Label(self.control_frame, text="Duration (ms):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.duration_var = tk.DoubleVar(value=self.current_params['duration'])
        duration_slider = ttk.Scale(
            self.control_frame,
            from_=0.0,
            to=1000.0,
            orient=tk.HORIZONTAL,
            variable=self.duration_var,
            length=200
        )
        duration_slider.grid(row=3, column=1, padx=5, pady=5)
        duration_slider.bind("<ButtonRelease-1>", self._update_parameters)
        
        # Stimulation control buttons
        ttk.Label(self.control_frame, text="Simulation Control:").grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)
        
        # Button frame
        button_frame = ttk.Frame(self.control_frame)
        button_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
        
        # Start simulation button
        self.start_button = ttk.Button(
            button_frame,
            text="Start Simulation",
            command=self.start_simulation
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Reset simulation button
        self.reset_button = ttk.Button(
            button_frame,
            text="Reset Simulation",
            command=self.reset_simulation
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Stimulation button
        self.stim_button = ttk.Button(
            button_frame,
            text="Apply Stimulation",
            command=self.apply_stimulation
        )
        self.stim_button.pack(side=tk.LEFT, padx=5)
        
        # Preset protocols
        ttk.Label(self.control_frame, text="Preset Protocols:").grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)
        
        # Protocol selection
        self.protocol_var = tk.StringVar()
        protocol_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.protocol_var,
            values=[
                "Stroke Rehabilitation",
                "Cognitive Enhancement",
                "Epilepsy Management",
                "Depression Treatment"
            ]
        )
        protocol_combo.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)
        protocol_combo.bind("<<ComboboxSelected>>", self._load_protocol)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.control_frame, text="Simulation Results")
        results_frame.grid(row=8, column=0, columnspan=2, padx=5, pady=10, sticky=tk.EW)
        
        # Results display
        self.results_text = tk.Text(results_frame, height=10, width=30)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Save and load buttons
        save_load_frame = ttk.Frame(self.control_frame)
        save_load_frame.grid(row=9, column=0, columnspan=2, padx=5, pady=5)
        
        # Save results button
        self.save_button = ttk.Button(
            save_load_frame,
            text="Save Results",
            command=self.save_results
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Load results button
        self.load_button = ttk.Button(
            save_load_frame,
            text="Load Results",
            command=self.load_results
        )
        self.load_button.pack(side=tk.LEFT, padx=5)
    
    def _setup_visualization_panel(self):
        """Set up the visualization panel."""
        # Create tabs for different visualizations
        self.notebook = ttk.Notebook(self.vis_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Neural activity tab
        self.activity_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.activity_tab, text="Neural Activity")
        
        # EEG tab
        self.eeg_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.eeg_tab, text="EEG Signal")
        
        # Plasticity tab
        self.plasticity_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plasticity_tab, text="Plasticity")
        
        # Create figures and canvases for each tab
        self._setup_activity_tab()
        self._setup_eeg_tab()
        self._setup_plasticity_tab()
    
    def _setup_activity_tab(self):
        """Set up the neural activity visualization tab."""
        # Create figure and axes
        self.activity_fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.activity_canvas = FigureCanvasTkAgg(self.activity_fig, master=self.activity_tab)
        self.activity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create three subplots: raster plot, voltage traces, and field potential
        self.raster_ax = self.activity_fig.add_subplot(311)
        self.voltage_ax = self.activity_fig.add_subplot(312)
        self.field_ax = self.activity_fig.add_subplot(313)
        
        # Set titles and labels
        self.raster_ax.set_title("Spike Raster Plot")
        self.raster_ax.set_ylabel("Neuron Index")
        
        self.voltage_ax.set_title("Membrane Potentials")
        self.voltage_ax.set_ylabel("Voltage (mV)")
        
        self.field_ax.set_title("Field Potential")
        self.field_ax.set_xlabel("Time (ms)")
        self.field_ax.set_ylabel("Amplitude (μV)")
        
        self.activity_fig.tight_layout()
    
    def _setup_eeg_tab(self):
        """Set up the EEG signal visualization tab."""
        # Create figure and axes
        self.eeg_fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.eeg_canvas = FigureCanvasTkAgg(self.eeg_fig, master=self.eeg_tab)
        self.eeg_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create two subplots: time domain and frequency domain
        self.eeg_time_ax = self.eeg_fig.add_subplot(211)
        self.eeg_freq_ax = self.eeg_fig.add_subplot(212)
        
        # Set titles and labels
        self.eeg_time_ax.set_title("EEG Time Domain")
        self.eeg_time_ax.set_ylabel("Amplitude (μV)")
        
        self.eeg_freq_ax.set_title("EEG Frequency Domain")
        self.eeg_freq_ax.set_xlabel("Frequency (Hz)")
        self.eeg_freq_ax.set_ylabel("Power")
        
        self.eeg_fig.tight_layout()
    
    def _setup_plasticity_tab(self):
        """Set up the plasticity visualization tab."""
        # Create figure and axes
        self.plasticity_fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.plasticity_canvas = FigureCanvasTkAgg(self.plasticity_fig, master=self.plasticity_tab)
        self.plasticity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create two subplots: weight changes and phase sensitivity
        self.weight_ax = self.plasticity_fig.add_subplot(211)
        self.phase_ax = self.plasticity_fig.add_subplot(212)
        
        # Set titles and labels
        self.weight_ax.set_title("Synaptic Weight Changes")
        self.weight_ax.set_ylabel("Normalized Change")
        
        self.phase_ax.set_title("Phase Sensitivity")
        self.phase_ax.set_xlabel("Phase (degrees)")
        self.phase_ax.set_ylabel("Plasticity Factor")
        
        # Plot phase sensitivity curve (constant)
        phases = np.linspace(0, 360, 100)
        sensitivity = np.zeros_like(phases)
        for i, phase in enumerate(phases):
            closest_phase = min(
                self.stim_effect.phase_sensitivity['alpha'].keys(),
                key=lambda x: abs(x - phase)
            )
            sensitivity[i] = self.stim_effect.phase_sensitivity['alpha'][closest_phase]
        
        self.phase_ax.plot(phases, sensitivity)
        self.phase_ax.set_xlim(0, 360)
        self.phase_ax.set_ylim(0, 2.0)
        
        self.plasticity_fig.tight_layout()
    
    def _update_parameters(self, event=None):
        """
        Update the current stimulation parameters from the GUI.
        
        Args:
            event: Tkinter event (unused)
        """
        self.current_params = {
            'amplitude': self.amplitude_var.get(),
            'frequency': self.frequency_var.get(),
            'phase': self.phase_var.get(),
            'duration': self.duration_var.get()
        }
        
        # Update simulator parameters
        self.simulator.set_stimulation_parameters(
            self.current_params['amplitude'],
            self.current_params['frequency'],
            self.current_params['phase'],
            self.current_params['duration']
        )
        
        # Update results display
        self._update_results_display()
    
    def _load_protocol(self, event=None):
        """
        Load a preset stimulation protocol.
        
        Args:
            event: Tkinter event (unused)
        """
        protocol = self.protocol_var.get()
        
        # Set parameters based on selected protocol
        if protocol == "Stroke Rehabilitation":
            params = {
                'amplitude': 2.0,
                'frequency': 20.0,
                'phase': 0.0,
                'duration': 500.0
            }
        elif protocol == "Cognitive Enhancement":
            params = {
                'amplitude': 1.0,
                'frequency': 40.0,
                'phase': 90.0,
                'duration': 300.0
            }
        elif protocol == "Epilepsy Management":
            params = {
                'amplitude': 3.0,
                'frequency': 1.0,
                'phase': 180.0,
                'duration': 1000.0
            }
        elif protocol == "Depression Treatment":
            params = {
                'amplitude': 2.5,
                'frequency': 10.0,
                'phase': 270.0,
                'duration': 800.0
            }
        else:
            return
        
        # Update GUI variables
        self.amplitude_var.set(params['amplitude'])
        self.frequency_var.set(params['frequency'])
        self.phase_var.set(params['phase'])
        self.duration_var.set(params['duration'])
        
        # Update current parameters
        self._update_parameters()
    
    def _update_results_display(self):
        """Update the results display with current parameters and estimated effects."""
        # Clear current text
        self.results_text.delete(1.0, tk.END)
        
        # Add current parameters
        self.results_text.insert(tk.END, "Current Parameters:\n")
        self.results_text.insert(tk.END, f"Amplitude: {self.current_params['amplitude']:.1f} mA\n")
        self.results_text.insert(tk.END, f"Frequency: {self.current_params['frequency']:.1f} Hz\n")
        self.results_text.insert(tk.END, f"Phase: {self.current_params['phase']:.1f} degrees\n")
        self.results_text.insert(tk.END, f"Duration: {self.current_params['duration']:.1f} ms\n\n")
        
        # Add estimated effects
        self.results_text.insert(tk.END, "Estimated Effects:\n")
        
        # Get dominant frequency band
        freq = self.current_params['frequency']
        band = self.stim_effect._get_frequency_band(freq)
        
        # Get phase sensitivity
        phase = self.current_params['phase']
        closest_phase = self.stim_effect._find_closest_phase(phase)
        phase_factor = self.stim_effect.phase_sensitivity.get(band, {}).get(closest_phase, 1.0)
        
        # Add band and plasticity factor
        self.results_text.insert(tk.END, f"Target Band: {band.capitalize()}\n")
        self.results_text.insert(tk.END, f"Plasticity Factor: {phase_factor:.2f}\n")
        
        # Add qualitative assessment
        self.results_text.insert(tk.END, "\nQualitative Assessment:\n")
        
        if phase_factor > 1.2:
            effect = "Strong enhancement of neuroplasticity"
        elif phase_factor > 0.8:
            effect = "Moderate enhancement of neuroplasticity"
        elif phase_factor > 0.5:
            effect = "Mild enhancement of neuroplasticity"
        else:
            effect = "Potential suppression of neuroplasticity"
        
        self.results_text.insert(tk.END, f"{effect}\n")
    
    def start_simulation(self):
        """Start the animated simulation."""
        if self.is_animating:
            return
        
        # Reset simulator
        self.simulator.reset()
        
        # Set stimulation parameters
        self.simulator.set_stimulation_parameters(
            self.current_params['amplitude'],
            self.current_params['frequency'],
            self.current_params['phase'],
            self.current_params['duration']
        )
        
        # Initialize visualization data
        self.sim_time = 0.0
        self.sim_data = {
            'times': [],
            'spikes': [],
            'membrane_potentials': [],
            'field_potentials': []
        }
        
        # Clear previous plots
        self.raster_ax.clear()
        self.voltage_ax.clear()
        self.field_ax.clear()
        
        # Set titles and labels
        self.raster_ax.set_title("Spike Raster Plot")
        self.raster_ax.set_ylabel("Neuron Index")
        
        self.voltage_ax.set_title("Membrane Potentials")
        self.voltage_ax.set_ylabel("Voltage (mV)")
        
        self.field_ax.set_title("Field Potential")
        self.field_ax.set_xlabel("Time (ms)")
        self.field_ax.set_ylabel("Amplitude (μV)")
        
        # Start animation
        self.is_animating = True
        self.animation = FuncAnimation(
            self.activity_fig,
            self._animate,
            interval=50,
            blit=False
        )
        
        self.activity_canvas.draw()
    
    def _animate(self, frame):
        """
        Animation function for updating plots.
        
        Args:
            frame: Frame index
        
        Returns:
            list: List of updated artists
        """
        # Run simulation for a few steps
        for _ in range(10):
            result = self.simulator.step()
            
            # Store results
            self.sim_data['times'].append(result['time'])
            self.sim_data['spikes'].append(result['spikes'])
            self.sim_data['membrane_potentials'].append(result['membrane_potentials'])
            self.sim_data['field_potentials'].append(result['field_potential'])
        
        # Keep only the last 1000 steps
        max_steps = 1000
        if len(self.sim_data['times']) > max_steps:
            self.sim_data['times'] = self.sim_data['times'][-max_steps:]
            self.sim_data['spikes'] = self.sim_data['spikes'][-max_steps:]
            self.sim_data['membrane_potentials'] = self.sim_data['membrane_potentials'][-max_steps:]
            self.sim_data['field_potentials'] = self.sim_data['field_potentials'][-max_steps:]
        
        # Update raster plot
        self.raster_ax.clear()
        self.raster_ax.set_title("Spike Raster Plot")
        self.raster_ax.set_ylabel("Neuron Index")
        
        # Plot spikes
        for t in range(len(self.sim_data['times'])):
            spike_indices = np.where(self.sim_data['spikes'][t])[0]
            if len(spike_indices) > 0:
                self.raster_ax.scatter(
                    [self.sim_data['times'][t]] * len(spike_indices),
                    spike_indices,
                    color='k',
                    marker='|',
                    s=10
                )
        
        self.raster_ax.set_xlim(
            self.sim_data['times'][0] if self.sim_data['times'] else 0,
            self.sim_data['times'][-1] if self.sim_data['times'] else 100
        )
        self.raster_ax.set_ylim(-1, self.simulator.n_neurons)
        
        # Update voltage traces
        self.voltage_ax.clear()
        self.voltage_ax.set_title("Membrane Potentials")
        self.voltage_ax.set_ylabel("Voltage (mV)")
        
        # Plot voltage traces for a subset of neurons
        n_plot = 5
        neuron_indices = np.linspace(0, self.simulator.n_neurons-1, n_plot, dtype=int)
        for i, neuron_idx in enumerate(neuron_indices):
            voltages = [v[neuron_idx] for v in self.sim_data['membrane_potentials']]
            self.voltage_ax.plot(self.sim_data['times'], voltages, label=f"Neuron {neuron_idx}")
        
        self.voltage_ax.set_xlim(
            self.sim_data['times'][0] if self.sim_data['times'] else 0,
            self.sim_data['times'][-1] if self.sim_data['times'] else 100
        )
        self.voltage_ax.set_ylim(-90, 40)
        
        # Update field potential
        self.field_ax.clear()
        self.field_ax.set_title("Field Potential")
        self.field_ax.set_xlabel("Time (ms)")
        self.field_ax.set_ylabel("Amplitude (μV)")
        
        self.field_ax.plot(
            self.sim_data['times'],
            self.sim_data['field_potentials'],
            color='b'
        )
        
        self.field_ax.set_xlim(
            self.sim_data['times'][0] if self.sim_data['times'] else 0,
            self.sim_data['times'][-1] if self.sim_data['times'] else 100
        )
        
        # Mark stimulation periods
        if self.simulator.stim_active:
            for ax in [self.raster_ax, self.voltage_ax, self.field_ax]:
                ax.axvspan(
                    self.simulator.stim_start_time,
                    self.simulator.stim_start_time + self.simulator.stim_duration,
                    alpha=0.2,
                    color='r'
                )
        
        self.activity_fig.tight_layout()
        
        # Update EEG plots if last frame
        if frame % 5 == 0:
            self._update_eeg_plots()
        
        return []
    
    def _update_eeg_plots(self):
        """Update the EEG visualization plots."""
        # Generate EEG-like signal from recent simulation data
        if len(self.sim_data['field_potentials']) < 100:
            return
        
        # Use the last second of field potentials as EEG
        eeg_signal = np.array(self.sim_data['field_potentials'][-1000:]) * 50  # Scale to μV
        times = np.array(self.sim_data['times'][-1000:])
        
        # Clear previous plots
        self.eeg_time_ax.clear()
        self.eeg_freq_ax.clear()
        
        # Plot time domain signal
        self.eeg_time_ax.set_title("EEG Time Domain")
        self.eeg_time_ax.set_ylabel("Amplitude (μV)")
        self.eeg_time_ax.plot(times, eeg_signal, color='b')
        self.eeg_time_ax.set_xlim(times[0], times[-1])
        
        # Calculate and plot frequency domain
        if len(eeg_signal) > 0:
            # Calculate power spectrum
            fs = 1000 / np.mean(np.diff(times))  # Sampling frequency
            freqs, psd = np.fft.rfftfreq(len(eeg_signal), 1/fs), np.abs(np.fft.rfft(eeg_signal))**2
            
            # Plot frequency domain
            self.eeg_freq_ax.set_title("EEG Frequency Domain")
            self.eeg_freq_ax.set_xlabel("Frequency (Hz)")
            self.eeg_freq_ax.set_ylabel("Power")
            self.eeg_freq_ax.plot(freqs, psd, color='g')
            self.eeg_freq_ax.set_xlim(0, 50)  # Limit to 0-50 Hz
        
        self.eeg_fig.tight_layout()
        self.eeg_canvas.draw()
    
    def reset_simulation(self):
        """Reset the simulation."""
        # Stop animation if running
        if self.is_animating:
            self.is_animating = False
            if self.animation is not None:
                self.animation.event_source.stop()
        
        # Reset simulator
        self.simulator.reset()
        
        # Clear plots
        self.raster_ax.clear()
        self.voltage_ax.clear()
        self.field_ax.clear()
        self.eeg_time_ax.clear()
        self.eeg_freq_ax.clear()
        
        # Reset titles and labels
        self.raster_ax.set_title("Spike Raster Plot")
        self.raster_ax.set_ylabel("Neuron Index")
        
        self.voltage_ax.set_title("Membrane Potentials")
        self.voltage_ax.set_ylabel("Voltage (mV)")
        
        self.field_ax.set_title("Field Potential")
        self.field_ax.set_xlabel("Time (ms)")
        self.field_ax.set_ylabel("Amplitude (μV)")
        
        self.eeg_time_ax.set_title("EEG Time Domain")
        self.eeg_time_ax.set_ylabel("Amplitude (μV)")
        
        self.eeg_freq_ax.set_title("EEG Frequency Domain")
        self.eeg_freq_ax.set_xlabel("Frequency (Hz)")
        self.eeg_freq_ax.set_ylabel("Power")
        
        # Redraw canvases
        self.activity_fig.tight_layout()
        self.eeg_fig.tight_layout()
        self.activity_canvas.draw()
        self.eeg_canvas.draw()
    
    def apply_stimulation(self):
        """Apply stimulation to the simulation."""
        if not self.is_animating:
            messagebox.showinfo("Stimulation", "Please start the simulation first.")
            return
        
        # Apply stimulation
        self.simulator.start_stimulation()
        
        # Update plasticity tab
        self._update_plasticity_plots()
    
    def _update_plasticity_plots(self):
        """Update the plasticity visualization plots."""
        # Clear weight change plot
        self.weight_ax.clear()
        self.weight_ax.set_title("Synaptic Weight Changes")
        self.weight_ax.set_ylabel("Normalized Change")
        
        # Calculate weight changes
        initial_weights = self.stim_effect.initial_weights
        current_weights = self.simulator.weights
        weight_changes = current_weights - initial_weights
        
        # Normalize changes
        if np.sum(np.abs(weight_changes)) > 0:
            weight_changes = weight_changes / np.max(np.abs(weight_changes))
        
        # Plot histogram of weight changes
        self.weight_ax.hist(weight_changes.flatten(), bins=50, alpha=0.7)
        self.weight_ax.axvline(0, color='r', linestyle='--')
        
        # Highlight current phase on phase sensitivity plot
        self.phase_ax.clear()
        self.phase_ax.set_title("Phase Sensitivity")
        self.phase_ax.set_xlabel("Phase (degrees)")
        self.phase_ax.set_ylabel("Plasticity Factor")
        
        # Plot phase sensitivity curve
        phases = np.linspace(0, 360, 100)
        sensitivity = np.zeros_like(phases)
        for i, phase in enumerate(phases):
            closest_phase = min(
                self.stim_effect.phase_sensitivity['alpha'].keys(),
                key=lambda x: abs(x - phase)
            )
            sensitivity[i] = self.stim_effect.phase_sensitivity['alpha'][closest_phase]
        
        self.phase_ax.plot(phases, sensitivity, color='b')
        
        # Mark current phase
        current_phase = self.current_params['phase']
        closest_phase = self.stim_effect._find_closest_phase(current_phase)
        phase_factor = self.stim_effect.phase_sensitivity.get('alpha', {}).get(closest_phase, 1.0)
        
        self.phase_ax.scatter(current_phase, phase_factor, color='r', s=100, zorder=10)
        self.phase_ax.axvline(current_phase, color='r', linestyle='--', alpha=0.5)
        
        # Set limits
        self.phase_ax.set_xlim(0, 360)
        self.phase_ax.set_ylim(0, 2.0)
        
        # Update canvas
        self.plasticity_fig.tight_layout()
        self.plasticity_canvas.draw()
    
    def save_results(self):
        """Save simulation results to a file."""
        if not self.sim_data or not self.sim_data['times']:
            messagebox.showinfo("Save Results", "No simulation data to save.")
            return
        
        # Open file dialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return  # User cancelled
        
        # Prepare data for saving
        save_data = {
            'parameters': self.current_params,
            'results': {
                'times': self.sim_data['times'][-100:],  # Save only the last 100 steps
                'field_potentials': self.sim_data['field_potentials'][-100:],
                'simulator_state': {
                    'time': self.simulator.time,
                    'stim_active': self.simulator.stim_active,
                    'stim_start_time': self.simulator.stim_start_time
                }
            }
        }
        
        # Calculate summary statistics
        if len(self.sim_data['times']) > 0:
            # Calculate average firing rate
            all_spikes = np.array(self.sim_data['spikes'])
            avg_firing_rate = np.mean(all_spikes) * 1000 / self.simulator.dt  # Spikes per second
            
            # Calculate power in different frequency bands
            if len(self.sim_data['field_potentials']) > 100:
                field_potential = np.array(self.sim_data['field_potentials'][-1000:])
                fs = 1000 / self.simulator.dt
                freqs, psd = np.fft.rfftfreq(len(field_potential), 1/fs), np.abs(np.fft.rfft(field_potential))**2
                
                # Calculate band powers
                band_powers = {}
                for band, (low, high) in self.eeg_processor.bands.items():
                    mask = (freqs >= low) & (freqs <= high)
                    if np.any(mask):
                        band_powers[band] = np.sum(psd[mask])
                
                save_data['results']['statistics'] = {
                    'avg_firing_rate': float(avg_firing_rate),
                    'band_powers': band_powers
                }
        
        # Save to file
        try:
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            messagebox.showinfo("Save Results", "Results successfully saved.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving results: {str(e)}")
    
    def load_results(self):
        """Load simulation results from a file."""
        # Open file dialog
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return  # User cancelled
        
        # Load from file
        try:
            with open(file_path, 'r') as f:
                load_data = json.load(f)
            
            # Extract parameters
            params = load_data.get('parameters', {})
            
            # Update GUI variables
            self.amplitude_var.set(params.get('amplitude', 1.0))
            self.frequency_var.set(params.get('frequency', 10.0))
            self.phase_var.set(params.get('phase', 0.0))
            self.duration_var.set(params.get('duration', 500.0))
            
            # Update current parameters
            self._update_parameters()
            
            # Display statistics if available
            results = load_data.get('results', {})
            stats = results.get('statistics', {})
            
            if stats:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "Loaded Results:\n\n")
                
                self.results_text.insert(tk.END, f"Average Firing Rate: {stats.get('avg_firing_rate', 0):.2f} Hz\n\n")
                
                if 'band_powers' in stats:
                    self.results_text.insert(tk.END, "EEG Band Powers:\n")
                    for band, power in stats['band_powers'].items():
                        self.results_text.insert(tk.END, f"{band.capitalize()}: {power:.2f}\n")
            
            messagebox.showinfo("Load Results", "Results successfully loaded.")
        except Exception as e:
            messagebox.showerror("Load Error", f"Error loading results: {str(e)}")


def main():
    """Main function to run the simulation application."""
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
