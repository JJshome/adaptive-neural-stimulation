import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import logging
import os
from datetime import datetime, timedelta
import copy

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class StimulationProtocol:
    """
    Base class for stimulation protocols.
    Provides a framework for defining and executing treatment protocols.
    """
    
    def __init__(self, protocol_id: str, name: str, description: str = ""):
        """
        Initialize the stimulation protocol.
        
        Args:
            protocol_id: Unique identifier for the protocol
            name: User-friendly name for the protocol
            description: Detailed description of the protocol
        """
        self.protocol_id = protocol_id
        self.name = name
        self.description = description
        self.version = "1.0.0"
        self.created_date = datetime.now().isoformat()
        self.last_modified = self.created_date
        self.target_condition = ""
        self.author = ""
        self.references = []
        self.tags = []
        self.parameters = {}
        self.steps = []
        self.safety_limits = {
            "max_amplitude": 5.0,  # mA
            "max_frequency": 100.0,  # Hz
            "max_duration": 60.0,  # minutes
            "max_daily_sessions": 3
        }
        self.contraindications = []
        self.notes = ""
        
    def to_dict(self) -> Dict:
        """
        Convert protocol to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the protocol
        """
        return {
            "protocol_id": self.protocol_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_date": self.created_date,
            "last_modified": self.last_modified,
            "target_condition": self.target_condition,
            "author": self.author,
            "references": self.references,
            "tags": self.tags,
            "parameters": self.parameters,
            "steps": self.steps,
            "safety_limits": self.safety_limits,
            "contraindications": self.contraindications,
            "notes": self.notes
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'StimulationProtocol':
        """
        Create a protocol instance from dictionary data.
        
        Args:
            data: Dictionary containing protocol data
            
        Returns:
            StimulationProtocol instance
        """
        protocol = cls(
            protocol_id=data["protocol_id"],
            name=data["name"],
            description=data.get("description", "")
        )
        
        # Copy all other fields
        for key, value in data.items():
            if key not in ["protocol_id", "name", "description"] and hasattr(protocol, key):
                setattr(protocol, key, value)
                
        return protocol
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the protocol for completeness and adherence to safety limits.
        
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        errors = []
        
        # Check required fields
        if not self.protocol_id:
            errors.append("Protocol ID is required")
        if not self.name:
            errors.append("Protocol name is required")
            
        # Check steps
        if not self.steps:
            errors.append("Protocol must contain at least one step")
            
        # Check if any parameters exceed safety limits
        for step in self.steps:
            if "parameters" in step:
                params = step["parameters"]
                
                # Check amplitude
                if "amplitude" in params and params["amplitude"] > self.safety_limits["max_amplitude"]:
                    errors.append(f"Step '{step.get('name', 'Unnamed')}' exceeds maximum amplitude of {self.safety_limits['max_amplitude']} mA")
                    
                # Check frequency
                if "frequency" in params and params["frequency"] > self.safety_limits["max_frequency"]:
                    errors.append(f"Step '{step.get('name', 'Unnamed')}' exceeds maximum frequency of {self.safety_limits['max_frequency']} Hz")
                    
                # Check duration
                if "duration" in params and params["duration"] > self.safety_limits["max_duration"]:
                    errors.append(f"Step '{step.get('name', 'Unnamed')}' exceeds maximum duration of {self.safety_limits['max_duration']} minutes")
                    
        return (len(errors) == 0, errors)
    
    def save(self, directory: str) -> str:
        """
        Save protocol to a JSON file.
        
        Args:
            directory: Directory to save the protocol file in
            
        Returns:
            Path to the saved protocol file
        """
        # Update last modified date
        self.last_modified = datetime.now().isoformat()
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Construct file path
        filename = f"{self.protocol_id}.json"
        filepath = os.path.join(directory, filename)
        
        # Convert to dictionary and save as JSON
        protocol_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(protocol_dict, f, indent=2)
            
        logger.info(f"Protocol '{self.name}' saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'StimulationProtocol':
        """
        Load protocol from a JSON file.
        
        Args:
            filepath: Path to the protocol JSON file
            
        Returns:
            StimulationProtocol instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        logger.info(f"Protocol loaded from {filepath}")
        return cls.from_dict(data)


class CognitiveEnhancementProtocol(StimulationProtocol):
    """
    Protocol specifically designed for cognitive enhancement.
    Includes specialized steps for memory, attention, and executive function enhancement.
    """
    
    def __init__(self, protocol_id: str, name: str, 
                cognitive_domain: str = "general", 
                description: str = ""):
        """
        Initialize the cognitive enhancement protocol.
        
        Args:
            protocol_id: Unique identifier for the protocol
            name: User-friendly name for the protocol
            cognitive_domain: Target cognitive domain (e.g., memory, attention, executive)
            description: Detailed description of the protocol
        """
        super().__init__(protocol_id, name, description)
        
        # Set cognitive domain-specific properties
        self.target_condition = f"Cognitive Enhancement - {cognitive_domain.capitalize()}"
        self.cognitive_domain = cognitive_domain
        self.tags = ["cognitive", cognitive_domain, "enhancement"]
        
        # Set default parameters based on cognitive domain
        self._set_default_parameters()
        
    def _set_default_parameters(self):
        """Set default parameters based on the cognitive domain."""
        # Base parameters
        self.parameters = {
            "amplitude": 1.5,  # mA
            "ramp_up": 30.0,   # seconds
            "ramp_down": 30.0, # seconds
            "session_duration": 20.0,  # minutes
            "sessions_per_week": 3,
            "total_weeks": 4
        }
        
        # Domain-specific parameters
        if self.cognitive_domain == "memory":
            self.parameters.update({
                "target_areas": ["left_dlpfc", "hippocampus"],
                "frequency": 5.0,  # theta band frequency in Hz
                "phase_difference": 0.0,  # in-phase stimulation
                "waveform": "sine"
            })
            self._create_memory_steps()
            
        elif self.cognitive_domain == "attention":
            self.parameters.update({
                "target_areas": ["right_ppc", "right_dlpfc"],
                "frequency": 40.0,  # gamma band frequency in Hz
                "phase_difference": 180.0,  # anti-phase stimulation
                "waveform": "gamma"
            })
            self._create_attention_steps()
            
        elif self.cognitive_domain == "executive":
            self.parameters.update({
                "target_areas": ["bilateral_dlpfc", "pre_sma"],
                "frequency": 20.0,  # beta band frequency in Hz
                "phase_difference": 90.0,  # quadrature stimulation
                "waveform": "sine"
            })
            self._create_executive_steps()
            
        else:  # general cognitive enhancement
            self.parameters.update({
                "target_areas": ["bilateral_dlpfc"],
                "frequency": 10.0,  # alpha band frequency in Hz
                "phase_difference": 180.0,  # anti-phase stimulation
                "waveform": "sine"
            })
            self._create_general_steps()
    
    def _create_memory_steps(self):
        """Create protocol steps specifically for memory enhancement."""
        self.steps = [
            {
                "name": "Preparation",
                "type": "setup",
                "description": "Set up equipment and prepare patient",
                "duration": 5.0,  # minutes
                "parameters": {}
            },
            {
                "name": "Baseline EEG",
                "type": "measurement",
                "description": "Record baseline EEG with eyes closed",
                "duration": 2.0,  # minutes
                "parameters": {}
            },
            {
                "name": "Theta Entrainment",
                "type": "stimulation",
                "description": "Entrain hippocampal-prefrontal theta oscillations",
                "duration": 5.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "ramp_up": self.parameters["ramp_up"],
                    "ramp_down": self.parameters["ramp_down"]
                }
            },
            {
                "name": "Memory Task with Stimulation",
                "type": "combined",
                "description": "Perform memory task while continuing stimulation",
                "duration": 10.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "task": "n_back_memory"
                }
            },
            {
                "name": "Consolidation Period",
                "type": "stimulation",
                "description": "Continued stimulation during rest for memory consolidation",
                "duration": 5.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "ramp_up": 0.0,
                    "ramp_down": self.parameters["ramp_down"]
                }
            },
            {
                "name": "Post-stimulation Assessment",
                "type": "measurement",
                "description": "Evaluate memory performance after stimulation",
                "duration": 3.0,  # minutes
                "parameters": {
                    "task": "memory_recall"
                }
            }
        ]
    
    def _create_attention_steps(self):
        """Create protocol steps specifically for attention enhancement."""
        self.steps = [
            {
                "name": "Preparation",
                "type": "setup",
                "description": "Set up equipment and prepare patient",
                "duration": 5.0,  # minutes
                "parameters": {}
            },
            {
                "name": "Baseline EEG",
                "type": "measurement",
                "description": "Record baseline EEG with eyes open",
                "duration": 2.0,  # minutes
                "parameters": {}
            },
            {
                "name": "Gamma Entrainment",
                "type": "stimulation",
                "description": "Entrain frontal-parietal gamma oscillations",
                "duration": 3.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "ramp_up": self.parameters["ramp_up"],
                    "ramp_down": 0.0
                }
            },
            {
                "name": "Sustained Attention Task with Stimulation",
                "type": "combined",
                "description": "Perform SART test while continuing stimulation",
                "duration": 12.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "task": "sustained_attention"
                }
            },
            {
                "name": "Selective Attention Training",
                "type": "combined",
                "description": "Perform selective attention task with stimulation",
                "duration": 5.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "task": "selective_attention"
                }
            },
            {
                "name": "Post-stimulation Assessment",
                "type": "measurement",
                "description": "Evaluate attention performance after stimulation",
                "duration": 3.0,  # minutes
                "parameters": {
                    "task": "attention_assessment"
                }
            }
        ]
    
    def _create_executive_steps(self):
        """Create protocol steps specifically for executive function enhancement."""
        self.steps = [
            {
                "name": "Preparation",
                "type": "setup",
                "description": "Set up equipment and prepare patient",
                "duration": 5.0,  # minutes
                "parameters": {}
            },
            {
                "name": "Baseline EEG",
                "type": "measurement",
                "description": "Record baseline EEG during simple cognitive task",
                "duration": 2.0,  # minutes
                "parameters": {
                    "task": "simple_reaction_time"
                }
            },
            {
                "name": "Beta Entrainment",
                "type": "stimulation",
                "description": "Entrain frontal beta oscillations",
                "duration": 3.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "ramp_up": self.parameters["ramp_up"],
                    "ramp_down": 0.0
                }
            },
            {
                "name": "Cognitive Flexibility Task",
                "type": "combined",
                "description": "Perform task-switching exercise with stimulation",
                "duration": 7.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "task": "task_switching"
                }
            },
            {
                "name": "Working Memory and Planning",
                "type": "combined",
                "description": "Perform planning task with stimulation",
                "duration": 8.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "task": "planning"
                }
            },
            {
                "name": "Inhibitory Control",
                "type": "combined",
                "description": "Perform go/no-go task with stimulation",
                "duration": 5.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "task": "inhibitory_control"
                }
            },
            {
                "name": "Post-stimulation Assessment",
                "type": "measurement",
                "description": "Evaluate executive function performance after stimulation",
                "duration": 5.0,  # minutes
                "parameters": {
                    "task": "executive_assessment"
                }
            }
        ]
    
    def _create_general_steps(self):
        """Create protocol steps for general cognitive enhancement."""
        self.steps = [
            {
                "name": "Preparation",
                "type": "setup",
                "description": "Set up equipment and prepare patient",
                "duration": 5.0,  # minutes
                "parameters": {}
            },
            {
                "name": "Baseline EEG",
                "type": "measurement",
                "description": "Record baseline EEG with alternating eyes open/closed",
                "duration": 3.0,  # minutes
                "parameters": {}
            },
            {
                "name": "Alpha Entrainment",
                "type": "stimulation",
                "description": "Entrain bilateral prefrontal alpha oscillations",
                "duration": 5.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "ramp_up": self.parameters["ramp_up"],
                    "ramp_down": 0.0
                }
            },
            {
                "name": "Cognitive Training with Stimulation",
                "type": "combined",
                "description": "Perform mixed cognitive tasks with stimulation",
                "duration": 15.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "task": "mixed_cognitive"
                }
            },
            {
                "name": "Ramp Down and Rest",
                "type": "stimulation",
                "description": "Gradually reduce stimulation during rest",
                "duration": 2.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "ramp_up": 0.0,
                    "ramp_down": self.parameters["ramp_down"]
                }
            },
            {
                "name": "Post-stimulation Assessment",
                "type": "measurement",
                "description": "Evaluate cognitive performance after stimulation",
                "duration": 5.0,  # minutes
                "parameters": {
                    "task": "cognitive_assessment"
                }
            }
        ]


class MotorRehabilitationProtocol(StimulationProtocol):
    """
    Protocol specifically designed for motor rehabilitation.
    Includes specialized steps for motor recovery after stroke or injury.
    """
    
    def __init__(self, protocol_id: str, name: str, 
                affected_side: str = "left",  # 'left', 'right', or 'bilateral'
                motor_function: str = "upper_limb",  # 'upper_limb', 'lower_limb', or 'fine_motor'
                description: str = ""):
        """
        Initialize the motor rehabilitation protocol.
        
        Args:
            protocol_id: Unique identifier for the protocol
            name: User-friendly name for the protocol
            affected_side: Side of the body affected ('left', 'right', or 'bilateral')
            motor_function: Type of motor function targeted
            description: Detailed description of the protocol
        """
        super().__init__(protocol_id, name, description)
        
        # Set motor rehabilitation specific properties
        self.target_condition = "Motor Rehabilitation"
        self.affected_side = affected_side
        self.motor_function = motor_function
        self.tags = ["motor", "rehabilitation", affected_side, motor_function]
        
        # Set default parameters based on target motor function
        self._set_default_parameters()
        
    def _set_default_parameters(self):
        """Set default parameters based on affected side and motor function."""
        # Base parameters
        self.parameters = {
            "amplitude": 2.0,  # mA
            "ramp_up": 30.0,   # seconds
            "ramp_down": 30.0, # seconds
            "session_duration": 30.0,  # minutes
            "sessions_per_week": 5,
            "total_weeks": 6
        }
        
        # Determine target areas based on affected side
        if self.affected_side == "left":
            motor_area = "right_m1"  # Contralateral primary motor cortex
            sensory_area = "right_s1"  # Contralateral primary sensory cortex
        elif self.affected_side == "right":
            motor_area = "left_m1"
            sensory_area = "left_s1"
        else:  # bilateral
            motor_area = "bilateral_m1"
            sensory_area = "bilateral_s1"
            
        # Function-specific parameters
        if self.motor_function == "upper_limb":
            target_areas = [motor_area, sensory_area, "sma"]  # Include supplementary motor area
            self.parameters.update({
                "target_areas": target_areas,
                "frequency": 20.0,  # beta band frequency in Hz
                "phase_difference": 0.0,  # in-phase stimulation
                "waveform": "sine"
            })
            self._create_upper_limb_steps()
            
        elif self.motor_function == "lower_limb":
            target_areas = [motor_area, sensory_area, "pm"]  # Include premotor cortex
            self.parameters.update({
                "target_areas": target_areas,
                "frequency": 15.0,  # lower beta band frequency in Hz
                "phase_difference": 0.0,  # in-phase stimulation
                "waveform": "sine"
            })
            self._create_lower_limb_steps()
            
        elif self.motor_function == "fine_motor":
            target_areas = [motor_area, sensory_area, "cerebellum"]  # Include cerebellum
            self.parameters.update({
                "target_areas": target_areas,
                "frequency": 35.0,  # gamma band frequency in Hz
                "phase_difference": 90.0,  # quadrature stimulation
                "waveform": "sine"
            })
            self._create_fine_motor_steps()
            
    def _create_upper_limb_steps(self):
        """Create protocol steps specifically for upper limb rehabilitation."""
        self.steps = [
            {
                "name": "Preparation",
                "type": "setup",
                "description": "Set up equipment and prepare patient",
                "duration": 5.0,  # minutes
                "parameters": {}
            },
            {
                "name": "Baseline EMG/EEG",
                "type": "measurement",
                "description": "Record baseline EMG of affected muscles and motor cortex EEG",
                "duration": 3.0,  # minutes
                "parameters": {
                    "measurement_type": "combined_emg_eeg",
                    "target_muscles": ["biceps", "triceps", "wrist_flexors", "wrist_extensors"]
                }
            },
            {
                "name": "Initial Stimulation",
                "type": "stimulation",
                "description": "Stimulate motor and sensory cortices",
                "duration": 5.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "ramp_up": self.parameters["ramp_up"],
                    "ramp_down": 0.0
                }
            },
            {
                "name": "Passive Movement with Stimulation",
                "type": "combined",
                "description": "Passive movement of affected limb with continued stimulation",
                "duration": 8.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "movement_type": "passive",
                    "movement_speed": "slow"
                }
            },
            {
                "name": "Active-Assisted Movement with Stimulation",
                "type": "combined",
                "description": "Active-assisted movement exercises with stimulation",
                "duration": 10.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "movement_type": "active_assisted",
                    "movement_complexity": "moderate"
                }
            },
            {
                "name": "Functional Task Practice",
                "type": "combined",
                "description": "Practice everyday tasks with continued stimulation",
                "duration": 7.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "task_type": "functional",
                    "task_examples": ["reaching", "grasping", "lifting"]
                }
            },
            {
                "name": "Post-stimulation Assessment",
                "type": "measurement",
                "description": "Evaluate motor function after stimulation",
                "duration": 5.0,  # minutes
                "parameters": {
                    "assessment_type": "upper_limb_function",
                    "scales": ["fugl_meyer", "grip_strength"]
                }
            }
        ]

    def _create_lower_limb_steps(self):
        """Create protocol steps specifically for lower limb rehabilitation."""
        self.steps = [
            {
                "name": "Preparation",
                "type": "setup",
                "description": "Set up equipment and prepare patient",
                "duration": 5.0,  # minutes
                "parameters": {}
            },
            {
                "name": "Baseline EMG/EEG",
                "type": "measurement",
                "description": "Record baseline EMG of affected muscles and motor cortex EEG",
                "duration": 3.0,  # minutes
                "parameters": {
                    "measurement_type": "combined_emg_eeg",
                    "target_muscles": ["quadriceps", "hamstrings", "tibialis_anterior", "gastrocnemius"]
                }
            },
            {
                "name": "Initial Stimulation",
                "type": "stimulation",
                "description": "Stimulate motor and sensory cortices",
                "duration": 5.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "ramp_up": self.parameters["ramp_up"],
                    "ramp_down": 0.0
                }
            },
            {
                "name": "Passive Movement with Stimulation",
                "type": "combined",
                "description": "Passive movement of affected leg with continued stimulation",
                "duration": 8.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "movement_type": "passive",
                    "movement_patterns": ["flexion-extension", "medial-lateral"]
                }
            },
            {
                "name": "Active-Assisted Movement with Stimulation",
                "type": "combined",
                "description": "Active-assisted movement exercises with stimulation",
                "duration": 10.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "movement_type": "active_assisted",
                    "movement_patterns": ["sit-to-stand", "stepping"]
                }
            },
            {
                "name": "Weight-Bearing and Balance",
                "type": "combined",
                "description": "Standing and weight-shifting exercises with stimulation",
                "duration": 7.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "task_type": "weight_bearing",
                    "support_level": "moderate"
                }
            },
            {
                "name": "Post-stimulation Assessment",
                "type": "measurement",
                "description": "Evaluate motor function after stimulation",
                "duration": 5.0,  # minutes
                "parameters": {
                    "assessment_type": "lower_limb_function",
                    "scales": ["berg_balance", "timed_up_and_go"]
                }
            }
        ]
            
    def _create_fine_motor_steps(self):
        """Create protocol steps specifically for fine motor rehabilitation."""
        self.steps = [
            {
                "name": "Preparation",
                "type": "setup",
                "description": "Set up equipment and prepare patient",
                "duration": 5.0,  # minutes
                "parameters": {}
            },
            {
                "name": "Baseline EMG/EEG",
                "type": "measurement",
                "description": "Record baseline EMG of hand muscles and motor cortex EEG",
                "duration": 3.0,  # minutes
                "parameters": {
                    "measurement_type": "combined_emg_eeg",
                    "target_muscles": ["finger_flexors", "finger_extensors", "thenar", "intrinsic_hand"]
                }
            },
            {
                "name": "Initial Stimulation",
                "type": "stimulation",
                "description": "Stimulate motor and sensory cortices and cerebellum",
                "duration": 5.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "ramp_up": self.parameters["ramp_up"],
                    "ramp_down": 0.0
                }
            },
            {
                "name": "Individual Finger Movements",
                "type": "combined",
                "description": "Practice individual finger movements with stimulation",
                "duration": 6.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "movement_type": "sequential",
                    "movement_patterns": ["finger_tapping", "finger_opposition"]
                }
            },
            {
                "name": "Precision Grip Training",
                "type": "combined",
                "description": "Practice precision grip and object manipulation",
                "duration": 8.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "task_type": "precision",
                    "task_examples": ["pinch", "key_grip", "tripod_grip"]
                }
            },
            {
                "name": "Dexterity Tasks",
                "type": "combined",
                "description": "Perform advanced dexterity tasks with stimulation",
                "duration": 8.0,  # minutes
                "parameters": {
                    "amplitude": self.parameters["amplitude"],
                    "frequency": self.parameters["frequency"],
                    "phase_difference": self.parameters["phase_difference"],
                    "waveform": self.parameters["waveform"],
                    "target_areas": self.parameters["target_areas"],
                    "task_type": "dexterity",
                    "task_examples": ["pegboard", "threading", "buttoning"]
                }
            },
            {
                "name": "Post-stimulation Assessment",
                "type": "measurement",
                "description": "Evaluate fine motor function after stimulation",
                "duration": 5.0,  # minutes
                "parameters": {
                    "assessment_type": "fine_motor_function",
                    "scales": ["purdue_pegboard", "jebsen_taylor_hand_function"]
                }
            }
        ]


class ProtocolLibrary:
    """
    Manage a collection of stimulation protocols.
    Provides methods for loading, saving, and selecting appropriate protocols.
    """
    
    def __init__(self, protocols_directory: str = None):
        """
        Initialize the protocol library.
        
        Args:
            protocols_directory: Directory containing protocol JSON files
        """
        self.protocols = {}
        self.protocols_directory = protocols_directory
        
        # Load protocols if directory is provided
        if protocols_directory and os.path.exists(protocols_directory):
            self.load_all_protocols()
            
    def add_protocol(self, protocol: StimulationProtocol) -> bool:
        """
        Add a protocol to the library.
        
        Args:
            protocol: StimulationProtocol instance
            
        Returns:
            True if added successfully, False if protocol ID already exists
        """
        if protocol.protocol_id in self.protocols:
            logger.warning(f"Protocol with ID '{protocol.protocol_id}' already exists in library")
            return False
            
        self.protocols[protocol.protocol_id] = protocol
        logger.info(f"Added protocol '{protocol.name}' to library")
        return True
        
    def get_protocol(self, protocol_id: str) -> Optional[StimulationProtocol]:
        """
        Get a protocol by ID.
        
        Args:
            protocol_id: ID of the protocol to retrieve
            
        Returns:
            StimulationProtocol instance or None if not found
        """
        return self.protocols.get(protocol_id)
        
    def load_protocol(self, filepath: str) -> Optional[StimulationProtocol]:
        """
        Load a protocol from a JSON file and add it to the library.
        
        Args:
            filepath: Path to the protocol JSON file
            
        Returns:
            Loaded StimulationProtocol instance or None if loading failed
        """
        try:
            protocol = StimulationProtocol.load(filepath)
            self.add_protocol(protocol)
            return protocol
        except Exception as e:
            logger.error(f"Failed to load protocol from {filepath}: {str(e)}")
            return None
            
    def load_all_protocols(self) -> int:
        """
        Load all protocol JSON files from the protocols directory.
        
        Returns:
            Number of protocols loaded
        """
        if not self.protocols_directory or not os.path.exists(self.protocols_directory):
            logger.warning("Protocols directory not set or doesn't exist")
            return 0
            
        count = 0
        for filename in os.listdir(self.protocols_directory):
            if filename.endswith('.json'):
                filepath = os.path.join(self.protocols_directory, filename)
                if self.load_protocol(filepath):
                    count += 1
                    
        logger.info(f"Loaded {count} protocols from {self.protocols_directory}")
        return count
        
    def save_protocol(self, protocol_id: str) -> bool:
        """
        Save a protocol to a JSON file.
        
        Args:
            protocol_id: ID of the protocol to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.protocols_directory:
            logger.error("Protocols directory not set")
            return False
            
        protocol = self.get_protocol(protocol_id)
        if not protocol:
            logger.error(f"Protocol with ID '{protocol_id}' not found in library")
            return False
            
        try:
            protocol.save(self.protocols_directory)
            return True
        except Exception as e:
            logger.error(f"Failed to save protocol '{protocol_id}': {str(e)}")
            return False
            
    def save_all_protocols(self) -> int:
        """
        Save all protocols to JSON files.
        
        Returns:
            Number of protocols saved
        """
        if not self.protocols_directory:
            logger.error("Protocols directory not set")
            return 0
            
        # Create directory if it doesn't exist
        os.makedirs(self.protocols_directory, exist_ok=True)
        
        count = 0
        for protocol_id in self.protocols:
            if self.save_protocol(protocol_id):
                count += 1
                
        logger.info(f"Saved {count} protocols to {self.protocols_directory}")
        return count
        
    def find_protocols(self, condition: str = None, tags: List[str] = None) -> List[StimulationProtocol]:
        """
        Find protocols matching specified criteria.
        
        Args:
            condition: Target condition to match
            tags: List of tags to match (protocols must have all tags)
            
        Returns:
            List of matching protocols
        """
        results = []
        
        for protocol in self.protocols.values():
            # Check condition if specified
            if condition and condition.lower() not in protocol.target_condition.lower():
                continue
                
            # Check tags if specified
            if tags:
                if not all(tag.lower() in [t.lower() for t in protocol.tags] for tag in tags):
                    continue
                    
            # If we get here, all conditions are met
            results.append(protocol)
            
        return results
        
    def create_custom_protocol(self, base_protocol_id: str, custom_params: Dict) -> Optional[StimulationProtocol]:
        """
        Create a custom protocol based on an existing one.
        
        Args:
            base_protocol_id: ID of the base protocol
            custom_params: Dictionary of parameters to customize
            
        Returns:
            New StimulationProtocol instance or None if base protocol not found
        """
        base_protocol = self.get_protocol(base_protocol_id)
        if not base_protocol:
            logger.error(f"Base protocol with ID '{base_protocol_id}' not found in library")
            return None
            
        # Create a deep copy of the base protocol
        custom_protocol = copy.deepcopy(base_protocol)
        
        # Generate a new ID for the custom protocol
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        custom_protocol.protocol_id = f"{base_protocol_id}_custom_{timestamp}"
        custom_protocol.name = f"Custom {base_protocol.name}"
        
        # Update parameters
        for param_key, param_value in custom_params.items():
            if hasattr(custom_protocol, param_key):
                setattr(custom_protocol, param_key, param_value)
            elif param_key in custom_protocol.parameters:
                custom_protocol.parameters[param_key] = param_value
                
        # Add to library
        self.add_protocol(custom_protocol)
        
        return custom_protocol
        
    def get_protocol_summary(self) -> List[Dict]:
        """
        Get a summary of all protocols in the library.
        
        Returns:
            List of dictionaries with protocol summary information
        """
        summary = []
        
        for protocol_id, protocol in self.protocols.items():
            summary.append({
                "protocol_id": protocol_id,
                "name": protocol.name,
                "target_condition": protocol.target_condition,
                "tags": protocol.tags,
                "duration": sum(step.get("duration", 0) for step in protocol.steps),
                "steps": len(protocol.steps)
            })
            
        return summary
