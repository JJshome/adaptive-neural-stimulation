import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any
import threading
import json
import os
import numpy as np

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class SystemController:
    """
    Central controller for the Adaptive Neural Stimulation System.
    Coordinates all system components and manages the overall system state.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the system controller.
        
        Args:
            config_path: Path to configuration file
        """
        self.components = {}  # Dictionary of component instances
        self.system_state = {
            "is_running": False,
            "session_active": False,
            "treatment_active": False,
            "error_state": False
        }
        self.config = self._load_config(config_path)
        self.event_handlers = {}  # Dictionary of event handler callbacks
        self.event_queue = []  # Queue of pending events
        self.event_thread = None
        self.event_thread_running = False
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Initialize event handling
        self._init_event_handling()
        
        logger.info("SystemController initialized")
        
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "system_name": "Adaptive Neural Stimulation System",
            "version": "1.0.0",
            "log_level": "INFO",
            "data_directory": "data",
            "safety_limits": {
                "max_current": 5.0,  # mA
                "max_frequency": 100.0,  # Hz
                "max_session_duration": 60.0  # minutes
            },
            "event_processing_interval": 0.1  # seconds
        }
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded configuration from {config_path}")
                    
                    # Merge with default config
                    merged_config = {**default_config, **config}
                    return merged_config
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                
        return default_config
        
    def _init_event_handling(self):
        """Initialize event handling system."""
        # Register default event handlers
        self.register_event_handler("system_start", self._handle_system_start)
        self.register_event_handler("system_stop", self._handle_system_stop)
        self.register_event_handler("error", self._handle_error)
        self.register_event_handler("component_added", self._handle_component_added)
        self.register_event_handler("component_removed", self._handle_component_removed)
        
    def register_component(self, component_id: str, component: Any) -> bool:
        """
        Register a system component.
        
        Args:
            component_id: Unique identifier for the component
            component: Component instance
            
        Returns:
            True if component registered successfully, False if already exists
        """
        with self.lock:
            if component_id in self.components:
                logger.warning(f"Component '{component_id}' already registered")
                return False
                
            self.components[component_id] = component
            
            # Trigger event
            self.trigger_event("component_added", {
                "component_id": component_id,
                "component_type": type(component).__name__
            })
            
            logger.info(f"Registered component: {component_id}")
            return True
            
    def unregister_component(self, component_id: str) -> bool:
        """
        Unregister a system component.
        
        Args:
            component_id: ID of the component to unregister
            
        Returns:
            True if component unregistered successfully, False if not found
        """
        with self.lock:
            if component_id not in self.components:
                logger.warning(f"Component '{component_id}' not found")
                return False
                
            component = self.components.pop(component_id)
            
            # Trigger event
            self.trigger_event("component_removed", {
                "component_id": component_id,
                "component_type": type(component).__name__
            })
            
            logger.info(f"Unregistered component: {component_id}")
            return True
            
    def get_component(self, component_id: str) -> Optional[Any]:
        """
        Get a registered component.
        
        Args:
            component_id: ID of the component to retrieve
            
        Returns:
            Component instance or None if not found
        """
        return self.components.get(component_id)
        
    def start_system(self) -> bool:
        """
        Start the system.
        
        Returns:
            True if system started successfully, False if already running
        """
        with self.lock:
            if self.system_state["is_running"]:
                logger.warning("System already running")
                return False
                
            # Start event processing thread
            self.event_thread_running = True
            self.event_thread = threading.Thread(
                target=self._event_processing_loop,
                daemon=True
            )
            self.event_thread.start()
            
            # Update system state
            self.system_state["is_running"] = True
            self.system_state["error_state"] = False
            
            # Trigger event
            self.trigger_event("system_start", {
                "timestamp": time.time(),
                "components": list(self.components.keys())
            })
            
            logger.info("System started")
            return True
            
    def stop_system(self) -> bool:
        """
        Stop the system.
        
        Returns:
            True if system stopped successfully, False if not running
        """
        with self.lock:
            if not self.system_state["is_running"]:
                logger.warning("System not running")
                return False
                
            # Stop event processing thread
            self.event_thread_running = False
            if self.event_thread and self.event_thread.is_alive():
                self.event_thread.join(timeout=2.0)
                
            # Update system state
            self.system_state["is_running"] = False
            self.system_state["session_active"] = False
            self.system_state["treatment_active"] = False
            
            # Trigger event
            self.trigger_event("system_stop", {
                "timestamp": time.time()
            })
            
            logger.info("System stopped")
            return True
            
    def register_event_handler(self, event_type: str, handler: callable) -> None:
        """
        Register a handler for a specific event type.
        
        Args:
            event_type: Type of event to handle
            handler: Callback function to handle the event
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
            
        if handler not in self.event_handlers[event_type]:
            self.event_handlers[event_type].append(handler)
            logger.debug(f"Registered handler for event type: {event_type}")
            
    def unregister_event_handler(self, event_type: str, handler: callable) -> bool:
        """
        Unregister an event handler.
        
        Args:
            event_type: Type of event
            handler: Handler function to unregister
            
        Returns:
            True if handler unregistered successfully, False if not found
        """
        if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
            logger.debug(f"Unregistered handler for event type: {event_type}")
            return True
        return False
        
    def trigger_event(self, event_type: str, event_data: dict = None) -> None:
        """
        Trigger an event.
        
        Args:
            event_type: Type of event to trigger
            event_data: Data associated with the event
        """
        event = {
            "type": event_type,
            "data": event_data or {},
            "timestamp": time.time()
        }
        
        with self.lock:
            self.event_queue.append(event)
            
        logger.debug(f"Triggered event: {event_type}")
        
    def _event_processing_loop(self) -> None:
        """Background loop for processing events."""
        logger.debug("Event processing loop started")
        
        while self.event_thread_running:
            try:
                # Process events in queue
                events_to_process = []
                with self.lock:
                    if self.event_queue:
                        events_to_process = self.event_queue.copy()
                        self.event_queue.clear()
                        
                for event in events_to_process:
                    self._process_event(event)
                    
                # Sleep for a short interval
                time.sleep(self.config.get("event_processing_interval", 0.1))
                
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                
        logger.debug("Event processing loop stopped")
        
    def _process_event(self, event: dict) -> None:
        """
        Process a single event.
        
        Args:
            event: Event dictionary with type, data, and timestamp
        """
        event_type = event["type"]
        event_data = event["data"]
        
        # Call handlers for this event type
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
                    
    def _handle_system_start(self, event_data: dict) -> None:
        """
        Handle system start event.
        
        Args:
            event_data: Event data
        """
        logger.info(f"System started with {len(self.components)} components")
        
    def _handle_system_stop(self, event_data: dict) -> None:
        """
        Handle system stop event.
        
        Args:
            event_data: Event data
        """
        logger.info("System stopped")
        
    def _handle_error(self, event_data: dict) -> None:
        """
        Handle error event.
        
        Args:
            event_data: Event data
        """
        error = event_data.get("error", "Unknown error")
        logger.error(f"System error: {error}")
        self.system_state["error_state"] = True
        
    def _handle_component_added(self, event_data: dict) -> None:
        """
        Handle component added event.
        
        Args:
            event_data: Event data
        """
        component_id = event_data.get("component_id")
        component_type = event_data.get("component_type")
        logger.info(f"Component added: {component_id} ({component_type})")
        
    def _handle_component_removed(self, event_data: dict) -> None:
        """
        Handle component removed event.
        
        Args:
            event_data: Event data
        """
        component_id = event_data.get("component_id")
        component_type = event_data.get("component_type")
        logger.info(f"Component removed: {component_id} ({component_type})")
        
    def start_session(self, session_config: dict) -> str:
        """
        Start a treatment session.
        
        Args:
            session_config: Session configuration
            
        Returns:
            Session ID if started successfully, empty string if failed
        """
        with self.lock:
            if not self.system_state["is_running"]:
                logger.error("Cannot start session: System not running")
                return ""
                
            if self.system_state["session_active"]:
                logger.warning("Session already active")
                return ""
                
            # Generate session ID
            session_id = f"session_{int(time.time())}"
            
            # Update system state
            self.system_state["session_active"] = True
            
            # Prepare session data
            session_data = {
                "session_id": session_id,
                "start_time": time.time(),
                "config": session_config,
                "user_id": session_config.get("user_id", "unknown"),
                "protocol_id": session_config.get("protocol_id", "unknown")
            }
            
            # Trigger event
            self.trigger_event("session_start", session_data)
            
            logger.info(f"Started session: {session_id}")
            return session_id
            
    def stop_session(self, session_id: str) -> bool:
        """
        Stop a treatment session.
        
        Args:
            session_id: ID of the session to stop
            
        Returns:
            True if session stopped successfully, False if not active
        """
        with self.lock:
            if not self.system_state["session_active"]:
                logger.warning("No active session")
                return False
                
            # Stop any active treatment
            if self.system_state["treatment_active"]:
                self.stop_treatment()
                
            # Update system state
            self.system_state["session_active"] = False
            
            # Trigger event
            self.trigger_event("session_stop", {
                "session_id": session_id,
                "end_time": time.time()
            })
            
            logger.info(f"Stopped session: {session_id}")
            return True
            
    def start_treatment(self, treatment_config: dict) -> bool:
        """
        Start a treatment within a session.
        
        Args:
            treatment_config: Treatment configuration
            
        Returns:
            True if treatment started successfully, False if failed
        """
        with self.lock:
            if not self.system_state["is_running"]:
                logger.error("Cannot start treatment: System not running")
                return False
                
            if not self.system_state["session_active"]:
                logger.error("Cannot start treatment: No active session")
                return False
                
            if self.system_state["treatment_active"]:
                logger.warning("Treatment already active")
                return False
                
            # Update system state
            self.system_state["treatment_active"] = True
            
            # Trigger event
            self.trigger_event("treatment_start", {
                "start_time": time.time(),
                "config": treatment_config,
                "protocol_id": treatment_config.get("protocol_id", "unknown")
            })
            
            logger.info("Started treatment")
            return True
            
    def stop_treatment(self) -> bool:
        """
        Stop the current treatment.
        
        Returns:
            True if treatment stopped successfully, False if not active
        """
        with self.lock:
            if not self.system_state["treatment_active"]:
                logger.warning("No active treatment")
                return False
                
            # Update system state
            self.system_state["treatment_active"] = False
            
            # Trigger event
            self.trigger_event("treatment_stop", {
                "end_time": time.time()
            })
            
            logger.info("Stopped treatment")
            return True
            
    def get_system_state(self) -> dict:
        """
        Get current system state.
        
        Returns:
            Dictionary with system state information
        """
        with self.lock:
            # Deep copy to avoid modification from outside
            return dict(self.system_state)
            
    def handle_error(self, error_message: str, component_id: str = None) -> None:
        """
        Handle system error.
        
        Args:
            error_message: Error message
            component_id: ID of the component that encountered the error
        """
        # Trigger error event
        self.trigger_event("error", {
            "error": error_message,
            "component_id": component_id,
            "timestamp": time.time()
        })
        
        # Update system state
        self.system_state["error_state"] = True
        
        logger.error(f"System error: {error_message}" + 
                  (f" (component: {component_id})" if component_id else ""))
                  
    def reset_error_state(self) -> bool:
        """
        Reset system error state.
        
        Returns:
            True if error state reset successfully, False if no error
        """
        with self.lock:
            if not self.system_state["error_state"]:
                return False
                
            self.system_state["error_state"] = False
            
            # Trigger event
            self.trigger_event("error_reset", {
                "timestamp": time.time()
            })
            
            logger.info("Error state reset")
            return True
