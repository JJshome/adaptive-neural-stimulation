import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Dict, List, Tuple, Optional, Union, Any
import io
import base64
import logging
from datetime import datetime, timedelta

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class StimulationVisualizer:
    """
    Class for visualizing stimulation patterns and neural response data.
    Provides real-time monitoring and analysis tools for the stimulation system.
    """
    
    def __init__(self, theme: str = 'light', dpi: int = 100, 
                default_figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the stimulation visualizer.
        
        Args:
            theme: Visualization theme ('light' or 'dark')
            dpi: DPI for rendered visualizations
            default_figsize: Default figure size in inches (width, height)
        """
        self.theme = theme
        self.dpi = dpi
        self.default_figsize = default_figsize
        
        # Set up matplotlib style based on theme
        if theme == 'dark':
            plt.style.use('dark_background')
            self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
            self.bg_color = '#1e1e1e'
            self.fg_color = '#ffffff'
            self.grid_color = '#555555'
        else:
            plt.style.use('default')
            self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
            self.bg_color = '#ffffff'
            self.fg_color = '#333333'
            self.grid_color = '#dddddd'
    
    def _create_figure(self, figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Any]:
        """
        Create a new figure with the current theme.
        
        Args:
            figsize: Figure size in inches (width, height)
            
        Returns:
            Tuple of (figure, axes)
        """
        if figsize is None:
            figsize = self.default_figsize
            
        fig = plt.figure(figsize=figsize, dpi=self.dpi, facecolor=self.bg_color)
        ax = fig.add_subplot(111)
        
        # Apply theme to axes
        ax.set_facecolor(self.bg_color)
        ax.spines['bottom'].set_color(self.fg_color)
        ax.spines['top'].set_color(self.fg_color)
        ax.spines['left'].set_color(self.fg_color)
        ax.spines['right'].set_color(self.fg_color)
        ax.tick_params(axis='both', colors=self.fg_color)
        ax.yaxis.label.set_color(self.fg_color)
        ax.xaxis.label.set_color(self.fg_color)
        ax.title.set_color(self.fg_color)
        
        return fig, ax
    
    def _figure_to_base64(self, fig: Figure) -> str:
        """
        Convert a matplotlib figure to base64 encoded string.
        
        Args:
            fig: Matplotlib figure object
            
        Returns:
            Base64 encoded string of the figure
        """
        canvas = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        plt.close(fig)
        return data
    
    def visualize_stimulation_pattern(self, patterns: Dict[int, np.ndarray], 
                                    time_axis: Optional[np.ndarray] = None,
                                    title: str = "Stimulation Pattern",
                                    channel_labels: Optional[List[str]] = None,
                                    as_base64: bool = False) -> Union[Figure, str]:
        """
        Visualize stimulation patterns across multiple channels.
        
        Args:
            patterns: Dictionary mapping channel indices to stimulation waveforms
            time_axis: Optional time axis values (if None, uses sample indices)
            title: Plot title
            channel_labels: Optional list of channel labels
            as_base64: Whether to return the figure as a base64 encoded string
            
        Returns:
            Matplotlib figure or base64 encoded string
        """
        # Get number of channels and samples
        channels = sorted(patterns.keys())
        n_channels = len(channels)
        
        if n_channels == 0:
            logger.warning("No channels to visualize")
            fig, ax = self._create_figure()
            ax.text(0.5, 0.5, "No channels to visualize", 
                  ha='center', va='center', color=self.fg_color)
            
            if as_base64:
                return self._figure_to_base64(fig)
            return fig
        
        # Get number of samples from first channel
        first_channel = next(iter(patterns.values()))
        n_samples = len(first_channel)
        
        # Create time axis if not provided
        if time_axis is None:
            time_axis = np.arange(n_samples)
            
        # Create channel labels if not provided
        if channel_labels is None:
            channel_labels = [f"Channel {ch}" for ch in channels]
            
        # Create figure
        fig, ax = self._create_figure(figsize=(10, 6 + 0.3 * n_channels))
        
        # Plot each channel with vertical offset for clarity
        for i, channel in enumerate(channels):
            offset = i * 2.5  # Vertical spacing between channels
            ax.plot(time_axis, patterns[channel] + offset, 
                  label=channel_labels[i], 
                  color=self.colors[i % len(self.colors)])
            
        # Add horizontal line at each channel's baseline
        for i in range(n_channels):
            ax.axhline(y=i * 2.5, color=self.grid_color, linestyle='--', alpha=0.5)
            
        # Set up axis labels and title
        ax.set_xlabel('Time', color=self.fg_color)
        ax.set_yticks([i * 2.5 for i in range(n_channels)])
        ax.set_yticklabels(channel_labels)
        ax.set_title(title, color=self.fg_color)
        ax.grid(True, alpha=0.3, color=self.grid_color)
        
        # Adjust layout
        plt.tight_layout()
        
        if as_base64:
            return self._figure_to_base64(fig)
        return fig
    
    def visualize_eeg_data(self, eeg_data: np.ndarray, 
                         sampling_rate: float = 250.0,
                         channel_names: Optional[List[str]] = None,
                         time_window: float = 5.0,  # seconds
                         title: str = "EEG Signal",
                         as_base64: bool = False) -> Union[Figure, str]:
        """
        Visualize EEG data across multiple channels.
        
        Args:
            eeg_data: Array of EEG data with shape (n_channels, n_samples)
            sampling_rate: Sampling rate in Hz
            channel_names: Optional list of channel names
            time_window: Time window to display in seconds
            title: Plot title
            as_base64: Whether to return the figure as a base64 encoded string
            
        Returns:
            Matplotlib figure or base64 encoded string
        """
        # Get dimensions
        n_channels, n_samples = eeg_data.shape
        
        # Calculate time axis
        time_axis = np.arange(n_samples) / sampling_rate
        
        # Create channel names if not provided
        if channel_names is None:
            channel_names = [f"Ch{i+1}" for i in range(n_channels)]
            
        # Create figure
        fig, ax = self._create_figure(figsize=(10, 6 + 0.3 * n_channels))
        
        # Determine display window
        max_time = time_axis[-1]
        if max_time <= time_window:
            # Show all data if it fits within the time window
            t_start = 0
            t_end = max_time
        else:
            # Show the most recent time_window seconds
            t_start = max_time - time_window
            t_end = max_time
            
        # Find indices corresponding to the time window
        idx_start = np.argmax(time_axis >= t_start)
        idx_end = np.argmax(time_axis >= t_end) if t_end < max_time else n_samples
        
        # Plot each channel with vertical offset
        scaling_factor = 50  # Scale to make the EEG signals visible
        for i in range(n_channels):
            offset = i * 3 * scaling_factor  # Vertical spacing between channels
            normalized_data = eeg_data[i, idx_start:idx_end] / scaling_factor
            ax.plot(time_axis[idx_start:idx_end], normalized_data + offset, 
                  label=channel_names[i], 
                  color=self.colors[i % len(self.colors)])
            
        # Add horizontal line at each channel's baseline
        for i in range(n_channels):
            ax.axhline(y=i * 3 * scaling_factor, color=self.grid_color, linestyle='--', alpha=0.5)
            
        # Set up axis labels and title
        ax.set_xlabel('Time (s)', color=self.fg_color)
        ax.set_yticks([i * 3 * scaling_factor for i in range(n_channels)])
        ax.set_yticklabels(channel_names)
        ax.set_title(title, color=self.fg_color)
        ax.grid(True, alpha=0.3, color=self.grid_color)
        
        # Set x-axis limits to the time window
        ax.set_xlim(t_start, t_end)
        
        # Adjust layout
        plt.tight_layout()
        
        if as_base64:
            return self._figure_to_base64(fig)
        return fig
    
    def visualize_spectral_data(self, freqs: np.ndarray, psd: np.ndarray,
                              channel_names: Optional[List[str]] = None,
                              freq_range: Optional[Tuple[float, float]] = None,
                              title: str = "Spectral Power",
                              as_base64: bool = False) -> Union[Figure, str]:
        """
        Visualize spectral power data across frequency bands and channels.
        
        Args:
            freqs: Array of frequency bins
            psd: Power spectral density array with shape (n_channels, n_freqs)
            channel_names: Optional list of channel names
            freq_range: Optional tuple of (min_freq, max_freq) to display
            title: Plot title
            as_base64: Whether to return the figure as a base64 encoded string
            
        Returns:
            Matplotlib figure or base64 encoded string
        """
        # Get dimensions
        n_channels, n_freqs = psd.shape
        
        # Check if frequencies match
        if len(freqs) != n_freqs:
            raise ValueError(f"Frequency array length ({len(freqs)}) doesn't match PSD frequency dimension ({n_freqs})")
            
        # Create channel names if not provided
        if channel_names is None:
            channel_names = [f"Ch{i+1}" for i in range(n_channels)]
            
        # Apply frequency range filter if provided
        if freq_range is not None:
            min_freq, max_freq = freq_range
            mask = (freqs >= min_freq) & (freqs <= max_freq)
            freqs = freqs[mask]
            psd = psd[:, mask]
            
        # Create figure
        fig, ax = self._create_figure()
        
        # Plot each channel
        for i in range(n_channels):
            ax.semilogy(freqs, psd[i], 
                      label=channel_names[i], 
                      color=self.colors[i % len(self.colors)])
            
        # Mark common frequency bands
        band_ranges = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 100)
        }
        
        # Shade frequency bands if within the displayed range
        for band_name, (f_min, f_max) in band_ranges.items():
            if (f_min <= freqs[-1]) and (f_max >= freqs[0]):
                # Adjust band boundaries to displayed frequency range
                f_min_adj = max(f_min, freqs[0])
                f_max_adj = min(f_max, freqs[-1])
                
                # Determine indices for shading
                idx_min = np.argmin(np.abs(freqs - f_min_adj))
                idx_max = np.argmin(np.abs(freqs - f_max_adj))
                
                # Shade the band area
                ax.axvspan(freqs[idx_min], freqs[idx_max], 
                         alpha=0.2, 
                         label=f"{band_name} ({f_min}-{f_max} Hz)")
                
        # Set up axis labels and title
        ax.set_xlabel('Frequency (Hz)', color=self.fg_color)
        ax.set_ylabel('Power Spectral Density (µV²/Hz)', color=self.fg_color)
        ax.set_title(title, color=self.fg_color)
        ax.grid(True, alpha=0.3, color=self.grid_color)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        if as_base64:
            return self._figure_to_base64(fig)
        return fig
    
    def visualize_brain_activity(self, channel_values: np.ndarray,
                               channel_positions: List[Tuple[float, float]],
                               channel_names: Optional[List[str]] = None,
                               title: str = "Brain Activity Map",
                               cmap: str = 'viridis',
                               head_outline: bool = True,
                               contour: bool = True,
                               as_base64: bool = False) -> Union[Figure, str]:
        """
        Visualize brain activity as a 2D topographic map with electrode positions.
        
        Args:
            channel_values: Array of values to visualize (one per channel)
            channel_positions: List of (x, y) positions for each channel
            channel_names: Optional list of channel names
            title: Plot title
            cmap: Matplotlib colormap name
            head_outline: Whether to draw head outline
            contour: Whether to draw contour lines
            as_base64: Whether to return the figure as a base64 encoded string
            
        Returns:
            Matplotlib figure or base64 encoded string
        """
        # Check if positions match number of channels
        n_channels = len(channel_values)
        if len(channel_positions) != n_channels:
            raise ValueError(f"Number of channel positions ({len(channel_positions)}) doesn't match number of channels ({n_channels})")
            
        # Create channel names if not provided
        if channel_names is None:
            channel_names = [f"Ch{i+1}" for i in range(n_channels)]
            
        # Extract x and y coordinates
        x = np.array([pos[0] for pos in channel_positions])
        y = np.array([pos[1] for pos in channel_positions])
        
        # Create figure
        fig, ax = self._create_figure()
        
        # Create a grid for interpolation
        xi = np.linspace(min(x) - 0.1, max(x) + 0.1, 100)
        yi = np.linspace(min(y) - 0.1, max(y) + 0.1, 100)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate values on the grid
        from scipy.interpolate import griddata
        zi = griddata((x, y), channel_values, (xi_grid, yi_grid), method='cubic')
        
        # Plot contour map
        if contour:
            contour = ax.contour(xi, yi, zi, 6, colors='k', alpha=0.4)
            
        # Plot color map
        im = ax.pcolormesh(xi, yi, zi, shading='auto', cmap=cmap)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activity', color=self.fg_color)
        cbar.ax.yaxis.set_tick_params(color=self.fg_color)
        cbar.outline.set_edgecolor(self.fg_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.fg_color)
        
        # Draw head outline
        if head_outline:
            circle = plt.Circle((0, 0), 1, fill=False, linewidth=2, color=self.fg_color)
            ax.add_patch(circle)
            # Add nose
            ax.plot([0, 0], [0.9, 1.1], color=self.fg_color, linewidth=2)
            # Add ears
            ax.plot([-1, -1.1], [0, 0], color=self.fg_color, linewidth=2)
            ax.plot([1, 1.1], [0, 0], color=self.fg_color, linewidth=2)
            
        # Plot channel positions
        ax.scatter(x, y, s=50, c='k', marker='o')
        
        # Add channel labels
        for i in range(n_channels):
            ax.text(x[i], y[i], channel_names[i], ha='center', va='center', 
                  fontsize=8, color='white', fontweight='bold',
                  bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
            
        # Set equal aspect ratio and remove axes
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, color=self.fg_color)
        
        # Adjust layout
        plt.tight_layout()
        
        if as_base64:
            return self._figure_to_base64(fig)
        return fig
    
    def visualize_treatment_progress(self, timestamps: List[datetime],
                                   metrics: Dict[str, List[float]],
                                   metric_labels: Optional[Dict[str, str]] = None,
                                   title: str = "Treatment Progress",
                                   as_base64: bool = False) -> Union[Figure, str]:
        """
        Visualize treatment progress over time with multiple metrics.
        
        Args:
            timestamps: List of datetime objects for each measurement point
            metrics: Dictionary mapping metric names to lists of values
            metric_labels: Optional dictionary mapping metric names to display labels
            title: Plot title
            as_base64: Whether to return the figure as a base64 encoded string
            
        Returns:
            Matplotlib figure or base64 encoded string
        """
        # Check if timestamps match the length of metrics
        n_points = len(timestamps)
        for metric_name, values in metrics.items():
            if len(values) != n_points:
                raise ValueError(f"Length of metric '{metric_name}' ({len(values)}) doesn't match number of timestamps ({n_points})")
                
        # Create metric labels if not provided
        if metric_labels is None:
            metric_labels = {name: name for name in metrics.keys()}
            
        # Create figure
        fig, ax = self._create_figure()
        
        # Plot each metric
        for i, (metric_name, values) in enumerate(metrics.items()):
            label = metric_labels.get(metric_name, metric_name)
            ax.plot(timestamps, values, 
                  label=label, 
                  color=self.colors[i % len(self.colors)],
                  marker='o')
            
        # Set up axis labels and title
        ax.set_xlabel('Date/Time', color=self.fg_color)
        ax.set_ylabel('Value', color=self.fg_color)
        ax.set_title(title, color=self.fg_color)
        ax.grid(True, alpha=0.3, color=self.grid_color)
        
        # Format x-axis as dates
        plt.gcf().autofmt_xdate()
        
        # Add legend
        ax.legend(loc='best')
        
        # Adjust layout
        plt.tight_layout()
        
        if as_base64:
            return self._figure_to_base64(fig)
        return fig
    
    def visualize_phase_locking(self, phase_data: np.ndarray,
                              channel_pairs: List[Tuple[int, int]],
                              channel_names: Optional[List[str]] = None,
                              time_axis: Optional[np.ndarray] = None,
                              title: str = "Phase Locking Value",
                              as_base64: bool = False) -> Union[Figure, str]:
        """
        Visualize phase locking values between channel pairs over time.
        
        Args:
            phase_data: Array of phase locking values with shape (n_pairs, n_timepoints)
            channel_pairs: List of (ch1, ch2) channel index pairs
            channel_names: Optional list of channel names
            time_axis: Optional time axis values (if None, uses sample indices)
            title: Plot title
            as_base64: Whether to return the figure as a base64 encoded string
            
        Returns:
            Matplotlib figure or base64 encoded string
        """
        # Check dimensions
        n_pairs, n_timepoints = phase_data.shape
        if len(channel_pairs) != n_pairs:
            raise ValueError(f"Number of channel pairs ({len(channel_pairs)}) doesn't match phase data first dimension ({n_pairs})")
            
        # Create time axis if not provided
        if time_axis is None:
            time_axis = np.arange(n_timepoints)
            
        # Create channel names if not provided
        if channel_names is None:
            channel_names = [f"Ch{i}" for i in range(max([max(p) for p in channel_pairs]) + 1)]
            
        # Create pair labels
        pair_labels = [f"{channel_names[p[0]]} - {channel_names[p[1]]}" 
                     for p in channel_pairs]
            
        # Create figure
        fig, ax = self._create_figure()
        
        # Plot each channel pair
        for i in range(n_pairs):
            ax.plot(time_axis, phase_data[i], 
                  label=pair_labels[i], 
                  color=self.colors[i % len(self.colors)])
            
        # Set up axis labels and title
        ax.set_xlabel('Time', color=self.fg_color)
        ax.set_ylabel('Phase Locking Value', color=self.fg_color)
        ax.set_title(title, color=self.fg_color)
        ax.grid(True, alpha=0.3, color=self.grid_color)
        
        # Set y-axis limits (PLV ranges from 0 to 1)
        ax.set_ylim(-0.05, 1.05)
        
        # Add legend
        if n_pairs <= 10:  # Only show legend if not too many pairs
            ax.legend(loc='best')
            
        # Adjust layout
        plt.tight_layout()
        
        if as_base64:
            return self._figure_to_base64(fig)
        return fig
    
    def visualize_device_status(self, device_status: Dict[str, Dict],
                              as_base64: bool = False) -> Union[Figure, str]:
        """
        Visualize status of stimulation devices.
        
        Args:
            device_status: Dictionary mapping device IDs to status dictionaries
            as_base64: Whether to return the figure as a base64 encoded string
            
        Returns:
            Matplotlib figure or base64 encoded string
        """
        # Extract device information
        device_ids = list(device_status.keys())
        n_devices = len(device_ids)
        
        if n_devices == 0:
            logger.warning("No devices to visualize")
            fig, ax = self._create_figure()
            ax.text(0.5, 0.5, "No devices to visualize", 
                  ha='center', va='center', color=self.fg_color)
            
            if as_base64:
                return self._figure_to_base64(fig)
            return fig
            
        # Create figure with subplots
        fig = plt.figure(figsize=(12, 4 + 2 * n_devices), dpi=self.dpi, facecolor=self.bg_color)
        
        # Create grid layout
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(n_devices, 3, figure=fig)
        
        # Create subplots for each device
        for i, device_id in enumerate(device_ids):
            status = device_status[device_id]
            
            # Device overview panel
            ax1 = fig.add_subplot(gs[i, 0])
            self._setup_axis_theme(ax1)
            self._plot_device_overview(ax1, device_id, status)
            
            # Amplitude/frequency panel
            ax2 = fig.add_subplot(gs[i, 1])
            self._setup_axis_theme(ax2)
            self._plot_stimulation_params(ax2, status)
            
            # Battery and connection panel
            ax3 = fig.add_subplot(gs[i, 2])
            self._setup_axis_theme(ax3)
            self._plot_battery_connection(ax3, status)
            
        # Add overall title
        plt.suptitle("Device Status Overview", fontsize=16, color=self.fg_color)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
        
        if as_base64:
            return self._figure_to_base64(fig)
        return fig
    
    def _setup_axis_theme(self, ax):
        """Apply theme to an axis."""
        ax.set_facecolor(self.bg_color)
        ax.spines['bottom'].set_color(self.fg_color)
        ax.spines['top'].set_color(self.fg_color)
        ax.spines['left'].set_color(self.fg_color)
        ax.spines['right'].set_color(self.fg_color)
        ax.tick_params(axis='both', colors=self.fg_color)
        ax.yaxis.label.set_color(self.fg_color)
        ax.xaxis.label.set_color(self.fg_color)
        ax.title.set_color(self.fg_color)
    
    def _plot_device_overview(self, ax, device_id: str, status: Dict):
        """Plot device overview panel."""
        # Clear existing elements
        ax.clear()
        
        # Set title
        ax.set_title(f"Device: {device_id}", color=self.fg_color)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Create text summary
        text = []
        text.append(f"Status: {'Active' if status.get('active', False) else 'Inactive'}")
        text.append(f"Connected: {'Yes' if status.get('connected', False) else 'No'}")
        text.append(f"Waveform: {status.get('waveform', 'N/A')}")
        
        if 'impedance' in status:
            imp_text = []
            for channel, value in status['impedance'].items():
                imp_text.append(f"{channel}: {value:.1f} kΩ")
            text.append(f"Impedance: {', '.join(imp_text)}")
            
        # Display text
        ax.text(0.5, 0.5, '\n'.join(text), 
              ha='center', va='center', 
              color=self.fg_color,
              fontsize=10)
        
        # Add a border around the panel
        for spine in ax.spines.values():
            spine.set_visible(True)
    
    def _plot_stimulation_params(self, ax, status: Dict):
        """Plot stimulation parameters panel."""
        # Clear existing elements
        ax.clear()
        
        # Set title
        ax.set_title("Stimulation Parameters", color=self.fg_color)
        
        # Check if device has stimulation parameters
        if 'current_amplitude' not in status and 'frequency' not in status:
            ax.text(0.5, 0.5, "No stimulation parameters available", 
                  ha='center', va='center', color=self.fg_color)
            return
            
        # Create bar chart for amplitude and frequency
        params = ['current_amplitude', 'frequency']
        values = [status.get(p, 0) for p in params]
        labels = ['Amplitude (mA)', 'Frequency (Hz)']
        
        # Normalize values for display
        norm_values = [values[0] / 5.0, values[1] / 100.0]  # Normalize to 0-1 range
        
        # Create horizontal bars
        bars = ax.barh([0, 1], norm_values, height=0.6, 
                     color=[self.colors[0], self.colors[1]])
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(max(0.05, norm_values[i] - 0.2), i, f"{v:.1f}", 
                  va='center', color='white', fontweight='bold')
            
        # Set axis labels
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        
        # Add phase information if available
        if 'phase' in status:
            ax.text(0.5, -0.2, f"Phase: {status['phase']:.1f}°", 
                  ha='center', va='center', color=self.fg_color,
                  transform=ax.transAxes)
            
        # Add a border around the panel
        for spine in ax.spines.values():
            spine.set_visible(True)
    
    def _plot_battery_connection(self, ax, status: Dict):
        """Plot battery and connection status panel."""
        # Clear existing elements
        ax.clear()
        
        # Set title
        ax.set_title("System Status", color=self.fg_color)
        
        # Check if device has battery and connection information
        has_battery = 'battery_level' in status
        has_connection = 'connection_strength' in status
        
        if not has_battery and not has_connection:
            ax.text(0.5, 0.5, "No status information available", 
                  ha='center', va='center', color=self.fg_color)
            return
            
        # Create a list of status items to display
        items = []
        values = []
        colors = []
        
        if has_battery:
            items.append('Battery')
            values.append(status['battery_level'] / 100.0)  # Normalize to 0-1
            
            # Color based on level
            if status['battery_level'] < 20:
                colors.append('red')
            elif status['battery_level'] < 50:
                colors.append('orange')
            else:
                colors.append('green')
                
        if has_connection:
            items.append('Signal')
            values.append(status['connection_strength'] / 100.0)  # Normalize to 0-1
            
            # Color based on strength
            if status['connection_strength'] < 30:
                colors.append('red')
            elif status['connection_strength'] < 70:
                colors.append('orange')
            else:
                colors.append('green')
                
        # Create horizontal bars
        y_pos = range(len(items))
        bars = ax.barh(y_pos, values, height=0.6, color=colors)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(max(0.05, v - 0.2), i, f"{int(v * 100)}%", 
                  va='center', color='white', fontweight='bold')
            
        # Set axis labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(items)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        
        # Add a border around the panel
        for spine in ax.spines.values():
            spine.set_visible(True)
