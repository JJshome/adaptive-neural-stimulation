# Neural Stimulation Simulation Tool

This directory contains a simulation tool for the Adaptive Phase-Differential Brain Nerve Stimulation System, allowing users to explore the effects of different stimulation parameters on neural circuit dynamics.

## Overview

The simulation tool provides an interactive environment for experimenting with various stimulation parameters and observing their effects on simulated neural circuits. This can be valuable for:

- Training clinicians and researchers on the system's capabilities
- Developing and refining stimulation protocols for specific conditions
- Exploring the theoretical underpinnings of phase-differential stimulation
- Planning patient-specific treatment approaches

## Features

- Interactive visualization of neural circuit activity
- Customizable stimulation parameters including:
  - Amplitude
  - Frequency
  - Phase differences
  - Electrode placements
  - Stimulation patterns
- Real-time feedback showing effects on neural plasticity
- Multiple circuit models representing different brain regions and pathologies
- Export functionality for saving and sharing simulation results

## Getting Started

1. Install the required dependencies (see `requirements.txt`)
2. Run `python simulation_app.py` to launch the application
3. Choose a neural circuit model from the available options
4. Adjust stimulation parameters using the interactive controls
5. Observe the effects on the simulated neural activity

## Technical Details

The simulation is based on biophysically realistic neural network models, incorporating:

- Hodgkin-Huxley neuron models for accurate action potential dynamics
- Spike-timing-dependent plasticity (STDP) for modeling synaptic changes
- Volume conductor models for simulating electric field propagation
- Realistic brain anatomy based on MRI-derived models

## Example Simulations

The `examples/` directory contains pre-configured simulations for common use cases:

- Stroke rehabilitation scenarios
- Cognitive enhancement protocols
- Epilepsy management approaches
- Depression treatment simulations

## Custom Models

Users can create custom neural circuit models by following the guidelines in the `custom_models.md` documentation.

## Limitations

It's important to note that while the simulation provides valuable insights, it has limitations:

- Simplified representation of complex neural dynamics
- Idealized tissue properties that may differ from real-world scenarios
- Limited validation against in vivo recordings
- Exclusion of certain biological mechanisms for computational feasibility

## References

The simulation models are based on published research in computational neuroscience. See `references.md` for a complete list of sources.
