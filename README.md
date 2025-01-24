# Project README

## Overview

This repository contains code, models, and resources for a simulation and training pipeline. Below is a detailed breakdown of the folder structure and its contents.

---

## Folder Structure

### Main Files and Directories

- **`__pycache__`**
  - Compiled Python bytecode files for optimized execution of Python modules.
  - Files:
    - `ot2_environment.cpython-311.pyc`
    - `sim_class.cpython-310.pyc`
    - `sim_class.cpython-311.pyc`

- **`meshes`**
  - Contains STL files for 3D models used in the simulation.
  - Files:
    - `base_link.stl`
    - `gantry_x1.stl`
    - `gantry_y1.stl`
    - `gantry_z1.stl`


- **`textures`**
  - Contains image files for applying textures to 3D objects in the simulation.
  - Files:
    - `01.png` to `10.png`
    - `_plates`: Subdirectory with additional texture files, including:
      - Corrected microscopy images (`*_Fish Eye Corrected.png`).

- **`wandb`**
  - Stores metadata and logs for `Weights & Biases` runs.
  - Subdirectories:
    - `run-YYYYMMDD_HHmmss-<run_id>`: Logs, metadata, and temporary files for each training session.
    - Links to log files for debugging and monitoring.

---

### Key Files

- **`model.zip`**
  - Compressed file containing a trained model or related data.

- **`ot2_gym_wrapper.py`**
  - Python script providing a Gym environment wrapper for simulations.

- **`ot_2_simulation_v6.urdf`**
  - URDF file describing the robotic system for use in simulations.

- **`sim.ipynb`**
  - Jupyter Notebook for interactive simulation tasks.

- **`sim_class.py`**
  - Python module defining the simulation class and associated methods.

- **`task11.ipynb`, `task12.ipynb`, `task13.ipynb`**
  - Jupyter Notebooks for specific tasks and/or ILO requirements.

- **`test_wrapper.py`**
  - Python script for testing the Gym wrapper.

- **`train.py`**
  - Main script for training models using the simulation pipeline.

- **`training.py`**
  - Supplementary training script with additional utilities.

- **`uvmapped_dish_large_comp.png`**
  - UV-mapped texture file for 3D visualization.

---

## How to Use

1. **Setup Environment:**
   - Install necessary Python dependencies listed in `requirements.txt` files located in `wandb` subdirectories.

2. **Run Simulations:**
   - Use `sim_class.py` and `ot2_gym_wrapper.py` to extend or customize the simulation environment.

3. **Training:**
   - Execute `train.py` or `training.py` to train models using the simulation.
   **warning** change the credentials to your own.

4. **Visualization:**
   - STL files in the `meshes` directory can be visualized using 3D modeling tools.
   - Apply textures from the `textures` directory for enhanced visuals.

5. **Logging and Monitoring:**
   - Use the `wandb` directory to analyze training performance and debug issues.

---

## Contributing

Contributions are welcome! Please follow the guidelines below:
- Fork the repository.
- Create a new branch for your feature or bugfix.
- Submit a pull request with detailed descriptions.



For questions or support, please contact 232997@buas.nl or joey.van.angelen@outlook.com
