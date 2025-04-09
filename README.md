# get-nwbfile-info

A tool to analyze NWB (Neurodata Without Borders) files and generate Python code for accessing their objects and fields.

## Installation

```bash
pip install get-nwbfile-info
```

## Usage

### Command Line

```bash
# For remote files
get-nwbfile-info ai-usage-script https://api.dandiarchive.org/api/assets/7423831f-100c-4103-9dde-73ac567d32fb/download/

# For local files
get-nwbfile-info ai-usage-script path/to/file.nwb
```

### Python

```python
from nwbinfo import analyze_nwb_file

# Analyze a remote file
code = analyze_nwb_file("https://api.dandiarchive.org/api/assets/7423831f-100c-4103-9dde-73ac567d32fb/download/")
print(code)

# Analyze a local file
code = analyze_nwb_file("path/to/file.nwb")
print(code)
```

## Output

The tool generates Python code that shows how to access objects and fields in the NWB file. For example:

```python
# This script shows how to load this in Python using PyNWB

import pynwb
import remfile
import h5py

# Load
path = "path/to/file.nwb"
nwb = pynwb.read_nwb(path=path)

nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["Fluorescence"].data[0:10, :] # Access first 10 rows
```

## Authors

- Ryan Ly
- Jeremy Magland
- Ben Dichter
