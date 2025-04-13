# get-nwbfile-info

A tool to analyze NWB (Neurodata Without Borders) files and generate Python code for accessing their objects and fields, especially for use by LLMs.

## Installation

```bash
pip install -e .
```

## Usage

### Command Line

```bash
# For remote files
get-nwbfile-info usage-script https://api.dandiarchive.org/api/assets/7423831f-100c-4103-9dde-73ac567d32fb/download/
# Prints a usage script

# For local files
get-nwbfile-info usage-script path/to/file.nwb
# Prints a usage script
```
