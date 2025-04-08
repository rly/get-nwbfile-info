#!/usr/bin/env python3
"""
Script to analyze an NWB file and print Python code for accessing its objects and fields.
Takes a URL to an NWB file on the DANDI Archive and generates Python code to access
all objects and their attributes/fields.
"""

import sys
import pynwb
import numpy as np
import h5py
from datetime import datetime
from pynwb import NWBHDF5IO
import inspect
from collections.abc import Iterable
import warnings
import remfile
import hdmf
import types

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def get_type_name(obj):
    """Get a string representation of the object's type."""
    if obj is None:
        return "None"

    if hasattr(obj, "__class__"):
        class_name = obj.__class__.__name__
        return class_name

    return str(type(obj))


def is_small_value(value):
    """Determine if a value is small enough to be displayed as a comment."""
    if value is None:
        return True

    # Handle basic types
    if isinstance(value, (str, int, float, bool)):
        return True

    # Handle datetime
    if isinstance(value, datetime):
        return True

    # Handle lists, tuples, etc. if they're small
    if isinstance(value, (list, tuple)) and len(value) < 10:
        return all(is_small_value(item) for item in value)

    # Handle numpy arrays if they're small
    if isinstance(value, np.ndarray) and value.size < 10:
        return True

    return False


def format_value(value):
    """Format a value for display in a comment."""
    if value is None:
        return "None"

    if isinstance(value, str):
        # Replace newlines with actual newline characters in the comment
        formatted = value.replace("\n", "\\n")
        if len(formatted) > 100:
            return formatted[:97] + "..."
        return formatted

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "[]"
        if all(isinstance(item, str) for item in value):
            return f"[{', '.join(repr(item) for item in value)}]"
        return str(value)

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return f"Empty array with shape {value.shape}"
        if value.size < 10:
            return str(value)
        return f"Array with shape {value.shape}; dtype {value.dtype}"

    return str(value)


def process_nwb_container(obj, path="nwb", visited=None):
    """
    Recursively process an NWB container and generate Python code to access its fields.

    Args:
        obj: The NWB object to process
        path: The Python code path to access this object
        visited: Set of already visited object IDs to avoid cycles

    Returns:
        List of strings with Python code to access the object and its fields
    """
    if visited is None:
        visited = set()

    # Avoid processing the same object twice (prevents infinite recursion)
    obj_id = id(obj)
    if obj_id in visited:
        return []

    visited.add(obj_id)
    results = []

    # Process NWBContainer or NWBData objects
    if isinstance(obj, hdmf.container.AbstractContainer):
        # Add a comment about the object type
        type_name = get_type_name(obj)
        results.append(f"{path} # ({type_name})")

        # Process each fieldobj.keys
        try:
            obj.fields.items()
        except:
            breakpoint()
        for field_name, field_value in obj.fields.items():
            # Skip private fields
            if field_name.startswith('_'):
                continue

            field_path = f"{path}.{field_name}"

            # Add the field with a comment if the value is small
            if isinstance(field_value, h5py.Dataset):
                # Add basic dataset info
                results.append(f"{field_path} # ({get_type_name(field_value)}) shape {field_value.shape}; dtype {field_value.dtype}")

                # Always add code to access the dataset
                if len(field_value.shape) == 1:
                    results.append(f"{field_path}[:] # Access all data")
                    results.append(f"{field_path}[0:10] # Access first 10 elements")
                elif len(field_value.shape) == 2:
                    results.append(f"{field_path}[:, :] # Access all data")
                    results.append(f"{field_path}[0:10, :] # Access first 10 rows")
                    results.append(f"{field_path}[:, 0:10] # Access first 10 columns")
                elif len(field_value.shape) >= 3:
                    results.append(f"{field_path}[:, :, :] # Access all data")
                    results.append(f"{field_path}[0, :, :] # Access first plane")

                # Try to read and display small datasets in comments
                try:
                    if field_value.size < 50:  # Only for reasonably small datasets
                        # For 1D datasets
                        if len(field_value.shape) == 1 and field_value.shape[0] > 0:
                            sample = field_value[:min(10, field_value.shape[0])]
                            results.append(f"# First few values of {field_path}: {sample}")
                        # For 2D datasets
                        elif len(field_value.shape) == 2 and field_value.shape[0] > 0 and field_value.shape[1] > 0:
                            sample = field_value[0, :min(10, field_value.shape[1])]
                            results.append(f"# First row sample of {field_path}: {sample}")
                except Exception as e:
                    # Raise a warning if we can't read the data
                    warnings.warn(f"Could not read data from {field_path}: {e}")
            elif is_small_value(field_value):  # non-h5py.Dataset
                type_name = get_type_name(field_value)
                value_str = format_value(field_value)
                results.append(f"{field_path} # ({type_name}) {value_str}")
            else:
                type_name = get_type_name(field_value)
                results.append(f"{field_path} # ({type_name})")

            # Recursively process the field value if it's a container
            if isinstance(obj, hdmf.container.AbstractContainer):
                results.extend(process_nwb_container(field_value, field_path, visited))

            # Special handling for LabelledDict objects (like acquisition, processing)
            elif hasattr(field_value, "items") and callable(getattr(field_value, "items")):
                results.extend(process_dict_like(field_value, field_path, visited))

    # Process dictionaries and dict-like objects (like acquisition, processing, etc.)
    elif isinstance(obj, dict) or (hasattr(obj, "items") and callable(getattr(obj, "items"))):
        results.extend(process_dict_like(obj, path, visited))

    # Process iterables (excluding strings)
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, dict, h5py.Dataset)):
        try:
            for i, item in enumerate(obj):
                if i >= 10:  # Limit to first 10 items
                    results.append(f"# ... more items in {path}")
                    break

                item_path = f"{path}[{i}]"
                results.extend(process_nwb_container(item, item_path, visited))
        except Exception as e:
            # Raise a warning if we can't iterate
            warnings.warn(f"Could not iterate through {path}: {e}")

    return results


def process_dict_like(obj, path, visited):
    """
    Process dictionary-like objects (including LabelledDict) and generate Python code to access their items.

    Args:
        obj: The dictionary-like object to process
        path: The Python code path to access this object
        visited: Set of already visited object IDs to avoid cycles

    Returns:
        List of strings with Python code to access the object's items
    """
    results = []

    # try:
    # Try to iterate through items
    for key, value in obj.items():
        # Skip private keys
        if isinstance(key, str) and key.startswith('_'):
            continue

        # Format the path based on the key type
        if isinstance(key, str):
            item_path = f"{path}[\"{key}\"]"
        else:
            item_path = f"{path}[{key}]"

        # Add a comment about the value type
        type_name = get_type_name(value)
        results.append(f"{item_path} # ({type_name})")

        # Recursively process the value
        results.extend(process_nwb_container(value, item_path, visited))
    # except (AttributeError, TypeError) as e:
    #     # If .items() doesn't work, raise a warning
    #     print(obj)
    #     breakpoint()
    #     print(type(obj), path)
    #     warnings.warn(f"Could not process dictionary-like object {path}: {e}")
    #     stop

    return results


def analyze_nwb_file(url):
    """
    Analyze an NWB file and print Python code to access its objects and fields.

    Args:
        url: URL to the NWB file
    """
    # Header lines
    header_lines = [
        "# This script shows how to load this in Python using PyNWB",
        "",
        "import pynwb",
        "import remfile",
        "import h5py",
        "",
        "# Load",
        f"path = \"{url}\""
    ]

    # Show different loading methods based on URL type
    if url.startswith(('http://', 'https://')):
        header_lines.extend([
            "# For remote files:",
            "file = remfile.File(path)",
            "f = h5py.File(file)",
            "io = pynwb.NWBHDF5IO(file=f)",
            "nwb = io.read()"
        ])
    else:
        header_lines.append("nwb = pynwb.read_nwb(path=path)")

    header_lines.append("")

    # Print header
    for line in header_lines:
        print(line)

    # Read the NWB file using remfile for remote URLs
    if url.startswith(('http://', 'https://')):
        # Open the remote file using remfile
        remote_file = remfile.File(url)
        h5_file = h5py.File(remote_file)
        io = NWBHDF5IO(file=h5_file)
    else:
        # For local files
        io = NWBHDF5IO(url, mode='r')

    # Read the NWB file
    nwb = io.read()

    # Process the NWB file - collect results in a list
    results_list = process_nwb_container(nwb)

    # Convert to a set to remove duplicates
    unique_results = set(results_list)

    # Convert back to a sorted list for output
    sorted_results = sorted(unique_results)

    # Print the results
    for line in sorted_results:
        print(line)

    # Close the file
    io.close()


def main():
    """Main function to parse arguments and analyze the NWB file."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <nwb_file_url>")
        print("Example: python get_nwbfile_info.py https://api.dandiarchive.org/api/assets/7423831f-100c-4103-9dde-73ac567d32fb/download/")
        sys.exit(1)

    url = sys.argv[1]
    analyze_nwb_file(url)


if __name__ == "__main__":
    main()
