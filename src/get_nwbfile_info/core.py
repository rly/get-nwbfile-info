"""Core functionality for analyzing NWB files."""

import warnings
import numpy as np
import h5py
import pynwb
import hdmf
from datetime import datetime
from collections.abc import Iterable

def get_type_name(obj):
    """Get a string representation of the object's type."""
    if obj is None:
        return "None"

    if hasattr(obj, "__class__"):
        return obj.__class__.__name__

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

def process_dict_like(obj, path, visited):
    """
    Process dictionary-like objects (including LabelledDict) and generate Python code to access their items.
    """
    results = []

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

        # AbstractContainer objects will be printed later in recursion
        if not isinstance(value, hdmf.container.AbstractContainer):
            type_name = get_type_name(value)
            results.append(f"{item_path} # ({type_name})")

        # Recursively process the value
        results.extend(process_nwb_container(value, item_path, visited))

    return results

def process_nwb_container(obj, path="nwb", visited=None):
    """
    Recursively process an NWB container and generate Python code to access its fields.
    """
    if visited is None:
        visited = set()

    # Avoid processing the same object twice (prevents infinite recursion)
    # Using path instead of id(obj) because I found that id(obj) is not unique sometimes, which surprised me.
    # This happened in https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/
    # where nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
    # had the same id(obj) as nwb.processing["ophys"].data_interfaces["EventAmplitude"].rois.table
    # Why???
    # It ended up leaving out a lot of details about nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
    # obj_id = id(obj) (not using this, see above)
    obj_id = path  # Using this instead (see above)
    if obj_id in visited:
        return []

    visited.add(obj_id)
    results = []

    # Process NWBContainer or NWBData objects
    if isinstance(obj, hdmf.container.AbstractContainer):
        # Add a comment about the object type
        type_name = get_type_name(obj)
        results.append(f"{path} # ({type_name})")

        # Process non container fields
        for field_name, field_value in obj.fields.items():
            # Skip private fields
            if field_name.startswith('_'):
                continue

            field_path = f"{path}.{field_name}"

            if isinstance(field_value, hdmf.container.AbstractContainer):
                # Don't process containers yet
                continue

            # Add the field with a comment if the value is small
            if isinstance(field_value, h5py.Dataset):
                # Add basic dataset info
                results.append(f"{field_path} # ({get_type_name(field_value)}) shape {field_value.shape}; dtype {field_value.dtype}")

                # Always add code to access the dataset
                # But comment it out because we don't want to actually download
                # the data if we run the script for testing.
                if len(field_value.shape) == 1:
                    results.append(f"# {field_path}[:] # Access all data")
                    results.append(f"# {field_path}[0:10] # Access first 10 elements")
                elif len(field_value.shape) == 2:
                    results.append(f"# {field_path}[:, :] # Access all data")
                    results.append(f"# {field_path}[0:10, :] # Access first 10 rows")
                    results.append(f"# {field_path}[:, 0:10] # Access first 10 columns")
                elif len(field_value.shape) >= 3:
                    results.append(f"# {field_path}[:, :, :] # Access all data")
                    results.append(f"# {field_path}[0, :, :] # Access first plane")

                # Try to read and display small datasets in comments
                try:
                    if field_value.size < 50:  # type: ignore
                        # Only for reasonably small datasets
                        # For 1D datasets
                        if len(field_value.shape) == 1 and field_value.shape[0] > 0:
                            sample = field_value[:min(10, field_value.shape[0])]
                            results.append(f"# First few values of {field_path}: {sample}")
                        # For 2D datasets
                        elif len(field_value.shape) == 2 and field_value.shape[0] > 0 and field_value.shape[1] > 0:
                            sample = field_value[0, :min(10, field_value.shape[1])]
                            results.append(f"# First row sample of {field_path}: {sample}")
                except Exception as e:
                    warnings.warn(f"Could not read data from {field_path}: {e}")
            elif is_small_value(field_value):  # non-h5py.Dataset
                type_name = get_type_name(field_value)
                value_str = format_value(field_value)
                results.append(f"{field_path} # ({type_name}) {value_str}")
            else:
                type_name = get_type_name(field_value)
                results.append(f"{field_path} # ({type_name})")

            # Special handling for LabelledDict objects
            if isinstance(field_value, hdmf.utils.LabelledDict):
                results.extend(process_dict_like(field_value, field_path, visited))

        # Process container fields
        for field_name, field_value in obj.fields.items():
            # Skip private fields
            if field_name.startswith('_'):
                continue

            field_path = f"{path}.{field_name}"

            # Recursively process the field value if it's a container
            if isinstance(field_value, hdmf.container.AbstractContainer):
                results.extend(process_nwb_container(field_value, field_path, visited))

    # Process dictionaries and dict-like objects
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
            warnings.warn(f"Could not iterate through {path}: {e}")

    return results

def get_nwbfile_usage_script(url_or_path):
    """
    Analyze an NWB file and return Python code to access its objects and fields.
    """
    is_url = url_or_path.startswith(('http://', 'https://'))
    is_lindi = url_or_path.endswith('.lindi.json') or url_or_path.endswith('.lindi.tar')

    # Header lines
    header_lines = [
        f"# This script shows how to load the NWB file at {url_or_path} in Python using PyNWB",
        "",
        "import pynwb",
        "import h5py"
    ]

    if is_url and not is_lindi:
        header_lines.extend([
            "import remfile",
        ])
    elif is_lindi:
        header_lines.extend([
            "import lindi",
        ])

    header_lines.extend([
        "",
        "# Load"
    ])

    if is_url:
        header_lines.extend([
            f"url = \"{url_or_path}\""
        ])
    else:
        header_lines.extend([
            f"path = \"{url_or_path}\"",
        ])

    # Show different loading methods based on URL type
    if is_url and not is_lindi:
        header_lines.extend([
            "file = remfile.File(url)",
            "f = h5py.File(file)",
            "io = pynwb.NWBHDF5IO(file=f)",
            "nwb = io.read()"
        ])
    elif is_url and is_lindi:
        header_lines.extend([
            "f = lindi.LindiH5pyFile.from_lindi_file(url)",
            "io = pynwb.NWBHDF5IO(file=f, mode='r')",
            "nwb = io.read()"
        ])
    elif not is_url and is_lindi:
        header_lines.extend([
            "f = lindi.LindiH5pyFile.from_lindi_file(path)",
            "io = pynwb.NWBHDF5IO(file=f, mode='r')",
            "nwb = io.read()"
        ])
    elif not is_url and not is_lindi:
        header_lines.extend([
            "nwb = pynwb.read_nwb(path=path)"
        ])
    else:
        raise ValueError("Impossible condition")

    header_lines.append("")

    # Read the NWB file using remfile for remote URLs
    if is_url and not is_lindi:
        import remfile
        remote_file = remfile.File(url_or_path)
        h5_file = h5py.File(remote_file)
        io = pynwb.NWBHDF5IO(file=h5_file)
    elif is_lindi:
        import lindi
        io = pynwb.NWBHDF5IO(file=lindi.LindiH5pyFile.from_lindi_file(url_or_path), mode='r')
    else:
        io = pynwb.read_nwb(path=url_or_path)  # type: ignore

    try:
        # Read the NWB file
        nwb = io.read()

        # Process the NWB file - collect results in a list
        results_list = process_nwb_container(nwb)

        if len(results_list) != len(set(results_list)):
            warnings.warn("Warning: Duplicate entries found in the results.")

        # Combine header and results
        all_lines = header_lines + results_list

        return "\n".join(all_lines)
    finally:
        io.close()
