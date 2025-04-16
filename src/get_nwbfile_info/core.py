"""Core functionality for analyzing NWB files."""

import warnings
import numpy as np
import h5py
import pynwb
import hdmf
from datetime import datetime
from collections.abc import Iterable
from hdmf.common import DynamicTable

# Limit the number of fields to show
# For example, see: get-nwbfile-info usage-script https://api.dandiarchive.org/api/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/download/
MAX_NUM_FIELDS_TO_SHOW = 15

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

def process_dict_like(obj, path):
    """
    Process dictionary-like objects (including LabelledDict) and generate Python code to access their items.
    """
    results = []
    num_shown_fields = 0
    unshown_field_names = []

    # Try to iterate through items
    for key, value in obj.items():
        # Skip private keys
        if isinstance(key, str) and key.startswith('_'):
            continue

        if num_shown_fields >= MAX_NUM_FIELDS_TO_SHOW:
            if isinstance(key, str):
                unshown_field_names.append(key)
            else:
                unshown_field_names.append(str(key))
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
        results.extend(process_nwb_container(value, item_path))

        num_shown_fields += 1

    if unshown_field_names:
        results.append("# ...")
        results.append(f"# Other fields: {', '.join(unshown_field_names)}")

    return results

def process_nwb_container(obj, path="nwb"):
    """
    Recursively process an NWB container and generate Python code to access its fields.
    """
    results = []

    # Process NWBContainer or NWBData objects
    if isinstance(obj, hdmf.container.AbstractContainer):
        # Add a comment about the object type
        type_name = get_type_name(obj)
        results.append(f"{path} # ({type_name})")

        # Get all field names upfront and filter out private ones
        field_names = [name for name in obj.fields.keys() if not name.startswith('_')]

        # Split fields into non-container and container fields
        non_container_fields = []
        container_fields = []
        for name in field_names:
            if isinstance(obj.fields[name], hdmf.container.AbstractContainer):
                container_fields.append(name)
            else:
                non_container_fields.append(name)

        # Process non-container fields
        for field_name in non_container_fields:
            field_value = obj.fields[field_name]
            field_path = f"{path}.{field_name}"

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
                            results.append(f"# First few values of {field_path}: {sample}".replace("\n", " "))
                        # For 2D datasets
                        elif len(field_value.shape) == 2 and field_value.shape[0] > 0 and field_value.shape[1] > 0:
                            sample = field_value[0, :min(10, field_value.shape[1])]
                            results.append(f"# First row sample of {field_path}: {sample}".replace("\n", " "))
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
                results.extend(process_dict_like(field_value, field_path))

        # Process container fields
        for field_name in container_fields:
            field_value = obj.fields[field_name]
            field_path = f"{path}.{field_name}"

            # Recursively process the field value
            results.extend(process_nwb_container(field_value, field_path))

        # Special handling for DynamicTable objects
        if isinstance(obj, DynamicTable):
            # Comment the dataframe code out because we don't want to download data if we run the script for testing
            results.append(f"# {path}.to_dataframe() # (DataFrame) Convert to a pandas DataFrame with {len(obj)} rows and {len(obj.columns)} columns")
            results.append(f"# {path}.to_dataframe().head() # (DataFrame) Show the first few rows of the pandas DataFrame")
            results.append(f'# Number of rows: {len(obj)}')
            # show each of the columns
            for colname in obj.colnames:
                results.append(f"{path}.{colname} # ({get_type_name(obj[colname])}) {obj[colname].description}")
                if get_type_name(obj[colname]) == "VectorIndex":
                    for j in range(len(obj[colname + "_index"])):
                        if j <= 3:
                            results.append(f"# {path}.{colname}_index[{j}] # ({get_type_name(obj[colname+"_index"][j])})")
                    if len(obj[colname + "_index"]) > 3:
                        results.append(f"# ...")


    # Process dictionaries and dict-like objects
    elif isinstance(obj, dict) or (hasattr(obj, "items") and callable(getattr(obj, "items"))):
        results.extend(process_dict_like(obj, path))

    # Process iterables (excluding strings)
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, dict, h5py.Dataset)):
        try:
            for i, item in enumerate(obj):
                if i >= 10:  # Limit to first 10 items
                    results.append(f"# ... more items in {path}")
                    break

                item_path = f"{path}[{i}]"
                results.extend(process_nwb_container(item, item_path))
        except Exception as e:
            warnings.warn(f"Could not iterate through {path}: {e}")

    return results

def get_nwbfile_usage_script(url_or_path):
    """
    Analyze an NWB file and return Python code to access its objects and fields.
    """
    is_url = url_or_path.startswith(('http://', 'https://'))
    is_lindi = url_or_path.endswith(('.lindi.json', '.lindi.tar'))

    # Header lines
    header_lines = [
        f"# This script shows how to load the NWB file at {url_or_path} in Python using PyNWB",
        "",
        "import pynwb",
        "import h5py"
    ]

    # Show different loading methods based on URL type
    if is_url and not is_lindi:
        header_lines.extend([
            "import remfile",
            "",
            "# Load",
            f"url = \"{url_or_path}\"",
            "remote_file = remfile.File(url)",
            "h5_file = h5py.File(remote_file)",
            "io = pynwb.NWBHDF5IO(file=h5_file)",
            "nwb = io.read()",
        ])
    elif is_url and is_lindi:
        header_lines.extend([
            "import lindi",
            "",
            "# Load",
            f"url = \"{url_or_path}\"",
            "f = lindi.LindiH5pyFile.from_lindi_file(url)",
            "io = pynwb.NWBHDF5IO(file=f, mode='r')",
            "nwb = io.read()",
        ])
    elif not is_url and is_lindi:
        header_lines.extend([
            "import lindi",
            "",
            "# Load",
            f"path = \"{url_or_path}\"",
            "f = lindi.LindiH5pyFile.from_lindi_file(path)",
            "io = pynwb.NWBHDF5IO(file=f, mode='r')",
            "nwb = io.read()",
        ])
    else:  # not is_url and not is_lindi:
        header_lines.extend([
            "",
            "# Load",
            f"path = \"{url_or_path}\"",
            "nwb = pynwb.read_nwb(path=path)",
        ])

    header_lines.append("")

    # Read the NWB file using remfile for remote URLs
    if is_url and not is_lindi:
        import remfile
        remote_file = remfile.File(url_or_path)
        h5_file = h5py.File(remote_file)
        io = pynwb.NWBHDF5IO(file=h5_file)
    elif is_lindi:
        import lindi
        f = lindi.LindiH5pyFile.from_lindi_file(url_or_path)
        io = pynwb.NWBHDF5IO(file=f, mode='r')
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
