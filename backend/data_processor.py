#!/usr/bin/env python3
"""
Data Processor for .hea/.dat files
Converts raw PhysioNet format to features for multimodal prediction
"""

import numpy as np
import struct
from pathlib import Path
import re
from typing import Dict, List, Tuple, Any

class DataProcessor:
    """Process .hea and .dat files from CTU-UHB database"""

    def __init__(self):
        # Expected clinical parameters from .hea files
        self.clinical_params = [
            'pH', 'BDecf', 'pCO2', 'BE', 'Apgar1', 'Apgar5',
            'NICU days', 'Seizures', 'HIE', 'Intubation',
            'Gest. weeks', 'Weight(g)', 'Sex', 'Age', 'Gravidity', 'Parity',
            'Diabetes', 'Hypertension', 'Preeclampsia', 'Liq. praecox',
            'Pyrexia', 'Meconium', 'Presentation', 'Induced',
            'I.stage', 'NoProgress', 'II.stage', 'Deliv. type'
        ]

        # Default values for missing parameters
        self.default_values = {
            'pH': 7.35, 'BDecf': 0.0, 'pCO2': 5.0, 'BE': 0.0,
            'Apgar1': 8, 'Apgar5': 9, 'NICU days': 0, 'Seizures': 0,
            'HIE': 0, 'Intubation': 0, 'Gest. weeks': 39,
            'Weight(g)': 3300, 'Sex': 1, 'Age': 28, 'Gravidity': 1,
            'Parity': 0, 'Diabetes': 0, 'Hypertension': 0,
            'Preeclampsia': 0, 'Liq. praecox': 0, 'Pyrexia': 0,
            'Meconium': 0, 'Presentation': 1, 'Induced': 0,
            'I.stage': 300, 'NoProgress': 0, 'II.stage': 30, 'Deliv. type': 1
        }

    async def process_files(self, hea_path: Path, dat_path: Path) -> Dict[str, Any]:
        """Process .hea and .dat files into features"""

        # Parse header file
        header_info = self.parse_hea_file(hea_path)

        # Extract clinical features
        clinical_features = self.extract_clinical_features(header_info)

        # Process signal data
        signal_data = self.parse_dat_file(dat_path, header_info)
        signal_features = self.extract_signal_features(signal_data)

        return {
            "clinical_features": clinical_features,
            "signal_features": signal_features,
            "processing_info": {
                "record_id": header_info.get("record_id"),
                "signal_length": len(signal_features),
                "clinical_params_count": len(clinical_features),
                "sampling_rate": header_info.get("sampling_rate", 4),
                "duration_minutes": len(signal_features) / (header_info.get("sampling_rate", 4) * 60)
            }
        }

    def parse_hea_file(self, hea_path: Path) -> Dict[str, Any]:
        """Parse .hea header file to extract metadata and clinical parameters"""

        header_info = {
            "record_id": hea_path.stem,
            "clinical_params": {},
            "signal_info": {}
        }

        with open(hea_path, 'r') as f:
            lines = f.readlines()

        # Parse first line for basic info
        if lines:
            first_line = lines[0].strip().split()
            if len(first_line) >= 4:
                header_info["record_id"] = first_line[0]
                header_info["num_signals"] = int(first_line[1])
                header_info["sampling_rate"] = int(first_line[2])
                header_info["num_samples"] = int(first_line[3])

        # Parse signal information
        signal_idx = 0
        for i, line in enumerate(lines[1:], 1):
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 9:
                    signal_name = parts[-1]  # Signal name is usually last
                    header_info["signal_info"][signal_idx] = {
                        "name": signal_name,
                        "format": parts[1] if len(parts) > 1 else "16",
                        "gain": parts[2].split('(')[0] if len(parts) > 2 else "100",
                        "offset": parts[4] if len(parts) > 4 else "0"
                    }
                    signal_idx += 1

        # Parse clinical parameters (lines starting with #)
        for line in lines:
            line = line.strip()
            if line.startswith('#') and not line.startswith('#--') and not line.startswith('#-'):
                # Extract parameter name and value
                match = re.match(r'#([^#\s]+)\s+(.+)', line)
                if match:
                    param_name = match.group(1).strip()
                    param_value = match.group(2).strip()

                    # Try to convert to numeric
                    try:
                        if '.' in param_value:
                            param_value = float(param_value)
                        else:
                            param_value = int(param_value)
                    except ValueError:
                        pass  # Keep as string

                    header_info["clinical_params"][param_name] = param_value

        return header_info

    def parse_dat_file(self, dat_path: Path, header_info: Dict) -> np.ndarray:
        """Parse binary .dat file to extract signal data"""

        num_signals = header_info.get("num_signals", 2)
        num_samples = header_info.get("num_samples", 19200)

        # Read binary data
        with open(dat_path, 'rb') as f:
            data = f.read()

        # PhysioNet format: 16-bit signed integers, little-endian
        # Each sample contains values for all signals
        sample_size = num_signals * 2  # 2 bytes per signal per sample
        expected_size = num_samples * sample_size

        if len(data) < expected_size:
            print(f"Warning: File size mismatch. Expected {expected_size}, got {len(data)}")
            num_samples = len(data) // sample_size

        # Unpack data
        format_string = f'<{num_signals * num_samples}h'  # Little-endian signed short
        try:
            unpacked = struct.unpack(format_string, data[:sample_size * num_samples])
        except struct.error as e:
            print(f"Error unpacking data: {e}")
            # Try with available data
            available_samples = len(data) // sample_size
            format_string = f'<{num_signals * available_samples}h'
            unpacked = struct.unpack(format_string, data[:sample_size * available_samples])
            num_samples = available_samples

        # Reshape into signals x samples
        signals = np.array(unpacked).reshape((num_samples, num_signals))

        # Extract FHR signal (usually first signal)
        fhr_signal = signals[:, 0]

        return fhr_signal

    def extract_clinical_features(self, header_info: Dict) -> List[float]:
        """Extract and normalize clinical features"""

        clinical_params = header_info.get("clinical_params", {})
        features = []

        for param in self.clinical_params:
            if param in clinical_params:
                value = clinical_params[param]
                # Convert to float if possible
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = self.default_values.get(param, 0.0)
            else:
                value = self.default_values.get(param, 0.0)

            features.append(value)

        return features

    def extract_signal_features(self, signal_data: np.ndarray) -> List[float]:
        """Extract and prepare FHR signal features"""

        # Remove invalid values (common in FHR: 0, negative values)
        valid_signal = signal_data[signal_data > 0]

        if len(valid_signal) == 0:
            print("Warning: No valid signal data found")
            return [0.0] * 5000  # Return zeros if no valid data

        # Interpolate to standard length (5000 samples = ~20 minutes at 4Hz)
        target_length = 5000

        if len(valid_signal) > target_length:
            # Downsample
            indices = np.linspace(0, len(valid_signal) - 1, target_length, dtype=int)
            resampled_signal = valid_signal[indices]
        else:
            # Upsample by interpolation
            old_indices = np.linspace(0, 1, len(valid_signal))
            new_indices = np.linspace(0, 1, target_length)
            resampled_signal = np.interp(new_indices, old_indices, valid_signal)

        # Basic preprocessing
        # Remove extreme outliers (< 50 bpm or > 200 bpm)
        resampled_signal = np.clip(resampled_signal, 50, 200)

        # Normalize (simple min-max scaling)
        min_val, max_val = resampled_signal.min(), resampled_signal.max()
        if max_val > min_val:
            normalized_signal = (resampled_signal - min_val) / (max_val - min_val)
        else:
            normalized_signal = np.zeros_like(resampled_signal)

        return normalized_signal.tolist()

    def validate_files(self, hea_path: Path, dat_path: Path) -> bool:
        """Validate that .hea and .dat files are properly formatted"""

        try:
            # Check file existence
            if not hea_path.exists() or not dat_path.exists():
                return False

            # Check file extensions
            if hea_path.suffix != '.hea' or dat_path.suffix != '.dat':
                return False

            # Check filename match
            if hea_path.stem != dat_path.stem:
                return False

            # Basic content validation
            with open(hea_path, 'r') as f:
                hea_content = f.read()
                if not hea_content.strip():
                    return False

            dat_size = dat_path.stat().st_size
            if dat_size == 0:
                return False

            return True

        except Exception:
            return False