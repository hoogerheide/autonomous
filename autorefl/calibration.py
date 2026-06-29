"""Intensity calibration for AutoRefl instruments.

Reads intensity scan files from the IntensityDatabase (via the reduction
template's SLIT loader node) and populates the instrument's interpolation
table so that instrument.intensity(x) returns calibrated counts/s.
"""

import numpy as np

from reductus.dataflow.core import Template
from reductus.dataflow.calc import process_template

from refl_tools.reduction import map_loader_modules_by_intent, generate_config_by_intent
from refl_tools.util import FileInfo


def calibrate_intensity(instrument, template: dict | Template, file_infos: list[FileInfo]) -> None:
    """Populate instrument.s1_intens_calib / intens_calib from SLIT scan files.

    Targets the SLIT loader node in the template directly — no monitor
    normalisation is applied. The raw loaded ReflData gives counts/s vs s1
    which is exactly what instrument.intensity(x) needs.

    Args:
        instrument: ReflectometerBase subclass instance to update in-place.
        template: Reductus reduction template (dict or Template). Must contain
            a loader node with intent 'intensity' (SLIT scans).
        file_infos: List of FileInfo dicts for the intensity scan nexus files,
            as returned by IntensityDatabase.read_file_info() under 'highQ'.
    """
    if isinstance(template, dict):
        template = Template(**template)

    intents = map_loader_modules_by_intent(template)
    slit_indices = intents.get('intensity', [])
    if not slit_indices:
        raise ValueError("No intensity (SLIT) loader node found in reduction template")

    slit_idx = slit_indices[0]
    config = generate_config_by_intent(file_infos, template)

    results = process_template(template, config['local_config'], target=(slit_idx, 'output'))

    # process_template returns a list of ReflData, one per file entry
    if not results:
        raise ValueError("SLIT loader returned no data — check file_infos and template")

    s1_parts = []
    v_parts = []
    for rd in results:
        x = np.asarray(rd.x, ndmin=1)
        v = np.asarray(rd.v, ndmin=1)
        s1_parts.append(x)
        v_parts.append(v)

    s1_all = np.concatenate(s1_parts)
    v_all = np.concatenate(v_parts, axis=0)  # shape (N,) or (N, D)

    order = np.argsort(s1_all)
    instrument.s1_intens_calib = s1_all[order]
    instrument.intens_calib = v_all[order] if v_all.ndim == 1 else v_all[order, :]
