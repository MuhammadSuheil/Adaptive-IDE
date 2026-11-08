"""English data labels with Indonesian explanatory text."""

from .data import DIAGNOSTICS, FAMILIES, LEVELS

LEVEL_NAMES = {level: level for level in LEVELS}
VARIANT_NAMES = {"good_only": "Good only", "good_acceptable": "Good + acceptable", "all_data": "All data"}
FAMILY_NAMES = {name: name for name in (*FAMILIES, "Signal quality")}
FEATURE_NAMES = {column: column for columns in FAMILIES.values() for column in columns}
FEATURE_NAMES.update({column: column for column in DIAGNOSTICS})
NOTE_ID = "Estimasi relatif sementara berbasis HRV dan mata; belum divalidasi dengan pengukuran beban kognitif yang independen."


def display_frame(frame):
    """Keep English source column names and values; format dataset IDs for display."""
    result = frame.copy()
    for column in ("variant", "Training dataset", "Dataset"):
        if column in result:
            result[column] = result[column].map(lambda value: VARIANT_NAMES.get(value, value) if isinstance(value, str) else value)
    return result
