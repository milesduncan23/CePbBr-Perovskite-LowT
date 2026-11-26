from pathlib import Path

# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "CSV_Data"
OUTPUT_DIR = BASE_DIR / "Output"

# ==========================================
# Samples
# ==========================================

FILES = ["ES073(undoped) @55K Fluence dependece",
         "ES074(10_ doped) @55K Fluence dependece",
         "ES075(20_ doped) @55K Fluence dependece",
         "ES076(30_ doped) @55K Fluence dependece",
         "ES077(40_ doped) @55K Fluence dependece"]

DOPE_LABELS = ["Undoped", "10% Doped", "20% Doped", "30% Doped", "40% Doped"] # Titles and output folder names
LI_LABELS = ["0% Li", "10% Li", "20% Li", "30% Li", "40% Li"]                 # Panel labels in the combined figures

SAMPLES = range(0, 5) # Which samples to run (index into FILES)

# ==========================================
# Peak Template
# ==========================================

# Guesses ----------------------------------------------------- ref
PEAK_OFFSETS = (-66.2, -55.7, -46.0, -39.1, -29.4, -19.4, -8.6, 0.0, 11.1) # meV
REF = 7 # Reference Peak Index (offset 0)
REF_POS = 2.3070
N_PEAKS = len(PEAK_OFFSETS)

# Points left out of the amplitude plots: (sample, power index, peak index), all counted from 0
GARBAGE_PEAKS = [(0, 0, 1)] # Undoped, 10 mW, peak 2

# ==========================================
# Plot Settings
# ==========================================

ENERGY_RANGE = (2.23, 2.35) # eV, zoom on the active region in the spectrum plots
FIG_SIZE = (12, 7)          # Single-panel figures
DPI = 300
