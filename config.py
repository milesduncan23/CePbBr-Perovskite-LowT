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

# Peak positions relative to B2 (the line that dominates at low power), measured from the 320 mW spectra of all
# five samples (every sample agrees within ~2 meV). Names follow the peak families of the group's RT paper
# (J. Mater. Chem. C, d6tc00159a, SI Fig. S7), whose A1-A4, B1, B2, C1 and C3 energies line up with these at 55 K
#                A1     A2     A3     A4    B1a    B1b    B'    B2    C1    C3
PEAK_OFFSETS = (-69.3, -58.2, -47.7, -41.1, -29.4, -23.2, -12.4, 0.0, 12.3, 20.4) # meV
REF_POS = 2.309 # eV, B2 position at 320 mW (each sample's first fit slides the whole template up to 6 meV to line up)
N_PEAKS = len(PEAK_OFFSETS)

# Detection limit: a fitted amplitude below this many noise units of its spectrum counts as not detected
# and is left out of the amplitude plots and slope fits (the fits themselves are kept as they are)
DETECTION_SNR = 3

# Extra points to leave out of the amplitude plots by hand: (sample, power index, peak index), all counted from 0
# The old (0, 0, 1) entry (Undoped, 10 mW, peak 2) is now caught by the detection limit
GARBAGE_PEAKS = []

# ==========================================
# Plot Settings
# ==========================================

ENERGY_RANGE = (2.23, 2.35) # eV, zoom on the active region in the spectrum plots
FIG_SIZE = (12, 7)          # Single-panel figures
DPI = 300