from plotting import*
from spectrum_export import *

Input = ['Cs_Papir_R', 'BG_04_21_R']
Cs_P, BG = get_Spectra_from_mca(Input)
Cs_noBG = subtract_spectra(Cs_P, BG)
plot_Spectrum([Cs_noBG, BG], 'linear', energy_min= 650, energy_max=670, show_uncertainty=True)