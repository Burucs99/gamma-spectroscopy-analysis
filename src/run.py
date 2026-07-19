from plotting import*
from spectrum_export import *

Input = ['Cs_Viz_R', 'BG_04_21_R']
Cs_P, BG = get_Spectra_from_mca(Input)
Cs_noBG = subtract_spectra(Cs_P, BG)
plot_Spectrum([Cs_P, Cs_noBG, BG], 'log', energy_min= 0, energy_max=1500, show_errors=True)