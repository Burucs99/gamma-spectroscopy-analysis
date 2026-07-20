from plotting import*
from spectrum_export import *

Input = ['Cs_Viz_R', 'BG_04_21_R']
Cs_P, BG = get_Spectra_from_mca(Input)

plot_Spectrum([Cs_P, BG], 'log', energy_min= 0, energy_max=1500, show_errors=True)