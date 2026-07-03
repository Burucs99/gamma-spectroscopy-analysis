from spectrum_export import get_Spectra_from_mca
import matplotlib.pyplot as plt
import numpy as np

def plot_spectra(FileName_list, yscale = 'linear', save_as = None, energy_min = 0, energy_max = 1e5):
    '''
    Plots multiple spectra on the same figure from .mca files

    Parameters
    ----------
    FileName_list: list of strings 
        List of the names of the .mca files placed in the MCA folder. The .mca extension at the end should be omitted.
    yscale: string, default = 'linear'
        Type of the yscale of the resulting plot (for valid values see https://matplotlib.org/stable/api/scale_api.html#builtin-scales).
    save_as: string, default = None
        The name of the saved output .png file. If not specified, the plot will not be saved. The .png should be omitted from the end. The output will be saved in the Plot_Output/ folder.
    energy_min: double, default = 0
        The minimum energy, where the plot should start.
    energy_max: double, default = 1e4
        The maximum energy, where the plot should end. 
    '''

    N = len(FileName_list)
    #Using get_Spectra_from_mca in spectrum_export.py
    cps_norm_list, E_bin_list = get_Spectra_from_mca(FileName_list)

    #Use Computer Modern fonts, which are the LaTeX default
    plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",  
    "mathtext.rm": "serif",
    "font.size": 12
    }) 

    plt.figure(figsize=(12,6))

    #Loop through every spectrum
    for i in range(N):
        #Create the energy mask
        mask_i = (E_bin_list[i] > energy_min) & (E_bin_list[i] < energy_max)
        #Apply the mask so only the values in the specified energy range are plotted
        cps_norm_mask_i = cps_norm_list[i][mask_i]
        E_bin_mask_i = E_bin_list[i][mask_i]

        #Plot the values
        plt.plot(E_bin_mask_i, cps_norm_mask_i, label = f'{FileName_list[i]}')
    
    #Apply the input yscale
    plt.yscale(yscale)
    plt.xlabel(r'$E \, (\mathrm{keV})$')
    plt.ylabel(r'$CPS/\Delta E \, (\mathrm{s^{-1}\cdot keV^{-1}})$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    #Save the file if required
    if save_as != None:
        plt.savefig(f'Plot_Output/{save_as}.png', dpi = 300)

    plt.show()

#Example usage
plot_spectra(['BG', 'Cs_Papir', 'Cs_Ures'], yscale='log', save_as='test', energy_min=5, energy_max=2000)

