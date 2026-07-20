from spectrum_export import get_Spectra_from_mca
import matplotlib.pyplot as plt
import numpy as np


def _as_energy_spectrum(spec):
    # Plotting is defined in energy space, so promote raw channel spectra lazily.
    if getattr(spec, 'calibrated', False):
        return spec
    return spec.to_energy()

def plot_Spectrum(Spectrum_list, yscale = 'linear', save_as = None, energy_min = 0, energy_max = 5e3, show_errors = True):
    '''
    Plots one or more spectra on the same figure from Spectrum objects. 

    Parameters
    ----------
    Spectra_list: list of Spectrum objects 
        A list containing one ore more Spectrum object(s) to be plotted
    yscale: string, default = 'linear' 
        Type of the yscale of the resulting plot (for valid values see https://matplotlib.org/stable/api/scale_api.html#builtin-scales).
    save_as: string, default = None
        The name of the saved output .png file. If not specified, the plot will not be saved. The .png should be omitted from the end. The output will be saved in the Plot_Output/ folder.
    energy_min: double, default = 0
        The minimum energy, where the plot should start.
    energy_max: double, default = 1e4
        The maximum energy, where the plot should end.
    show_errors: boolean, default = True
        Decides if errors should be shown or not. 
    '''
    N = len(Spectrum_list)

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

        # Plot the spectrum using the calibrated view when the input is still in channels.
        spec = _as_energy_spectrum(Spectrum_list[i])
        edges = spec.bin_edge_list
        y = spec.cps_list
        sigma = spec.cps_error_list

        #Plot the spectrum using filled bin edges and counts per bin
        plt.stairs(y, edges, label = f'{[i]}, mean = {spec.E_mean:.2f} keV')
        if show_errors:
            plt.fill_between(edges[:-1], y - sigma, y + sigma, step = 'post', alpha = 0.25)
    
    #Apply the input yscale
    plt.yscale(yscale)
    plt.xlim(energy_min, energy_max)
    plt.xlabel(r'$E \, (\mathrm{keV})$')
    plt.ylabel(r'$CPS/\Delta E \, (\mathrm{s^{-1}\cdot keV^{-1}})$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    #Save the file if required
    if save_as != None:
        plt.savefig(f'Plot_Output/{save_as}.png', dpi = 300)

    plt.show()

def plot_from_files(FileName_list, yscale = 'linear', save_as = None, energy_min = 0, energy_max = 5e3, show_errors = True):
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
    energy_max: double, default = 5e3
        The maximum energy, where the plot should end.
    show_errors: boolean, default = True
        Decides if errors should be shown or not
    '''

    N = len(FileName_list)
    #Using get_Spectra_from_mca in spectrum_export.py
    Spectrum_list = get_Spectra_from_mca(FileName_list)
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
        # Convert the imported raw MCA spectrum before plotting it.
        spec = _as_energy_spectrum(Spectrum_list[i])
        edges = spec.bin_edge_list
        y = spec.cps_list
        sigma = spec.cps_error_list

        #Plot the spectrum using filled bin edges and counts per bin
        plt.stairs(y, edges, label = f'{FileName_list[i]}, mean = {spec.E_mean:.2f} keV')
        if show_errors:
            plt.fill_between(edges[:-1], y - sigma, y + sigma, step = 'post', alpha = 0.25)

    #Apply the input yscale
    plt.yscale(yscale)
    plt.xlim(energy_min, energy_max)
    plt.xlabel(r'$E \, (\mathrm{keV})$')
    plt.ylabel(r'$CPS/\Delta E \, (\mathrm{s^{-1}\cdot keV^{-1}})$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    #Save the file if required
    if save_as != None:
        plt.savefig(f'Plot_Output/{save_as}.png', dpi = 300)

    plt.show()

def plot_isotope(Isotope_Name):
    '''Plots all the shieldings with this isotope on one plot.
    
    Parameters
    ----------
    Isotope_Name: string
        Name of the isotope whished to be plotted.
    '''
    shield_list = ['Ures', 'Papir', 'Viz', 'Fem']
    isotope_shield_list = shield_list
    for i in range(len(isotope_shield_list)):
        isotope_shield_list[i] = f'{Isotope_Name}_{isotope_shield_list[i]}_R'
    plot_from_files(isotope_shield_list, yscale='log', save_as=f'{Isotope_Name}_All_Shielding', energy_min=0, energy_max=2000)
        

