import numpy as np
from scipy.optimize import curve_fit

#TODO: Error propagation for spectrum subtraction
class Spectrum:
    '''Stores one spectrum with bin edges, normalized counts, and the mean energy.

    Parameters
    -----------
    bin_edge_list : 1DArray
        The energy values corresponding to the bin edges in the spectrum.
    cps_list : 1DArray
        The counts per second in each bin of the spectrum normalized by the bin width in energy.
    '''

    def __init__(self, bin_edge_list, cps_list, cps_error_list):
        self.bin_edge_list = bin_edge_list
        self.cps_list = cps_list
        self.cps_error_list = cps_error_list
        E_mean = 0
        for i in range(len(cps_list)):
            E_curr = (bin_edge_list[i]+bin_edge_list[i+1])/2
            E_mean += E_curr*cps_list[i]
        self.E_mean = E_mean/np.sum(cps_list)

    def bin_widths(self):
        return np.diff(self.bin_edge_list)

    def to_common_energy_grid(self, target_edges):
        '''Rebins the spectrum onto a new energy grid while preserving bin integrals.

        Parameters
        ----------
        target_edges : 1DArray
            Monotonic energy bin edges for the new spectrum.

        Returns
        -------
        Spectrum
            A new Spectrum object on the requested energy grid.
        '''
        source_edges = np.asarray(self.bin_edge_list)
        target_edges = np.asarray(target_edges)
        source_widths = np.diff(source_edges)
        target_widths = np.diff(target_edges)

        source_cps_integral = np.asarray(self.cps_list) * source_widths
        source_error_integral = np.asanyarray(self.cps_error_list) * source_widths
        source_variance_integral = source_error_integral ** 2

        # Cumulative counts per second and cumulative variance at the source bin edges.
        # Interpolating the cumulative integrals preserves the total area of the histogram,
        # instead of interpolating the densities directly.
        cumulative_counts = np.concatenate(([0.0], np.cumsum(source_cps_integral)))
        cumulative_variance = np.concatenate(([0.0], np.cumsum(source_variance_integral)))

        # Interpolate the cumulative integrals at the new edges.
        rebinned_cumulative = np.interp(target_edges, source_edges, cumulative_counts)
        rebinned_cumulative_variance = np.interp(target_edges, source_edges, cumulative_variance)

        # Recover the rebinned bin integrals from successive cumulative values.
        rebinned_cps_integral = np.diff(rebinned_cumulative)
        # Variance of each rebinned bin comes from the change in cumulative variance.
        rebinned_error_integral = np.sqrt(np.clip(np.diff(rebinned_cumulative_variance), 0.0, None))

        # Convert the rebinned integrals back to densities.
        rebinned_density = rebinned_cps_integral / target_widths
        rebinned_error = rebinned_error_integral / target_widths
        return Spectrum(target_edges, rebinned_density, rebinned_error)

# A linear function for fitting
def cfit_lin(x,a,b):
    return a*x + b

def spectrum_from_mca(MCA_input):
    '''Imports data from an .mca file

    Parameters
    -----------
    MCA_input : string
        Name of the .mca file.
    
    Returns
    -------
    Spectrum: Spectrum
        A Spectrum object containing the energy bin edges and the normalized counts per second in each bin.
    '''

    #Data will go here
    Data_list = []
    
    #Booleans to help reading the files
    Data_start = False
    Calib_start = False
    REAL_TIME = 0
    
    #The given calibration value pairs
    Calib_ch = []
    Calib_E = []

    with open(MCA_input, "r") as file:

        for line in file:
            line_str = line.strip()

            #Gets the real time from the mca file
            if line_str.split()[0] == "REAL_TIME":
                REAL_TIME = int(line_str.split()[2])
            
            #Checks if calibration pairs are started
            if line_str == "<<CALIBRATION>>":
                Calib_start = True
                continue
            
            #Checks if the calibration pairs ended
            if line_str == "<<ROI>>":
                if Calib_start == False:
                    print(f'ERROR: No calibration data in {MCA_input} file')
                    return -1
                Calib_start = False
                
                continue

            #Checks where the DATA starts in the input file
            if line_str == "<<DATA>>":
                Data_start = True
                continue
            
            #Checks if DATA listing is ended
            if line_str == "<<END>>":
                Data_start = False
                continue
            

            #Puts the hit data into a list
            if Data_start == True:
                Data_list.append(int(line_str))

            #Imports the calibration pairs
            if Calib_start == True:
                #Useless row, ignored
                if line_str.split(" ")[0] == "LABEL":
                    continue
                #Useless row, ignored
                elif line_str[-1] == 'f':
                    continue
                #Takes the pair and puts them in the lists
                else: 
                    Calib_ch.append(float(line_str.split(" ")[0]))
                    Calib_E.append(float(line_str.split(" ")[1]))
    #Convert to np arrays            
    Calib_ch_np = np.array(Calib_ch)
    Calib_E_np = np.array(Calib_E)

    #Fits a line on the given pairs using curve_fit
    popt, _ = curve_fit(cfit_lin, Calib_ch_np, Calib_E_np)
    a, b = popt

    #Creates the channel number array
    ch_edges = np.arange(len(Data_list) + 1) - 0.5
    #Converts channel numbers to energy values
    E_edges = cfit_lin(ch_edges, a, b)

    #Calculates cps
    cps_data = np.array(Data_list)/REAL_TIME
    cps_err = np.sqrt(np.array(Data_list))/REAL_TIME
    #This is the width of each bin in energy
    #TODO: Should make this more general if calibration is not linear
    dE = a
    #Divide the counts by the bin width so different calibrations
    #can be shown together
    cps_norm = cps_data/dE
    cps_norm_err = cps_err/dE
    Spect = Spectrum(E_edges, cps_norm, cps_norm_err)
    #return [cps_norm, E_bins]
    return Spect

def get_Spectra_from_mca(FileName_list_input):
    '''
    Creates spectra from multiple .mca files.

    Parameters
    -----------
    FileName_list_input : list of strings
        List containing the names of the .mca files that are in the MCA folder. The .mca extension at the end should be omitted.
    
    Returns
    -------
    Spectrum_list: list of Spectrum
        List of Spectrum objects for each input file.
    '''
    Spectrum_list = []
    #Looping through the input files
    for file in FileName_list_input:
        #Using the previous function to get the spectra
        Spect_curr = spectrum_from_mca(f'MCA/{file}.mca')
        #Storing the spectra in the predefined lists
        Spectrum_list.append(Spect_curr)
    return Spectrum_list

#spectrum_from_mca('MCA/BG.mca')

def common_energy_edges(*spectra, step=None):
    '''Build a shared energy grid for spectra subtraction.

    Parameters
    ----------
    spectra : Spectrum
        One or more spectra to align.
    step : float, default = None
        Grid spacing in keV. If omitted, the smallest median bin width among the
        input spectra is used.

    Returns
    -------
    1D numpy array
        Monotonic energy bin edges spanning the common overlap region.
    '''
    if len(spectra) < 2:
        raise ValueError('At least two spectra are required')

    left_edge = max(np.min(spectrum.bin_edge_list) for spectrum in spectra)
    right_edge = min(np.max(spectrum.bin_edge_list) for spectrum in spectra)

    if right_edge <= left_edge:
        raise ValueError('Spectra do not overlap in energy')

    
    if step is None:
        # This is the minimum of the median bin width across spectra
        # It is chosen because if one of the spectra has fine details it mostly conserves those but it's not overly fine so nothing get's overly interpolated
        step = min(np.median(spectrum.bin_widths()) for spectrum in spectra)

    if step <= 0:
        raise ValueError('Grid spacing must be positive')
    
    # How many steps make up the histogram
    n_steps = int(np.floor((right_edge - left_edge) / step))
    # New edges -> Steps from the left edge
    edges = left_edge + step * np.arange(n_steps + 1)

    #Puts the right edge at the end if neccesary
    if edges[-1] < right_edge:
        edges = np.append(edges, right_edge)

    return edges


def subtract_spectra(spectrum, background, target_edges=None, step=None):
    '''Subtract one spectrum from another in energy space.

    Parameters
    ----------
    spectrum : Spectrum
        The foreground spectrum.
    background : Spectrum
        The background spectrum to subtract.
    target_edges : 1DArray, default = None
        Shared energy grid. If omitted, a common grid is created automatically.
    step : float, default = None
        Grid spacing used when target_edges is not supplied.

    Returns
    -------
    Spectrum
        The background-subtracted spectrum on the shared energy grid.
    '''
    if target_edges is None:
        target_edges = common_energy_edges(spectrum, background, step=step)

    #Rebin the spectrum and the BG to a commmon energy grid using the target_edges
    spectrum_rebinned = spectrum.to_common_energy_grid(target_edges)
    background_rebinned = background.to_common_energy_grid(target_edges)
    #Subtracting the cps on that new energy grid
    subtracted_density = spectrum_rebinned.cps_list - background_rebinned.cps_list

    subtracted_error = np.sqrt((spectrum_rebinned.cps_error_list)**2 + (background_rebinned.cps_error_list)**2)
    return Spectrum(target_edges, subtracted_density, subtracted_error)
