import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

class Spectrum:
    '''Stores one spectrum together with optional linear calibration metadata.

    Parameters
    -----------
    bin_edge_list : 1DArray
        The x-axis values corresponding to the bin edges in the spectrum.
    cps_list : 1DArray
        The counts per second in each bin of the spectrum.
    cps_error_list : 1DArray
        The uncertainty on cps_list.
    ROI_list : 1DArray
        Regions of interest on the same x-axis as bin_edge_list.
    calibration_slope : float, default = None
        Linear calibration slope used to convert channels to energy.
    calibration_offset : float, default = None
        Linear calibration offset used to convert channels to energy.
    calibrated : bool, default = False
        True if bin_edge_list and ROI_list already live in energy space.
    '''

    def __init__(self, bin_edge_list, cps_list, cps_error_list, ROI_list, calibration_slope=None, calibration_offset=None, calibrated=False):
        self.bin_edge_list = np.asarray(bin_edge_list, dtype=float)
        self.cps_list = np.asarray(cps_list, dtype=float)
        self.cps_error_list = np.asarray(cps_error_list, dtype=float)
        self.ROI_list = None if ROI_list is None else np.asarray(ROI_list, dtype=int)
        # Keep calibration metadata on the object so the same spectrum can be viewed
        # either in raw channel space or converted to energy on demand.
        self.calibration_slope = calibration_slope
        self.calibration_offset = calibration_offset
        self.calibrated = calibrated

        self.E_mean = self._compute_mean_axis_value()

    def _compute_mean_axis_value(self):
        if len(self.cps_list) == 0:
            return 0.0

        bin_centers = (self.bin_edge_list[:-1] + self.bin_edge_list[1:]) / 2
        if self.calibrated or self.calibration_slope is None or self.calibration_offset is None:
            axis_centers = bin_centers
        else:
            axis_centers = self.calibration_slope * bin_centers + self.calibration_offset

        total_weight = np.sum(self.cps_list)
        if total_weight == 0:
            return 0.0

        return np.sum(axis_centers * self.cps_list) / total_weight

    def bin_widths(self):
        return np.diff(self.bin_edge_list)

    def _roi_to_energy(self, roi_list):
        if roi_list is None:
            return None
        if self.calibration_slope is None or self.calibration_offset is None:
            raise ValueError('Calibration constants are required to convert ROIs to energy')

        # Convert each ROI boundary with the same linear calibration as the spectrum.
        roi_array = np.asarray(roi_list, dtype=float)
        energy_roi = np.empty_like(roi_array, dtype=float)
        energy_roi[:, 0] = self.calibration_slope * roi_array[:, 0] + self.calibration_offset
        energy_roi[:, 1] = self.calibration_slope * roi_array[:, 1] + self.calibration_offset
        return energy_roi[np.argsort(energy_roi[:, 0])]

    def to_energy(self):
        '''Return a calibrated copy of the spectrum in energy space.'''
        if self.calibration_slope is None or self.calibration_offset is None:
            raise ValueError('No calibration constants are stored on this spectrum')

        if self.calibrated:
            # Already in energy space, so only return a detached copy.
            return Spectrum(
                self.bin_edge_list.copy(),
                self.cps_list.copy(),
                self.cps_error_list.copy(),
                None if self.ROI_list is None else self.ROI_list.copy(),
                calibration_slope=self.calibration_slope,
                calibration_offset=self.calibration_offset,
                calibrated=True,
            )

        energy_edges = self.calibration_slope * self.bin_edge_list + self.calibration_offset
        source_widths = self.bin_widths()
        energy_widths = np.diff(energy_edges)

    # Re-express the bin contents as densities on the calibrated axis.
        source_integral = self.cps_list * source_widths
        source_error_integral = self.cps_error_list * source_widths

        energy_density = source_integral / energy_widths
        energy_error = source_error_integral / energy_widths

        return Spectrum(
            energy_edges,
            energy_density,
            energy_error,
            self._roi_to_energy(self.ROI_list),
            calibration_slope=self.calibration_slope,
            calibration_offset=self.calibration_offset,
            calibrated=True,
        )

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

        # Accumulate the contribution of each source bin into each target bin by overlap length.
        rebinned_cps_integral = np.zeros(len(target_widths))
        # For Poisson data, a fractional overlap contributes variance proportional to the overlap fraction.
        rebinned_error_integral = np.zeros(len(target_widths))

        for i in range(len(target_widths)):
            left_edge = target_edges[i]
            right_edge = target_edges[i + 1]

            source_start = np.searchsorted(source_edges, left_edge, side='right') - 1
            source_end = np.searchsorted(source_edges, right_edge, side='left')

            source_start = max(source_start, 0)
            source_end = min(source_end, len(source_widths))

            for j in range(source_start, source_end + 1):
                if j < 0 or j >= len(source_widths):
                    continue

                overlap_left = max(left_edge, source_edges[j])
                overlap_right = min(right_edge, source_edges[j + 1])
                overlap_width = overlap_right - overlap_left

                if overlap_width <= 0:
                    continue

                # A source bin only contributes by the fraction that lies inside the target bin.
                overlap_fraction = overlap_width / source_widths[j]
                rebinned_cps_integral[i] += source_cps_integral[j] * overlap_fraction
                rebinned_error_integral[i] += source_error_integral[j] ** 2 * overlap_fraction

            # Convert variance to standard deviation after summing all source-bin contributions.
        rebinned_error_integral = np.sqrt(rebinned_error_integral)

        # Convert the rebinned integrals back to densities.
        rebinned_density = rebinned_cps_integral / target_widths
        rebinned_error = rebinned_error_integral / target_widths
        return Spectrum(
            target_edges,
            rebinned_density,
            rebinned_error,
            None if self.ROI_list is None else self.ROI_list.copy(),
            calibration_slope=self.calibration_slope,
            calibration_offset=self.calibration_offset,
            calibrated=True,
        )

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
        A Spectrum object containing the channel bin edges, counts per second, and stored linear calibration constants.
    '''

    #Data will go here
    Data_list = []
    ROI_list_ch = []
    #Booleans to help reading the files
    Data_start = False
    Calib_start = False
    ROI_start = False
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
                ROI_start = True
                continue

            #Checks where the DATA starts in the input file
            if line_str == "<<DATA>>":
                ROI_start = False
                Data_start = True
                continue
            
            #Checks if DATA listing is ended
            if line_str == "<<END>>":
                Data_start = False
                continue
            
            if ROI_start == True:
                ROI = list(map(int, line_str.split(" ")))
                ROI_list_ch.append(ROI)

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
    ROI_list_ch_np = np.array(ROI_list_ch)

    #Fits a line on the given pairs using curve_fit
    popt, _ = curve_fit(cfit_lin, Calib_ch_np, Calib_E_np)
    a, b = popt

    #Creates the channel number array
    ch_edges = np.arange(len(Data_list) + 1) - 0.5
    #Calculates cps
    cps_data = np.array(Data_list)/REAL_TIME
    cps_err = np.sqrt(np.array(Data_list))/REAL_TIME
    # Store the raw channel histogram together with the calibration constants.
    ROI_list_ch_np = np.asarray(ROI_list_ch, dtype=int)
    sort_indeces = np.argsort(ROI_list_ch_np[:,0])
    ROI_list_ch_sorted = ROI_list_ch_np[sort_indeces]
    Spect = Spectrum(ch_edges, cps_data, cps_err, ROI_list_ch_sorted, calibration_slope=a, calibration_offset=b, calibrated=False)
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
    #Keeps the ROI list of the spectrum, throws away the BG
    return Spectrum(target_edges, subtracted_density, subtracted_error, spectrum.ROI_list)


def ROI_check(Spectrum, Target_Peak_list):
    ROI_list = Spectrum.ROI_list
    Peaks_Found_list = []
    for i in range(len(Target_Peak_list)):
        peak = Target_Peak_list[i]
        Peak_found = False
        for j in range(len(ROI_list)):
            #print(j)
            if ROI_list[j][0] < peak and ROI_list[j][1] > peak:
                Peak_found = True
            if ROI_list[j][0] > peak:
                break
        Peaks_Found_list.append(Peak_found)
    #print(Target_Peak_list)
    #print(ROI_list)
    #print(Peaks_Found_list)

    return Peaks_Found_list

def Check_Isotope_Peaks(IsotopeName):
    peak_df = pd.read_csv('src/Target_Peaks.csv')
    Isotope_Peaks = peak_df.loc[peak_df['Isotope'] == IsotopeName, 'Energy'].values

    shield_list = ['Ures', 'Papir', 'Viz', 'Fem']
    rows = []
    for i in range(len(shield_list)):
        mca_name = f'{IsotopeName}_{shield_list[i]}'
        # This check is performed in energy space, so convert the imported spectrum here.
        Spec = get_Spectra_from_mca([f'{mca_name}_R'])[0].to_energy()
        peak_checks = ROI_check(Spec, Isotope_Peaks)
        rows.append([mca_name, *peak_checks])

    result_df = pd.DataFrame(rows, columns=['MCA_Name', *Isotope_Peaks.tolist()])
    return result_df
    
Eu_df = Check_Isotope_Peaks('Eu')
Ba_df = Check_Isotope_Peaks('Ba')
Am_df = Check_Isotope_Peaks('Am')
Cs_df = Check_Isotope_Peaks('Cs')


Eu_df.to_csv('Eu_Peaks_Found.csv', index = None)
Ba_df.to_csv('Ba_Peaks_Found.csv', index = None)
Am_df.to_csv('Am_Peaks_Found.csv', index = None)
Cs_df.to_csv('Cs_Peaks_Found.csv', index = None)
