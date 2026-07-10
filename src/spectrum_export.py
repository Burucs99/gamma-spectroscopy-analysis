import numpy as np
from scipy.optimize import curve_fit

class Spectrum:
    '''Stores one spectrum with bin edges, normalized counts, and the mean energy.

    Parameters
    -----------
    bin_edge_list : 1DArray
        The energy values corresponding to the bin edges in the spectrum.
    cps_list : 1DArray
        The counts per second in each bin of the spectrum normalized by the bin width in energy.
    '''

    def __init__(self, bin_edge_list, cps_list):
        self.bin_edge_list = bin_edge_list
        self.cps_list = cps_list
        E_mean = 0
        for i in range(len(cps_list)):
            E_curr = (bin_edge_list[i]+bin_edge_list[i+1])/2
            E_mean += E_curr*cps_list
        self.E_mean = E_mean/np.sum(cps_list)

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
                    print("ERROR: No calibration data in .mca file")
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
    ch_bins = np.arange(len(Data_list))
    ch_edges = np.arange(len(Data_list) + 1) - 0.5
    #Converts channel numbers to energy values
    E_bins = cfit_lin(ch_bins, a, b)
    E_edges = cfit_lin(ch_edges, a, b)
    #Calculates cps
    cps_data = np.array(Data_list)/REAL_TIME
    
    #This is the width of each bin in energy
    #TODO: Should make this more general if calibration is not linear
    dE = a
    #Divide the counts by the bin width so different calibrations
    #can be shown together
    cps_norm = cps_data/dE

    Spect = Spectrum(E_edges, cps_norm)
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

spectrum_from_mca('MCA/BG.mca')

