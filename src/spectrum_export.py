import numpy as np
from scipy.optimize import curve_fit

# A linear function for fitting
def cfit_lin(x,a,b):
    return a*x + b

def spectrum_from_mca(MCA_input):
    '''Imports data from an .mca file

    Parameters
    -----------
    MCA_input : string
        Name of the .mca file 
    
    Returns
    -------
    cps_norm: 1DArray
        The counts per second in each bin in the spectrum normalized by the bin width in energy
    E_bins: 1DArray
        The energy values corresponding to each bin center in the spectrum
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
    #Converts channel numbers to energy values
    E_bins = cfit_lin(ch_bins, a, b)
    #Calculates cps
    cps_data = np.array(Data_list)/REAL_TIME
    
    #This is the width of each bin in energy
    dE = a
    #Divide the counts by the bin width so different calibrations
    #can be shown together
    cps_norm = cps_data/dE

    return [cps_norm, E_bins]


def get_Spectra_from_mca(FileName_list_input):
    '''
    Creates spectra from multiple .mca files

    Parameters
    -----------
    FileName_list_input : list of strings
        List containing the names of the .mca files that are in the MCA folder. The .mca extension at the end should be omitted.
    
    Returns
    -------
    cps_norm_list: list of 1DArray
        List of the normalized cps values, that are the counts per second in each bin in the spectrum normalized by the bin width in energy
    E_bin_list: list of 1DArray
        List of the energy bin centers for each spectrum
    '''
    cps_norm_list = []
    E_bin_list = []
    #Looping through the input files
    for file in FileName_list_input:
        #Using the previous function to get the spectra
        cps_norm, E_bin = spectrum_from_mca(f'MCA/{file}.mca')
        #Storing the spectra in the predefined lists
        cps_norm_list.append(cps_norm)
        E_bin_list.append(E_bin)

    return [cps_norm_list, E_bin_list]
