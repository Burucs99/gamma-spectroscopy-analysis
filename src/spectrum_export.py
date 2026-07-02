import numpy as np
from scipy.optimize import curve_fit

#A linear function for fitting
def cfit_lin(x,a,b):
    return a*x + b

def import_from_mca(MCA_input):
    '''Imports data from an .mca file

    Parameters
    -----------
    MCA_input : string
        Name of the .mca file 
    
    Returns
    -------
    cps_data: 1DArray
        The counts per second in each bin in the spectrum
    E_list: 1DArray
        The energy values corresponding to each bin in the spectrum
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
    #Convert tu np arrays            
    Calib_ch_np = np.array(Calib_ch)
    Calib_E_np = np.array(Calib_E)

    #Fits a line on the given pairs using curve_fit
    popt, _ = curve_fit(cfit_lin, Calib_ch_np, Calib_E_np)
    a, b = popt

    #Creates the channel number array
    ch_list = np.arange(len(Data_list))
    #Converts channel numbers to energy values
    E_list = cfit_lin(ch_list, a, b)
    #Calculates cps
    cps_data = np.array(Data_list)/REAL_TIME
    
    return [cps_data, E_list]

