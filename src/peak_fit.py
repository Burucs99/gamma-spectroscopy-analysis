import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc
from spectrum_export import *
from plotting import *

def peak_plus_bkg_function(x,Lambda,x0,delta,A,beta,S, B, C):
    return Lambda*(np.exp(-((x-x0)/delta)**2) + A/2.*np.exp((x-x0)/beta)*erfc(delta/2./beta+(x-x0)/delta)+ S/2.*erfc((x-x0)/delta)) + B*(x-x0) + C

def peak_function(x,Lambda,x0,delta,A,beta,S, B, C):
    return Lambda*(np.exp(-((x-x0)/delta)**2) + A/2.*np.exp((x-x0)/beta)*erfc(delta/2./beta+(x-x0)/delta))

def bkg_function(x,Lambda,x0,delta,A,beta,S, B, C):
    return Lambda*(S/2.*erfc((x-x0)/delta)) + B*(x-x0) + C

def get_peak_area(Lambda,x0,delta,A,beta,S, B, C):
    return Lambda*(delta*np.sqrt(np.pi) + A*beta*np.exp(-(delta/2./beta)**2))

def get_gradient(Lambda,x0,delta,A,beta,S, B, C):
    return np.array([
        (delta*np.sqrt(np.pi) + A*beta*np.exp(-(delta/2./beta)**2)),
        0,
        Lambda*(np.sqrt(np.pi) - A*beta*np.exp(-(delta/2./beta)**2)* (delta/2./(beta**2))),
        Lambda*(beta*np.exp(-(delta/2./beta)**2)),
        Lambda*A*(1+1/2.*(delta/beta)**2)*np.exp(-(delta/2./beta)**2),
        0,
        0,
        0
    ])

#This function fits a single gamma peak with a fit function inspired by DOI: 10.1007/s10967-017-5589-z
def fit_single_peak(chData, sortedROIs, Real_time, plotOut=False, filenameOut='', wRoiSearch = 10):
    #Generates a list for all the data, that will be returned
    REAL_TIME_list = [Real_time]*len(sortedROIs)
    Net_area_list = []
    Total_area_list = [0]*len(sortedROIs)
    Error_list = []
    centre_list=[]
    popt_list=[]
    pcov_list=[]
    perr_list=[]

    # This array will hold the new, recommended ROI borders
    recOutput=[] 
    locChData=np.array(chData)
    # CAREFUL!! indexing is shifted by one for second derivative function with respect to chData
    # Numerical second derivative of the spectrum
    locSecondDerivative=locChData[:-2]+locChData[2:]-2*locChData[1:-1] 
    # Error of the numerical second derivative
    locSecondDerivativeErr=np.max([np.sqrt(locChData[:-2]+locChData[2:]+4*locChData[1:-1]), np.ones(len(locChData)-2)], axis=0)
    # Significance defined as the ratio of the second derivative and it's error
    locSecondDerivativeSignificance=np.abs(locSecondDerivative)/locSecondDerivativeErr
    wRoiCheck=5 # Number of checked channels around the border, must be odd
    # wRoiSearch=20  Number of channels that will be searched (max.) outward from original ROI
    secondDerivativeToleranceLevel = 2.5 # how many sigma deviation is acceptable in the second derivative value 

    for currROI in sortedROIs:
        # First find an extended fit range
        recommendedROI=[currROI[0],currROI[1]]

        # Looping through possible border positions and checking the significance condition at each point
        alternativeROIChecksLow=np.array([np.sum(locSecondDerivativeSignificance[currROI[0]-shift-1-int(wRoiCheck/2):currROI[0]-shift+int(wRoiCheck/2)]>secondDerivativeToleranceLevel) for shift in range(1,wRoiSearch+1)])
        conditionMetLow=np.where(alternativeROIChecksLow==0)[0] # acceptable ROI positions
        if (len(conditionMetLow)==0):
            print("No good alternative ROI found between "+str(currROI[0])+" and "+str(currROI[0]-wRoiSearch)+" !")
        else:
            firstGoodRoi=conditionMetLow[-1] # selecting the last acceptable position
            recommendedROI[0]=int(currROI[0]-1-firstGoodRoi) # Saving new ROI bound
        # Looping through possible border positions and checking the significance condition at each point
        alternativeROIChecksUp=np.array([np.sum(locSecondDerivativeSignificance[currROI[1]+shift-1-int(wRoiCheck/2):currROI[1]+shift+int(wRoiCheck/2)]>secondDerivativeToleranceLevel) for shift in range(1,wRoiSearch+1)])
        conditionMetUp=np.where(alternativeROIChecksUp==0)[0] # acceptable ROI positions
        if (len(conditionMetUp)==0):
            print("No good alternative ROI found between "+str(currROI[1])+" and "+str(currROI[1]+wRoiSearch)+" !")
        else:
            firstGoodRoi=conditionMetUp[-1] # selecting the last acceptable position
            recommendedROI[1]=int(currROI[1]+1+firstGoodRoi)

        recOutput.append(recommendedROI)

        # defining fit parameter bounds and some original parameter estimates
        fit_bounds=([0,recommendedROI[0],0.01, 0, 0.01, 0, -np.inf, 0],[np.inf,recommendedROI[1],50, 30, 100, 0.5, np.inf, np.inf])
        estimated_slope=(locChData[recommendedROI[1]]-locChData[recommendedROI[0]])/(recommendedROI[1]-recommendedROI[0])
        est_C=(locChData[recommendedROI[0]]+locChData[recommendedROI[1]])/2.

        
        #Perform fitting
        popt_tmp,pcov_tmp=curve_fit(peak_plus_bkg_function,np.linspace(recommendedROI[0],recommendedROI[1],recommendedROI[1]-recommendedROI[0]+1),
                                    locChData[recommendedROI[0]:recommendedROI[1]+1],
                                    sigma=np.max([np.sqrt(locChData[recommendedROI[0]:recommendedROI[1]+1]), np.ones(recommendedROI[1]+1-recommendedROI[0])], axis=0), 
                                    absolute_sigma=True,
                                    p0=[
                                        int(np.max(locChData[recommendedROI[0]:recommendedROI[1]+1])),
                                        int(recommendedROI[0]+np.where(locChData[recommendedROI[0]:recommendedROI[1]+1]==np.max(locChData[recommendedROI[0]:recommendedROI[1]+1]))[0][0]),
                                        3, 0.05, 1.5, 0.001, float(estimated_slope), float(np.max([est_C,0]))
                                    ], bounds=fit_bounds, maxfev=1e4)
        
        # Save the fit results
        centre_list.append(popt_tmp[1])
        Net_area_list.append(get_peak_area(*popt_tmp))
        popt_list.append(popt_tmp)
        pcov_list.append(pcov_tmp)
        perr_list.append(np.sqrt(np.diag(pcov_tmp)))
        grad_tmp=get_gradient(*popt_tmp)
        Error_list.append(np.sqrt(np.dot(grad_tmp,np.matmul(pcov_tmp,grad_tmp))))
        # If requested, save the fit to pdf files
        if plotOut:
            tmpXaxis=np.linspace(0,len(locChData)-1, len(locChData))
            plotFilter=(tmpXaxis<recommendedROI[1]+10)*(recommendedROI[0]-10<tmpXaxis)
            tmpXaxisFine=np.linspace(tmpXaxis[plotFilter][0],tmpXaxis[plotFilter][-1],1000)
            plt.errorbar(tmpXaxis[plotFilter], locChData[plotFilter], yerr=np.sqrt(locChData[plotFilter]), linestyle='', marker='o', markersize=3, capsize=2, color='C0', label="Measured spectrum")
            #plt.errorbar(tmpXaxis[plotFilter], locSecondDerivative[plotFilter[1:-1]], yerr=locSecondDerivativeErr[plotFilter[1:-1]], linestyle='', marker='o', markersize=3, capsize=2)
            plt.vlines(currROI,colors='red', ymin=0, ymax=np.max(locChData[plotFilter]), label="Input ROI")
            plt.vlines(recommendedROI,colors='green', ymin=0, ymax=np.max(locChData[plotFilter]), linestyles="dashed", label="New, fitted ROI")
            plt.xlabel('Channel number')
            plt.ylabel('Counts')
            plt.ylim(0,)
            plt.plot(tmpXaxisFine,peak_plus_bkg_function(tmpXaxisFine,*popt_tmp), label="Total fitted function")
            plt.plot(tmpXaxisFine,peak_function(tmpXaxisFine,*popt_tmp), label="Fitted peak component")
            plt.plot(tmpXaxisFine,bkg_function(tmpXaxisFine,*popt_tmp), label="Fitted background component")
            plt.grid()
            plt.legend(fontsize=6)
            plt.savefig('../plot_outputs/'+filenameOut+'_ROI_'+str(currROI[0])+'_'+str(currROI[1])+'_fit.pdf')
            plt.clf()

    popt_list=np.array(popt_list)
    pcov_list=np.array(pcov_list)
    perr_list=np.array(perr_list)
    name_list=[r"$\Gamma$",r"$x_0$",r"$\delta$",r"$A$",r"$\beta$","$S$", "$B$", "$C$"]

    # for each parameter plot the fitted values as a function of fit position
    for i in range(0,len(popt_list[0])):
        if i==1: 
            continue    
        
        plt.errorbar(popt_list[:,1],popt_list[:,i],yerr=perr_list[:,i], xerr=perr_list[:,1], linestyle="", marker="o", markersize=3, capsize=2)
        plt.xlabel(name_list[1])
        plt.ylabel(name_list[i])
        plt.ylim(np.min(popt_list[:,i])-np.abs(np.min(popt_list[:,i]))*0.2,np.max(popt_list[:,i])+np.abs(np.max(popt_list[:,i]))*0.2)
        plt.grid()
        plt.savefig('../plot_outputs/'+filenameOut+'_'+str(i)+'_variables.pdf')
        plt.clf()


    #print([REAL_TIME_list, recOutput, centre_list, Total_area_list, Net_area_list, Error_list])
    return [REAL_TIME_list, recOutput, centre_list, Total_area_list, Net_area_list, Error_list]

#TODO: Check if it works
#TODO: Apply to the formalism used here
def fit_single_peak_expand_to_next_peak(Spectrum_in, plotOut=True, plotLang = 'eng' , filenameOut='', wRoiSearch = 100, wRoiCheck = 5, secondDerivativeToleranceLevel = 2.5):
    print('Using custom ROI expansion script (CsP)')
    """
    Same as fit_single_peak but ROI borders are expanded outward until a next peak
    is encountered (detected via the second derivative significance). If no next
    peak is found within wRoiSearch channels the border is extended to array edge.
    """
    #Generates a list for all the data, that will be returned
    cpsData = Spectrum_in.cps_list
    cpsError = Spectrum_in.cps_error_list
    sortedROIs = Spectrum_in.ROI_list
    Net_area_list = []
    Total_area_list = [0]*len(sortedROIs)
    Error_list = []
    centre_list=[]
    popt_list=[]
    pcov_list=[]
    perr_list=[]

    # This array will hold the new, recommended ROI borders
    recOutput=[] 
    locChData=np.array(cpsData)
    # CAREFUL!! indexing is shifted by one for second derivative function with respect to chData
    # Numerical second derivative of the spectrum
    locSecondDerivative=locChData[:-2]+locChData[2:]-2*locChData[1:-1] 
    # Error of the numerical second derivative
    locSecondDerivativeErr=np.max([np.sqrt(locChData[:-2]+locChData[2:]+4*locChData[1:-1]), np.ones(len(locChData)-2)], axis=0)
    # Significance defined as the ratio of the second derivative and it's error
    locSecondDerivativeSignificance=np.abs(locSecondDerivative)/locSecondDerivativeErr
    wRoiCheck=5 # Number of checked channels around the border, must be odd
    # wRoiSearch=20  Number of channels that will be searched (max.) outward from original ROI
    #Lower because there are smaller bumps
    secondDerivativeToleranceLevel = 1.5 # how many sigma deviation is acceptable in the second derivative value 

    for currROI in sortedROIs:
        # First find an extended fit range
        recommendedROI=[currROI[0],currROI[1]]

        # Looping through possible border positions and checking the significance condition at each point
        alternativeROIChecksLow = np.array([
            np.sum(
                locSecondDerivativeSignificance[
                    max(0, currROI[0] - shift - 1 - int(wRoiCheck / 2)):
                    min(len(locSecondDerivativeSignificance), currROI[0] - shift + int(wRoiCheck / 2))
                ] > secondDerivativeToleranceLevel
            )
            for shift in range(1, wRoiSearch + 1)
        ])
        '''for shift in range(1, wRoiSearch+1):
            print(f'shift: {shift}')
            print(locSecondDerivativeSignificance[
                    max(0, currROI[0] - shift - 1 - int(wRoiCheck / 2)):
                    min(len(locSecondDerivativeSignificance), currROI[0] - shift + int(wRoiCheck / 2))
                ])'''
        print(f'{currROI}: {alternativeROIChecksLow}')

        #Going to the next peak, so the first value with 2 is needed after a series of 0s
        FoundEdgeOfPeak = False
        lastGoodRoi = 0
        for i in range(len(alternativeROIChecksLow)):
            if (alternativeROIChecksLow[i] == 0) and (not FoundEdgeOfPeak):
                for j in range(i, len(alternativeROIChecksLow)):
                    #print(f'j: {j}')
                    if alternativeROIChecksLow[j] != 0 or j == len(alternativeROIChecksLow) - 1:
                        FoundEdgeOfPeak = True
                        break
            if alternativeROIChecksLow[i] > 0 and FoundEdgeOfPeak:
                lastGoodRoi = i - 1
                break
        
        recommendedROI[0]=int(currROI[0]-1-lastGoodRoi) # Saving new ROI bound

        #Checking Upper border    
        # Looping through possible border positions and checking the significance condition at each point
        alternativeROIChecksUp = np.array([
            np.sum(
            locSecondDerivativeSignificance[
                max(0, currROI[1] + shift - 1 - int(wRoiCheck / 2)):
                min(len(locSecondDerivativeSignificance), currROI[1] + shift + int(wRoiCheck / 2))
            ] > secondDerivativeToleranceLevel
            )
            for shift in range(1, wRoiSearch + 1)
        ])

        #Going to the next peak, so the first value with 2 is needed after a series of 0s
        FoundEdgeOfPeak = False
        lastGoodRoi = 0
        for i in range(len(alternativeROIChecksUp)):
            if (alternativeROIChecksUp[i] == 0) and (not FoundEdgeOfPeak):
                for j in range(i, len(alternativeROIChecksUp)):
                    #print(f'j: {j}')
                    if alternativeROIChecksUp[j] != 0 or j == len(alternativeROIChecksUp) - 1:
                        FoundEdgeOfPeak = True
                        break
            if alternativeROIChecksUp[i] > 0 and FoundEdgeOfPeak:
                lastGoodRoi = i - 1
                break

        recommendedROI[1]=int(currROI[1]+1+lastGoodRoi)

        recOutput.append(recommendedROI)

        # defining fit parameter bounds and some original parameter estimates
        fit_bounds=([0,recommendedROI[0],0.01, 0, 0.01, 0, -np.inf, 0],[np.inf,recommendedROI[1],50, 30, 100, 0.5, np.inf, np.inf])
        estimated_slope=(locChData[recommendedROI[1]]-locChData[recommendedROI[0]])/(recommendedROI[1]-recommendedROI[0])
        est_C=(locChData[recommendedROI[0]]+locChData[recommendedROI[1]])/2.

        
        #Perform fitting
        popt_tmp,pcov_tmp=curve_fit(peak_plus_bkg_function,np.linspace(recommendedROI[0],recommendedROI[1],recommendedROI[1]-recommendedROI[0]+1),
                                    locChData[recommendedROI[0]:recommendedROI[1]+1],
                                    sigma=np.max([cpsError[recommendedROI[0]:recommendedROI[1]+1], np.ones(recommendedROI[1]+1-recommendedROI[0])], axis=0), 
                                    absolute_sigma=True,
                                    p0=[
                                        int(np.max(locChData[recommendedROI[0]:recommendedROI[1]+1])),
                                        int(recommendedROI[0]+np.where(locChData[recommendedROI[0]:recommendedROI[1]+1]==np.max(locChData[recommendedROI[0]:recommendedROI[1]+1]))[0][0]),
                                        3, 0.05, 1.5, 0.001, float(estimated_slope), float(np.max([est_C,0]))
                                    ], bounds=fit_bounds, maxfev=1e4)
        
        # Save the fit results
        centre_list.append(popt_tmp[1])
        Net_area_list.append(get_peak_area(*popt_tmp))
        popt_list.append(popt_tmp)
        pcov_list.append(pcov_tmp)
        perr_list.append(np.sqrt(np.diag(pcov_tmp)))
        grad_tmp=get_gradient(*popt_tmp)
        Error_list.append(np.sqrt(np.dot(grad_tmp,np.matmul(pcov_tmp,grad_tmp))))
        # If requested, save the fit to pdf files
        if plotOut:
            if plotLang == 'eng':
                tmpXaxis=np.linspace(0,len(locChData)-1, len(locChData))
                plotFilter=(tmpXaxis<recommendedROI[1]+10)*(recommendedROI[0]-10<tmpXaxis)
                tmpXaxisFine=np.linspace(tmpXaxis[plotFilter][0],tmpXaxis[plotFilter][-1],1000)
                plt.errorbar(tmpXaxis[plotFilter], locChData[plotFilter], yerr=np.sqrt(locChData[plotFilter]), linestyle='', marker='o', markersize=3, capsize=2, color='C0', label="Measured spectrum")
                #plt.errorbar(tmpXaxis[plotFilter], locSecondDerivative[plotFilter[1:-1]], yerr=locSecondDerivativeErr[plotFilter[1:-1]], linestyle='', marker='o', markersize=3, capsize=2)
                plt.vlines(currROI,colors='red', ymin=0, ymax=np.max(locChData[plotFilter]), label="Input ROI")
                plt.vlines(recommendedROI,colors='green', ymin=0, ymax=np.max(locChData[plotFilter]), linestyles="dashed", label="New, fitted ROI")
                plt.xlabel('Channel number')
                plt.ylabel('Counts')
                plt.ylim(0,)
                plt.plot(tmpXaxisFine,peak_plus_bkg_function(tmpXaxisFine,*popt_tmp), label="Total fitted function")
                plt.plot(tmpXaxisFine,peak_function(tmpXaxisFine,*popt_tmp), label="Fitted peak component")
                plt.plot(tmpXaxisFine,bkg_function(tmpXaxisFine,*popt_tmp), label="Fitted background component")
                plt.grid()
                plt.legend(fontsize=6)
                plt.savefig('Fit_Output/'+filenameOut+'_ROI_'+str(currROI[0])+'_'+str(currROI[1])+'_expand_fit.pdf')
                plt.clf()
            elif plotLang == 'hun':
                plt.rcParams.update({
                "font.family": "serif",
                "mathtext.fontset": "cm",   # use Computer Modern fonts
                "mathtext.rm": "serif",
                "font.size": 14
                })
                tmpXaxis=np.linspace(0,len(locChData)-1, len(locChData))
                plotFilter=(tmpXaxis<recommendedROI[1]+10)*(recommendedROI[0]-10<tmpXaxis)
                tmpXaxisFine=np.linspace(tmpXaxis[plotFilter][0],tmpXaxis[plotFilter][-1],1000)
                plt.errorbar(tmpXaxis[plotFilter], locChData[plotFilter], yerr=np.sqrt(locChData[plotFilter]), linestyle='', marker='o', markersize=3, capsize=2, color='C0', label="Mért spektrum")
                #plt.errorbar(tmpXaxis[plotFilter], locSecondDerivative[plotFilter[1:-1]], yerr=locSecondDerivativeErr[plotFilter[1:-1]], linestyle='', marker='o', markersize=3, capsize=2)
                plt.vlines(currROI,colors='red', ymin=0, ymax=np.max(locChData[plotFilter]), label="Input ROI")
                plt.vlines(recommendedROI,colors='green', ymin=0, ymax=np.max(locChData[plotFilter]), linestyles="dashed", label="Kitolt ROI")
                plt.xlabel('Csatornaszám')
                plt.ylabel('Beütések')
                plt.ylim(0,)
                plt.plot(tmpXaxisFine,peak_plus_bkg_function(tmpXaxisFine,*popt_tmp), label="Teljes illesztett függvény")
                plt.plot(tmpXaxisFine,peak_function(tmpXaxisFine,*popt_tmp), label="Illesztett csúcs komponens")
                plt.plot(tmpXaxisFine,bkg_function(tmpXaxisFine,*popt_tmp), label="Illesztett háttér")
                plt.grid()
                plt.legend(fontsize=10)
                plt.tight_layout()
                plt.savefig('Fit_Output/'+filenameOut+'_ROI_'+str(currROI[0])+'_'+str(currROI[1])+'_expand_fit.pdf')
                plt.clf()
            else:
                print('Not supported plot language')

    popt_list=np.array(popt_list)
    pcov_list=np.array(pcov_list)
    perr_list=np.array(perr_list)
    name_list=[r"$\Gamma$",r"$x_0$",r"$\delta$",r"$A$",r"$\beta$","$S$", "$B$", "$C$"]

    # for each parameter plot the fitted values as a function of fit position
    for i in range(0,len(popt_list[0])):
        if i==1: 
            continue    
        
        plt.errorbar(popt_list[:,1],popt_list[:,i],yerr=perr_list[:,i], xerr=perr_list[:,1], linestyle="", marker="o", markersize=3, capsize=2)
        plt.xlabel(name_list[1])
        plt.ylabel(name_list[i])
        plt.ylim(np.min(popt_list[:,i])-np.abs(np.min(popt_list[:,i]))*0.2,np.max(popt_list[:,i])+np.abs(np.max(popt_list[:,i]))*0.2)
        plt.grid()
        plt.savefig('Fit_Output/'+filenameOut+'_'+str(i)+'_variables.pdf')
        plt.clf()


    #print([REAL_TIME_list, recOutput, centre_list, Total_area_list, Net_area_list, Error_list])
    return [recOutput, centre_list, Total_area_list, Net_area_list, Error_list]
        

Spec = get_Spectra_from_mca(['Cs_Papir_R'])[0]
print(Spec.ROI_list)
fit_single_peak_expand_to_next_peak(Spec, plotOut=True, filenameOut='asd')