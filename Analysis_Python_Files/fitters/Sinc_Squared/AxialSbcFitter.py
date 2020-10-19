import numpy as np
import uncertainties.unumpy as unp
from fitters.Sinc_Squared import sinc_sq, arb_sinc_sq_sum
from fitters import rabi_equation

numSinc = 2

class AxialSbcFitter:
    time = 0.06 # in ms
    maxCarrierAmp = 0.8 # this is measureable. 
    
    def f(self, x, Offset, Amp1, Amp2, Spread, sidebandSigma, 
                   carrierCenter, Omega):
        """
        The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
        :return:
        """
        penalty = 10**10 * np.ones(len(x))
        ss_params = [Offset, Amp1, carrierCenter - Spread/2, sidebandSigma, Amp2, carrierCenter + Spread/2, sidebandSigma]
        
        for i in range(numSinc):
            if ss_params[3*i+1] < 0:
                # Penalize negative amplitude fits.
                #print('negative amp!')
                return penalty
            #if not (min(x) < ss_params[3*i+2] < max(x)):
                # penalize fit centers outside of the data range (assuming if you want to see these that you've
                # at least put the gaussian in the scan)
            #    print('fit center ' + str(ss_params[3*i+2]) + ' outside range!')
            #    return penalty
        if ss_params[0] < 0:
            # penalize negative offset
            #print('negative offset!')
            return penalty
        
        # rabi_equation expects delta in angular units
        return arb_sinc_sq_sum.f(x, *ss_params) + rabi_equation.f(self.time, Omega, (carrierCenter - x)*2*np.pi, self.maxCarrierAmp)
    
    def fitCharacter( self, params ):
        Offset, Amp1, Amp2, Spread, sidebandSigma, carrierCenter, Omega = params
        #params_ = [Offset, Amp1, Center - Spread/2, sidebandSigma, Amp2, Center, Sigma2, Amp3, Center + Spread/2, sidebandSigma]
        # for raman spectra, assuming fits are in order from left to right, i.e. first fit is lowest freq
        r = Amp2 / Amp1
        return r / ( 1 - r ) if not ( r >= 1 ) else np.inf

    def fitCharacterErr(self, params, errs):
        Offset, Amp1, Amp2, Spread, sidebandSigma, carrierCenter, Omega = params
        Offset_e, Amp1_e, Amp2_e, Spread_e, sidebandSigma_e, carrierCenter_e, Omega_e = errs
        r = Amp2 / Amp1
        errR = np.sqrt(Amp2_e**2/Amp1**2 + Amp1_e**2 * (r**2/Amp1**2) )
        return errR/(1-r)**2
        #r = params[4]/params[1]
        #errR = np.sqrt(errs[4]**2/params[1]**2 + errs[1]**2 * (r**2/params[1]**2) )
        #return errR/(1-r)**2

    def args(self):
        arglist = ["Offset", "Amp1", "Amp2", "Spread", "sidebandSigma", 
                   "carrierCenter", "Omega"]
        return arglist

    def getFitCharacterString(self):
        return r'$\bar{n}$'

    #def f_raw(self, x, *params):
    #    """
    #    The raw function call, performs no checks on valid parameters..
    #    :return:
    #    """
    #    return arb_sinc_sq_sum.f(x, *params)

    def f_unc(self, x, Offset, Amp1, Amp2, Spread, sidebandSigma, 
                   carrierCenter, Omega):
        """
        similar to the raw function call, but uses unp instead of np for uncertainties calculations.
        :return:
        """
        ss_params = [Offset, Amp1, carrierCenter - Spread/2, sidebandSigma, Amp2, carrierCenter + Spread/2, sidebandSigma]
        # rabi_equation expects delta in angular units
        return arb_sinc_sq_sum.f_unc(x, *ss_params) + rabi_equation.f_unc(self.time, Omega, (carrierCenter - x)*2*np.pi, self.maxCarrierAmp)
        #params = [Offset, Amp1, Center - Spread/2, sidebandSigma, Amp2, Center, Sigma2, Amp3, Center + Spread/2, sidebandSigma]
        #return arb_sinc_sq_sum.f_unc(x, *params)

    def guess(self, key, values):
        """
        Returns guess values for the parameters of this function class based on the input. Used for fitting using this class.
        """
        a = (max(values)-min(values))/10
        return [min(values),
                a, 
                a, 3,
                a,
                30, 70, 3]

    #def sbcGuess(self):
    #    return [[0, 0.3, 0.3, 20, 0.3, 115, 60, 20]]
    def sbcGuess(self):
         return [[0.1, 0.5, 0.1, 60, # spread in cyclic units
                  20, # sideband width in cyclic units
                  125, # in khz
                  11*2*np.pi # in angular units
                 ]]
    
