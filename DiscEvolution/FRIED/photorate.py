import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import DiscEvolution.constants as cst
import matplotlib.colors as colors
import argparse

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

"""
Parent class that holds the grid and returns the photoevaporation rate
"""
class FRIEDgrid(object):
    def __init__(self):
        # Fixed parameters of the grid
        self._floor = 1e-10
        self._R_min = 1
        self._R_max = 400
        self._Mfrac_min = 3.2e-5
        self._Mfrac_max = 0.2
        
        # Load data from FRIED table (Haworth et al 2018)
        ## Data listed as M_star, UV, M_disc, Sigma_disc, R_disc, M_dot
        ## Values given in linear space apart from M_dot which is given as its base 10 logarithm
        data_dir = os.path.join(os.path.dirname(__file__))
        # Take M_star, UV, M_disc, Sigma_disc, R_disc to build parameter space
        self.grid_parameters = np.genfromtxt(os.path.join(data_dir, "friedgrid.dat"),skip_header=1,usecols=(0,1,2,3,4))
        # Import M_dot
        self.grid_rate = np.genfromtxt(os.path.join(data_dir, "friedgrid.dat"),skip_header=1,usecols=5)

        # Calculate mass within 400 AU and disc-to-star mass ratio and add to grid as columns 5/6
        M_400  = 2*np.pi*self.grid_parameters[:,3]*self.grid_parameters[:,4]*400*cst.AU**2/cst.Mjup
        M_frac = M_400*cst.Mjup/cst.Msun/self.grid_parameters[:,0]
        M_400  = np.reshape(M_400,(np.size(M_400),1))
        M_frac = np.reshape(M_frac,(np.size(M_frac),1))
        self.grid_parameters = np.hstack((self.grid_parameters,M_400))
        self.grid_parameters = np.hstack((self.grid_parameters,M_frac))

    def Sigma_min(self, R):
        return self._Mfrac_min * self._M_star*cst.Msun / (2*np.pi*R*400*cst.AU**2)
        
    def Sigma_max(self, R):
        return self._Mfrac_max * self._M_star*cst.Msun / (2*np.pi*R*400*cst.AU**2)

    def PE_rate(self, query_inputs, extrapolate=False):
        query_log = tuple(np.log10(query_inputs))           # Take logarithm of input values
        M_dot = self.M_dot_interp(query_log)                # Perform the interpolation; must be implemented in a subclass
        M_dot = np.power(10,M_dot)                          # Exponentiate
        M_dot[(query_inputs[1]<self._R_min)] = self._floor  # Fix values inside 1 AU (outside FRIED) to FRIED floor
        #M_dot[(query_inputs[1]>self._R_min)] = self._floor  # Fix values outside 400 AU (outside FRIED) to FRIED floor   ## Can probably do better extrapolating off of R=400 value using M_dot~R*Sigma
        return M_dot                                        # Return exponentiated mass rate

"""
Base class for 2D interpolation
All interpolators are 2D (no stellar mass or UV for computational speed) in log space
"""
class FRIED_2DInterpolator(FRIEDgrid):
    def __init__(self, M_star, UV, use_keys):
        super().__init__()
        
        # Select correct subgrid and build the interpolator
        self._M_star = M_star
        self._UV    = UV
        select_mass = (np.abs(self.grid_parameters[:,0] - M_star)<0.001) # Filter based on ones with the correct mass
        select_UV   = (np.abs(self.grid_parameters[:,1] - UV)<0.001)     # Filter based on ones with the correct UV
        select_MUV  = select_mass * select_UV

        if sum(select_mass)==0 and sum(select_UV)==0:
            raise NotImplementedError("This stellar mass and FUV field are not in the FRIED grid... currently not set up to pre-interpolate between in 4D")
        
        elif sum(select_mass)==0:
            # Pre-interpolate mass
            if 6 not in use_keys:
                raise NotImplementedError("This stellar mass ({}) is not in the FRIED grid... currently not set up to interpolate between unless using fractional mass".format(M_star))
            print("Pre-interpolating mass")
            grid_inputs_3D = self.grid_parameters[select_UV,:]    # Apply filter
            values_3D = self.grid_rate[select_UV]
            M_dot_interp_3D = interpolate.LinearNDInterpolator(grid_inputs_3D[:,(0,use_keys[0],use_keys[1])], values_3D)
            self.selected_inputs = grid_inputs_3D[:,(use_keys[0],use_keys[1])]
            self.selected_rate = M_dot_interp_3D(tuple((np.full_like(self.selected_inputs[:,0],M_star),self.selected_inputs[:,0],self.selected_inputs[:,1])))
            self.M_dot_interp = interpolate.LinearNDInterpolator(np.log10(self.selected_inputs),self.selected_rate) # Build interpolator on log of inputs

        elif sum(select_UV)==0:
            # Pre-interpolate UV
            print("Pre-interpolating UV")
            grid_inputs_3D = self.grid_parameters[select_mass,:]    # Apply filter
            values_3D = self.grid_rate[select_mass]
            M_dot_interp_3D = interpolate.LinearNDInterpolator(np.log10(grid_inputs_3D[:,(1,use_keys[0],use_keys[1])]), values_3D)
            self.selected_inputs = grid_inputs_3D[:,(use_keys[0],use_keys[1])]
            self.selected_rate = M_dot_interp_3D(tuple((np.log10(np.full_like(self.selected_inputs[:,0],UV)),np.log10(self.selected_inputs[:,0]),np.log10(self.selected_inputs[:,1]))))
            self.M_dot_interp = interpolate.LinearNDInterpolator(np.log10(self.selected_inputs),self.selected_rate) # Build interpolator on log of inputs            

        else:
            # Usual case
            grid_inputs_2D = self.grid_parameters[select_MUV,:] # Apply filter
            self.selected_inputs = grid_inputs_2D[:,(use_keys[0],use_keys[1])]
            self.selected_rate = self.grid_rate[select_MUV]
            self.M_dot_interp = interpolate.LinearNDInterpolator(np.log10(self.selected_inputs),self.selected_rate) # Build interpolator on log of inputs
            self.Sigma_inputs = grid_inputs_2D[:,(3,4)] # Select only the columns necessary - Sigma_disc, R_disc; just for plotting in tests

    def extrapolate_master(self, query_inputs, calc_rates):
        # The instances call PE_rate with the maximum of Sigma, Sigma_min, the extrapolation uses the true sigma to rescale
        # The instances call PE_rate with the minimum of Sigma, Sigma_max, no further calculations needed

        # At low surface densities and large enough radii, use scaling law M_dot \propto R \Sigma
        low_Sigma = ( query_inputs[0] < self.Sigma_min(query_inputs[1]) )
        ot_regime = low_Sigma * (calc_rates > self._floor)
        scaling_factor = (query_inputs[0]/self.Sigma_min(query_inputs[1]))
        calc_rates[ot_regime] *= scaling_factor[ot_regime]
        
        # At high surface densities, clip to top of grid
        envelope_regime = ( query_inputs[0] > self.Sigma_max(query_inputs[1]) ) * (query_inputs[1] > self._R_min) * (query_inputs[1] < self._R_max)

        return ot_regime, envelope_regime, calc_rates

"""
1st Order linear interpolators on different mass measures - either disc mass (M), surface density (S), extrapolated mass within 400 au (M400) or extrapolated fractional mass within 400 au (M400)
"""
class FRIED_2DS(FRIED_2DInterpolator):
    #Interpolates on surface density (S)
    def __init__(self, M_star, UV):
        super().__init__(M_star, UV, [3,4])
    #Extrapolation routine works here
    def extrapolate(self,query_inputs,calc_rates):
        return self.extrapolate_master(query_inputs,calc_rates)

class FRIED_2DM(FRIED_2DInterpolator):
    # Interpolates on mass (M)
    def __init__(self, M_star, UV):
        super().__init__(M_star, UV, [2,4])
    #Extrapolation routine doesn't work here
    def extrapolate(self,query_inputs,calc_rates):
        print("Extrapolation not valid when interpolating on mass")

class FRIED_2DM400(FRIED_2DInterpolator):
    # Interpolates on mass (M400)
    def __init__(self, M_star, UV):
        super().__init__(M_star, UV, [5,4])
    #Extrapolation routine doesn't work here
    def extrapolate(self,query_inputs,calc_rates):
        print("Extrapolation not valid when interpolating on mass")

class FRIED_2DfM400(FRIED_2DInterpolator):
    # Interpolates on fractional mass (fM400)
    def __init__(self, M_star, UV):
        super().__init__(M_star, UV, [6,4])
    #Extrapolation routine doesn't work here
    def extrapolate(self,query_inputs,calc_rates):
        print("Extrapolation not valid when interpolating on mass")
        
"""
2nd Order linear interpolators on different mass measures
"""
class FRIED_2DMS(FRIED_2DM):
    # Interpolates on mass (M) but is provided with surface density (S)
    def PE_rate(self, query_inputs,extrapolate=True):
        new_query = np.array(query_inputs) # New array to hold modified query
        # Clip densities to ones in grid for calculating rates
        if extrapolate:
            re_Sigma = np.minimum(query_inputs[0], self.Sigma_max(query_inputs[1]))
            re_Sigma = np.maximum(re_Sigma, self.Sigma_min(query_inputs[1]))
        else:
            re_Sigma = query_inputs[0]
        # Convert sigma to a disc mass (for 1/R profile) and replace in query
        Mass_calc = 2*np.pi * re_Sigma * (query_inputs[1]*cst.AU)**2 / (cst.Mjup)
        new_query[0] = Mass_calc
        # Calculate rates
        calc_rates = super().PE_rate(new_query)
        # Adjust calculated rates according to extrapolation prescription using actual density
        if extrapolate:
            _, _, calc_rates = self.extrapolate(query_inputs,calc_rates)
        return calc_rates
    #Extrapolation routine works here
    def extrapolate(self,query_inputs,calc_rates):
        return self.extrapolate_master(query_inputs,calc_rates)

class FRIED_2DM400S(FRIED_2DM400):
    # Interpolates on mass at 400 AU (M400) but is provided with surface density (S)
    def PE_rate(self, query_inputs,extrapolate=True):
        new_query = np.array(query_inputs) # New array to hold modified query
        # Clip densities to ones in grid for calculating rates
        if extrapolate:
            re_Sigma = np.minimum(query_inputs[0], self.Sigma_max(query_inputs[1]))
            re_Sigma = np.maximum(re_Sigma, self.Sigma_min(query_inputs[1]))
        else:
            re_Sigma = query_inputs[0]
        # Convert sigma to a disc mass at 400 AU (for 1/R profile) and replace in query
        Mass_400 = 2*np.pi * re_Sigma * (query_inputs[1]*cst.AU) * (400*cst.AU) / (cst.Mjup)
        new_query[0] = Mass_400 # Replace first query parameter with mass
        # Calculate rates
        calc_rates =  super().PE_rate(new_query)
        # Adjust calculated rates according to extrapolation prescription using actual density
        if extrapolate:
            _, _, calc_rates = self.extrapolate(query_inputs,calc_rates)
        return calc_rates
    #Extrapolation routine works here
    def extrapolate(self,query_inputs,calc_rates):
        return self.extrapolate_master(query_inputs,calc_rates)

class FRIED_2DfM400S(FRIED_2DfM400):
    # Interpolates on fractional mass at 400 AU (fM400) but is provided with surface density (S)
    def PE_rate(self, query_inputs,extrapolate=True):
        new_query = np.array(query_inputs) # New array to hold modified query
        # Clip densities to ones in grid for calculating rates
        if extrapolate:
            re_Sigma = np.minimum(query_inputs[0], self.Sigma_max(query_inputs[1],self._M_star))
            re_Sigma = np.maximum(re_Sigma, self.Sigma_min(query_inputs[1],self._M_star))
        else:
            re_Sigma = query_inputs[0]
        # Convert sigma to a disc mass at 400 AU (for 1/R profile) and replace in query
        fMass_400 = 2*np.pi * re_Sigma * (query_inputs[1]*cst.AU) * (400*cst.AU) / (cst.Msun * self._M_star)
        new_query[0] = fMass_400 # Replace first query parameter with mass
        # Calculate rates
        calc_rates =  super().PE_rate(new_query)
        # Adjust calculated rates according to extrapolation prescription using actual density
        if extrapolate:
            _, _, calc_rates = self.extrapolate(query_inputs,calc_rates)
        return calc_rates
    #Extrapolation routine works here
    def extrapolate(self,query_inputs,calc_rates):
        return self.extrapolate_master(query_inputs,calc_rates)

class FRIED_2DM400M(FRIED_2DM400):
    # Interpolates on mass at 400 AU (M400) but is provided with mass (M)
    def PE_rate(self, query_inputs,extrapolate=False):
        new_query = np.array(query_inputs) # New array to hold modified query
        # Convert to a disc mass at 400 AU (for 1/R profile) and replace in query
        Mass_400 = query_inputs[0] * (400 / query_inputs[1])
        new_query[0] = Mass_400 # Replace first query parameter with mass
        # Calculate rates
        calc_rates =  super().PE_rate(new_query)
        return calc_rates
    #Extrapolation routine works here
    def extrapolate(self,query_inputs,calc_rates):
        print("Extrapolation not valid when interpolating on mass")

"""
Functions for testing
"""
def D2_space(interp_type = 'M400S', extrapolate=True, UV=1000, M_star = 1.0, title=False, markers=False, ax=None, discs=[None]):
        # Function for plotting mass loss rates as function of R and Sigma

        # Setup interpolator
        if (interp_type == 'S'):
            photorate = FRIED_2DS(M_star,UV)
        if (interp_type == 'MS'):
            photorate = FRIED_2DMS(M_star,UV)
        elif (interp_type == 'M400S'):
            photorate = FRIED_2DM400S(M_star,UV)

        # Setup interpolation grid
        R = np.linspace(1,500,2000,endpoint=True)
        Sigma = np.logspace(-5,5,101)
        (R_interp, Sigma_interp) = np.meshgrid(R,Sigma)

        # Interpolate
        rates = photorate.PE_rate((Sigma_interp,R_interp), extrapolate=extrapolate)

        # Plot
        n_levels = 25#*(2-save) 
        pcm = ax.contourf(R_interp,Sigma_interp,rates,levels=np.logspace(-16,-4,n_levels),norm=colors.LogNorm(vmin=1e-16,vmax=1e-4,clip=True),extend='min')
        Sig_max = photorate.Sigma_max(R)
        Sig_min = photorate.Sigma_min(R)
        ax.plot(R,Sig_max,linestyle='--',color='red',label='$\Sigma_{max}$')
        ax.plot(R,Sig_min,linestyle='--',color='red',label='$\Sigma_{min}$')

        # Can show the actual points where the calculations are made    
        if markers:
            grid_inputs_2D = photorate.Sigma_inputs
            ax.plot(grid_inputs_2D[:,1],grid_inputs_2D[:,0],marker='x',color='black',linestyle='None')
            
        # Superpose a disc profile
        if not discs[0] is None:
            if len(discs)>3:
                Rmax = []
                Smax = []
                for disc in discs:
                    discrates = photorate.PE_rate((disc[1],disc[0]), extrapolate=extrapolate)
                    locmax = np.nanargmax(discrates)
                    Rmax.append(disc[0][locmax])
                    Smax.append(disc[1][locmax])
                ax.plot(Rmax, Smax, color='white', linestyle='-.')
            else:
                for disc in discs:
                    p, = ax.plot(disc[0], disc[1], color='white', linestyle='--')
                    discrates = photorate.PE_rate((disc[1],disc[0]), extrapolate=extrapolate)
                    locmax = np.nanargmax(discrates)
                    ax.plot(disc[0][locmax], disc[1][locmax], color=p.get_color(), linestyle='', marker='+')


        # Adorn plot
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$R~/\mathrm{AU}$',fontsize=14)
        ax.set_ylabel('$\Sigma~/\mathrm{g~cm}^{-2}$', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim([1,500])
        ax.set_ylim([1e-5,1e5])
        bar_label='Mass Loss Rate ($M_\odot~\mathrm{yr}^{-1}$)'
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.ax.tick_params(labelsize=12) 
        cbar.set_label(label=bar_label, fontsize=14)
    
        # Add a title, if desired
        if title:
            if (interp_type == 'S'):
                ax.title("Interpolation on $\Sigma$",fontsize=24)
            elif (interp_type == 'MS'):
                ax.title("Interpolation on $M(\Sigma)$",fontsize=24)
            elif (interp_type == 'M400S'):
                ax.title("Interpolation on $M_{400}(\Sigma)$",fontsize=24)

if __name__ == "__main__":
    # If run as main, create plot showing the interpolation as function of R and Sigma
    plt.rcParams['text.usetex'] = "True"
    plt.rcParams['font.family'] = "serif"
    cm = 1/2.54
    fig, ax = plt.subplots(1,1,figsize=(6*cm,6*cm))

    parser = argparse.ArgumentParser()
    parser.add_argument("--FUV", "-u", type=float, default=1000)
    parser.add_argument("--save", "-s", action='store_true')
    args = parser.parse_args()

    D2_space(interp_type='M400S', extrapolate=True, UV=args.FUV, markers=True, ax=ax)

    # Save and/or show the figure
    fig.tight_layout()
    if args.save:
        fig.savefig('Interpolation_'+'M400S'+'_'+str(UV)+'.png')
        fig.savefig('Interpolation_'+'M400S'+'_'+str(UV)+'.pdf')
    plt.show()
    
