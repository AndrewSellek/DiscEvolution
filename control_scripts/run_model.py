# run_model.py
#
# Author: R. Booth
# Date: 4 - Jun - 2018
#
# Run a disc evolution model with transport and absorption / desorption but
# no other chemical reactions. 
###############################################################################

import os
import json
import numpy as np
from DiscEvolution.constants import Msun, AU, yr, Mjup
from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar, PhotoStar
from DiscEvolution.eos  import IrradiatedEOS, LocallyIsothermalEOS, TanhAlphaEOS, ExternalHeatEOS
from DiscEvolution.dust import DustGrowthTwoPop
from DiscEvolution.opacity import Tazzari2016
from DiscEvolution.viscous_evolution import ViscousEvolution, ViscousEvolutionFV
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.dust import SingleFluidDrift
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.driver import DiscEvolutionDriver
from DiscEvolution.io import Event_Controller, DiscReader
from DiscEvolution.disc_utils import mkdir_p
from DiscEvolution.internal_photo import EUVDiscAlexander, XrayDiscOwen, XrayDiscPicogna
from DiscEvolution.history import History
import DiscEvolution.photoevaporation as photoevaporation

from DiscEvolution.chemistry import (
    ChemicalAbund, MolecularIceAbund,
    SimpleCNOAtomAbund, SimpleCNOMolAbund, SimpleH2OAtomAbund, SimpleH2OMolAbund, SimpleAtomAbund, SimpleMolAbund,
    EquilibriumCNOChemOberg, TimeDepCNOChemOberg,
    EquilibriumChemMINDS,
    EquilibriumCNOChemMadhu,
    EquilibriumH2OChemKalyaan,
)

###############################################################################
# Global Constants
###############################################################################

DefaultModel = "DiscConfig_default.json"

def LBP_profile(R,R_C,Sigma_C):
    # For profile fitting
    x = R/R_C
    return np.log(Sigma_C) - np.log(x)-x

###############################################################################
# Setup Functions
###############################################################################

#
def setup_wrapper(model, restart, output=True):
    # Setup basics
    disc = setup_disc(model)
    if model['disc']['d2g'] > 0:
        dust = True
        d_thresh = model['dust']['radii_thresholds']
    else:
        dust = False
        d_thresh = None
    
    history = History(dust, d_thresh, disc.chem)

    # Setup model
    if restart:
        disc, history, time, photo_type, R_hole = restart_model(model, disc, history, restart)       
        driver = setup_model(model, disc, history, time, internal_photo_type=photo_type, R_hole=R_hole)
    else:
        driver = setup_model(model, disc, history)

    # Setup outputs
    if output:
        output_name, io_control, output_times = setup_output(model)
        plot_name = model['output']['plot_name']
    else:
        output_name, io_control, plot_name = None, None, None

    # Truncate disc at base of wind
    if driver.photoevaporation_external and not restart and driver.photoevaporation_external._tshield<0:
        print("Truncate initial disc")
        if (isinstance(driver.photoevaporation_external,photoevaporation.FRIEDExternalEvaporationMS)):
            driver.photoevaporation_external.optically_thin_weighting(disc)
            optically_thin = (disc.R > driver.photoevaporation_external._Rot)
        else:
            initial_trunk = photoevaporation.FRIEDExternalEvaporationMS(disc)
            initial_trunk.optically_thin_weighting(disc)
            optically_thin = (disc.R > initial_trunk._Rot)

        print("Truncating initial disc at", driver.photoevaporation_external._Rot)
        print("({} cells)".format(np.sum(optically_thin)))
        disc._Sigma[optically_thin] = 0.
        try:
            disc._eps[:,optically_thin] = 0.
        except AttributeError:
            pass
        if disc.chem:
            for spec in disc.chem.gas.species:
                disc.chem.gas[spec][optically_thin] = 0.
            for spec in disc.chem.ice.species:
                disc.chem.ice[spec][optically_thin] = 0.
    
    Dt_nv = np.zeros_like(disc.R)
    if driver.photoevaporation_external:
        # Perform estimate of evolution for non-viscous case
        (_, _, M_cum, Dt_nv) = driver.photoevaporation_external.get_timescale(disc)

    return disc, driver, output_name, io_control, plot_name, Dt_nv

##
def setup_disc(model):
    '''Create disc object from initial conditions'''

    # Setup the grid
    p = model['grid']
    grid = Grid(p['R0'], p['R1'], p['N'], spacing=p['spacing'])

    # Setup the star with photoionizing luminosity if provided and non-zero
    p = model['star']
    try:
        if model['x-ray']['L_X'] > 0:
            star = PhotoStar(LX=model['x-ray']['L_X'], Phi=0, M=p['mass'], R=p['radius'], T_eff=p['T_eff'])
        elif model['euv']['Phi'] > 0:
            star = PhotoStar(LX=0, Phi=model['euv']['Phi'], M=p['mass'], R=p['radius'], T_eff=p['T_eff'])
        else:
            star = SimpleStar(M=p['mass'], R=p['radius'], T_eff=p['T_eff'])
    except KeyError:
        star = SimpleStar(M=p['mass'], R=p['radius'], T_eff=p['T_eff'])

    # Setup the external FUV irradiation
    try:
        p = model['fuv']
        FUV_field = p['fuv_field']
    except KeyError:
        p = model['uv']
        FUV_field = p['uv_field']
    
    # Setup the equation of state
    p = model['eos']
    try:
        mu = p['mu']
    except KeyError:
        mu = 2.4
    try:
        R_alpha = model['disc']['Ra']
    except KeyError:
        try:
            R_alpha = model['disc']['RDZ']
        except KeyError:
            R_alpha = None
    if R_alpha:
        assert len(model['disc']['alpha']) - len(R_alpha) == 1, "Need to specify one fewer radius than alpha value."
    try:
        if model['perturbation']['Type']!="None":
            perturbation = True
            ptbn_kwargs = model['perturbation']
            ptbn_kwargs['Star Mass'] = model['star']['mass']
        else:
            perturbation = False
    except:
        perturbation = False
    if p['type'] == 'irradiated':
        assert p['opacity'] == 'Tazzari2016'
        kappa = Tazzari2016()
        eos = IrradiatedEOS(star, model['disc']['alpha'], kappa=kappa, mu=mu)
    elif p['type'] == 'external':
        try:
            T_ext = model['fuv']['T_ext']
        except KeyError:
            T_ext = 39000
        if perturbation:
            eos = ExternalHeatEOS(star, p['h0'], p['q'], model['disc']['alpha'], mu=mu, G_0=FUV_field, T_ext=T_ext, ptbn_kwargs=ptbn_kwargs)
        else:
            eos = ExternalHeatEOS(star, p['h0'], p['q'], model['disc']['alpha'], mu=mu, G_0=FUV_field, T_ext=T_ext)
    elif p['type'] == 'iso':
        if R_alpha:
            eos = TanhAlphaEOS(star, p['h0'], p['q'], 
                                   model['disc']['alpha'], mu=mu, R_alpha=R_alpha)
        elif perturbation:
            eos = LocallyIsothermalEOS(star, p['h0'], p['q'], 
                                   model['disc']['alpha'], mu=mu, ptbn_kwargs=ptbn_kwargs)        
        else:
            eos = LocallyIsothermalEOS(star, p['h0'], p['q'], 
                                   model['disc']['alpha'], mu=mu)
    else:
        raise ValueError("Error: eos::type not recognised")
    eos.set_grid(grid)

    # Setup the physical part of the disc
    p = model['disc']
    if (('profile' in model['disc']) == False):
        Sigma = np.exp(-grid.Rc / p['Rc']) / (grid.Rc) # Catch missing profile by assuming Lynden-Bell & Pringle, gamma=1
    elif model['disc']['profile'] == 'LBP':
        try:
            gamma_visc = model['disc']['gamma']
        except:
            gamma_visc = 1
        Sigma = np.exp(-(grid.Rc / p['Rc'])**(2-gamma_visc)) / (grid.Rc**gamma_visc) # Lynden-Bell & Pringle
    else:
        try:
            gamma_visc = model['disc']['gamma']
        except:
            gamma_visc = 1.5 + 2 * model['eos']['q'] # Set gamma to steady state
        Sigma = 1.0 / (grid.Rc**gamma_visc)          # R^-gamma Power Law
    if perturbation:
        if ptbn_kwargs["Initial"]:
            Sigma = eos.update(0, Sigma, star=star)       # Divide out perturbation profile if required
    else:
        eos.update(0, Sigma)
    Sigma *= p['mass'] / np.trapz(Sigma, np.pi*grid.Rc**2)          # Normalise to correct mass
    if (p['unit']=='jup'):                                          # Disc mass given in Jupiter masses
        Sigma *= Mjup / AU**2
    elif (p['unit']=='sol') or (p['unit']=='sun'):                  # Disc mass given in Solar masses
        Sigma *= Msun / AU**2
    else:
        raise NotImplementedError

    try:
        feedback = model['disc']['feedback']
    except KeyError:
        feedback = True

    # If non-zero dust, set up a two population model, else use a simple accretion disc
    if model['disc']['d2g'] > 0:
        # If model dust parameters not specified, resort to default
        try:
            disc = DustGrowthTwoPop(grid, star, eos, p['d2g'], Sigma=Sigma, FUV=FUV_field,
                    rho_s=model['dust']['density'], Sc=model['disc']['Schmidt'], feedback=feedback, uf_0=model['dust']['dust_frag_v'], uf_ice=model['dust']['ice_frag_v'], f_grow=model['dust']['f_grow'], distribution_slope=model['dust']['p'])
        except:
            disc = DustGrowthTwoPop(grid, star, eos, p['d2g'], Sigma=Sigma, FUV=FUV_field, Sc=model['disc']['Schmidt'], feedback=feedback, uf_0=model['dust']['dust_frag_v'], uf_ice=model['dust']['ice_frag_v'])
    else:
        disc = AccretionDisc(grid, star, eos, Sigma=Sigma, FUV=FUV_field)

    # Setup the chemical part of the disc
    try:
        chem = model['chemistry']
    except KeyError:
        chem = False
    disc.chem = None
    if chem:
        if model['chemistry']["on"]:
            if model['chemistry']['type'] == 'krome':
                raise NotImplementedError("Haven't configured krome chemistry in this version")
            else:
                disc.chem = setup_init_abund_simple(model, disc)
                disc.update_ices(disc.chem.ice)

    return disc

###
def setup_init_abund_simple(model, disc):
    # Define abundances
    if model['chemistry']['type']=="Kalyaan":
        X_atom = SimpleH2OAtomAbund(model['grid']['N'])
        X_atom.set_Kalyaan_abundances()
    elif model['chemistry']['type']=="MINDS":
        X_atom = SimpleAtomAbund(model['grid']['N'])
        X_atom.set_adopted_abundances()
    else:
        X_atom = SimpleCNOAtomAbund(model['grid']['N'])
        X_atom.set_solar_abundances()    

    # Put into a chemical model
    chemistry = get_simple_chemistry_model(model)

    # Iterate adsorption/desorption equilibrium as the ice fraction changes the dust-to-gas ratio
    # Note that mu also changes the disc temperature, but not worrying about this for now
    for i in range(10):
        chem = chemistry.equilibrium_chem(disc.T,
                                          disc.midplane_gas_density,
                                          disc.dust_frac.sum(0),
                                          disc.dust_frac[0]/disc.dust_frac.sum(0),
                                          disc.R,
                                          disc.Sigma_G,
                                          X_atom)
        disc.initialize_dust_density(chem.ice.total_abund)
    print("Resulting gas phase mu: {}-{}".format(min(chem.gas.mu()),max(chem.gas.mu())))
    print("Resulting ice phase mu: {}-{}".format(min(chem.ice.mu()),max(chem.ice.mu())))
        
    return chem

####
def get_simple_chemistry_model(model):
    chem_type = model['chemistry']['type']

    grain_size = 1e-5
    try:
        grain_size = model['chemistry']['fixed_grain_size']
    except KeyError:
        pass

    # Non-thermal chemistry
    nonThermal = False
    nonThermal_dict = {}
    if 'CR_desorb' in model['chemistry'].keys() and model['chemistry']['CR_desorb']:
        nonThermal = True
        nonThermal_dict['CR_desorb']    = model['chemistry']['CR_desorb']
        if 'CR_rate' in model['chemistry'].keys():
            nonThermal_dict['CR_rate']  = model['chemistry']['CR_rate']
    if 'UV_desorb' in model['chemistry'].keys() and model['chemistry']['UV_desorb']:
        nonThermal = True
        nonThermal_dict['UV_desorb']    = model['chemistry']['UV_desorb']
        nonThermal_dict['G0']           = model['fuv']['fuv_field']
        nonThermal_dict['AV_rad']       = (model['fuv']['photoevaporation'] == "None")
    if 'X_desorb' in model['chemistry'].keys() and model['chemistry']['X_desorb']:
        nonThermal = True
        nonThermal_dict['X_desorb']     = model['chemistry']['X_desorb']
        nonThermal_dict['Mstar']        = model['star']['mass']
    
    if chem_type == 'TimeDep':
        chemistry = TimeDepCNOChemOberg(a=grain_size)
    elif chem_type == 'Madhu':
        chemistry = EquilibriumCNOChemMadhu(fix_ratios=False,  a=grain_size, nonThermal=nonThermal, nonThermal_dict=nonThermal_dict)
    elif chem_type == 'Oberg':
        chemistry = EquilibriumCNOChemOberg(fix_ratios=False,  a=grain_size, nonThermal=nonThermal, nonThermal_dict=nonThermal_dict)
    elif chem_type == 'Kalyaan':
        chemistry = EquilibriumH2OChemKalyaan(fix_ratios=True, a=grain_size, nonThermal=nonThermal, nonThermal_dict=nonThermal_dict)
    elif chem_type == 'NoReact':
        chemistry = EquilibriumCNOChemOberg(fix_ratios=True,   a=grain_size, nonThermal=nonThermal, nonThermal_dict=nonThermal_dict)
    elif chem_type == 'MINDS':
        chemistry = EquilibriumChemMINDS(fix_ratios=True,      a=grain_size, nonThermal=nonThermal, nonThermal_dict=nonThermal_dict)
    else:
        raise ValueError("Unknown chemical model type")

    return chemistry

##
def restart_model(model, disc, history, snap_number):
    # Resetup model
    out = model['output']
    reader = DiscReader(out['directory'], out['base'], out['format'])

    snap = reader[snap_number]

    # Surface density
    disc.Sigma[:] = snap.Sigma

    # Dust
    try:
        disc.dust_frac[:] = snap.dust_frac
        disc.grain_size[:] = snap.grain_size
    except:
        pass

    # Chem
    try:
        chem = snap.chem
        disc.chem.gas.data[:] = chem.gas.data
        disc.chem.ice.data[:] = chem.ice.data
    except AttributeError as e:
        if model['chemistry']['on']:
            raise e

    time = snap.time * yr       # Convert real time (years) to code time

    disc.update(0)

    # Revise and write history
    infile = model['output']['directory']+"/"+"discproperties.dat"
    history.restart(infile, snap_number)

    # Find current location of hole, if appropriate
    try:
        R_hole = history._Rh[-1]
        if np.isnan(R_hole):
            R_hole = None
        else:
            print("Hole is at: {} AU".format(R_hole))
    except:
        R_hole = None

    return disc, history, time, snap.photo_type, R_hole     # Return disc objects, history, time (code units), input data and internal photoevaporation type

##
def setup_model(model, disc, history, start_time=0, internal_photo_type="Primordial", R_hole=None):
    '''Setup the physics of the model'''
    
    gas       = None
    dust      = None
    diffuse   = None
    chemistry = None

    if model['transport']['gas']:
        try:
            gas = ViscousEvolution(boundary=model['grid']['outer_bound'], in_bound=model['grid']['inner_bound'])
        except KeyError:
            print("Default boundaries")
            gas = ViscousEvolution(boundary='Mdot_out')
        
    if model['transport']['diffusion']:
        diffuse = TracerDiffusion(Sc=model['disc']['Schmidt'])
    if model['transport']['radial drift']:
        dust = SingleFluidDrift(diffuse)
        diffuse = None

    try:
        chem = model['chemistry']
    except KeyError:
        chem = False
    if chem:    
        if model['chemistry']['on']:
            if  model['chemistry']['type'] == 'krome':
                raise NotImplementedError("Haven't configured krome chemistry in this version")
            else:
                chemistry = setup_simple_chem(model)

    # Inititate the correct external photoevaporation routine
    # FRIED should be considered default 
    try:
        p = model['fuv']
    except KeyError:
        p = model['uv']
    try:
        tshield = p['t_shield']
    except:
        tshield = 0
    try:
        evolvedDust = p['dust']=="evolved"
    except:
        evolvedDust = True
    if start_time>0 and p['photoevaporation']!="None":
        _, Mcum_gas  = history.mass
        _, Mcum_dust = history.mass_dust
        Mcum_gas  = Mcum_gas[-1]
        Mcum_dust = Mcum_dust[-1]
    elif p['photoevaporation']!="None":
        Mcum_gas  = 0.0
        Mcum_dust = 0.0
    if (p['photoevaporation'] == "Constant"):
        photoevap = photoevaporation.FixedExternalEvaporation(disc, 1e-9, tshield)
    elif (p['photoevaporation'] == "FRIED" and disc.FUV>0):
        # Using 2DMS at 400 au
        photoevap = photoevaporation.FRIEDExternalEvaporationMS(disc, tshield=tshield, Mcum_gas = Mcum_gas, Mcum_dust = Mcum_dust, evolvedDust = evolvedDust)
    elif (p['photoevaporation'] == "FRIEDv2" and disc.FUV>0):
        # Using 2DMS at 400 au
        photoevap = photoevaporation.FRIEDExternalEvaporationMS(disc, tshield=tshield, Mcum_gas = Mcum_gas, Mcum_dust = Mcum_dust, evolvedDust = evolvedDust, versionFRIED=2)
    elif (p['photoevaporation'] == "Integrated"):
        # Using integrated M(<R), extrapolated to M400
        photoevap = photoevaporation.FRIEDExternalEvaporationM(disc, tshield=tshield, Mcum_gas = Mcum_gas, Mcum_dust = Mcum_dust)
    elif (disc.FUV<=0):
        photoevap = None
    elif (p['photoevaporation'] == "None"):
        photoevap = None
    else:
        print("Photoevaporation Mode Unrecognised: Default to 'None'")
        photoevap = None

    # Add internal photoevaporation
    try:
        if model['x-ray']['L_X'] > 0:
            try:
                photomodel = model['x-ray']['model']
            except KeyError:
                photomodel = 'Picogna'
            InnerHole = internal_photo_type.startswith('InnerHole')
            if InnerHole:
                if photomodel=='Picogna':
                    internal_photo = XrayDiscPicogna(disc,Type='InnerHole',R_hole=R_hole)
                elif photomodel=='Owen':
                    internal_photo = XrayDiscOwen(disc,Type='InnerHole',R_hole=R_hole)
                else:
                    print("Photoevaporation Mode Unrecognised: Default to 'None'")
                    internal_photo = None
            else:
                if photomodel=='Picogna':
                    internal_photo = XrayDiscPicogna(disc)
                elif photomodel=='Owen':
                    internal_photo = XrayDiscOwen(disc)
                else:
                    print("Photoevaporation Mode Unrecognised: Default to 'None'")
                    internal_photo = None
                if internal_photo and R_hole:
                    internal_photo._Hole=True
        elif model['euv']['Phi'] > 0:
            InnerHole = internal_photo_type.startswith('InnerHole')
            if InnerHole:
                internal_photo = EUVDiscAlexander(disc,Type='InnerHole')
            else:
                internal_photo = EUVDiscAlexander(disc)
                if R_hole:
                    internal_photo._Hole=True
        else:
            internal_photo = None
    except KeyError:
        internal_photo = None    

    return DiscEvolutionDriver(disc, 
                               gas=gas, dust=dust, diffusion=diffuse,
                               chemistry=chemistry,
                               ext_photoevaporation=photoevap, int_photoevaporation=internal_photo,
                               history=history, t0=start_time)

###
def setup_simple_chem(model):
    return get_simple_chemistry_model(model)

##
def setup_output(model):
    
    out = model['output']

    # For explicit control of output times
    if (out['arrange'] == 'explicit'):

        # Setup of the output controller
        output_times = np.arange(out['first'], out['last'], out['interval']) * yr
        if not np.allclose(out['last'], output_times[-1], 1e-12):
            output_times = np.append(output_times, out['last'] * yr)
            output_times = np.insert(output_times,0,0) # Prepend 0

        # Setup of the plot controller
        if out['plot'] and out['plot_times']!=[0]:
            plot = np.array([0]+out["plot_times"]) * yr
        elif out['plot']:
            plot = output_times
        else:
            plot = []

        # Setup of the history controller
        if out['history'] and out['history_times']!=[0]:
            history = np.array([0]+out['history_times']) * yr
        elif out['history']:
            history = output_times
        else:
            history = []

    # For regular, logarithmic output times
    elif (out['arrange'] == 'log'):
        print("Logarithmic spacing of outputs chosen - overrides the anything entered manually for plot/history times.")

        # Setup of the output controller
        if out['interval']<10:
            perdec = 10
        else:
            perdec = out['interval']
        first_log = np.floor( np.log10(out['first']) * perdec ) / perdec
        last_log  = np.floor( np.log10(out['last'])  * perdec ) / perdec
        no_saves = int((last_log-first_log)*perdec+1)
        output_times = np.logspace(first_log,last_log,no_saves,endpoint=True,base=10,dtype=int) * yr
        output_times = np.insert(output_times,0,0) # Prepend 0
        if not np.allclose(out['last'], output_times[-1], 1e-12):
            output_times = np.append(output_times, out['last'] * yr)

        # Setup of the plot controller
        if out['plot']:
            plot = output_times
        else:
            plot = []      

        # Setup of the history controller
        if out['history']:
            history = output_times
        else:
            history = []      

    EC = Event_Controller(save=output_times, plot=plot, history=history)
    
    # Base string for output:
    mkdir_p(out['directory'])
    base_name = os.path.join(out['directory'], out['base'] + '_{:04d}')

    format = out['format']
    if format.lower() == 'hdf5':
        base_name += '.h5'
    elif format.lower() == 'ascii':
        base_name += '.dat'
    else:
        raise ValueError ("Output format {} not recognized".format(format))

    return base_name, EC, output_times / yr

###############################################################################
# Run
###############################################################################    

def run(model, io, base_name, all_in, restart, verbose=True, n_print=1000, end_low=False):
    external_mass_loss_mode = all_in['fuv']['photoevaporation']

    save_no = 0
    end = False     # Flag to set in order to end computation
    first = True    # Avoid duplicating output during hole clearing
    hole_open = 0   # Flag to set to snapshot hole opening
    hole_save = 0   # Flag to set to snapshot hole opening
    if all_in['transport']['radial drift']:
        hole_snap_no = 1e5
    else:
        hole_snap_no = 1e4
    hole_switch = False

    if restart:
        save_no = restart+1
        # Skip evolution already completed
        while not io.finished():
            ti = io.next_event_time()
            
            if ti > model.t:
                break
            else:
                io.pop_events(model.t)

    while not io.finished():
        ti = io.next_event_time()
        while (model.t < ti and end==False):
            """
            External photoevaporation - if present, model terminates when all cells at (or below) the base rate as unphysical (and prevents errors).
            Internal photoevaporation - if present, model terminates once the disc is empty.
            Accretion - optionally, the model terminates once unobservably low accretion rates (10^-11 solar mass/year)
            """

            # External photoevaporation -  Read mass loss rates
            if model.photoevaporation_external:
                not_empty = (model.disc.Sigma_G > 0)
                Mdot_evap = model.photoevaporation_external.mass_loss_rate(model.disc,not_empty)
                # Stopping condition
                if (np.amax(Mdot_evap)<=model.photoevaporation_external.floor):
                    print ("Photoevaporation rates below FRIED floor... terminating calculation at ~ {:.0f} yr".format(model.t/yr))
                    end = True
                elif external_mass_loss_mode == 'Constant' and model.photoevaporation_external._empty:
                    print ("Photoevaporation has cleared entire disc... terminating calculation at ~ {:.0f} yr".format(model.t/yr))
                    end = True                

            # Internal photoevaporation
            if model._internal_photo:
                # Stopping condition
                if model.photoevaporation_internal._empty:
                    print ("No valid Hole radius as disc is depleted... terminating calculation at ~ {:.0f} yr".format(model.t/yr))
                    end = True
                # Check if need to reset the hole or if have switched to direct field
                elif model.photoevaporation_internal._Thin and not hole_switch:
                    hole_open = np.inf
                    hole_switch = True
                elif model.photoevaporation_internal._reset:
                    hole_open = 0
                    model.photoevaporation_internal._reset = False
                    model.history.clear_hole()
                # If the hole has opened, count steps and determine whether to do extra snapshot
                if model.photoevaporation_internal._Hole:
                    hole_open += 1
                    if (hole_open % hole_snap_no) == 1 and not first:
                        ti = model.t
                        break

            # Viscous evolution - Calculate accretion rate
            if model.gas and end_low:
                M_visc_out = 2*np.pi * model.disc.grid.Rc[0] * model.disc.Sigma[0] * model._gas.viscous_velocity(model.disc)[0] * (AU**2)
                Mdot_acc = -M_visc_out*(yr/Msun)
                # Stopping condition
                if (Mdot_acc<1e-11):
                    print ("Accretion rates below observable limit... terminating calculation at ~ {:.0f} yr".format(model.t/yr))
                    end = True
                    
            if end:
                ### Stop model ###
                last_save=0
                last_plot=0
                last_history=0
                # If there are save times left
                if np.size(io.event_times('save'))>0:
                    last_save = io.event_times('save')[-1]
                # If there are plot times left 
                if np.size(io.event_times('plot'))>0:
                    last_plot = io.event_times('plot')[-1]
                # If there are history times left 
                if np.size(io.event_times('history'))>0:
                    last_history = io.event_times('history')[-1]
                # Remove all events up to the end
                last_t = max(last_save,last_plot,last_history)
                io.pop_events(last_t)

            else:
                ### Evolve model and return timestep ###
                dt = model(ti)
                first = False

            ### Printing
            if verbose and (model.num_steps % n_print) == 0:
                print('Nstep: {}'.format(model.num_steps))
                print('Time: {} yr'.format(model.t / yr))
                print('dt: {} yr'.format(dt / yr))
                if model.photoevaporation_internal and model.photoevaporation_internal._Hole:
                    print("Column density to hole is N = {} g cm^-2".format(model._internal_photo._N_hole))
                    print("Empty cells: {}".format(np.sum(model.disc.Sigma_G<=0)))
                
        grid = model.disc.grid
        
        ### Saving
        if io.check_event(model.t, 'save') or end or (hole_open % hole_snap_no)==1:
            # Print message to record this
            if (hole_open % hole_snap_no)==1:
                print ("Taking extra snapshot of properties while hole is clearing")
                hole_save+=1
            elif end:
                print ("Taking snapshot of final disc state")
            else:
                print ("Making save at {} yr".format(model.t/yr))
            if base_name.endswith('.h5'):
                    model.dump_hdf5( base_name.format(save_no))
            else:
                    model.dump_ASCII(base_name.format(save_no))
            save_no+=1
        if io.check_event(model.t, 'history') or end or (hole_open % hole_snap_no)==1:
            # Measure disc properties and record
            model.history(model)
            # Save state
            model.history.save(model,all_in['output']['directory'])

        io.pop_events(model.t)


def main():
    # Retrieve model from inputs
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=DefaultModel, help='specify the model input json file')
    parser.add_argument("--restart", "-r", type=int, default=0, help='specify a save number from which to restart')
    parser.add_argument("--end", "-e", action="store_true", help='include in order to stop when below observable accretion rates')
    args = parser.parse_args()
    model = json.load(open(args.model, 'r'))
    
    # Do all setup
    disc, driver, output_name, io_control, plot_name, Dt_nv = setup_wrapper(model, args.restart)

    # Run model
    run(driver, io_control, output_name, model, args.restart, end_low=args.end)
        
    # Save disc properties
    outputdata = driver.history.save(driver,model['output']['directory'])

if __name__ == "__main__":
    main()
    
