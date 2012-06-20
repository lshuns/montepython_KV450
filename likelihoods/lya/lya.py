from likelihood_class import likelihood
import os
import numpy as np
from math import exp

class lya(likelihood):

  def __init__(self,path,data,command_line,log_flag,default):

    likelihood.__init__(self,path,data,command_line,log_flag,default)

    ############ read data   ###########

    self.z = np.zeros(self.num_z,'float64')
    self.k = np.zeros(self.num_k,'float64')
    self.pk_obs = np.zeros((self.num_k,self.num_z),'float64')
    self.npk_obs = np.zeros((self.num_k,self.num_z),'float64')
    covmat = np.zeros((self.num_k*self.num_z,self.num_k*self.num_z),'float64')
    invcovmat = np.zeros((self.num_k*self.num_z,self.num_k*self.num_z),'float64')
    self.inv_covmat = np.zeros((self.num_k,self.num_z,self.num_k,self.num_z),'float64')
    self.dampcorr = np.zeros((self.num_k,self.num_z),'float64')

    # k, z, pk, noise

    inputfile = open(self.data_directory+self.table,'r')

    for zz in range(self.num_z):
      for kk in range(self.num_k):

        line = inputfile.readline()

        zzz = float(line.split()[0])
        if self.z[zz] == 0.:
          self.z[zz]=zzz
        else:
          if self.z[zz] != zzz:
            print 'input data table not arranged as expected',self.z[zz],zzz
            exit()

        kkk = float(line.split()[1])
        if self.k[kk] == 0.:
          self.k[kk]=kkk
        else:
          if self.k[kk] != kkk:
            print 'input data table not arranged as expected',self.k[kk],kkk
            exit()

        self.pk_obs[kk,zz] = line.split()[2]
        self.npk_obs[kk,zz] = line.split()[4]

    inputfile.close()    

    # covariance matrix

    inputfile = open(self.data_directory+self.covar,'r')

    for np1 in range(self.num_k*self.num_z):
      line = inputfile.readline()
      for np2 in range(self.num_k*self.num_z):
        covmat[np1,np2] = float(line.split()[np2])

    inputfile.close() 
    
    # invert covariance matrix and store it with good indicing

    invcovmat = np.linalg.inv(covmat)

    for zz1 in range(self.num_z):
      for kk1 in range(self.num_k):
        for zz2 in range(self.num_z):
          for kk2 in range(self.num_k):
            self.inv_covmat[kk1,zz1,kk2,zz2]=invcovmat[zz1*self.num_k+kk1,zz2*self.num_k+kk2]

    # damping factors

    inputfile = open(self.data_directory+self.damp,'r')

    for zz in range(self.num_z):
      for kk in range(self.num_k):
        line = inputfile.readline()
        self.dampcorr[kk,zz] = float(line)

    inputfile.close() 

    ############ read theory ###########

    # index of parameters entering in calculation of theoretical flux spectrum (Taylor expansion parameters)

    index = 0
    self.index_s8 = index
    index+=1
    self.index_ns = index
    index+=1
    self.index_Om = index
    index+=1
    self.index_H0 = index
    index+=1
#    self.index_zr = index
#    index+=1
    self.index_g = index
    index+=1
    self.index_tf = index
    index+=1
    self.index_t0 = index
    index+=1
    self.index_num = index

    # read Taylor coefficients for these parameters

    # each table contains:
    #
    # [k1, z1, 2nd order]  ......  [k1, zmax, 2nd order]
    # [k2, z1, 2nd order]  ......  [k2, zmax, 2nd order]
    # ...............................................
    # [kmax, z1, 2nd order] ...... [kmax, zmax, 2nd order]
    # [k1, z1, 1st order]  ......  [k1, zmax, 1st order]
    # [k2, z1, 1st order]  ......  [k2, zmax, 1st order]
    # ...............................................
    # [kmax, z1, 1st order] ...... [kmax, zmax, 1st order]

    self.taylor = np.zeros((self.index_num,self.num_k,self.num_z,2),'float64')

    for param in range(self.index_num):

      if (param == self.index_s8):
        table_name = 's8_gamma1_new.dat'
      if (param == self.index_ns):
        table_name = 'n_gamma1_new.dat'
      if (param == self.index_Om):
        table_name = 'om_gamma1_new.dat'
      if (param == self.index_H0):
        table_name = 'h_gamma1_new.dat'
      if (param == self.index_g):
        table_name = 'coeff_gammaEQ1_all_pk_new.dat'
      if (param == self.index_tf):
        table_name = 'meanflux_gamma1_new.dat'
      if (param == self.index_t0):
        table_name = 't0_gamma1_new.dat'
#     if (param == self.index_zr):
#        table_name = 'highzreion.dat'

      inputfile = open(self.data_directory+table_name,'r')

      # reio table is special: only 1st order coefs, only 2nd half of table
#      if (param == self.index_zr):
#        for kk in range(self.num_k):
#          for zz in range(self.num_z):
#            self.taylor[param,kk,zz,1] = 0. # 2nd order fixed to zero
#        for kk in range(self.num_k):
#          line = inputfile.readline()for kk in range(self.num_k):
#          for zz in range(self.num_z):
#            self.taylor[param,kk,zz,0] = line.split()[zz]-1. # 1st order gets correction by one
#      else:
      # other tables

      for kk in range(self.num_k):
        line = inputfile.readline()
        for zz in range(self.num_z):
          self.taylor[param,kk,zz,1] = float(line.split()[zz])
      for kk in range(self.num_k):
        line = inputfile.readline()
        for zz in range(self.num_z):
          self.taylor[param,kk,zz,0] = float(line.split()[zz])

      inputfile.close()

    # read best-fit values around which Taylor expansion is performed
    self.taylor_expansion_point = np.zeros((self.index_num,self.num_z),'float64')

    self.taylor_expansion_point[self.index_s8,0] = 0.85
    self.taylor_expansion_point[self.index_ns,0] = 0.95
    self.taylor_expansion_point[self.index_Om,0] = 0.26
    self.taylor_expansion_point[self.index_H0,0] = 72.
#    self.taylor_expansion_point[self.index_zr,0] = 6.

    self.taylor_expansion_point[self.index_tf,0] = 0.178000
    self.taylor_expansion_point[self.index_tf,1] = 0.2192
    self.taylor_expansion_point[self.index_tf,2] = 0.2714000
    self.taylor_expansion_point[self.index_tf,3] = 0.3285330
    self.taylor_expansion_point[self.index_tf,4] = 0.379867
    self.taylor_expansion_point[self.index_tf,5] = 0.42900
    self.taylor_expansion_point[self.index_tf,6] = 0.513000
    self.taylor_expansion_point[self.index_tf,7] = 0.600400
    self.taylor_expansion_point[self.index_tf,8] = 0.657800
    self.taylor_expansion_point[self.index_tf,9] = 0.756733
    self.taylor_expansion_point[self.index_tf,10] = 0.896000

    self.taylor_expansion_point[self.index_g,0] =  1.05
    self.taylor_expansion_point[self.index_g,1] =  1.05
    self.taylor_expansion_point[self.index_g,2] =  1.05
    self.taylor_expansion_point[self.index_g,3] =  1.05
    self.taylor_expansion_point[self.index_g,4] =  1.05
    self.taylor_expansion_point[self.index_g,5] =  1.05
    self.taylor_expansion_point[self.index_g,6] =  1.04
    self.taylor_expansion_point[self.index_g,7] =  1.04
    self.taylor_expansion_point[self.index_g,8] =  1.03
    self.taylor_expansion_point[self.index_g,9] =  1.03
    self.taylor_expansion_point[self.index_g,10] = 1.03

    self.taylor_expansion_point[self.index_t0,0] =  20600
    self.taylor_expansion_point[self.index_t0,1] =  21100
    self.taylor_expansion_point[self.index_t0,2] =  21600
    self.taylor_expansion_point[self.index_t0,3] =  22000
    self.taylor_expansion_point[self.index_t0,4] =  22300
    self.taylor_expansion_point[self.index_t0,5] =  22492
    self.taylor_expansion_point[self.index_t0,6] =  22600
    self.taylor_expansion_point[self.index_t0,7] =  22600
    self.taylor_expansion_point[self.index_t0,8] =  22505
    self.taylor_expansion_point[self.index_t0,9] = 22200
    self.taylor_expansion_point[self.index_t0,10] = 21529

    # flux spectrum of best-fit model

    self.pkth = np.zeros((self.num_k,self.num_z),'float64')

    inputfile = open(self.data_directory+'B2_gamma1_all.txt','r')

    for zz in range(self.num_z):
      for kk in range(self.num_k):
        self.pkth[kk,zz] = float(inputfile.readline())

    inputfile.close()    

    # resolution box size correction

    inputfile = open(self.data_directory+'res_box_table_all.txt','r')

    for kk in range(self.num_k):
      line = inputfile.readline()
      for zz in range(self.num_z):
        rb = float(line.split()[zz])
        self.pkth[kk,zz] *= rb

    inputfile.close() 

    ######## end read theory ##########

    # necessary class parameters. mPk and k_max=2h/Mpc needed to have sigma_8 computed and accurate !

    self.need_Class_arguments(data,{'output':'mPk', 'P_k_max_h/Mpc':2})
 
    return
  
  # compute likelihood

  def loglkl(self,_cosmo,data):

    # noise for each data point

    noise = np.zeros((self.num_k,self.num_z),'float64')

    for kk in range(self.num_k):     
      noise[kk,0] = float(data.mcmc_parameters['noise_z0']['current']) * self.npk_obs[kk,0]
      noise[kk,1] = float(data.mcmc_parameters['noise_z1']['current']) * self.npk_obs[kk,1]
      noise[kk,2] = float(data.mcmc_parameters['noise_z2']['current']) * self.npk_obs[kk,2]
      noise[kk,3] = float(data.mcmc_parameters['noise_z3']['current']) * self.npk_obs[kk,3]
      noise[kk,4] = float(data.mcmc_parameters['noise_z4']['current']) * self.npk_obs[kk,4]
      noise[kk,5] = float(data.mcmc_parameters['noise_z5']['current']) * self.npk_obs[kk,5]
      noise[kk,6] = float(data.mcmc_parameters['noise_z6']['current']) * self.npk_obs[kk,6]
      noise[kk,7] = float(data.mcmc_parameters['noise_z7']['current']) * self.npk_obs[kk,7]
      noise[kk,8] = float(data.mcmc_parameters['noise_z8']['current']) * self.npk_obs[kk,8]
      noise[kk,9] = float(data.mcmc_parameters['noise_z9']['current']) * self.npk_obs[kk,9]
      noise[kk,10]= float(data.mcmc_parameters['noise_z10']['current']) * self.npk_obs[kk,10]

    ######### compute theory  ############## 

    # displacement vector of Taylor expansion:

    delta = np.zeros(self.index_num,'float64')

    # theoretical spectrum including noise corrections

    pk_th = np.zeros((self.num_k,self.num_z),'float64')

    # displacement for cosmo parameters:

    delta[self.index_H0] = _cosmo._h()*100.  - self.taylor_expansion_point[self.index_H0,0]
    delta[self.index_ns] = _cosmo._n_s()     - self.taylor_expansion_point[self.index_ns,0]
    delta[self.index_Om] = _cosmo._Omega_m() - self.taylor_expansion_point[self.index_Om,0]
    delta[self.index_s8] = _cosmo._sigma8()  - self.taylor_expansion_point[self.index_s8,0]
#    delta[self.index_zr] = (_cosmo._z_reio() - self.taylor_expansion_point[self.index_zr,0])/7.

    for zz in range(self.num_z):
      
      # redshift-dependent displacement for thermo params

      if (zz<5): 
        gpower = float(data.mcmc_parameters['gvals']['current'])
        t0power = float(data.mcmc_parameters['t0vals']['current'])
      else:
        gpower = float(data.mcmc_parameters['gvals2']['current'])
        t0power = float(data.mcmc_parameters['t0vals2']['current'])

      tfpower = float(data.mcmc_parameters['teffvals']['current'])

      delta[self.index_g] = float(data.mcmc_parameters['gvala']['current']) * ((1.+self.z[zz])/4)**gpower - self.taylor_expansion_point[self.index_g,zz]
      delta[self.index_tf] = (float(data.mcmc_parameters['teffvala']['current']) * ((1.+self.z[zz])/4)**tfpower - self.taylor_expansion_point[self.index_tf,zz])/self.taylor_expansion_point[self.index_tf,zz]
      delta[self.index_t0] = (float(data.mcmc_parameters['t0vala']['current']) * ((1.+self.z[zz])/4)**t0power - self.taylor_expansion_point[self.index_t0,zz])/1.e3

      for kk in range(self.num_k):

        # get theoretical spectrum (Taylor expansion)

        delta_pf = float(data.mcmc_parameters['damp']['current']) * self.dampcorr[kk,zz]

        for param in range(self.index_num):
          delta_pf += self.taylor[param,kk,zz,0]*delta[param] + self.taylor[param,kk,zz,1]*delta[param]**2

        pk_th[kk,zz] = (self.pk_obs[kk,zz]*(1.+delta_pf) + noise[kk,zz]) * exp(-float(data.mcmc_parameters['alpha']['current']) * self.k[kk]**2)

# chi2: priors

    chi2 = ((float(data.mcmc_parameters['damp']['current'])-1.)/0.3)**2 
    chi2 += (float(data.mcmc_parameters['alpha']['current'])/49.)**2
    chi2 += (float(data.mcmc_parameters['noise_z0']['current'])/0.05)**2
    chi2 += (float(data.mcmc_parameters['noise_z1']['current'])/0.05)**2
    chi2 += (float(data.mcmc_parameters['noise_z2']['current'])/0.05)**2
    chi2 += (float(data.mcmc_parameters['noise_z3']['current'])/0.05)**2
    chi2 += (float(data.mcmc_parameters['noise_z4']['current'])/0.05)**2
    chi2 += (float(data.mcmc_parameters['noise_z5']['current'])/0.05)**2
    chi2 += (float(data.mcmc_parameters['noise_z6']['current'])/0.05)**2
    chi2 += (float(data.mcmc_parameters['noise_z7']['current'])/0.05)**2
    chi2 += (float(data.mcmc_parameters['noise_z8']['current'])/0.05)**2
    chi2 += (float(data.mcmc_parameters['noise_z9']['current'])/0.05)**2
    chi2 += (float(data.mcmc_parameters['noise_z10']['current'])/0.05)**2

# chi2: fit

    for zz1 in range(self.num_z):
      for kk1 in range(self.num_k):
        for zz2 in range(self.num_z):
          for kk2 in range(self.num_k):
            chi2 += (self.pk_obs[kk1,zz1]-pk_th[kk1,zz1])*self.inv_covmat[kk1,zz1,kk2,zz2]*(self.pk_obs[kk2,zz2]-pk_th[kk2,zz2])

    return - chi2/2.