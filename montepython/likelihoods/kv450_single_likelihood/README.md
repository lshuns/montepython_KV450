This repository contains the likelihood module modified from 'kv450_cf_likelihood_public', the KiDS+VIKING-450 (in short: KV450) correlation function measurements from [Hildebrandt et al. 2018 (arXiv:1812.06076)](http://adsabs.harvard.edu/abs/2018arXiv181206076H).

This version of likelihood allows to analyse a signle reproduced data vector from shear catalogues.

The likelihood structure is not changed, so the purpose of this module is to verify the reproduced data vector.

Changelog
---------

    rename kv450_cf_likelihood_public to kv450_single_likelihood

    remove data existence check in the beginning of __init__.py

    remove saving masked covariance in list format

    remove __write_out_vector_in_list_format, self.write_out_theory (redundant for mcmc running)

    remove __load_public_theory_vector, covariance list operation (covariance is supposed to be provided in matrix form)

    remove self.bootstrap_photoz_errors (not even used)

    rename all data-loading-related file names

    use self.sample_tag in kv450_single_likelihood.data file to control different samples
