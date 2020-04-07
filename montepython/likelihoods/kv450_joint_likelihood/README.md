This repository contains the likelihood module modified from 'kv450_cf_likelihood_public', the KiDS+VIKING-450 (in short: KV450) correlation function measurements from [Hildebrandt et al. 2018 (arXiv:1812.06076)](http://adsabs.harvard.edu/abs/2018arXiv181206076H).

This version of likelihood allows to analyse two data vectors at the same time with a common set of fixed parameters (including the cosmological parameters or necessary nuisance parameters). For the set of free parameters, both duplicated sets and common set are supported.

Changelog
---------

2020-04-06: 
    
    module built with copy from 'kv450_cf_likelihood_public'

    rename all 'kv450_cf_likelihood_public' into 'kv450_diff_likelihood'
    
    comments info added to __init__.py
    
    -------------- kv450_diff_likelihood built

2020-04-07:
    
    rename to 'kv450_joint_likelihood'

    -------------- move to kv450_joint_likelihood