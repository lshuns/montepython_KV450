This repository contains the likelihood module modified to do null test with fixed cosmo. That is cosmology code (CLASS) only calculated once. In this way, likelihood can be calculated faster.

This version of likelihood allows to analyse two data vectors at the same time with a common set of fixed parameters (including the cosmological parameters or necessary nuisance parameters). For the set of free parameters, both duplicated sets and common set are supported.

Changelog
---------

    remove baryon feedback model

    All cosmo-related calculation moved to __init__

    Remove derived cosmo-para in kv450_joint_fix_cosmo.param, due to cosmo-related bug in sampler.py 

    New redshift offset strategy: mean and shift

    A_IA is duplicated (exp_IA is removed, since it is not used in KV450 fiducial run)