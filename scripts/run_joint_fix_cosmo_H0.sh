#!/bin/bash/

# model selection (null test) with joint likelihood
# H0 model (common nuisance parameters)

# running: nohup bash run_joint_fix_cosmo_H0.sh > nohup_H0_Dz_IA.out &
# output: nohup_H0_Dz_IA.out


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ KV450
wd=/disks/shear15/ssli/CosmicShear/KV450_H0_Dz_IA

test ! -d $wd && mkdir $wd

start=`date +%s`

# ## a short chain with Metropolis Hastings
# python2.7 \
#    ../montepython/MontePython.py run \
#    -p ../input/kv450_joint_fix_cosmo_H0.param \
#    -o $wd \
#    --conf ./kv450_cf.conf \
#    -N 10

# mpi MultiNest run
export OMP_NUM_THREADS=4 # the number of cores used for each chain
mpirun -np 5 python2.7 ../montepython/MontePython.py run \
    `# supply relative path from working directory (wd) to param-file` \
    -p ../input/kv450_joint_fix_cosmo_H0.param \
    `# supply relative path from wd to output folder` \
    -o $wd \
    `# supply relative path from wd to correctly set config-file (otherwise default.conf from MontePython will be used)` \
    --conf ./kv450_cf.conf \
    `# choose the MultiNest sampler (nested sampling)` \
    -m NS \
    `# set an arbitrary but large number of steps (run should converge well before!)` \
    --NS_max_iter 10000000 \
    `# flag for using importance nested sampling (we did not use it, but it might be faster when using it!)` \
    --NS_importance_nested_sampling False \
    `# for parameter estimation use 0.8 (0.3 recommended for more accurate evidences)` \
    --NS_sampling_efficiency 0.3 \
    `# the more live points the smoother the contours, empirical number, experiment (depends also on hardware available)` \
    --NS_n_live_points 1000 \
    `# run will finish/is converged if ln(Z_i) - ln(Z_j) <= NS_evidence_tolerance for i>j (0.1 is for evidences)` \
    --NS_evidence_tolerance 0.1

end=`date +%s`

runtime=$((end-start))
echo runtime, $runtime

echo -e "\033[0;34m=======================================\033[0m"
echo -e "\033[0;34m=======================================\033[0m"
echo -e "\033[0;34m=======================================\033[0m"




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Planck
wd=/disks/shear15/ssli/CosmicShear/Planck_H0_Dz_IA

test ! -d $wd && mkdir $wd

start=`date +%s`

# ## a short chain with Metropolis Hastings
# python2.7 \
#    ../montepython/MontePython.py run \
#    -p ../input/Planck_joint_fix_cosmo_H0.param \
#    -o $wd \
#    --conf ./kv450_cf.conf \
#    -N 10

# mpi MultiNest run
export OMP_NUM_THREADS=4 # the number of cores used for each chain
mpirun -np 5 python2.7 ../montepython/MontePython.py run \
    `# supply relative path from working directory (wd) to param-file` \
    -p ../input/Planck_joint_fix_cosmo_H0.param \
    `# supply relative path from wd to output folder` \
    -o $wd \
    `# supply relative path from wd to correctly set config-file (otherwise default.conf from MontePython will be used)` \
    --conf ./kv450_cf.conf \
    `# choose the MultiNest sampler (nested sampling)` \
    -m NS \
    `# set an arbitrary but large number of steps (run should converge well before!)` \
    --NS_max_iter 10000000 \
    `# flag for using importance nested sampling (we did not use it, but it might be faster when using it!)` \
    --NS_importance_nested_sampling False \
    `# for parameter estimation use 0.8 (0.3 recommended for more accurate evidences)` \
    --NS_sampling_efficiency 0.3 \
    `# the more live points the smoother the contours, empirical number, experiment (depends also on hardware available)` \
    --NS_n_live_points 1000 \
    `# run will finish/is converged if ln(Z_i) - ln(Z_j) <= NS_evidence_tolerance for i>j (0.1 is for evidences)` \
    --NS_evidence_tolerance 0.1

end=`date +%s`

runtime=$((end-start))
echo runtime, $runtime



