#!/bin/bash/
dataDir=/disks/shear15/ssli/CosmicShear/kv450_public_reproduced

rm -r $dataDir/*

# ## a short chain with Metropolis Hastings
# python2 \
#    ./montepython/MontePython.py run \
#    -p ./input/kv450_cf.param \
#    -o $dataDir \
#    --conf ./kv450_cf.conf \
#    -N 10


# Fiducial MutiNest run
start=`date +%s`

python2 ./montepython/MontePython.py run \
    `# supply relative path from working directory (wd) to param-file` \
    -p ./input/kv450_cf.param \
    `# supply relative path from wd to output folder` \
    -o $dataDir \
    `# supply relative path from wd to correctly set config-file (otherwise default.conf from MontePython will be used)` \
    --conf ./kv450_cf.conf \
    `# choose the MultiNest sampler (nested sampling)` \
    -m NS \
    `# set an arbitrary but large number of steps (run should converge well before!)` \
    --NS_max_iter 10000000 \
    `# flag for using importance nested sampling (we did not use it, but it might be faster when using it!)` \
    --NS_importance_nested_sampling False \
    `# for parameter estimation use 0.8 (0.3 recommended for more accurate evidences)` \
    --NS_sampling_efficiency 0.8 \
    `# the more live points the smoother the contours, empirical number, experiment (depends also on hardware available)` \
    --NS_n_live_points 1000 \
    `# run will finish/is converged if ln(Z_i) - ln(Z_j) <= NS_evidence_tolerance for i>j (0.1 is for evidences)` \
    --NS_evidence_tolerance 0.5

end=`date +%s`

runtime=$((end-start))
echo runtime, $runtime
