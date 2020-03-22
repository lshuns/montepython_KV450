#!/bin/bash/
# Running fixed maximum-likelihood values from public mcmc chain
dataDir=/disks/shear15/ssli/CosmicShear/kv450_public_reproduced

rm $dataDir/*

## a short chain with Metropolis Hastings
python2 \
   ./montepython/MontePython.py run \
   -p ./input/kv450_best.param \
   -o $dataDir \
   --conf ./kv450_cf.conf \
   -N 10
