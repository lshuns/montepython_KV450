#!/bin/bash

in_prefix=KiDS_2018-07-26_deepspecz_photoz_10th_BLIND_specweight_1000_4
in_suffix=TBgt3_blindB_Nz.asc
in_names=("ZB0p1t0p3" "ZB0p3t0p5" "ZB0p5t0p7" "ZB0p7t0p9" "ZB0p9t1p2")


out_prefix=Nz_DIR
out_suffix=_greater3.asc
out_names=("z0.1t0.3" "z0.3t0.5" "z0.5t0.7" "z0.7t0.9" "z0.9t1.2")


for i in ${!in_names[@]};
do
    in_name=${in_prefix}_${in_names[$i]}_${in_suffix}
    out_name=${out_prefix}_${out_names[$i]}${out_suffix}
    cp $in_name $out_name
done
