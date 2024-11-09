#!/bin/bash

# Check if results_dir is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <results_dir>"
    exit 1
fi

results_dir=$1
output_file="${results_dir}.rocx"

# Run the mawk command for all files in results_dir
mawk -f bench.noselfhit.awk scop_lookup.fix.tsv <(cat ${results_dir}/*) > ${output_file}
#mawk -f bench.noselfhit.awk scop_lookup.fix.tsv <(cat ${results_dir}) > ${output_file}

# Calculate number of lines in the concatenated input files and num_lines / 11209
num_lines=$(cat ${results_dir}/* | wc -l)
#num_lines=$(cat ${results_dir} | wc -l)
division_result=$(echo "$num_lines / 6504" | bc -l)

# Print out num_lines / 11209
echo "Number of proteins with results (should be integer; should be 93 if run on entire val set): $division_result"

# Calculate scaling factor: 3566 * (11209 / num_lines)
scaling_factor=$(echo "3566 * (6504 / $num_lines)" | bc -l)

# Run the awk command to compute scaled averages
awk -v scaling_factor=$scaling_factor \
'{ famsum+=$3; supfamsum+=$4; foldsum+=$5 } \
 END { \
     print (famsum/NR)*scaling_factor, (supfamsum/NR)*scaling_factor, (foldsum/NR)*scaling_factor \
 }' ${output_file}
