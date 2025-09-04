#!/bin/gnuplot

set terminal dumb 136 36;
set datafile separator ',';

stats 'data/u0_hop_lumi_float32.csv' nooutput;
num_cols = STATS_columns;

set multiplot layout 2, 3;
# This hides the tics from the plot, but doesn't remove the labels
set tic scale 0;
set offset graph 0.05, graph 0.05, graph 0.05, graph 0.05;

plot for [i=0:8] 'data/f'.i.'_hop_lumi_float32.csv'     using num_cols/2 with linespoints notitle;
plot             'data/u0_hop_lumi_float32.csv'         using num_cols/2 with linespoints notitle;
plot             'data/rho_hop_lumi_float32.csv'        using num_cols/2 with linespoints notitle;

plot for [i=0:8] 'data/f'.i.'_torch_lumi_float32.csv'   using num_cols/2 with linespoints notitle;
plot             'data/u0_torch_lumi_float32.csv'       using num_cols/2 with linespoints notitle;
plot             'data/rho_torch_lumi_float32.csv'      using num_cols/2 with linespoints notitle;
