#!/bin/gnuplot

#set terminal dumb 136 36;
# This hides the tics from the plot, but doesn't remove the labels
# Uncomment for dumb terminal
#set tic scale 0;

set terminal pngcairo size 3864,2160 enhanced font 'Helvetica,30';
set datafile separator ',';
set output outfile

# filename1 and filename2 are input arguments

set style line 1 linecolor rgb '#006400' linewidth 4.5 pointtype  6 pointsize 3.0;
set style line 2 linecolor rgb '#bc8f8f' linewidth 4.5 pointtype  2 pointsize 3.0;
set style line 3 linecolor rgb '#ff4500' linewidth 4.5 pointtype  4 pointsize 3.0;
set style line 4 linecolor rgb '#ffd700' linewidth 4.5 pointtype  1 pointsize 3.0;
set style line 5 linecolor rgb '#00ff00' linewidth 4.5 pointtype  8 pointsize 3.0;
set style line 6 linecolor rgb '#00ffff' linewidth 4.5 pointtype 10 pointsize 3.0;
set style line 7 linecolor rgb '#a020f0' linewidth 4.5 pointtype 12 pointsize 3.0;
set style line 8 linecolor rgb '#1e90ff' linewidth 4.5 pointtype 14 pointsize 3.0;
set style line 9 linecolor rgb '#ff1493' linewidth 4.5 pointtype 19 pointsize 3.0;

set border lw 8.0;
set grid;

set multiplot layout 1, 3;
# left, right, top, bottom
set offset graph 0.01, graph 0.01, graph 0.01, graph 0.01;

plot for [i=1:9] filename1   \
        using 1:i + 4 \
        with lines \
        title 'hop f'.(i-1) \
        linestyle i, \
     for [i=1:9] filename2 \
        every 100::10*i \
        using 1:i + 4 \
        with points \
        title 'torch f'.(i-1) \
        linestyle i;

plot filename1   \
        using 1:3 \
        with lines \
        title 'hop u0' \
        linestyle 1, \
     filename2 \
        every 40 \
        using 1:3 \
        with points \
        title 'torch u0' \
        linestyle 1;

plot filename1   \
        using 1:2 \
        with lines \
        title 'hop rho' \
        linestyle 1, \
     filename2 \
        every 40 \
        using 1:2 \
        with points \
        title 'torch rho' \
        linestyle 1;
