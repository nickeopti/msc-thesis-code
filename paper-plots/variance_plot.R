#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(ggplot2)
# library(tikzDevice)

data <- read_csv('ll_variance_results.csv', col_types = 'fdd')

# tikz(file = 'll_variance_results.tex', width=3.25, height=2.5)

data |>
    # filter(method %in% c('analytical', 'pedersen', 'pedersen_bridge')) |>
    ggplot() +
    theme_gray(base_size = 8) +
    aes(x = sigma, y = ll) +
    facet_wrap(~method, scales = 'free_y') +
    geom_line() +
    scale_x_log10() +
    xlab(expression(sigma^2)) +
    ylab('(approximated) log-likelihood curves')

# dev.off()

ggsave('ll_variance_results.png', width = 3.25, height = 2.5, units = 'in', dpi = 600)
