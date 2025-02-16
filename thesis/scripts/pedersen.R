#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(ggplot2)

data <- read_csv('pedersen.csv', col_types = 'fdd')

names = c(
    analytical = 'analytical',
    stable_analytical = 'stable analytical',
    stable_analytical_offset = 'stable analytical, offset',
    pedersen = 'simulated likelihood estimate',
    pedersen_bridge = 'importance sampled estimate',
    pedersen_bridge_reverse = 'importance sampled estimate, reverse',
    pedersen_bridge_reverse_old = 'importance sampled estimate, reverse, old'
)

data |>
    # filter(method %in% c('analytical', 'pedersen', 'pedersen_bridge')) |>
    ggplot() +
    aes(x = parameter, y = ll) +
    facet_wrap(~method, scales = 'free_y', labeller = as_labeller(names)) +
    geom_line() +
    scale_x_log10() +
    xlab(expression(sigma^2)) +
    ylab('log-likelihood')

ggsave('pedersen.png', width = 20, height = 8, units = 'cm', dpi = 600)
