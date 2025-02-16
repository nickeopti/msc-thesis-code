#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(ggplot2)
library(tikzDevice)

args <- commandArgs(trailingOnly = TRUE)

data <- read_csv(args[1], col_types = 'ffdd') |> filter(method == 'proposed')
maxs <- data |> group_by(species) |> summarise(top = max(ll), argmax = sigma[which.max(ll)])

tikz(file = 'll_variance_skull.tex', width=3.25, height=2.5)

data |>
    ggplot() +
    theme_gray(base_size = 8) +
    aes(x = sigma, y = ll) +
    facet_wrap(~ species) +
    geom_line() +
    geom_segment(data = maxs, aes(x = argmax, xend = argmax, y = -Inf, yend = top), linewidth = 0.25, show.legend = FALSE) +
    scale_x_log10(
        breaks = scales::trans_breaks("log10", function(x) 10^x, n = 3),
        labels = scales::trans_format("log10", scales::math_format(10^.x))
    ) +
    xlab('$ \\sigma^2 $') +
    ylab('log-likelihood curve')

dev.off()
