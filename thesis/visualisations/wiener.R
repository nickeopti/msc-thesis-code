#!/usr/bin/env Rscript

library(readr)
library(ggplot2)

data <- read_csv('wiener.csv', col_types = 'fdd')

data |>
    ggplot() +
    aes(x = t, y = y, colour = i, group = i) +
    geom_line(show.legend = FALSE) +
    xlab('t') +
    ylab('y')

ggsave('wiener.png', width = 20, height = 8, units = 'cm', dpi = 600)
