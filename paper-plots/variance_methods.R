#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(ggplot2)
# library(ggh4x)
# library(tikzDevice)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
    data <- read_csv('ll_variance_results.csv', col_types = 'fdd')
} else {
    data_file <- args[1]
    data <- read_csv(data_file, col_types = 'fffdd')
}

maxs <- data |> group_by(steps, landmarks, method) |> reframe(top = max(ll), argmax = sigma[which.max(ll)])

# tikz(file = 'll_variance_results.tex', width=3.25, height=2.5)

data |>
    group_by(steps, landmarks, method) |>
    mutate(ll = (ll - min(ll, na.rm = TRUE)) / (max(ll, na.rm = TRUE) - min(ll, na.rm = TRUE))) |>
    ggplot() +
    aes(x = sigma, y = ll, linetype = steps) +
    geom_segment(data = maxs, aes(x = argmax, xend = argmax, y = -Inf, yend = 1, linetype = steps), linewidth = 0.25, show.legend = FALSE) +
    geom_line() +
    # geom_line(linetype = 'dashed') +
    scale_x_log10() +
    facet_grid(
        landmarks ~ method, 
        scales = "free_y",
        # independent = "y",
        switch = 'y'
    ) +
    theme_gray(base_size = 8) +
    xlab(expression(sigma^2)) +
    ylab('dimensions') +
    theme(
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank()
    )


# data |>
#     group_by(steps, landmarks, method) |>
#     mutate(ll = (ll - min(ll, na.rm = TRUE)) / (max(ll, na.rm = TRUE) - min(ll, na.rm = TRUE))) |>
#     ggplot() +
#     aes(x = sigma, y = ll, colour = method) +
#     geom_segment(data = maxs, aes(x = argmax, xend = argmax, y = -Inf, yend = 1, colour = method), linewidth = 0.25, position = position_dodge(width = 0.05), show.legend = FALSE) +
#     geom_line(position = position_dodge(width = 0.05)) +
#     # geom_line(linetype = 'dashed') +
#     scale_x_log10() +
#     facet_grid(
#         steps ~ landmarks, 
#         scales = "free_y",
#         # independent = "y",
#         switch = 'y'
#     ) +
#     theme_gray(base_size = 8) +
#     xlab(expression(sigma^2)) +
#     ylab('dimensions') +
#     theme(
#         axis.text.y = element_blank(),
#         axis.ticks.y = element_blank()
#     )

# dev.off()

ggsave('ll_variance_results.png', width = 5.25, height = 2.5, units = 'in', dpi = 600)
