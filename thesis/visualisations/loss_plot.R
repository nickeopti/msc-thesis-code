#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
v_num <- args[2]
if (length(args) == 1) {
    cutoff <- 10
} else {
    cutoff <- strtoi(args[3])
}

metrics_path <- paste(
    "logs",
    args[1],
    paste("version_", v_num, sep = ""),
    "metrics.csv",
    sep = "/"
)

metrics <- read_csv(metrics_path)

metrics %>%
    pivot_longer(!epoch, names_to = "loss", values_to = "value") %>%
    filter(epoch >= cutoff) %>%
    ggplot() +
        aes(x = epoch, y = value) +
        facet_wrap(~loss, scales = "free_y") +
        # ylim(0, NA) +
        geom_line() +
        geom_smooth() +
        xlab("Epoch") +
        ylab("Loss")

ggsave(
    paste(
        "logs",
        args[1],
        paste("version_", v_num, sep = ""),
        "loss_plot.png",
        sep = "/"
    ),
    # paste("plots", paste(args[1], "_", args[2], ".png", sep = ""), sep = "/"),
    width = 30,
    height = 16,
    units = "cm",
    dpi = 600
)
