library(readr)
library(dplyr)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)

data <- read_csv(args[1])

data |>
    bind_rows(
        data |>
            filter(process == 'forwards') |>
            mutate(
                process = 'reversed forwards',
                t = 1 - t
            )
    ) |>
    mutate(across(process, ~factor(., levels = c('original', 'backwards', 'forwards', 'reversed forwards')))) |>
    mutate(iteration = i) |>
    ggplot() +
        aes(x = t, y = y, colour = iteration) +
        facet_wrap(~process) +
        geom_line() +
        scale_colour_gradient(low = '#000000', high='#aaaaaa')

ggsave('time_perspectives.png', height = 8, width= 20, units = 'cm', dpi = 600)
