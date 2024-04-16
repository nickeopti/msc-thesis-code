library(tidyverse)

data <- read_csv('lls.csv', col_types = 'dffid')

data |>
    ggplot() +
        aes(x = time, y = loglikelihood, group = interaction(point, path), colour = point) +
        scale_colour_discrete() +
        facet_wrap(~variance, scales = 'free_y') +
        geom_line(alpha = 0.3)
        # scale_y_log10()

ggsave('lls.png', width=30, height=12, units='cm', dpi=300)
