library(tidyverse)

data <- read_csv('lls_step_vis.csv', col_types = 'idffid')

data |>
    filter(n_steps > 1) |>
    mutate(variance = round(log10(variance), 2)) |>
    ggplot() +
        aes(x = step, y = loglikelihood, group = interaction(point, path), colour = point) +
        scale_colour_discrete() +
        facet_grid(variance ~ n_steps, scales = 'free', labeller = labeller(variance = \(v) paste('1e', v, sep = ''))) +
        # facet_wrap(~variance, scales = 'free_y') +
        geom_line(alpha = 0.3)
        # scale_y_log10()

ggsave('lls_step_vis.png', width=100, height=150, units='cm', dpi=100, limitsize = FALSE)
