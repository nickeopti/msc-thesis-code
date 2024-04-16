library(tidyverse)

data <- read_csv('lls_step.csv', col_types = 'fdd')

data |>
    # filter(variance >= 1e-2) |>
    ggplot() +
        aes(x = variance, y = loglikelihood) +
        scale_x_log10() +
        # scale_y_log10() +
        facet_wrap(~n_steps, scales = 'free_y', labeller = label_both) +
        geom_line() +
        geom_vline(xintercept = 0.1, linetype = 'dashed', alpha = 0.7)
        # scale_y_log10()

ggsave('lls_step.png', width=30, height=12, units='cm', dpi=300)
