library(ggplot2)

df <- read.csv("tuning_results.csv")
df$base_n_filters <- factor(df$base_n_filters)
df$additional_filters_per_block <- factor(df$additional_filters_per_block)

df[which.min(df$min_val_loss), ]

table(df$base_n_filters, df$additional_filters_per_block)

df$date <- as.Date(df$config_path)

p <- (ggplot(df, aes(x=dropout_rate,
                     y=min_val_loss,
                     color=base_n_filters,
                     shape=additional_filters_per_block,
                     linetype=additional_filters_per_block)) +
      scale_color_discrete() +
      geom_point(size=3.5) +
      ## geom_smooth() +
      theme_bw())
p
