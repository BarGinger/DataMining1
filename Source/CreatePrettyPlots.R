# Load necessary libraries
library(ggplot2)
library(reshape2)
library(dplyr)
library(tidyr)
library(stringr)
library(pheatmap)
library(tibble)
library(purrr)
library(broom)
library(MASS)

# constants
DATA_FOLDR <- "Data/"
OUTPUT_FOLDR <- "Output/"

# Specify the path to the zip file
zipfile <- paste(DATA_FOLDR, "promise-2_0a-packages-csv.zip", sep="")

# Function to read and preprocess CSV files
read_and_preprocess <- function(zipfile, columns_to_keep) {
  files_in_zip <- unzip(zipfile, list = TRUE)
  csv_files <- files_in_zip$Name[grep("\\.csv$", files_in_zip$Name)]
  
  df_list <- list()
  
  for (csv_file in csv_files) {
    temp_df <- read.csv(unz(zipfile, csv_file), sep = ";", stringsAsFactors = FALSE)
    temp_df$label <- ifelse(temp_df$post > 1, 1, 0)
    temp_df <- temp_df[, columns_to_keep]
    temp_df <- temp_df[, sapply(temp_df, is.numeric)]
    temp_df$release_version <- as.numeric(gsub(".*-([0-9]+\\.[0-9]+)\\.csv$", "\\1", csv_file))
    df_list[[csv_file]] <- temp_df
  }
  
  return(do.call(rbind, df_list))
}

# Define columns to keep
columns_to_keep <- c(
  'ACD_avg', 'ACD_max', 'ACD_sum', 'FOUT_avg', 'FOUT_max', 'FOUT_sum',
  'MLOC_avg', 'MLOC_max', 'MLOC_sum', 'NBD_avg', 'NBD_max', 'NBD_sum',
  'NOCU', 'NOF_avg', 'NOF_max', 'NOF_sum', 'NOI_avg', 'NOI_max', 'NOI_sum',
  'NOM_avg', 'NOM_max', 'NOM_sum', 'NOT_avg', 'NOT_max', 'NOT_sum', 
  'NSF_avg', 'NSF_max', 'NSF_sum', 'NSM_avg', 'NSM_max', 'NSM_sum',
  'PAR_avg', 'PAR_max', 'PAR_sum', 'pre', 'TLOC_avg', 'TLOC_max', 'TLOC_sum', 
  'VG_avg', 'VG_max', 'VG_sum', 'label'
)

# Read and preprocess the data
df_combined <- read_and_preprocess(zipfile, columns_to_keep)

## Linear Discriminant Analysis (LDA)
perform_lda <- function(df) {
  lda_model <- lda(label ~ ., data = df)
  lda_result <- predict(lda_model)$x
  return(lda_result)
}

lda_result <- as.data.frame(perform_lda(df_combined))  
lda_result$label <- df_combined$label
lda_result$release_version <- df_combined$release_version  

plot <- ggplot(lda_result, aes(x = LD1, fill = factor(label))) +
  geom_density(alpha = 0.6) +
  labs(title = "LDA: Distribution of LD1 by Release Version", 
       x = "Linear Discriminant 1", fill = "Class Label") +
  theme_minimal() +
  facet_wrap(~ release_version)

ggsave(paste(OUTPUT_FOLDR, "Distribution of LD1 by Release Version.png", sep=""), 
       plot = plot, bg = "white", width = 10, height = 6)
plot

# Adjusted Fisher Score Calculation by Label, Faceted by Release Version
calculate_fisher_score <- function(df) {
  labels <- df$label
  df <- df[, !(names(df) %in% c("release_version", "label"))]
  
  means <- colMeans(df[labels == 1, ]) - colMeans(df[labels == 0, ])
  sds <- apply(df, 2, function(col) sd(col[labels == 1]) + sd(col[labels == 0]))
  fisher_score <- means^2 / sds
  
  return(fisher_score)
}

# Split data by release_version to calculate Fisher scores for each version
fisher_scores_by_version <- lapply(split(df_combined, df_combined$release_version), function(df) {
  fisher_scores <- calculate_fisher_score(df)
  return(data.frame(
    attribute = names(fisher_scores), 
    fisher_score = fisher_scores, 
    release_version = unique(df$release_version)
  ))
})

# Combine the Fisher scores across all release versions into one data frame
# Add a new column to identify if the attribute is 'TLOC_sum'
fisher_scores_df <- do.call(rbind, fisher_scores_by_version)
fisher_scores_df <- fisher_scores_df %>%
  mutate(is_last = (attribute == 'TLOC_sum'))

# Plot the Fisher scores, faceted by release_version with adjusted text positioning
plot <- ggplot(fisher_scores_df, aes(x = reorder(attribute, -fisher_score), 
                                     y = fisher_score, fill = factor(release_version))) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  coord_flip() +
  geom_text(aes(label = sprintf("%.3f", fisher_score), 
                hjust = ifelse(is_last, 1, -0.1)), 
            size = 3.5, color = "black") +
  facet_wrap(~ release_version, scales = "free_y") +
  labs(title = "Fisher Score of Attributes for Labels Faceted by Release Version", 
       x = "Attribute", y = "Fisher Score", fill = "Release Version") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 10), 
        axis.title.y = element_text(margin = margin(r = 10)),
        axis.text.x = element_text(margin = margin(t = 5)),
        strip.text = element_text(size = 10), 
        plot.margin = margin(10, 10, 10, 10),  
        panel.spacing = unit(1, "lines"),  
        legend.position = "right")

# Save the plot with an increased width
ggsave(paste(OUTPUT_FOLDR, "Fisher Score Faceted by Release Version.png", sep = ""), 
       plot = plot, bg = "white", width = 14, height = 6)  
plot







## Correlation Matrix of Attributes by Label
correlation_matrix <- df_combined %>%
  dplyr::select(-release_version) %>%
  cor(use = "complete.obs")

plot <- pheatmap(correlation_matrix, cluster_rows = TRUE, cluster_cols = TRUE)
ggsave(paste(OUTPUT_FOLDR, "Feature correlation heatmap.png", sep=""), plot = plot, 
       bg = "white", width = 10, height = 6)
plot

## Correlation between each feature and the label
numeric_df <- df_combined %>% select_if(is.numeric)
correlation_with_label <- cor(numeric_df, use = "complete.obs")

correlation_df <- data.frame(
  feature = rownames(correlation_with_label)[-which(colnames(correlation_with_label) %in% 
                                                      c("label", "release_version"))],
  correlation = correlation_with_label[, "label"][-which(colnames(correlation_with_label) %in% 
                                                           c("label", "release_version"))]
)

plot <- ggplot(correlation_df, aes(x = reorder(feature, correlation), y = correlation, fill = correlation)) +
  geom_bar(stat = "identity", width = 0.7) +
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", 
                       midpoint = 0, limits = c(-1, 1), na.value = "grey") +
  coord_flip() +
  ylim(-1, 1) +
  geom_text(aes(label = sprintf("%.3f", correlation)), 
            hjust = -0.55, vjust = 0.5, color = "black", size = 3) +
  theme_minimal() +
  labs(title = "Correlation of Features with Label", 
       x = "Features", y = "Correlation with Label") +
  theme(axis.text.y = element_text(size = 10), 
        axis.title.y = element_text(margin = margin(r = 10)), 
        axis.text.x = element_text(margin = margin(t = 5)))

ggsave(paste(OUTPUT_FOLDR, "Correlation of Features with Label.png", sep=""), plot = plot, 
       bg = "white", width = 10, height = 6)
plot

## Correlation plot for each release version
correlation_list <- list()

for (version in unique(df_combined$release_version)) {
  filtered_df <- df_combined %>% filter(release_version == version)
  filtered_numeric_df <- filtered_df %>% select_if(is.numeric)
  filtered_numeric_df <- filtered_numeric_df[, sapply(filtered_numeric_df, var, na.rm = TRUE) > 0]
  
  if (ncol(filtered_numeric_df) > 1) {
    correlation_with_label <- cor(filtered_numeric_df, use = "complete.obs")
    correlation_df_version <- data.frame(
      feature = rownames(correlation_with_label)[-which(colnames(correlation_with_label) == "label")],  
      correlation = correlation_with_label[, "label"][-which(colnames(correlation_with_label) == "label")]
    )
    
    if (nrow(correlation_df_version) > 0) {
      correlation_df_version$release_version <- version
      correlation_list[[as.character(version)]] <- correlation_df_version
    }
  } else {
    message(paste("Skipping version:", version, "due to insufficient data for correlation."))
  }
}

all_correlation_df <- do.call(rbind, Filter(Negate(is.null), correlation_list))

plot <- ggplot(all_correlation_df, aes(x = reorder(feature, correlation), y = correlation, fill = correlation)) +
  geom_bar(stat = "identity", width = 0.7) +
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", 
                       midpoint = 0, limits = c(-1, 1), na.value = "grey") +
  coord_flip() +
  ylim(-1, 1) +
  geom_text(aes(label = sprintf("%.3f", correlation)), 
            hjust = -0.55, vjust = 0.5, color = "black", size = 3) +
  theme_minimal() +
  labs(title = "Correlation of Features with Label by Release Version", 
       x = "Features", y = "Correlation with Label") +
  facet_wrap(~release_version) +
  theme(axis.text.y = element_text(size = 10), 
        axis.title.y = element_text(margin = margin(r = 10)), 
        axis.text.x = element_text(margin = margin(t = 5)))

ggsave(paste(OUTPUT_FOLDR, "Correlation of Features with Label by Release Version.png", sep=""), 
       plot = plot, bg = "white", width = 10, height = 6)
plot