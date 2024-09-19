library(ggplot2)
library(reshape2)

# Specify the path to the zip file and the name of the file within the zip
zipfile <- "Data/promise-2_0a-packages-csv.zip"

# List all files in the zip archive
files_in_zip <- unzip(zipfile, list = TRUE)

# Filter only CSV files (assuming the extension is '.csv')
csv_files <- files_in_zip$Name[grep("\\.csv$", files_in_zip$Name)]

# Initialize an empty list to store each data frame
df_list <- list()

# Define the columns to keep
columns_to_keep <- c(
  'ACD_avg', 'ACD_max', 'ACD_sum', 'FOUT_avg', 'FOUT_max', 'FOUT_sum',
  'MLOC_avg', 'MLOC_max', 'MLOC_sum', 'NBD_avg', 'NBD_max', 'NBD_sum',
  'NOCU', 'NOF_avg', 'NOF_max', 'NOF_sum',
  'NOI_avg', 'NOI_max', 'NOI_sum', 'NOM_avg', 'NOM_max', 'NOM_sum',
  'NOT_avg', 'NOT_max', 'NOT_sum', 'NSF_avg', 'NSF_max', 'NSF_sum',
  'NSM_avg', 'NSM_max', 'NSM_sum', 'PAR_avg', 'PAR_max', 'PAR_sum',
  'pre', 'TLOC_avg', 'TLOC_max', 'TLOC_sum', 'VG_avg', 'VG_max', 'VG_sum'
)

# Loop through each CSV file in the zip
for (csv_file in csv_files) {
  # Read the CSV file
  temp_df <- read.csv(unz(zipfile, csv_file), sep = ";", stringsAsFactors = FALSE)
  
  
  # Keep only the specified columns
  temp_df <- temp_df[, columns_to_keep]
  
  # Drop non-numeric columns
  temp_df <- temp_df[, sapply(temp_df, is.numeric)]
  
  # Add a new column with the file name (without the ".csv" extension)
  temp_df$source_file <- gsub("\\.csv$", "", csv_file)
  
  # Append the data frame to the list
  df_list[[csv_file]] <- temp_df
}

# Combine all data frames into one
combined_df <- do.call(rbind, df_list)

# View the combined data frame
head(combined_df)

# Exclude the 'source_file' column
all_columns_except_source <- setdiff(colnames(combined_df), "source_file")

# Calculate variance for all columns except 'source_file' by grouping with 'source_file'
variance_df <- aggregate(. ~ source_file, combined_df[, c(all_columns_except_source, "source_file")], var)

# Assuming combined_df is your data frame and source_file is the column to group by
# Calculate variance for each attribute
variance_df <- aggregate(. ~ source_file, combined_df[, c(all_columns_except_source, "source_file")], var)

# Remove the 'source_file' column from variance_df for further processing
variances <- variance_df[, -which(names(variance_df) == "source_file")]

# Compute the mean variance for each attribute
mean_variances <- colMeans(variances, na.rm = TRUE)

# Get the indices of top 5 highest and lowest variance attributes
top5_indices <- order(mean_variances, decreasing = TRUE)[1:10]
bottom5_indices <- order(mean_variances, decreasing = FALSE)[1:10]

# Get the names of these attributes
top5_attributes <- names(mean_variances)[top5_indices]
bottom5_attributes <- names(mean_variances)[bottom5_indices]

# Create a filtered data frame with only the interesting attributes
filtered_df <- combined_df[, c(top5_attributes, "source_file")]

# Optionally, you can compute variances again for the filtered data frame
filtered_variances_df <- aggregate(. ~ source_file, filtered_df, var)

# Convert to long format for easier plotting
variance_long <- reshape2::melt(filtered_df, id.vars = "source_file")

# Plot the variance for each attribute across different 'source_file'
ggplot(variance_long, aes(x = variable, y = value, fill = source_file)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Top 10 Most Variance Attributes by Source File", x = "Attributes", y = "Variance")


# Create a filtered data frame with only the interesting attributes
filtered_df_btm <- combined_df[, c(bottom5_attributes, "source_file")]

# Optionally, you can compute variances again for the filtered data frame
filtered_variances_df_btm <- aggregate(. ~ source_file, filtered_df_btm, var)

# Convert to long format for easier plotting
variance_long_btm <- reshape2::melt(filtered_df_btm, id.vars = "source_file")

# Plot the variance for each attribute across different 'source_file'
ggplot(variance_long_btm, aes(x = variable, y = value, fill = source_file)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Top 10 Least Variance Attributes by Source File", x = "Attributes", y = "Variance")


# Select some important numeric columns for correlation analysis
selected_columns <- combined_df[, c("ACD_avg", "FOUT_avg", "MLOC_avg", "VG_avg", "source_file")]

# Calculate correlation for these columns
cor_matrix <- cor(selected_columns[ , -ncol(selected_columns)], use = "complete.obs")

# Convert correlation matrix to long format
cor_long <- melt(cor_matrix)

# Plot heatmap
ggplot(cor_long, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1)) +
  theme_minimal() +
  labs(title = "Correlation Heatmap", x = "Attribute 1", y = "Attribute 2")

library(ggplot2)

# Subset numeric columns for clustering
subset_df <- combined_df[, c("MLOC_sum", "FOUT_sum")]

# Apply K-means clustering
set.seed(123)
kmeans_result <- kmeans(subset_df, centers = 3)

# Add cluster results to data frame
combined_df$cluster <- as.factor(kmeans_result$cluster)

# Plot clusters
ggplot(combined_df, aes(x = MLOC_sum, y = FOUT_sum, color = cluster)) +
  geom_point() +
  labs(title = "K-means Clustering on MLOC_sum and FOUT_sum", x = "MLOC_sum", y = "FOUT_sum")


