#!/usr/bin/env Rscript
# =============================================================================
# Create minimal sample from stkcd_year embeddings for testing/debugging
# 从 stkcd_year_embeddings 中抽取样本用于测试和调试
# =============================================================================

# Load required package
if (!require("data.table", quietly = TRUE)) {
  install.packages("data.table")
  library(data.table)
}

# Configuration - paths relative to sample directory
model_suffix <- "_minilm"
input_file <- sprintf("output/stkcd_year%s_embeddings.csv", model_suffix)
cit_input_file <- sprintf("output/stkcd_year_citweighted%s_embeddings.csv", model_suffix)
output_dir <- "sample/data"

# Create output directory
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Check if main output exists
if (!file.exists(input_file)) {
  stop(sprintf("Main embeddings file not found: %s\nPlease run the main pipeline first.", input_file))
}

# Read full data
message("Loading embeddings from main output...")
embeddings <- fread(input_file)

# =============================================================================
# Strategy: Select companies with different year spans for diverse testing
# =============================================================================

# Count years per company
year_counts <- embeddings[, .(n_years = .N, year_range = paste(min(p_year), max(p_year), sep = "-")), by = stkcd]
setorder(year_counts, -n_years)

# Select samples:
# 1. One company with many years (>=20) - for testing lag-3 and cumulative
# 2. One company with medium years (5-10) - for testing normal cases
# 3. One company with few years (2-3) - for testing edge cases (lag-3 = NA)
# 4. One company with only 1 year - for testing edge cases (all NA)

sample_stkcd <- c(
  year_counts[n_years >= 20, stkcd][1], # Many years
  year_counts[n_years >= 5 & n_years <= 10, stkcd][1], # Medium years
  year_counts[n_years >= 2 & n_years <= 3, stkcd][1], # Few years
  year_counts[n_years == 1, stkcd][1] # Single year
)

sample_stkcd <- sample_stkcd[!is.na(sample_stkcd)]
message(sprintf("Selected sample companies: %s", paste(sample_stkcd, collapse = ", ")))

# Extract sample data
sample_data <- embeddings[stkcd %in% sample_stkcd]
setorder(sample_data, stkcd, p_year)

# Save sample embeddings
sample_output <- file.path(output_dir, sprintf("sample%s_embeddings.csv", model_suffix))
fwrite(sample_data, sample_output)
message(sprintf("Sample embeddings saved to: %s (%d rows)", sample_output, nrow(sample_data)))

# Also save citation-weighted if exists
if (file.exists(cit_input_file)) {
  embeddings_cit <- fread(cit_input_file)
  sample_cit <- embeddings_cit[stkcd %in% sample_stkcd]
  setorder(sample_cit, stkcd, p_year)
  sample_cit_output <- file.path(output_dir, sprintf("sample_citweighted%s_embeddings.csv", model_suffix))
  fwrite(sample_cit, sample_cit_output)
  message(sprintf("Sample citation-weighted embeddings saved to: %s", sample_cit_output))
}

# Print sample structure
message("\n========== Sample Structure ==========")
for (s in sample_stkcd) {
  years <- sample_data[stkcd == s, p_year]
  message(sprintf("Company %s: %d years (%s)", s, length(years), paste(range(years), collapse = "-")))
}

message("\nSample creation complete!")
