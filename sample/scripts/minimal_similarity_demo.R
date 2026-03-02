#!/usr/bin/env Rscript
# =============================================================================
# Minimal Similarity Calculation Demo / 最小相似度计算示例
#
# This script demonstrates the core similarity calculation logic with a
# small sample dataset for easy understanding and debugging.
#
# 本脚本使用小型样本数据演示核心相似度计算逻辑，便于理解和调试。
# =============================================================================

# -----------------------------------------------------------------------------
# Part 1: Setup / 设置
# -----------------------------------------------------------------------------

if (!require("data.table", quietly = TRUE)) {
  install.packages("data.table")
  library(data.table)
}

# Safe cosine similarity with detailed logging
safe_cosine_similarity <- function(v1, v2, verbose = FALSE) {
  if (length(v1) == 0L || length(v2) == 0L) {
    if (verbose) message("  [Empty vector]")
    return(NA_real_)
  }
  if (length(v1) != length(v2)) {
    if (verbose) message("  [Dimension mismatch]")
    return(NA_real_)
  }

  v1 <- as.numeric(v1)
  v2 <- as.numeric(v2)

  if (any(!is.finite(v1)) || any(!is.finite(v2))) {
    if (verbose) message("  [Non-finite values]")
    return(NA_real_)
  }

  n1 <- sqrt(sum(v1 * v1))
  n2 <- sqrt(sum(v2 * v2))

  if (n1 <= 1e-12 || n2 <= 1e-12) {
    if (verbose) message(sprintf("  [Zero norm: n1=%.2e, n2=%.2e]", n1, n2))
    return(NA_real_)
  }

  sim <- sum(v1 * v2) / (n1 * n2)
  if (verbose) message(sprintf("  [OK] sim=%.4f", sim))
  return(sim)
}

# -----------------------------------------------------------------------------
# Part 2: Load Sample Data / 加载样本数据
# -----------------------------------------------------------------------------

sample_file <- "sample/data/sample_minilm_embeddings.csv"
if (!file.exists(sample_file)) {
  stop("Sample file not found. Run create_sample_embeddings.R first.")
}

data <- fread(sample_file)
embedding_cols <- grep("^emb_", names(data), value = TRUE)
message(sprintf("Loaded %d rows with %d embedding dimensions\n", nrow(data), length(embedding_cols)))

# -----------------------------------------------------------------------------
# Part 3: Demonstrate Calculation for One Company / 演示单个公司的计算
# -----------------------------------------------------------------------------

# Pick a company with multiple years for demonstration
demo_stkcd <- data[, .N, by = stkcd][N >= 5, stkcd][1]
demo_data <- data[stkcd == demo_stkcd][order(p_year)]

message(sprintf("Demo Company: %s (%d years)", demo_stkcd, nrow(demo_data)))
message(paste(rep("=", 60), collapse = ""))

# Extract embedding matrix
mat <- as.matrix(demo_data[, ..embedding_cols])
n <- nrow(mat)

# Storage for results
results <- data.table(
  p_year = demo_data$p_year,
  n_patents = demo_data$n_patents,
  cos_sim_lag1 = NA_real_,
  cos_sim_lag3 = NA_real_,
  cos_sim_cumulative = NA_real_
)

# -----------------------------------------------------------------------------
# Detailed Walkthrough / 详细计算过程
# -----------------------------------------------------------------------------

message("\n--- Lag-1 Similarity (vs previous year) ---")
for (i in 2:n) {
  message(sprintf(
    "\nYear %d (row %d) vs Year %d (row %d):",
    demo_data$p_year[i], i, demo_data$p_year[i - 1], i - 1
  ))
  results$cos_sim_lag1[i] <- safe_cosine_similarity(
    v1 = mat[i, ],
    v2 = mat[i - 1, ],
    verbose = TRUE
  )
}

message("\n--- Lag-3 Similarity (vs mean of previous 3 years) ---")
for (i in 4:n) {
  prev3_idx <- (i - 3):(i - 1)
  prev3_mean <- colMeans(mat[prev3_idx, , drop = FALSE])
  message(sprintf(
    "\nYear %d vs mean of years %s:",
    demo_data$p_year[i],
    paste(demo_data$p_year[prev3_idx], collapse = ", ")
  ))
  results$cos_sim_lag3[i] <- safe_cosine_similarity(
    v1 = mat[i, ],
    v2 = prev3_mean,
    verbose = TRUE
  )
}

message("\n--- Cumulative Similarity (vs mean of all previous years) ---")
for (i in 2:n) {
  prev_mean <- colMeans(mat[1:(i - 1), , drop = FALSE])
  message(sprintf(
    "\nYear %d vs mean of years %s:",
    demo_data$p_year[i],
    paste(demo_data$p_year[1:(i - 1)], collapse = ", ")
  ))
  results$cos_sim_cumulative[i] <- safe_cosine_similarity(
    v1 = mat[i, ],
    v2 = prev_mean,
    verbose = TRUE
  )
}

# -----------------------------------------------------------------------------
# Part 4: Results Summary / 结果汇总
# -----------------------------------------------------------------------------

message("\n")
message(paste(rep("=", 60), collapse = ""))
message("RESULTS SUMMARY / 结果汇总")
message(paste(rep("=", 60), collapse = ""))
print(results)

# Statistics
message("\n--- Statistics / 统计信息 ---")
stats <- data.table(
  metric = c("lag-1", "lag-3", "cumulative"),
  mean = c(
    mean(results$cos_sim_lag1, na.rm = TRUE),
    mean(results$cos_sim_lag3, na.rm = TRUE),
    mean(results$cos_sim_cumulative, na.rm = TRUE)
  ),
  sd = c(
    sd(results$cos_sim_lag1, na.rm = TRUE),
    sd(results$cos_sim_lag3, na.rm = TRUE),
    sd(results$cos_sim_cumulative, na.rm = TRUE)
  ),
  n_valid = c(
    sum(!is.na(results$cos_sim_lag1)),
    sum(!is.na(results$cos_sim_lag3)),
    sum(!is.na(results$cos_sim_cumulative))
  )
)
print(stats)

# -----------------------------------------------------------------------------
# Part 5: Edge Cases Demonstration / 边界情况演示
# -----------------------------------------------------------------------------

message("\n")
message(paste(rep("=", 60), collapse = ""))
message("EDGE CASES / 边界情况")
message(paste(rep("=", 60), collapse = ""))

# Show edge cases for different company sizes
edge_cases <- data[, .N, by = stkcd][order(N)]
message("\nCompanies by number of years:")
print(edge_cases)

# Single year company (all NAs)
single_year <- edge_cases[N == 1, stkcd][1]
if (!is.na(single_year)) {
  message(sprintf("\n--- Single Year Company (%s) ---", single_year))
  message("Expected: All similarities = NA (no previous years to compare)")
  single_data <- data[stkcd == single_year]
  message(sprintf("Years available: %s", paste(single_data$p_year, collapse = ", ")))
}

# Two year company (only lag-1 possible)
two_year <- edge_cases[N == 2, stkcd][1]
if (!is.na(two_year)) {
  message(sprintf("\n--- Two Year Company (%s) ---", two_year))
  message("Expected: lag-1 has value, lag-3 = NA, cumulative has value")
  two_data <- data[stkcd == two_year][order(p_year)]
  message(sprintf("Years available: %s", paste(two_data$p_year, collapse = ", ")))
}

# Three year company (only lag-1 and cumulative possible)
three_year <- edge_cases[N == 3, stkcd][1]
if (!is.na(three_year)) {
  message(sprintf("\n--- Three Year Company (%s) ---", three_year))
  message("Expected: lag-1 and cumulative have values, lag-3 = NA")
  three_data <- data[stkcd == three_year][order(p_year)]
  message(sprintf("Years available: %s", paste(three_data$p_year, collapse = ", ")))
}

message("\n========== Demo Complete ==========")
