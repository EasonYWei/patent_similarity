#!/usr/bin/env Rscript
# =============================================================================
# Calculate Similarity for Sample Data / 计算样本数据的相似度
#
# This script computes lag-1, lag-3 and cumulative cosine similarities
# for sample firm-year embeddings.
# =============================================================================

# Load required package
if (!require("data.table", quietly = TRUE)) {
  install.packages("data.table")
  library(data.table)
}

SAFE_COSINE_TOLERANCE <- 1e-12

safe_cosine_similarity <- function(v1, v2, tolerance = SAFE_COSINE_TOLERANCE) {
  if (length(v1) == 0L || length(v2) == 0L) {
    return(NA_real_)
  }
  if (length(v1) != length(v2)) {
    return(NA_real_)
  }
  v1 <- as.numeric(v1)
  v2 <- as.numeric(v2)
  if (any(!is.finite(v1)) || any(!is.finite(v2))) {
    return(NA_real_)
  }
  n1 <- sqrt(sum(v1 * v1))
  n2 <- sqrt(sum(v2 * v2))
  if (n1 <= tolerance || n2 <= tolerance) {
    return(NA_real_)
  }
  sum(v1 * v2) / (n1 * n2)
}

rolling_mean <- function(mat, rows) {
  if (length(rows) == 0L) {
    return(rep(NA_real_, ncol(mat)))
  }
  colMeans(mat[rows, , drop = FALSE], na.rm = FALSE)
}

assert_valid_embeddings <- function(data, embedding_cols, file_path) {
  required <- c("stkcd", "p_year", "n_patents", "n_texts_used")
  missing_required <- setdiff(required, names(data))
  if (length(missing_required) > 0L) {
    stop(sprintf("%s missing required columns: %s", file_path, paste(missing_required, collapse = ", ")))
  }
  if (length(embedding_cols) == 0L) {
    stop(sprintf("%s has no embedding columns matching ^emb_", file_path))
  }
  nonnumeric <- names(data[, ..embedding_cols])[!vapply(data[, ..embedding_cols], is.numeric, logical(1L))]
  if (length(nonnumeric) > 0L) {
    stop(sprintf("%s embedding columns contain non-numeric values", file_path))
  }
}

calculate_similarities <- function(data, embedding_cols) {
  assert_valid_embeddings(data, embedding_cols, deparse(substitute(data)))
  data[, (embedding_cols) := lapply(.SD, as.numeric), .SDcols = embedding_cols]

  by_stkcd <- function(stkcd_vec, p_year_vec, n_patents_vec, n_texts_used_vec, mat) {
    n <- nrow(mat)
    cos_sim_lag1 <- rep(NA_real_, n)
    cos_sim_lag3 <- rep(NA_real_, n)
    cos_sim_cumulative <- rep(NA_real_, n)

    if (n >= 2L) {
      for (i in 2L:n) {
        cos_sim_lag1[i] <- safe_cosine_similarity(v1 = mat[i, ], v2 = mat[i - 1, ])
        prev_mean <- rolling_mean(mat, seq_len(i - 1L))
        cos_sim_cumulative[i] <- safe_cosine_similarity(v1 = mat[i, ], v2 = prev_mean)
      }
    }

    if (n >= 4L) {
      for (i in 4L:n) {
        prev3_mean <- rolling_mean(mat, (i - 3L):(i - 1L))
        cos_sim_lag3[i] <- safe_cosine_similarity(v1 = mat[i, ], v2 = prev3_mean)
      }
    }

    data.table(
      p_year = p_year_vec,
      n_patents = n_patents_vec,
      n_texts_used = n_texts_used_vec,
      cos_sim_lag1 = cos_sim_lag1,
      cos_sim_lag3 = cos_sim_lag3,
      cos_sim_cumulative = cos_sim_cumulative
    )
  }

  result <- data[
    ,
    by_stkcd(
      stkcd_vec = stkcd,
      p_year_vec = p_year,
      n_patents_vec = n_patents,
      n_texts_used_vec = n_texts_used,
      mat = as.matrix(.SD)
    ),
    by = stkcd,
    .SDcols = embedding_cols
  ]
  return(result[])
}

# =============================================================================
# Main execution
# =============================================================================

model_suffix <- "_minilm"
data_dir <- "sample/data"
output_dir <- "sample/output"

# Create output directory if needed
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

simple_input <- file.path(data_dir, sprintf("sample%s_embeddings.csv", model_suffix))
cit_input <- file.path(data_dir, sprintf("sample_citweighted%s_embeddings.csv", model_suffix))
simple_output <- file.path(output_dir, sprintf("sample_similarity%s.csv", model_suffix))
cit_output <- file.path(output_dir, sprintf("sample_similarity_citweighted%s.csv", model_suffix))
merged_output <- file.path(output_dir, sprintf("sample_similarity_merged%s.csv", model_suffix))

if (!file.exists(simple_input)) {
  stop(sprintf("Sample file not found: %s\nPlease run create_sample_embeddings.R first.", simple_input))
}

message("Loading sample embeddings...")
embeddings <- fread(simple_input)
embedding_cols <- grep("^emb_", names(embeddings), value = TRUE)
assert_valid_embeddings(embeddings, embedding_cols, simple_input)

ordered_check <- embeddings[order(stkcd, p_year), .(stkcd, p_year)]
is_sorted_like <- identical(embeddings[, .(stkcd, p_year)], ordered_check)
setorder(embeddings, stkcd, p_year)
if (!is_sorted_like) {
  warning("Input files are not ordered by stkcd, p_year; results are based on sorted order.")
}

message(sprintf("Loaded sample embeddings: %d rows, %d dimensions", nrow(embeddings), length(embedding_cols)))

message("\nComputing simple similarities...")
result_simple <- calculate_similarities(embeddings, embedding_cols)
fwrite(result_simple, simple_output)
message(sprintf("Simple similarities saved to: %s", simple_output))

if (file.exists(cit_input)) {
  message("Citation-weighted embeddings found. Computing weighted similarities...")
  embeddings_cit <- fread(cit_input)
  embedding_cols_cit <- grep("^emb_", names(embeddings_cit), value = TRUE)
  assert_valid_embeddings(embeddings_cit, embedding_cols_cit, cit_input)
  setorder(embeddings_cit, stkcd, p_year)

  result_cit <- calculate_similarities(embeddings_cit, embedding_cols_cit)
  setnames(
    result_cit,
    c("cos_sim_lag1", "cos_sim_lag3", "cos_sim_cumulative"),
    c("cos_sim_lag1_citw", "cos_sim_lag3_citw", "cos_sim_cumulative_citw")
  )
  fwrite(result_cit, cit_output)
  message(sprintf("Citation-weighted similarities saved to: %s", cit_output))

  merged <- merge(
    result_simple,
    result_cit[, .(stkcd, p_year, cos_sim_lag1_citw, cos_sim_lag3_citw, cos_sim_cumulative_citw)],
    by = c("stkcd", "p_year"),
    all = TRUE,
    suffixes = c("", "_cit")
  )
  fwrite(merged, merged_output)
  message(sprintf("Merged output written to: %s", merged_output))
}

message("\n========== Similarity Summary ==========")
message(sprintf(
  "Simple: Lag-1 mean=%.4f sd=%.4f n=%d",
  mean(result_simple$cos_sim_lag1, na.rm = TRUE),
  sd(result_simple$cos_sim_lag1, na.rm = TRUE),
  sum(!is.na(result_simple$cos_sim_lag1))
))
message(sprintf(
  "Simple: Lag-3 mean=%.4f sd=%.4f n=%d",
  mean(result_simple$cos_sim_lag3, na.rm = TRUE),
  sd(result_simple$cos_sim_lag3, na.rm = TRUE),
  sum(!is.na(result_simple$cos_sim_lag3))
))
message(sprintf(
  "Simple: Cumulative mean=%.4f sd=%.4f n=%d",
  mean(result_simple$cos_sim_cumulative, na.rm = TRUE),
  sd(result_simple$cos_sim_cumulative, na.rm = TRUE),
  sum(!is.na(result_simple$cos_sim_cumulative))
))

if (file.exists(cit_input)) {
  message(sprintf(
    "Cit-wtd: Lag-1 mean=%.4f sd=%.4f n=%d",
    mean(result_cit$cos_sim_lag1_citw, na.rm = TRUE),
    sd(result_cit$cos_sim_lag1_citw, na.rm = TRUE),
    sum(!is.na(result_cit$cos_sim_lag1_citw))
  ))
  message(sprintf(
    "Cit-wtd: Lag-3 mean=%.4f sd=%.4f n=%d",
    mean(result_cit$cos_sim_lag3_citw, na.rm = TRUE),
    sd(result_cit$cos_sim_lag3_citw, na.rm = TRUE),
    sum(!is.na(result_cit$cos_sim_lag3_citw))
  ))
  message(sprintf(
    "Cit-wtd: Cumulative mean=%.4f sd=%.4f n=%d",
    mean(result_cit$cos_sim_cumulative_citw, na.rm = TRUE),
    sd(result_cit$cos_sim_cumulative_citw, na.rm = TRUE),
    sum(!is.na(result_cit$cos_sim_cumulative_citw))
  ))
}

message("\nRows where a vector is zero/NA/Inf or history is short are emitted as NA by design.")
message("Undefined cosine similarities are intentionally not coerced to 0.")
message("\n========== Done ==========")
