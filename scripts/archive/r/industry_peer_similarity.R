#!/usr/bin/env Rscript
# =============================================================================
# Industry Peer Similarity Calculation
# Computes cosine similarity between a firm and its industry peers
# from t-1, t-2, t-3 years
# =============================================================================

pacman::p_load(data.table, readxl)

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

calculate_industry_peer_similarity <- function(embeddings_data, embedding_cols) {
  # Ensure data is sorted
  setorder(embeddings_data, Ind, stkcd, p_year)
  
  # Get unique industry-year combinations
  embeddings_data[, industry_year := paste(Ind, p_year, sep = "_")]
  
  # Function to calculate max similarity with peers for a single row
  calc_peer_sim <- function(stkcd_val, year_val, ind_val, emb_vec, all_data) {
    results <- list(
      n_peers_t1 = 0L, n_peers_t2 = 0L, n_peers_t3 = 0L,
      peer_sim_t1 = NA_real_, peer_sim_t2 = NA_real_, peer_sim_t3 = NA_real_
    )
    
    # Get peers from same industry, excluding self
    for (lag in 1:3) {
      target_year <- year_val - lag
      peers <- all_data[Ind == ind_val & p_year == target_year & stkcd != stkcd_val]
      
      if (nrow(peers) == 0) {
        next
      }
      
      peer_mat <- as.matrix(peers[, ..embedding_cols])
      n_peers <- nrow(peer_mat)
      
      # Calculate similarity with each peer and take maximum
      sims <- sapply(1:n_peers, function(i) {
        safe_cosine_similarity(emb_vec, peer_mat[i, ])
      })
      
      max_sim <- max(sims, na.rm = TRUE)
      if (!is.finite(max_sim)) {
        max_sim <- NA_real_
      }
      
      if (lag == 1) {
        results$n_peers_t1 <- n_peers
        results$peer_sim_t1 <- max_sim
      } else if (lag == 2) {
        results$n_peers_t2 <- n_peers
        results$peer_sim_t2 <- max_sim
      } else {
        results$n_peers_t3 <- n_peers
        results$peer_sim_t3 <- max_sim
      }
    }
    
    return(results)
  }
  
  # Process by industry for efficiency
  industries <- unique(embeddings_data$Ind)
  all_results <- list()
  
  for (ind in industries) {
    ind_data <- embeddings_data[Ind == ind]
    if (nrow(ind_data) == 0) next
    
    # Get unique companies in this industry
    companies <- unique(ind_data$stkcd)
    n_companies <- length(companies)
    
    if (n_companies < 2) {
      # Only one company in industry, all peer similarities are NA
      ind_data[, n_peers_t1 := 0L]
      ind_data[, n_peers_t2 := 0L]
      ind_data[, n_peers_t3 := 0L]
      ind_data[, peer_sim_t1 := NA_real_]
      ind_data[, peer_sim_t2 := NA_real_]
      ind_data[, peer_sim_t3 := NA_real_]
      all_results[[length(all_results) + 1]] <- ind_data
      next
    }
    
    # Process each row
    n_rows <- nrow(ind_data)
    n_peers_t1_vec <- integer(n_rows)
    n_peers_t2_vec <- integer(n_rows)
    n_peers_t3_vec <- integer(n_rows)
    peer_sim_t1_vec <- numeric(n_rows)
    peer_sim_t2_vec <- numeric(n_rows)
    peer_sim_t3_vec <- numeric(n_rows)
    
    # Convert to matrix for faster access
    emb_matrix <- as.matrix(ind_data[, ..embedding_cols])
    stkcd_vec <- ind_data$stkcd
    year_vec <- ind_data$p_year
    
    for (i in 1:n_rows) {
      results <- calc_peer_sim(
        stkcd_vec[i], 
        year_vec[i], 
        ind, 
        emb_matrix[i, ],
        ind_data
      )
      
      n_peers_t1_vec[i] <- results$n_peers_t1
      n_peers_t2_vec[i] <- results$n_peers_t2
      n_peers_t3_vec[i] <- results$n_peers_t3
      peer_sim_t1_vec[i] <- results$peer_sim_t1
      peer_sim_t2_vec[i] <- results$peer_sim_t2
      peer_sim_t3_vec[i] <- results$peer_sim_t3
    }
    
    ind_data[, n_peers_t1 := n_peers_t1_vec]
    ind_data[, n_peers_t2 := n_peers_t2_vec]
    ind_data[, n_peers_t3 := n_peers_t3_vec]
    ind_data[, peer_sim_t1 := peer_sim_t1_vec]
    ind_data[, peer_sim_t2 := peer_sim_t2_vec]
    ind_data[, peer_sim_t3 := peer_sim_t3_vec]
    
    all_results[[length(all_results) + 1]] <- ind_data
    
    message(sprintf("  Processed industry %s: %d firms, %d firm-years", 
                    ind, n_companies, n_rows))
  }
  
  result <- rbindlist(all_results, use.names = TRUE)
  result[, industry_year := NULL]
  
  return(result[])
}

process_model <- function(model_suffix, industry_info) {
  simple_input <- sprintf("./output/stkcd_year%s_embeddings.csv", model_suffix)
  cit_input <- sprintf("./output/stkcd_year_citweighted%s_embeddings.csv", model_suffix)
  simple_output <- sprintf("./output/industry_peer_similarity%s.csv", model_suffix)
  cit_output <- sprintf("./output/industry_peer_similarity_citweighted%s.csv", model_suffix)
  merged_output <- sprintf("./output/industry_peer_similarity_merged%s.csv", model_suffix)
  
  if (!file.exists(simple_input)) {
    stop(sprintf("Input file not found: %s", simple_input))
  }
  
  message(sprintf("\n========== Processing model: %s ==========", model_suffix))
  
  # Load and merge with industry info
  message("Loading embeddings and merging with industry info...")
  embeddings <- fread(simple_input)
  
  # Merge with industry info (note: stkcd_info uses 'year', embeddings use 'p_year')
  embeddings <- merge(
    embeddings, 
    industry_info[, .(stkcd, year, Ind)], 
    by.x = c("stkcd", "p_year"), 
    by.y = c("stkcd", "year"),
    all.x = TRUE
  )
  
  # Check for missing industry info
  missing_ind <- sum(is.na(embeddings$Ind))
  if (missing_ind > 0) {
    warning(sprintf("%d rows missing industry info, will be excluded", missing_ind))
    embeddings <- embeddings[!is.na(Ind)]
  }
  
  embedding_cols <- grep("^emb_", names(embeddings), value = TRUE)
  message(sprintf("Loaded: %d rows, %d dimensions, %d unique industries", 
                  nrow(embeddings), length(embedding_cols), 
                  length(unique(embeddings$Ind))))
  
  # Calculate simple peer similarities
  message("Calculating simple peer similarities...")
  result_simple <- calculate_industry_peer_similarity(embeddings, embedding_cols)
  
  # Select output columns
  output_cols <- c("stkcd", "p_year", "Ind", "n_patents", "n_texts_used",
                   "n_peers_t1", "n_peers_t2", "n_peers_t3",
                   "peer_sim_t1", "peer_sim_t2", "peer_sim_t3")
  result_simple <- result_simple[, ..output_cols]
  fwrite(result_simple, simple_output)
  message(sprintf("Simple peer similarity written to: %s", simple_output))
  
  # Process citation-weighted if available
  if (file.exists(cit_input)) {
    message("Processing citation-weighted embeddings...")
    embeddings_cit <- fread(cit_input)
    
    embeddings_cit <- merge(
      embeddings_cit, 
      industry_info[, .(stkcd, year, Ind)], 
      by.x = c("stkcd", "p_year"), 
      by.y = c("stkcd", "year"),
      all.x = TRUE
    )
    
    embeddings_cit <- embeddings_cit[!is.na(Ind)]
    embedding_cols_cit <- grep("^emb_", names(embeddings_cit), value = TRUE)
    
    message("Calculating citation-weighted peer similarities...")
    result_cit <- calculate_industry_peer_similarity(embeddings_cit, embedding_cols_cit)
    
    # Rename columns
    setnames(result_cit, 
             c("n_peers_t1", "n_peers_t2", "n_peers_t3",
               "peer_sim_t1", "peer_sim_t2", "peer_sim_t3"),
             c("n_peers_t1_citw", "n_peers_t2_citw", "n_peers_t3_citw",
               "peer_sim_t1_citw", "peer_sim_t2_citw", "peer_sim_t3_citw"))
    
    result_cit <- result_cit[, c("stkcd", "p_year", "Ind",
                                  "n_peers_t1_citw", "n_peers_t2_citw", "n_peers_t3_citw",
                                  "peer_sim_t1_citw", "peer_sim_t2_citw", "peer_sim_t3_citw"),
                              with = FALSE]
    fwrite(result_cit, cit_output)
    message(sprintf("Citation-weighted peer similarity written to: %s", cit_output))
    
    # Merge simple and weighted results
    merged <- merge(
      result_simple,
      result_cit,
      by = c("stkcd", "p_year", "Ind"),
      all = TRUE
    )
    fwrite(merged, merged_output)
    message(sprintf("Merged output written to: %s", merged_output))
  }
  
  # Print summary
  message("\n---------- Summary ----------")
  message(sprintf("Simple peer_sim_t1: mean=%.4f, sd=%.4f, n=%d",
                  mean(result_simple$peer_sim_t1, na.rm = TRUE),
                  sd(result_simple$peer_sim_t1, na.rm = TRUE),
                  sum(!is.na(result_simple$peer_sim_t1))))
  message(sprintf("Simple peer_sim_t2: mean=%.4f, sd=%.4f, n=%d",
                  mean(result_simple$peer_sim_t2, na.rm = TRUE),
                  sd(result_simple$peer_sim_t2, na.rm = TRUE),
                  sum(!is.na(result_simple$peer_sim_t2))))
  message(sprintf("Simple peer_sim_t3: mean=%.4f, sd=%.4f, n=%d",
                  mean(result_simple$peer_sim_t3, na.rm = TRUE),
                  sd(result_simple$peer_sim_t3, na.rm = TRUE),
                  sum(!is.na(result_simple$peer_sim_t3))))
}

# =============================================================================
# Main execution
# =============================================================================

message("Loading industry info from data/stkcd_info.xlsx...")
industry_info <- as.data.table(read_excel("./data/stkcd_info.xlsx"))

# Rename columns to lowercase except Ind
setnames(industry_info, tolower(names(industry_info)))
setnames(industry_info, "ind", "Ind")  # Keep Ind as uppercase

# Ensure stkcd and year are integer to match embeddings format
industry_info[, stkcd := as.integer(stkcd)]
industry_info[, year := as.integer(year)]

message(sprintf("Loaded industry info: %d rows, %d unique industries", 
                nrow(industry_info), length(unique(industry_info$Ind))))

# Process both models
process_model("_minilm", industry_info)
process_model("_distiluse", industry_info)

message("\n========== All Done ==========")
