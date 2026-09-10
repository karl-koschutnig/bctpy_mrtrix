#!/usr/bin/env Rscript
#' Generalized Additive Models for Network Metrics
#' ================================================
#'
#' This script fits Generalized Additive Models (GAMs) to network metrics
#' to model smooth changes across timepoints.
#'
#' Usage:
#'   Rscript gam_models.R --input-file metrics.csv --output-dir outputs/gam --timepoint-col session
#'
#' Author: TDD Implementation
#' Date: 2026-09-10


# ============================================================================
# LOAD LIBRARIES
# ============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(mgcv)
  library(jsonlite)
  library(argparse)
  library(R.matlab)  # For reading .mat files if needed
})

# Check if packages are available
if (!requireNamespace("mgcv", quietly = TRUE)) {
  stop("mgcv package required for GAM modeling. Install with: install.packages('mgcv')")
}


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parse_args <- function() {
  parser <- argparse::ArgumentParser(description = "GAM modeling for network metrics")
  
  parser$add_argument("--input-file", type = "character", required = TRUE,
                     help = "Input CSV file with network metrics")
  parser$add_argument("--output-dir", type = "character", default = "outputs/gam_results",
                     help = "Directory to save GAM results")
  parser$add_argument("--timepoint-col", type = "character", default = "session",
                     help = "Column name for timepoints")
  parser$add_argument("--group-col", type = "character", default = "group",
                     help = "Column name for groups")
  parser$add_argument("--config", type = "character", default = NULL,
                     help = "Path to configuration JSON file")
  parser$add_argument("--metric-cols", type = "character", nargs = "+", default = NULL,
                     help = "List of metric columns to model")
  
  args <- parser$parse_args()
  return(args)
}


# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

load_config <- function(config_path) {
  if (is.null(config_path)) {
    return(list())
  }
  
  if (!file.exists(config_path)) {
    stop(paste("Configuration file not found:", config_path))
  }
  
  config <- fromJSON(config_path)
  return(config)
}


# ============================================================================
# LOAD DATA
# ============================================================================

load_data <- function(input_file) {
  if (grepl("\\.csv$", input_file)) {
    data <- read_csv(input_file)
  } else if (grepl("\\.parquet$", input_file)) {
    if (!requireNamespace("arrow", quietly = TRUE)) {
      stop("arrow package required for parquet support")
    }
    data <- arrow::read_parquet(input_file)
  } else {
    stop(paste("Unsupported file format:", input_file))
  }
  
  return(data)
}


# ============================================================================
# DATA PREPARATION
# ============================================================================

prepare_data <- function(data, timepoint_col, group_col, metric_cols = NULL) {
  # Convert timepoint to factor
  data <- data %>%
    mutate(!!sym(timepoint_col) := factor(!!sym(timepoint_col)))
  
  # Convert group to factor
  data <- data %>%
    mutate(!!sym(group_col) := factor(!!sym(group_col)))
  
  # If metric_cols not specified, use all numeric columns except metadata
  if (is.null(metric_cols)) {
    metadata_cols <- c(timepoint_col, group_col, "participant_id", "age", "sex")
    metadata_cols <- metadata_cols[metadata_cols %in% names(data)]
    metric_cols <- setdiff(names(data)[sapply(data, is.numeric)], metadata_cols)
  }
  
  # Ensure metric_cols are numeric
  metric_cols <- metric_cols[sapply(data[metric_cols], is.numeric)]
  
  return(list(data = data, metric_cols = metric_cols))
}


# ============================================================================
# FIT GAM MODELS
# ============================================================================

#' Fit a single GAM model
#'
#' @param data Data frame with metrics and covariates
#' @param formula GAM formula
#' @param metric_name Name of the metric being modeled
#' @return List with model, summary, and predictions
fit_gam_model <- function(data, formula, metric_name) {
  # Fit GAM with REML estimation
  model <- mgcv::gam(formula, data = data, method = "REML")
  
  # Get summary
  summary <- summary(model)
  
  # Get predictions
  predictions <- predict(model, newdata = data, se.fit = TRUE)
  
  return(list(
    model = model,
    summary = summary,
    predictions = predictions$fit,
    se = predictions$se.fit,
    metric = metric_name
  ))
}


#' Fit GAM models for all metrics
#'
#' @param data Data frame with all metrics
#' @param timepoint_col Column name for timepoints
#' @param group_col Column name for groups
#' @param metric_cols Vector of metric column names
#' @return List of model results
fit_all_gam_models <- function(data, timepoint_col, group_col, metric_cols) {
  results <- list()
  
  for (metric in metric_cols) {
    # Create formula: metric ~ s(timepoint) + group + sex + age
    formula <- as.formula(paste(metric, "~ s(", timepoint_col, ") + ", 
                               group_col, " + sex + age", sep = ""))
    
    cat("Fitting GAM for:", metric, "...\n")
    
    tryCatch({
      model_result <- fit_gam_model(data, formula, metric)
      results[[metric]] <- model_result
      cat("  ✓ Model fitted successfully\n")
    }, error = function(e) {
      warning(paste("Failed to fit GAM for", metric, ":", e$message))
      results[[metric]] <- NULL
    })
  }
  
  return(results)
}


# ============================================================================
# EXTRACT RESULTS
# ============================================================================

#' Extract model statistics
#'
#' @param model_result Result from fit_gam_model
#' @return Data frame with model statistics
extract_model_stats <- function(model_result) {
  if (is.null(model_result)) {
    return(NULL)
  }
  
  model <- model_result$model
  summary <- model_result$summary
  metric <- model_result$metric
  
  # Extract coefficients
  coefs <- summary$coefficients
  
  # Extract R-sq (adjusted)
  r_sq <- summary$r.sq
  adj_r_sq <- summary$adj.r.sq
  
  # Extract F-statistic
  f_stat <- summary$f
  p_value <- summary$p
  
  # Extract smooth terms
  smooth_terms <- grep("s(", names(coefs), value = TRUE)
  
  stats <- data.frame(
    metric = metric,
    r_sq = r_sq,
    adj_r_sq = adj_r_sq,
    f_statistic = f_stat,
    p_value = p_value,
    edf = summary$edf,
    stringsAsFactors = FALSE
  )
  
  # Add smooth term statistics
  if (length(smooth_terms) > 0) {
    for (term in smooth_terms) {
      stats[paste0(term, "_est")] <- coefs[term]
      stats[paste0(term, "_se")] <- coefs[paste0(term, ".se")]
      stats[paste0(term, "_t")] <- coefs[paste0(term, ".t")]
    }
  }
  
  return(stats)
}


#' Extract predictions for visualization
#'
#' @param model_result Result from fit_gam_model
#' @param data Original data
#' @return Data frame with predictions
extract_predictions <- function(model_result, data) {
  if (is.null(model_result)) {
    return(NULL)
  }
  
  predictions <- model_result$predictions
  se <- model_result$se
  metric <- model_result$metric
  
  data$prediction <- predictions
  data$se <- se
  
  return(data)
}


# ============================================================================
# SAVE RESULTS
# ============================================================================

save_results <- function(results, output_dir, data, timepoint_col, group_col, metric_cols) {
  output_dir <- path.expand(output_dir)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Save model summaries
  all_stats <- list()
  all_predictions <- list()
  
  for (metric in metric_cols) {
    if (!is.null(results[[metric]])) {
      stats <- extract_model_stats(results[[metric]])
      if (!is.null(stats)) {
        all_stats[[metric]] <- stats
      }
      
      predictions <- extract_predictions(results[[metric]], data)
      if (!is.null(predictions)) {
        all_predictions[[metric]] <- predictions
      }
    }
  }
  
  # Combine all statistics
  if (length(all_stats) > 0) {
    stats_df <- do.call(rbind, all_stats)
    write_csv(stats_df, file.path(output_dir, "gam_model_statistics.csv"))
    write.csv(stats_df, file.path(output_dir, "gam_model_statistics.RData"))
  }
  
  # Save individual metric predictions
  for (metric in names(all_predictions)) {
    pred_df <- all_predictions[[metric]]
    write_csv(pred_df, file.path(output_dir, paste0("predictions_", metric, ".csv")))
  }
  
  # Save combined predictions
  if (length(all_predictions) > 0) {
    # Get all unique columns
    all_cols <- unique(unlist(lapply(all_predictions, names)))
    combined <- data.frame(matrix(nrow = nrow(data), ncol = 0))
    names(combined) <- character(0)
    
    for (metric in names(all_predictions)) {
      pred_df <- all_predictions[[metric]]
      for (col in names(pred_df)) {
        if (col %in% c("prediction", "se")) {
          col_name <- paste0(metric, "_", col)
          combined[[col_name]] <- pred_df[[col]]
        }
      }
    }
    
    write_csv(combined, file.path(output_dir, "all_predictions.csv"))
  }
  
  # Save R objects for later use
  saveRDS(results, file.path(output_dir, "gam_models.RDS"))
  
  cat("Results saved to:", output_dir, "\n")
}


# ============================================================================
# MAIN FUNCTION
# ============================================================================

main <- function() {
  args <- parse_args()
  
  # Load configuration
  config <- load_config(args$config)
  
  # Override args with config values
  if (!is.null(config$temporal) && !is.null(config$temporal$timepoint_col)) {
    args$timepoint_col <- config$temporal$timepoint_col
  }
  if (!is.null(config$temporal) && !is.null(config$temporal$group_col)) {
    args$group_col <- config$temporal$group_col
  }
  if (!is.null(config$gam) && !is.null(config$gam$smooth_terms)) {
    # Use config metric_cols if specified
    if (!is.null(args$metric_cols)) {
      args$metric_cols <- config$gam$metric_cols
    }
  }
  
  # Load data
  cat("Loading data from:", args$input_file, "\n")
  data <- load_data(args$input_file)
  cat("Loaded", nrow(data), "rows\n")
  
  # Prepare data
  prepared <- prepare_data(data, args$timepoint_col, args$group_col, args$metric_cols)
  data <- prepared$data
  metric_cols <- prepared$metric_cols
  
  cat("Metrics to model:", paste(metric_cols, collapse = ", "), "\n")
  
  # Fit GAM models
  results <- fit_all_gam_models(data, args$timepoint_col, args$group_col, metric_cols)
  
  # Save results
  save_results(results, args$output_dir, data, args$timepoint_col, args$group_col, metric_cols)
  
  # Print summary
  cat("\n=== GAM MODELING SUMMARY ===\n")
  cat("Timepoints:", length(levels(factor(data[[args$timepoint_col]]))), "\n")
  cat("Groups:", length(levels(factor(data[[args$group_col]]))), "\n")
  cat("Metrics modeled:", length(metric_cols), "\n")
  cat("Models fitted:", sum(sapply(results, function(x) !is.null(x))), "/", length(results), "\n")
  
  # Show significant smooth terms
  cat("\n=== SIGNIFICANT SMOOTH TERMS ===\n")
  for (metric in metric_cols) {
    if (!is.null(results[[metric]])) {
      summary <- results[[metric]]$summary
      if (summary$p < 0.05) {
        cat(metric, ": p =", round(summary$p, 4), "(R-sq =", round(summary$r.sq, 3), ")\n")
      }
    }
  }
}


# ============================================================================
# RUN MAIN
# ============================================================================

if (interactive()) {
  # Running interactively
  args <- list(
    input_file = "data/processed/metrics.csv",
    output_dir = "outputs/temporal_analysis/gam_results",
    timepoint_col = "session",
    group_col = "group"
  )
  
  # Load data
  data <- load_data(args$input_file)
  
  # Prepare data
  prepared <- prepare_data(data, args$timepoint_col, args$group_col)
  data <- prepared$data
  metric_cols <- prepared$metric_cols
  
  # Fit models
  results <- fit_all_gam_models(data, args$timepoint_col, args$group_col, metric_cols)
  
  # Save results
  save_results(results, args$output_dir, data, args$timepoint_col, args$group_col, metric_cols)
  
  cat("Done!\n")
} else {
  # Running from command line
  main()
}
