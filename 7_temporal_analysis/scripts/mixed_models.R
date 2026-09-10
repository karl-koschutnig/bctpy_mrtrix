#!/usr/bin/env Rscript
#' Linear Mixed-Effects Models for Network Metrics
#' ================================================
#'
#' Fits Group x Time linear mixed-effects models (subject random
#' intercept) to network metrics from a repeated-measures intervention
#' study, and computes pairwise group contrasts per timepoint via
#' emmeans.
#'
#' Replaces the age-based GAM approach used by the source
#' lifespan-turning-points pipeline, which requires a continuous
#' covariate with many distinct values and does not apply to a
#' 3-timepoint design.
#'
#' Usage:
#'   Rscript mixed_models.R --input-file metrics.csv --output-dir outputs/mixed_model_results \
#'     --timepoint-col session --group-col group --participant-col participant_id

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(jsonlite)
  library(argparse)
})

if (!requireNamespace("lme4", quietly = TRUE)) {
  stop("lme4 package required for mixed-effects modeling. Install with: install.packages('lme4')")
}


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parse_args <- function() {
  parser <- argparse::ArgumentParser(description = "Group x Time mixed-effects modeling for network metrics")

  parser$add_argument("--input-file", type = "character", required = TRUE,
                     help = "Input CSV file with network metrics")
  parser$add_argument("--output-dir", type = "character", default = "outputs/mixed_model_results",
                     help = "Directory to save model results")
  parser$add_argument("--timepoint-col", type = "character", default = "session",
                     help = "Column name for timepoints")
  parser$add_argument("--group-col", type = "character", default = "group",
                     help = "Column name for groups")
  parser$add_argument("--participant-col", type = "character", default = "participant_id",
                     help = "Column name for the subject identifier (random effect grouping)")
  parser$add_argument("--config", type = "character", default = NULL,
                     help = "Path to configuration JSON file")
  parser$add_argument("--metric-cols", type = "character", nargs = "+", default = NULL,
                     help = "List of metric columns to model")

  parser$parse_args()
}


# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

load_config <- function(config_path) {
  if (is.null(config_path)) return(list())
  if (!file.exists(config_path)) stop(paste("Configuration file not found:", config_path))
  fromJSON(config_path)
}


# ============================================================================
# LOAD DATA
# ============================================================================

load_data <- function(input_file) {
  if (grepl("\\.csv$", input_file)) {
    data <- read_csv(input_file, show_col_types = FALSE)
  } else if (grepl("\\.parquet$", input_file)) {
    if (!requireNamespace("arrow", quietly = TRUE)) {
      stop("arrow package required for parquet support")
    }
    data <- arrow::read_parquet(input_file)
  } else {
    stop(paste("Unsupported file format:", input_file))
  }
  data
}


# ============================================================================
# DATA PREPARATION
# ============================================================================

prepare_data <- function(data, timepoint_col, group_col, participant_col, metric_cols = NULL) {
  data <- data %>%
    mutate(
      !!sym(timepoint_col)   := factor(!!sym(timepoint_col)),
      !!sym(group_col)       := factor(!!sym(group_col)),
      !!sym(participant_col) := factor(!!sym(participant_col))
    )

  if (is.null(metric_cols)) {
    metadata_cols <- c(timepoint_col, group_col, participant_col, "age", "sex")
    metadata_cols <- metadata_cols[metadata_cols %in% names(data)]
    metric_cols <- setdiff(names(data)[sapply(data, is.numeric)], metadata_cols)
  }
  metric_cols <- metric_cols[sapply(data[metric_cols], is.numeric)]

  list(data = data, metric_cols = metric_cols)
}


# ============================================================================
# FIT MIXED-EFFECTS MODELS
# ============================================================================

#' Fit metric ~ group * session + (1 | participant) for one metric.
fit_mixed_model <- function(data, group_col, timepoint_col, participant_col, metric_name) {
  formula <- as.formula(
    paste0(metric_name, " ~ ", group_col, " * ", timepoint_col, " + (1 | ", participant_col, ")")
  )
  model <- lmerTest::lmer(formula, data = data)
  contrasts <- emmeans::emmeans(
    model, specs = as.formula(paste0("pairwise ~ ", group_col, " | ", timepoint_col))
  )

  list(model = model, anova = anova(model), contrasts = contrasts, metric = metric_name)
}


#' Fit mixed models for all metrics, skipping (with a warning) any that fail to converge
#' or that don't have enough non-missing data to support a group x session model.
fit_all_mixed_models <- function(data, group_col, timepoint_col, participant_col, metric_cols,
                                  min_obs_per_cell = 2) {
  results <- list()

  # Minimum floor: a couple of observations per group x session cell, so
  # lmer()/emmeans() aren't asked to fit/contrast an empty or near-empty cell.
  n_cells <- max(1, nlevels(data[[group_col]]) * nlevels(data[[timepoint_col]]))
  min_obs <- n_cells * min_obs_per_cell

  for (metric in metric_cols) {
    required_cols <- c(metric, group_col, timepoint_col, participant_col)
    n_complete <- sum(complete.cases(data[required_cols]))

    if (n_complete < min_obs) {
      msg <- paste0(
        "Skipping ", metric, ": only ", n_complete, "/", nrow(data),
        " observations non-NaN after removing missing data, insufficient for ",
        "group x session model (need >= ", min_obs, ")"
      )
      cat(msg, "\n")
      warning(msg)
      next
    }

    cat("Fitting mixed model for:", metric, "...\n")
    result <- tryCatch(
      fit_mixed_model(data, group_col, timepoint_col, participant_col, metric),
      error = function(e) {
        warning(paste("Failed to fit mixed model for", metric, ":", e$message))
        NULL
      }
    )
    if (!is.null(result)) {
      results[[metric]] <- result
      cat("  ✓ Model fitted successfully\n")
    }
  }
  results
}


# ============================================================================
# EXTRACT RESULTS
# ============================================================================

#' Extract the Type III ANOVA table (F, df, p) for group, session, group:session.
extract_model_stats <- function(model_result) {
  if (is.null(model_result)) return(NULL)

  anova_tbl <- as.data.frame(model_result$anova)
  anova_tbl$term <- rownames(anova_tbl)
  anova_tbl$metric <- model_result$metric
  rownames(anova_tbl) <- NULL
  anova_tbl
}


#' Extract pairwise group contrasts per timepoint as a tidy data frame.
extract_contrasts <- function(model_result) {
  if (is.null(model_result)) return(NULL)

  contrasts_df <- as.data.frame(model_result$contrasts$contrasts)
  contrasts_df$metric <- model_result$metric
  contrasts_df
}


# ============================================================================
# SAVE RESULTS
# ============================================================================

save_results <- function(results, output_dir, metric_cols) {
  output_dir <- path.expand(output_dir)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  all_stats <- list()
  all_contrasts <- list()

  for (metric in metric_cols) {
    if (!is.null(results[[metric]])) {
      stats <- tryCatch(
        extract_model_stats(results[[metric]]),
        error = function(e) {
          warning(paste("Failed to extract model statistics for", metric, ":", e$message))
          NULL
        }
      )
      if (!is.null(stats)) all_stats[[metric]] <- stats

      contrasts <- tryCatch(
        extract_contrasts(results[[metric]]),
        error = function(e) {
          warning(paste("Failed to extract contrasts for", metric, ":", e$message))
          NULL
        }
      )
      if (!is.null(contrasts)) all_contrasts[[metric]] <- contrasts
    }
  }

  if (length(all_stats) > 0) {
    write_csv(bind_rows(all_stats), file.path(output_dir, "mixed_model_statistics.csv"))
  }
  if (length(all_contrasts) > 0) {
    write_csv(bind_rows(all_contrasts), file.path(output_dir, "emmeans_contrasts.csv"))
  }

  saveRDS(results, file.path(output_dir, "mixed_models.RDS"))
  cat("Results saved to:", output_dir, "\n")
}


# ============================================================================
# MAIN FUNCTION
# ============================================================================

main <- function() {
  args <- parse_args()
  config <- load_config(args$config)

  if (!is.null(config$temporal$timepoint_col))  args$timepoint_col <- config$temporal$timepoint_col
  if (!is.null(config$temporal$group_col))      args$group_col     <- config$temporal$group_col
  if (!is.null(config$mixed_model$metric_cols)) args$metric_cols   <- config$mixed_model$metric_cols

  cat("Loading data from:", args$input_file, "\n")
  data <- load_data(args$input_file)
  cat("Loaded", nrow(data), "rows\n")

  prepared <- prepare_data(data, args$timepoint_col, args$group_col, args$participant_col, args$metric_cols)
  data <- prepared$data
  metric_cols <- prepared$metric_cols
  cat("Metrics to model:", paste(metric_cols, collapse = ", "), "\n")

  results <- fit_all_mixed_models(data, args$group_col, args$timepoint_col, args$participant_col, metric_cols)
  save_results(results, args$output_dir, metric_cols)

  cat("\n=== MIXED MODEL SUMMARY ===\n")
  cat("Timepoints:", nlevels(data[[args$timepoint_col]]), "\n")
  cat("Groups:", nlevels(data[[args$group_col]]), "\n")
  cat("Metrics modeled:", sum(sapply(results, Negate(is.null))), "/", length(metric_cols), "\n")
}

if (!interactive()) {
  main()
}
