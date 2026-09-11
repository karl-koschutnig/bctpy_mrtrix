#!/usr/bin/env Rscript
#' Difference-score models: the sturdy sanity check against the latent growth
#' curve. For each measure, delta = (last timepoint - first timepoint) per
#' subject, then  lm(delta ~ group + covariates).  One row per subject, no
#' latent variables, no 3-wave identification problem - if the LGCM group->slope
#' effect is real it should show here too.
#'
#' Usage:
#'   Rscript difference_scores.R --input-file OM.csv --output-dir OUT \
#'     [--timepoint-col session] [--group-col group] [--participant-col participant_id] \
#'     [--covariate-cols age sex] [--metric-cols ...]

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(argparse)
})

p <- argparse::ArgumentParser(description = "Pre-post difference-score models")
p$add_argument("--input-file", required = TRUE)
p$add_argument("--output-dir", default = "outputs/difference_scores")
p$add_argument("--timepoint-col", default = "session")
p$add_argument("--group-col", default = "group")
p$add_argument("--participant-col", default = "participant_id")
p$add_argument("--covariate-cols", nargs = "+", default = NULL)
p$add_argument("--metric-cols", nargs = "+", default = NULL)
args <- p$parse_args()

df <- readr::read_csv(args$input_file, show_col_types = FALSE)
tp <- args$timepoint_col; grp <- args$group_col; id <- args$participant_col
covs <- args$covariate_cols[args$covariate_cols %in% names(df)]

waves <- sort(unique(df[[tp]]))
first <- waves[1]; last <- waves[length(waves)]

always_meta <- c(id, tp, grp, "age", "sex", "height_cm", "weight_kg")
metrics <- args$metric_cols
if (is.null(metrics)) {
  metrics <- setdiff(names(df)[sapply(df, is.numeric)], union(always_meta, covs))
}
metrics <- setdiff(metrics, covs)

subj <- df %>%
  filter(.data[[tp]] %in% c(first, last)) %>%
  mutate(.w = ifelse(.data[[tp]] == first, "t0", "t1"))

# subject-level covariate + group table (constant within subject)
meta_tbl <- df %>% group_by(.data[[id]]) %>%
  summarise(across(all_of(c(grp, covs)), ~ .x[!is.na(.x)][1]), .groups = "drop")

results <- list()
for (m in metrics) {
  wide <- subj %>% select(all_of(c(id, ".w")), value = all_of(m)) %>%
    pivot_wider(names_from = ".w", values_from = "value")
  if (!all(c("t0", "t1") %in% names(wide))) next
  wide <- wide %>% mutate(delta = t1 - t0) %>%
    inner_join(meta_tbl, by = id) %>% filter(!is.na(delta))
  if (nrow(wide) < 10 || dplyr::n_distinct(wide[[grp]]) < 2) next
  wide[[grp]] <- factor(wide[[grp]])
  for (cv in covs) if (is.character(wide[[cv]])) wide[[cv]] <- factor(wide[[cv]])

  rhs <- grp
  if (length(covs)) rhs <- paste(c(grp, covs), collapse = " + ")
  fit <- tryCatch(lm(as.formula(paste("delta ~", rhs)), data = wide), error = function(e) NULL)
  if (is.null(fit)) next

  aov_tbl <- as.data.frame(anova(fit))
  aov_tbl$term <- rownames(aov_tbl); aov_tbl$metric <- m
  co <- as.data.frame(summary(fit)$coefficients)
  co$term <- rownames(co); co$metric <- m
  names(co) <- make.names(names(co))
  results[[m]] <- list(aov = aov_tbl, coef = co)
}

dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
if (length(results)) {
  aov_all <- bind_rows(lapply(results, `[[`, "aov"))
  coef_all <- bind_rows(lapply(results, `[[`, "coef"))
  readr::write_csv(aov_all, file.path(args$output_dir, "difference_score_anova.csv"))
  readr::write_csv(coef_all, file.path(args$output_dir, "difference_score_coefficients.csv"))
  cat(sprintf("Difference-score models: %d metrics, delta = %s - %s\n",
              length(results), last, first))
} else {
  cat("Difference-score models: nothing fit (insufficient data)\n")
}
