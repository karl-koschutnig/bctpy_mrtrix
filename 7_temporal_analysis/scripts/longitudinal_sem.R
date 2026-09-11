#!/usr/bin/env Rscript
#' Longitudinal SEM for network organizational measures
#' ====================================================
#'
#' A structural-equation complement to mixed_models.R. Two analyses:
#'
#'   1. Univariate latent growth curve model (LGCM) per measure
#'      - latent intercept (loadings 1,1,1) + latent linear slope (0,1,2)
#'      - group (dummy-coded, control = reference) predicts BOTH growth factors:
#'          group -> intercept  = baseline arm differences
#'          group -> slope      = arm effect on the trajectory
#'      - reports growth-factor means/variances, intercept-slope covariance,
#'        every group path, and fit indices (CFI/RMSEA/SRMR). FIML for missing
#'        values (e.g. path length undefined for a disconnected subject).
#'      - also fits an intercept-only model and reports the LRT vs. the linear
#'        model, i.e. "is there any systematic change at all".
#'
#'   2. Bivariate parallel-process LGCM within measure domains
#'      - integration  : global_efficiency + path_length
#'      - segregation  : modularity + clustering_coef_mean
#'      - key output is the SLOPE-SLOPE correlation (do the measures change
#'        together within a person) - the thing univariate lme cannot ask.
#'
#' Every measure is z-standardized (pooled across all person-waves) before
#' fitting so scales are comparable and estimation is stable.
#'
#' 3 waves only supports a linear (or latent-basis) slope, not a quadratic.
#' Near-constant measures (k-core, s-core here) and measures that are mostly
#' missing for an atlas are skipped automatically.
#'
#' Usage:
#'   Rscript longitudinal_sem.R --input-file organizational_measures_results.csv \
#'     --output-dir outputs/sem_results \
#'     --timepoint-col session --group-col group --participant-col participant_id

suppressPackageStartupMessages({
  library(tidyverse)
  library(lavaan)
  library(jsonlite)
  library(argparse)
})


# ============================================================================
# ARGS
# ============================================================================

parse_args <- function() {
  p <- argparse::ArgumentParser(description = "Longitudinal SEM for network measures")
  p$add_argument("--input-file", required = TRUE)
  p$add_argument("--output-dir", default = "outputs/sem_results")
  p$add_argument("--timepoint-col", default = "session")
  p$add_argument("--group-col", default = "group")
  p$add_argument("--participant-col", default = "participant_id")
  p$add_argument("--metric-cols", nargs = "+", default = NULL)
  p$add_argument("--covariate-cols", nargs = "+", default = NULL,
                 help = "Subject-level covariates to regress the latent intercept and slope on (e.g. age sex)")
  p$add_argument("--latent-basis", action = "store_true", default = FALSE,
                 help = "Free the interior slope loading(s) instead of fixing them 0,1,2 - relaxes the linear-shape assumption")
  p$add_argument("--min-var", type = "double", default = 1e-8,
                 help = "Skip a measure whose pooled variance is below this")
  p$add_argument("--max-missing", type = "double", default = 0.5,
                 help = "Skip a measure with more than this fraction missing")
  p$parse_args()
}


# ============================================================================
# DATA PREP
# ============================================================================

load_long <- function(path) {
  if (grepl("\\.parquet$", path)) {
    stopifnot(requireNamespace("arrow", quietly = TRUE))
    arrow::read_parquet(path)
  } else {
    readr::read_csv(path, show_col_types = FALSE)
  }
}

#' Long -> wide, z-standardized, with control-reference group dummies.
#' Returns list(data = wide df, waves = character vector, dummies = char vec,
#'              time_scores = numeric).
prepare <- function(df, tp_col, grp_col, id_col, metric_cols,
                    min_var, max_missing, covariate_cols = NULL) {
  waves <- sort(unique(df[[tp_col]]))
  if (length(waves) < 3) stop("Need at least 3 timepoints for an LGCM.")
  time_scores <- seq_along(waves) - 1  # 0, 1, 2, ...

  # sex -> numeric 0/1 so lavaan can regress on it
  if ("sex" %in% names(df) && !is.numeric(df[["sex"]])) {
    df[["sex"]] <- as.integer(factor(df[["sex"]])) - 1L
  }
  covariate_cols <- covariate_cols[covariate_cols %in% names(df)]

  meta <- c(id_col, tp_col, grp_col, "age", "sex", "height_cm", "weight_kg", covariate_cols)
  meta <- meta[meta %in% names(df)]
  if (is.null(metric_cols)) {
    metric_cols <- setdiff(names(df)[sapply(df, is.numeric)], meta)
  }
  metric_cols <- setdiff(metric_cols[metric_cols %in% names(df)], covariate_cols)

  # z-standardize each measure across all person-waves (pooled)
  usable <- character(0)
  for (m in metric_cols) {
    v <- df[[m]]
    frac_missing <- mean(is.na(v))
    if (frac_missing > max_missing) {
      message(sprintf("  skip %-28s (%.0f%% missing)", m, 100 * frac_missing))
      next
    }
    if (!is.finite(var(v, na.rm = TRUE)) || var(v, na.rm = TRUE) < min_var) {
      message(sprintf("  skip %-28s (near-constant)", m))
      next
    }
    df[[m]] <- as.numeric(scale(v))
    usable <- c(usable, m)
  }

  # group dummies, control as reference
  g <- as.character(df[[grp_col]])
  ref <- if ("ctrl" %in% g) "ctrl" else sort(unique(g))[1]
  levs <- setdiff(sort(unique(g)), ref)
  dummy_names <- paste0("grp_", make.names(levs))
  for (i in seq_along(levs)) df[[dummy_names[i]]] <- as.integer(g == levs[i])

  # z-standardize numeric covariates (pooled) so their SEM paths are comparable
  for (cv in covariate_cols) {
    if (is.numeric(df[[cv]]) && length(unique(na.omit(df[[cv]]))) > 2) {
      df[[cv]] <- as.numeric(scale(df[[cv]]))
    }
  }

  # wide: one row per participant, <measure>_w1.. plus dummies + covariates
  # (dummies and covariates are constant within id, so they stay row identifiers)
  keep <- c(id_col, ".wave", usable, dummy_names, covariate_cols)
  wide <- df %>%
    mutate(.wave = paste0("w", match(.data[[tp_col]], waves))) %>%
    select(all_of(keep)) %>%
    pivot_wider(names_from = ".wave",
                values_from = all_of(usable),
                names_glue = "{.value}_{.wave}")

  list(data = as.data.frame(wide), waves = waves, usable = usable,
       dummies = dummy_names, covariates = covariate_cols,
       time_scores = time_scores, ref = ref, levs = levs)
}


# ============================================================================
# MODEL SPECS
# ============================================================================

#' Slope-loading terms. Linear: fixed 0,1,2,... Latent basis: first fixed 0,
#' last fixed 1, interior loading(s) freed (with a linear start value).
slope_loadings <- function(ind, time_scores, latent_basis) {
  n <- length(ind)
  if (!latent_basis || n < 3) return(sprintf("%g*%s", time_scores, ind))
  span <- time_scores[n] - time_scores[1]
  out <- character(n)
  out[1] <- sprintf("0*%s", ind[1])
  out[n] <- sprintf("1*%s", ind[n])
  for (k in 2:(n - 1)) out[k] <- sprintf("start(%.3f)*%s", (time_scores[k] - time_scores[1]) / span, ind[k])
  out
}

lgcm_spec <- function(measure, waves_n, time_scores, dummies, with_slope = TRUE,
                      covariates = NULL, latent_basis = FALSE) {
  ind <- paste0(measure, "_w", seq_len(waves_n))
  i_load <- paste(sprintf("1*%s", ind), collapse = " + ")
  lines <- c(sprintf("i =~ %s", i_load))
  if (with_slope) {
    s_load <- paste(slope_loadings(ind, time_scores, latent_basis), collapse = " + ")
    lines <- c(lines, sprintf("s =~ %s", s_load))
  }
  preds <- c(dummies, covariates)
  if (length(preds)) {
    rhs <- paste(preds, collapse = " + ")
    lines <- c(lines, sprintf("i ~ %s", rhs))
    if (with_slope) lines <- c(lines, sprintf("s ~ %s", rhs))
  }
  paste(lines, collapse = "\n")
}

parallel_spec <- function(m1, m2, waves_n, time_scores, dummies, covariates = NULL,
                          latent_basis = FALSE) {
  spec <- function(m, pre) {
    ind <- paste0(m, "_w", seq_len(waves_n))
    c(sprintf("%si =~ %s", pre, paste(sprintf("1*%s", ind), collapse = " + ")),
      sprintf("%ss =~ %s", pre, paste(slope_loadings(ind, time_scores, latent_basis), collapse = " + ")))
  }
  lines <- c(spec(m1, "a"), spec(m2, "b"))
  preds <- c(dummies, covariates)
  if (length(preds)) {
    rhs <- paste(preds, collapse = " + ")
    lines <- c(lines, sprintf("ai ~ %s", rhs), sprintf("as ~ %s", rhs),
               sprintf("bi ~ %s", rhs), sprintf("bs ~ %s", rhs))
  }
  # free covariances among the four growth factors (lavaan growth() adds these,
  # but be explicit about the one we care about)
  lines <- c(lines, "as ~~ bs")
  paste(lines, collapse = "\n")
}


# ============================================================================
# FIT + EXTRACT
# ============================================================================

fit_fit_indices <- function(fit) {
  m <- tryCatch(fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "rmsea", "srmr", "npar")),
                error = function(e) rep(NA_real_, 7))
  as.list(setNames(as.numeric(m), c("chisq", "df", "pvalue", "cfi", "rmsea", "srmr", "npar")))
}

#' TRUE only if the solution converged AND passed lavaan's post-fit admissibility
#' checks (no negative variances, covariance matrices positive definite).
#' Parameter estimates from an inadmissible solution are not trustworthy.
is_admissible <- function(fit) {
  if (is.null(fit) || !lavInspect(fit, "converged")) return(FALSE)
  ok <- tryCatch(lavInspect(fit, "post.check"), error = function(e) FALSE)
  isTRUE(ok)
}

run_univariate <- function(prep, out_dir, latent_basis = FALSE) {
  waves_n <- length(prep$waves)
  params <- list(); fits <- list(); lrts <- list()
  slope_spec <- if (latent_basis) "latent_basis" else "linear"

  for (m in prep$usable) {
    spec_lin <- lgcm_spec(m, waves_n, prep$time_scores, prep$dummies, with_slope = TRUE,
                          covariates = prep$covariates, latent_basis = latent_basis)
    spec_int <- lgcm_spec(m, waves_n, prep$time_scores, prep$dummies, with_slope = FALSE,
                          covariates = prep$covariates, latent_basis = latent_basis)

    fit_lin <- suppressWarnings(tryCatch(
      lavaan::growth(spec_lin, data = prep$data, missing = "fiml", estimator = "ML"),
      error = function(e) { message(sprintf("  %s: linear model failed - %s", m, e$message)); NULL }))
    converged <- !is.null(fit_lin) && lavInspect(fit_lin, "converged")
    admissible <- is_admissible(fit_lin)
    if (!converged) {
      fits[[m]] <- data.frame(metric = m, converged = FALSE, admissible = FALSE)
      next
    }

    pe <- parameterEstimates(fit_lin, standardized = TRUE) %>%
      filter((op == "~" & rhs %in% c(prep$dummies, prep$covariates)) |
             (op == "~1" & lhs %in% c("i", "s")) |
             (op == "~~" & lhs %in% c("i", "s") & rhs %in% c("i", "s"))) %>%
      transmute(metric = m, lhs, op, rhs, est, se, z, pvalue,
                std = std.all, admissible = admissible, slope_spec = slope_spec)
    params[[m]] <- pe

    fi <- fit_fit_indices(fit_lin)
    fits[[m]] <- data.frame(metric = m, converged = TRUE, admissible = admissible,
                            slope_spec = slope_spec,
                            n = lavInspect(fit_lin, "ntotal"),
                            chisq = fi$chisq, df = fi$df, pvalue = fi$pvalue,
                            cfi = fi$cfi, rmsea = fi$rmsea, srmr = fi$srmr, npar = fi$npar)

    fit_int <- tryCatch(
      lavaan::growth(spec_int, data = prep$data, missing = "fiml", estimator = "ML"),
      error = function(e) NULL)
    if (!is.null(fit_int) && lavInspect(fit_int, "converged")) {
      lr <- tryCatch(lavaan::lavTestLRT(fit_int, fit_lin), error = function(e) NULL)
      if (!is.null(lr)) {
        lrts[[m]] <- data.frame(metric = m,
                                chisq_diff = lr[["Chisq diff"]][2],
                                df_diff = lr[["Df diff"]][2],
                                p_slope_improves_fit = lr[["Pr(>Chisq)"]][2])
      }
    }
  }

  if (length(params)) readr::write_csv(bind_rows(params), file.path(out_dir, "sem_growth_parameters.csv"))
  if (length(fits))   readr::write_csv(bind_rows(fits),   file.path(out_dir, "sem_fit_indices.csv"))
  if (length(lrts))   readr::write_csv(bind_rows(lrts),   file.path(out_dir, "sem_slope_lrt.csv"))
}

run_parallel <- function(prep, out_dir, latent_basis = FALSE) {
  domains <- list(
    integration = c("global_efficiency", "path_length"),
    segregation = c("modularity", "clustering_coef_mean")
  )
  waves_n <- length(prep$waves)
  rows <- list()
  for (dom in names(domains)) {
    ms <- domains[[dom]]
    if (!all(ms %in% prep$usable)) {
      message(sprintf("  parallel-process %s: skipped (need %s)", dom, paste(ms, collapse = " + ")))
      next
    }
    spec <- parallel_spec(ms[1], ms[2], waves_n, prep$time_scores, prep$dummies,
                          covariates = prep$covariates, latent_basis = latent_basis)
    fit <- suppressWarnings(tryCatch(
      lavaan::growth(spec, data = prep$data, missing = "fiml", estimator = "ML"),
      error = function(e) { message(sprintf("  parallel %s failed: %s", dom, e$message)); NULL }))
    if (is.null(fit) || !lavInspect(fit, "converged")) next
    admissible <- is_admissible(fit)

    pe <- parameterEstimates(fit, standardized = TRUE)
    slope_cov <- pe %>% filter(op == "~~", lhs == "as", rhs == "bs")
    grp_paths <- pe %>% filter(op == "~", lhs %in% c("ai", "as", "bi", "bs"), rhs %in% prep$dummies)
    fi <- fit_fit_indices(fit)

    rows[[dom]] <- bind_rows(
      data.frame(domain = dom, measure_a = ms[1], measure_b = ms[2],
                 term = "slope_slope_cov", est = slope_cov$est, se = slope_cov$se,
                 z = slope_cov$z, pvalue = slope_cov$pvalue, std = slope_cov$std.all,
                 admissible = admissible,
                 cfi = fi$cfi, rmsea = fi$rmsea, srmr = fi$srmr),
      grp_paths %>% transmute(domain = dom, measure_a = ms[1], measure_b = ms[2],
                              term = paste(lhs, "~", rhs), est, se, z, pvalue,
                              std = std.all, admissible = admissible,
                              cfi = fi$cfi, rmsea = fi$rmsea, srmr = fi$srmr)
    )
  }
  if (length(rows)) readr::write_csv(bind_rows(rows), file.path(out_dir, "sem_parallel_process.csv"))
}


# ============================================================================
# MAIN
# ============================================================================

main <- function() {
  args <- parse_args()
  dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

  df <- load_long(args$input_file)
  cat(sprintf("Loaded %d rows from %s\n", nrow(df), args$input_file))

  prep <- prepare(df, args$timepoint_col, args$group_col, args$participant_col,
                  args$metric_cols, args$min_var, args$max_missing,
                  covariate_cols = args$covariate_cols)
  cat(sprintf("Waves: %s | groups vs '%s': %s | measures: %s\n",
              paste(prep$waves, collapse = ","), prep$ref,
              paste(prep$levs, collapse = ","),
              paste(prep$usable, collapse = ", ")))
  if (length(prep$covariates)) cat(sprintf("Covariates on i & s: %s\n", paste(prep$covariates, collapse = ", ")))
  if (isTRUE(args$latent_basis)) cat("Slope shape: latent basis (interior loadings freed)\n")

  cat("\n== Univariate latent growth curve models ==\n")
  run_univariate(prep, args$output_dir, latent_basis = isTRUE(args$latent_basis))

  cat("\n== Bivariate parallel-process models ==\n")
  run_parallel(prep, args$output_dir, latent_basis = isTRUE(args$latent_basis))

  cat(sprintf("\nResults -> %s\n", args$output_dir))
}

if (!interactive()) main()
