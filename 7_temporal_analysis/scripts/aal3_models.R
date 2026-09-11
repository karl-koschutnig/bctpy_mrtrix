#!/usr/bin/env Rscript
#' Factorial mixed-effects analysis of the AAL3 network trajectories.
#'
#' The design is a 2 x 2 factorial (social context x duration) plus a
#' no-running control, so it is analysed with four orthogonal planned
#' contrasts rather than four arbitrary dummy comparisons against control:
#'
#'   c_running      any running vs control
#'   c_group        group running vs solo running
#'   c_duration     4 weeks vs 2 weeks
#'   c_group_x_dur  social context x duration
#'
#' Each contrast is crossed with session; the contrast x session terms are the
#' intervention effects. Models are fitted at three levels of stringency:
#'
#'   base      outcome ~ contrasts * session + age + sex + (1 | id)
#'   yield     ... + total_streamlines            (tractography-yield control)
#'   backbone  same model on consistency-thresholded measures, where the edge
#'             set is fixed and only edge weights can move
#'
#' A permutation test (arm labels shuffled between subjects, within-subject
#' structure preserved) gives an exact reference distribution for the primary
#' contrast x session tests, which parametric F tests with n = 15-41 per arm
#' cannot be trusted to give.
#'
#' Usage:
#'   Rscript aal3_models.R --input-file outputs/aal3_reanalysis/aal3_analysis_table.csv \
#'     --output-dir outputs/aal3_reanalysis [--n-perm 2000]

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(tidyr)
  library(lme4); library(lmerTest); library(argparse)
})

p <- argparse::ArgumentParser(description = "AAL3 factorial mixed models")
p$add_argument("--input-file", required = TRUE)
p$add_argument("--output-dir", default = "outputs/aal3_reanalysis")
p$add_argument("--n-perm", type = "integer", default = 2000)
p$add_argument("--seed", type = "integer", default = 42)
args <- p$parse_args()
set.seed(args$seed)
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

CONTRASTS <- c("c_running", "c_group", "c_duration", "c_group_x_dur")

d <- readr::read_csv(args$input_file, show_col_types = FALSE) %>%
  mutate(session = factor(session),
         participant_id = factor(participant_id),
         sex = factor(sex))

# Component signs are arbitrary in PCA. Flip them so that "higher" reads in the
# natural direction for each axis, which only changes the sign of an effect,
# never its magnitude or p-value:
#   PC1 loads -density/-global efficiency/+path length -> flip to "integration"
#   PC2 loads -clustering/-local efficiency/-sigma     -> flip to "segregation"
#   PC3 loads -betweenness/-core-periphery             -> flip to "centralisation"
for (pc in c("PC1", "PC2", "PC3")) if (pc %in% names(d)) d[[pc]] <- -d[[pc]]

# Primary outcomes: the three orthogonal components of the measure battery,
# plus density on the full graph (continuity with the exploratory analysis)
# and the backbone weight measures (yield-immune).
PRIMARY   <- c("PC1", "PC2", "PC3")
SECONDARY <- c("full_density", "full_global_efficiency", "full_modularity",
               "bb_strength_mean", "bb_clustering_coef_mean", "bb_modularity",
               "bb_global_efficiency", "bb_total_weight", "bb_mean_weight")
OUTCOMES <- intersect(c(PRIMARY, SECONDARY), names(d))

rhs_terms <- function(spec) {
  base <- paste(sprintf("%s * session", CONTRASTS), collapse = " + ")
  base <- paste(base, "+ age + sex")
  if (spec == "yield") base <- paste(base, "+ scale(total_streamlines)")
  base
}

fit_one <- function(outcome, spec) {
  f <- as.formula(sprintf("%s ~ %s + (1 | participant_id)", outcome, rhs_terms(spec)))
  m <- tryCatch(suppressMessages(lmerTest::lmer(f, data = d)), error = function(e) NULL)
  if (is.null(m)) return(NULL)
  a <- as.data.frame(anova(m))
  a$term <- rownames(a); rownames(a) <- NULL
  a$outcome <- outcome; a$spec <- spec
  # standardized slope of each contrast x session (linear session score)
  a
}

cat("== Fitting factorial models ==\n")
res <- list()
for (o in OUTCOMES) {
  for (s in c("base", "yield")) {
    r <- fit_one(o, s)
    if (!is.null(r)) res[[paste(o, s)]] <- r
    cat(sprintf("  %-26s %-6s %s\n", o, s, if (is.null(r)) "FAILED" else "ok"))
  }
}
anova_tbl <- bind_rows(res) %>%
  rename(p_value = `Pr(>F)`, F_value = `F value`) %>%
  select(outcome, spec, term, F_value, NumDF, DenDF, p_value)
readr::write_csv(anova_tbl, file.path(args$output_dir, "factorial_anova.csv"))

# ---- effect direction: contrast x linear-session slope estimates -------------
cat("\n== Contrast x time slope estimates ==\n")
d$t <- as.numeric(d$session) - 1          # 0, 1, 2
slope_rows <- list()
for (o in OUTCOMES) {
  f <- as.formula(sprintf(
    "%s ~ (%s) * t + age + sex + (1 | participant_id)",
    o, paste(CONTRASTS, collapse = " + ")))
  m <- tryCatch(suppressMessages(lmerTest::lmer(f, data = d)), error = function(e) NULL)
  if (is.null(m)) next
  co <- as.data.frame(summary(m)$coefficients)
  co$term <- rownames(co); rownames(co) <- NULL
  names(co) <- make.names(names(co))
  sdy <- sd(d[[o]], na.rm = TRUE)
  co <- co %>% filter(grepl(":t$", term)) %>%
    transmute(outcome = o, term = sub(":t$", "", term),
              est = Estimate, se = Std..Error, df = df,
              t_value = t.value, p_value = Pr...t..,
              est_per_sd = Estimate / sdy)
  slope_rows[[o]] <- co
}
slopes <- bind_rows(slope_rows)
readr::write_csv(slopes, file.path(args$output_dir, "contrast_time_slopes.csv"))
print(slopes %>% filter(p_value < 0.10) %>% mutate(across(where(is.numeric), ~round(.x, 4))))

# ---- permutation test on the primary outcomes -------------------------------
cat(sprintf("\n== Permutation test (%d permutations, primary outcomes) ==\n", args$n_perm))
subj <- d %>% distinct(participant_id, .keep_all = TRUE) %>%
  select(participant_id, all_of(CONTRASTS))

# F statistics only - skip the Satterthwaite denominator-df computation, which
# dominates runtime and is not needed for a permutation reference distribution.
# R names an interaction term by the order variables entered the formula, which
# is not always "contrast:session" (it can come back "session:c_group") - match
# on the unordered pair of names rather than the exact string.
find_interaction_row <- function(rownames_, contrast) {
  parts <- strsplit(rownames_, ":")
  hit <- vapply(parts, function(p) length(p) == 2 && setequal(p, c(contrast, "session")), logical(1))
  if (!any(hit)) return(NA_character_)
  rownames_[which(hit)[1]]
}

obs_F <- function(outcome, data) {
  want <- paste0(CONTRASTS, ":session")
  f <- as.formula(sprintf("%s ~ %s + (1 | participant_id)", outcome, rhs_terms("base")))
  m <- tryCatch(suppressMessages(lmerTest::lmer(f, data = data)), error = function(e) NULL)
  if (is.null(m)) return(setNames(rep(NA_real_, length(want)), want))
  a <- tryCatch(as.data.frame(anova(m, type = 3, ddf = "lme4")), error = function(e) NULL)
  if (is.null(a)) return(setNames(rep(NA_real_, length(want)), want))
  rows <- vapply(CONTRASTS, function(cn) find_interaction_row(rownames(a), cn), character(1))
  if (any(is.na(rows))) return(setNames(rep(NA_real_, length(want)), want))
  setNames(a[rows, "F value"], want)
}

perm_out <- list()
for (o in intersect(PRIMARY, OUTCOMES)) {
  observed <- obs_F(o, d)
  null_mat <- matrix(NA_real_, nrow = args$n_perm, ncol = length(observed),
                     dimnames = list(NULL, names(observed)))
  for (i in seq_len(args$n_perm)) {
    shuffled <- subj
    shuffled[CONTRASTS] <- subj[sample(nrow(subj)), CONTRASTS]
    dp <- d %>% select(-all_of(CONTRASTS)) %>%
      left_join(shuffled, by = "participant_id")
    null_mat[i, ] <- obs_F(o, dp)
  }
  pperm <- sapply(names(observed), function(k)
    (1 + sum(null_mat[, k] >= observed[k], na.rm = TRUE)) / (1 + sum(!is.na(null_mat[, k]))))
  perm_out[[o]] <- data.frame(outcome = o, term = names(observed),
                              F_observed = as.numeric(observed),
                              p_permutation = as.numeric(pperm))
  cat(sprintf("  %s done\n", o))
}
if (length(perm_out)) {
  perm <- bind_rows(perm_out)
  readr::write_csv(perm, file.path(args$output_dir, "permutation_tests.csv"))
  print(perm %>% mutate(across(where(is.numeric), ~round(.x, 4))))
}

# ---- FDR over the primary family --------------------------------------------
primary_tests <- anova_tbl %>%
  filter(spec == "base", outcome %in% PRIMARY, grepl(":session$", term)) %>%
  mutate(q_value = p.adjust(p_value, method = "BH"))
readr::write_csv(primary_tests, file.path(args$output_dir, "primary_family_fdr.csv"))
cat("\n== Primary family (3 components x 4 contrasts, BH-corrected) ==\n")
print(primary_tests %>% mutate(across(where(is.numeric), ~round(.x, 4))) %>% as.data.frame())

cat(sprintf("\nResults -> %s\n", args$output_dir))
