#!/usr/bin/env Rscript
#' Baseline-dependence sensitivity analysis for the AAL3 factorial models.
#'
#' The social-context arms do not start level: on the integration component the
#' group-running arm begins below the solo arm. A contrast x session effect can
#' therefore arise either from the intervention or from regression toward the
#' mean in an arm that happened to start low. Two analyses separate them:
#'
#'   ANCOVA        post-baseline sessions only, with each participant's own
#'                 baseline value as a covariate. If the effect is regression
#'                 to the mean, conditioning on baseline removes it.
#'   Baseline test contrast effects on the baseline session alone, to quantify
#'                 how unbalanced the arms were before the intervention.
#'
#' Usage:
#'   Rscript aal3_baseline_adjusted.R --input-file .../aal3_analysis_table.csv \
#'     --output-dir outputs/aal3_reanalysis

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(lme4); library(lmerTest); library(argparse)
})

p <- argparse::ArgumentParser(description = "Baseline-dependence sensitivity analysis")
p$add_argument("--input-file", required = TRUE)
p$add_argument("--output-dir", default = "outputs/aal3_reanalysis")
args <- p$parse_args()
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

CONTRASTS <- c("c_running", "c_group", "c_duration", "c_group_x_dur")
OUTCOMES <- c("PC1", "PC2", "PC3", "full_density", "full_global_efficiency",
              "full_modularity", "bb_strength_mean", "bb_total_weight")

d <- read_csv(args$input_file, show_col_types = FALSE) %>%
  mutate(session = factor(session), participant_id = factor(participant_id),
         sex = factor(sex))
for (pc in c("PC1", "PC2", "PC3")) if (pc %in% names(d)) d[[pc]] <- -d[[pc]]
base_level <- levels(d$session)[1]

# ---- 1. how unbalanced are the arms at baseline? ----------------------------
base_rows <- list()
for (o in intersect(OUTCOMES, names(d))) {
  dd <- d %>% filter(session == base_level)
  f <- as.formula(sprintf("%s ~ %s + age + sex", o, paste(CONTRASTS, collapse = " + ")))
  m <- tryCatch(lm(f, data = dd), error = function(e) NULL)
  if (is.null(m)) next
  co <- as.data.frame(summary(m)$coefficients); co$term <- rownames(co)
  names(co) <- make.names(names(co)); rownames(co) <- NULL
  sdy <- sd(dd[[o]], na.rm = TRUE)
  base_rows[[o]] <- co %>% filter(term %in% CONTRASTS) %>%
    transmute(outcome = o, term, est = Estimate, se = Std..Error,
              est_per_sd = Estimate / sdy, p_value = Pr...t..)
}
baseline_tbl <- bind_rows(base_rows)
write_csv(baseline_tbl, file.path(args$output_dir, "baseline_balance.csv"))
cat("== Contrast effects at baseline (ses-1 only) ==\n")
print(as.data.frame(baseline_tbl %>% mutate(across(where(is.numeric), ~round(.x, 4)))))

# ---- 2. ANCOVA on post-baseline sessions ------------------------------------
anc_rows <- list(); slope_rows <- list()
for (o in intersect(OUTCOMES, names(d))) {
  bl <- d %>% filter(session == base_level) %>% select(participant_id, .base = all_of(o))
  dd <- d %>% left_join(bl, by = "participant_id") %>%
    filter(session != base_level) %>% mutate(session = droplevels(session))
  f <- as.formula(sprintf(
    "%s ~ (%s) * session + .base + age + sex + (1 | participant_id)",
    o, paste(CONTRASTS, collapse = " + ")))
  m <- tryCatch(suppressMessages(lmerTest::lmer(f, data = dd)), error = function(e) NULL)
  if (is.null(m)) next
  a <- as.data.frame(anova(m)); a$term <- rownames(a); rownames(a) <- NULL
  names(a) <- make.names(names(a))
  anc_rows[[o]] <- a %>% transmute(outcome = o, term, F_value = F.value,
                                   NumDF, DenDF, p_value = Pr..F.)
  co <- as.data.frame(summary(m)$coefficients); co$term <- rownames(co)
  names(co) <- make.names(names(co)); rownames(co) <- NULL
  sdy <- sd(dd[[o]], na.rm = TRUE)
  slope_rows[[o]] <- co %>% filter(term %in% CONTRASTS) %>%
    transmute(outcome = o, term, est = Estimate, se = Std..Error,
              est_per_sd = Estimate / sdy, p_value = Pr...t..)
}
anc <- bind_rows(anc_rows)
write_csv(anc, file.path(args$output_dir, "baseline_adjusted_anova.csv"))
write_csv(bind_rows(slope_rows), file.path(args$output_dir, "baseline_adjusted_effects.csv"))

cat("\n== ANCOVA (post-baseline sessions, baseline as covariate): contrast terms ==\n")
print(as.data.frame(anc %>% filter(grepl("^c_", term) | grepl("c_", term)) %>%
                      mutate(across(where(is.numeric), ~round(.x, 4)))))
cat("\n== Adjusted contrast effects (per SD) ==\n")
print(as.data.frame(bind_rows(slope_rows) %>% mutate(across(where(is.numeric), ~round(.x, 4)))))
cat(sprintf("\nResults -> %s\n", args$output_dir))
