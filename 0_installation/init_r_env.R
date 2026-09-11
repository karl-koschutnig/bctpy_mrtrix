#!/usr/bin/env Rscript
# One-time setup: initializes renv and creates renv.lock for the R
# packages the temporal analysis pipeline needs. Run once from the repo
# root:
#   Rscript 0_installation/init_r_env.R
# Re-run (from repo root) after adding a new library() call anywhere in
# 7_temporal_analysis/ to refresh the lock file.

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}
if (!file.exists("renv.lock")) {
  renv::init(bare = TRUE)
}

required_packages <- c(
  "mgcv", "lme4", "lmerTest", "emmeans", "tidyverse",
  "jsonlite", "argparse", "R.matlab", "arrow", "lavaan"
)
install.packages(required_packages, repos = "https://cloud.r-project.org")

renv::snapshot(packages = required_packages, prompt = FALSE)
cat("\nrenv.lock written. Commit renv.lock, .Rprofile, and renv/activate.R.\n")
