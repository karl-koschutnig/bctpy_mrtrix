#!/usr/bin/env Rscript
# Restores the R environment from renv.lock. Run from the repo root
# (install.sh does this automatically). Fast and idempotent once the
# renv cache is warm.

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}
if (!file.exists("renv.lock")) {
  stop("renv.lock not found. Run 'Rscript 0_installation/init_r_env.R' once first.")
}
renv::restore(prompt = FALSE)
