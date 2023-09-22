rm(list = ls())

pkgTest <- function(x)
{
  if (!require(x,character.only = TRUE))
  {
    install.packages(x,dep=TRUE)
    if(!require(x,character.only = TRUE)) stop("Package not found")
  }
  return("OK")
}

global.libraries <- c("tidyverse", "modelsummary", "fixest")
results <- sapply(as.list(global.libraries), pkgTest)

rootdir <- "YOUR_PATH_TO_REPO"
dir <- paste(rootdir, "/kml_covid_family_migration/code/", sep="")
setwd(dir)
getwd()


y <- read.csv(file = '../data/processed/twfe_y.csv')
df <- read.csv(file = '../data/processed/twfe.csv', colClasses=c("CBSA"="character"))
df <- cbind(df, y)
clfe1 <- feols(y ~ Constant + FamilyxPostCovid | CBSA + PostCovid, data = df)
clfe2 <- feols(y ~ Constant + FamilyxPostCovid + PopxPostCovid + PopDenxPostCovid | CBSA + PostCovid, data = df)
clfe3 <- feols(y ~ Constant + FamilyxPostCovid + PopxPostCovid + PopDenxPostCovid + HomeValuexPostCovid + IncomexPostCovid + JobsxPostCovid | CBSA + PostCovid, data = df)
clfe4 <- feols(y ~ Constant + FamilyxPostCovid + PopxPostCovid + PopDenxPostCovid + HomeValuexPostCovid + SFHxPostCovid + IncomexPostCovid + JobsxPostCovid | CBSA + PostCovid, data = df)
models <- list()
models[['(1)']] <- clfe1
models[['(2)']] <- clfe2
models[['(3)']] <- clfe3
models[['(4)']] <- clfe4
msummary(models, stars = c('*' = .1, '**' = .05, '***' = .01))
modelsummary(models, stars = c('*' = .1, '**' = .05, '***' = .01), output = "latex")
