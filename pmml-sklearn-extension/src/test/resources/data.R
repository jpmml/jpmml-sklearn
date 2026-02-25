library("survival")

df = lung
df$status = df$status - 1

feature_cols = setdiff(names(df), c("status", "time"))
df = df[, c(feature_cols, "status", "time")]

write.csv(df, "csv/Lung.csv", row.names = FALSE, quote = FALSE, na = "N/A")