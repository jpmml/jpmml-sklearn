from sklift.datasets import fetch_hillstrom

import pandas

dataset = fetch_hillstrom(target_col = "visit")

df = pandas.concat([dataset.treatment.rename("segment"), dataset.data, dataset.target.rename("visit")], axis = 1)

df["segment"] = df["segment"].map({
	"Mens E-Mail": "mens_email",
	"Womens E-Mail": "womens_email",
	"No E-Mail": "control",
})

df = df.drop(columns = ["history_segment"])
df["zip_code"] = df["zip_code"].replace("Surburban", "Suburban")

df = df.groupby("segment").sample(n = 2000, random_state = 42)

df.to_csv("csv/Email.csv", index = False, sep = ",")