import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

print("Adarsh Shukla T117")

df = pd.read_csv(r"C:\Users\Adarsh\Downloads\retail_sales_dataset.csv")

print("\nDataset:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

transactions = []

for _, row in df.iterrows():
    basket = []

    for col in df.columns:
        name = col.lower()

        if "transaction" in name or "customer id" in name or name == "date":
            continue

        value = row[col]

        if pd.isna(value):
            continue

        if col.lower() == "age":
            age = int(value)

            if age < 25:
                basket.append("Age=Young")
            elif age < 40:
                basket.append("Age=Adult")
            elif age < 60:
                basket.append("Age=Middle")
            else:
                basket.append("Age=Senior")

        elif "price" in name or "amount" in name:
            continue

        else:
            basket.append(f"{col}={value}")

    transactions.append(set(basket))

min_support = 0.05
min_confidence = 0.5

total_transactions = len(transactions)

all_items = sorted(set().union(*transactions))

frequent_itemsets = {}

for item in all_items:
    support = sum(item in t for t in transactions) / total_transactions

    if support >= min_support:
        frequent_itemsets[frozenset([item])] = support

current_level = list(frequent_itemsets.keys())
k = 2

while current_level:
    previous_items = sorted(set().union(*current_level))

    candidates = list(combinations(previous_items, k))

    next_level = []

    for candidate in candidates:
        candidate_set = frozenset(candidate)

        support = sum(
            candidate_set.issubset(transaction)
            for transaction in transactions
        ) / total_transactions

        if support >= min_support:
            frequent_itemsets[candidate_set] = support
            next_level.append(candidate_set)

    current_level = next_level
    k += 1

itemset_results = []

for itemset, support in frequent_itemsets.items():
    if len(itemset) >= 2:
        itemset_results.append({
            "Itemset": " , ".join(sorted(itemset)),
            "Support": support
        })

itemset_df = pd.DataFrame(itemset_results)

if not itemset_df.empty:
    itemset_df = itemset_df.sort_values(
        "Support",
        ascending=False
    ).reset_index(drop=True)

print("\nFrequent Itemsets:")
if itemset_df.empty:
    print("No frequent itemsets found.")
else:
    print(itemset_df.head(20).to_string(index=False))

rules = []

for itemset, itemset_support in frequent_itemsets.items():

    if len(itemset) < 2:
        continue

    for size in range(1, len(itemset)):

        for antecedent_tuple in combinations(itemset, size):

            antecedent = frozenset(antecedent_tuple)
            consequent = itemset - antecedent

            antecedent_support = sum(
                antecedent.issubset(transaction)
                for transaction in transactions
            ) / total_transactions

            consequent_support = sum(
                consequent.issubset(transaction)
                for transaction in transactions
            ) / total_transactions

            if antecedent_support == 0:
                continue

            confidence = itemset_support / antecedent_support

            if consequent_support > 0:
                lift = confidence / consequent_support
            else:
                lift = 0

            if confidence >= min_confidence:
                rules.append({
                    "Antecedent": " , ".join(sorted(antecedent)),
                    "Consequent": " , ".join(sorted(consequent)),
                    "Support": itemset_support,
                    "Confidence": confidence,
                    "Lift": lift
                })

rules_df = pd.DataFrame(rules)

if not rules_df.empty:
    rules_df = rules_df.drop_duplicates()
    rules_df = rules_df.sort_values(
        ["Confidence", "Lift"],
        ascending=False
    ).reset_index(drop=True)

print("\nAssociation Rules:")

if rules_df.empty:
    print("No association rules found.")
else:
    print(rules_df.head(20).to_string(index=False))

    print("\nBest Rule:")
    print(
        rules_df.iloc[0]["Antecedent"],
        " -> ",
        rules_df.iloc[0]["Consequent"]
    )

    print(
        "Support:",
        round(rules_df.iloc[0]["Support"], 3)
    )

    print(
        "Confidence:",
        round(rules_df.iloc[0]["Confidence"], 3)
    )

    print(
        "Lift:",
        round(rules_df.iloc[0]["Lift"], 3)
    )

if not itemset_df.empty:

    top_items = itemset_df.head(10)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_items["Itemset"],
        top_items["Support"]
    )

    plt.xlabel("Support")
    plt.ylabel("Frequent Itemsets")
    plt.title("Top Frequent Itemsets")

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

if not rules_df.empty:

    plt.figure(figsize=(8, 6))

    plt.scatter(
        rules_df["Support"],
        rules_df["Confidence"],
        s=rules_df["Lift"] * 80,
        alpha=0.7
    )

    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules: Support vs Confidence")

    plt.grid(True)
    plt.tight_layout()
    plt.show()
