# Program 6: Market Basket Analysis using Association Rules (Apriori, implemented from scratch)
from itertools import combinations
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
transactions = [
    ["bread", "milk", "butter"],
    ["bread", "diapers", "beer", "eggs"],
    ["milk", "diapers", "beer", "cola"],
    ["bread", "milk", "diapers", "beer"],
    ["bread", "milk", "diapers", "cola"],
    ["bread", "milk", "butter", "eggs"],
    ["milk", "butter", "eggs"],
    ["bread", "butter"],
    ["diapers", "beer", "cola"],
    ["bread", "milk", "diapers", "butter"],
    ["milk", "diapers", "beer"],
    ["bread", "eggs", "butter"],
]
n = len(transactions)
tx_sets = [frozenset(t) for t in transactions]
print("Number of transactions:", n)
 
MIN_SUPPORT = 0.25
MIN_CONFIDENCE = 0.6
 
def support(itemset):
    return sum(1 for t in tx_sets if itemset <= t) / n
 
# --- Apriori: frequent itemsets ---
items = sorted({i for t in tx_sets for i in t})
current = [frozenset([i]) for i in items]
frequent = {}
k = 1
while current:
    level = {s: support(s) for s in current}
    level = {s: v for s, v in level.items() if v >= MIN_SUPPORT}
    frequent.update(level)
    keys = list(level.keys())
    # candidate generation
    nxt = set()
    for a in keys:
        for b in keys:
            u = a | b
            if len(u) == k + 1 and all(frozenset(sub) in level for sub in combinations(u, k)):
                nxt.add(u)
    current = list(nxt)
    k += 1
 
fi = pd.DataFrame([{"itemsets": ", ".join(sorted(s)), "size": len(s), "support": round(v, 3)}
                   for s, v in frequent.items()]).sort_values(["size", "support"], ascending=[True, False])
print(f"\nFrequent itemsets (min_support = {MIN_SUPPORT}):")
print(fi.to_string(index=False))
 
# --- Association rules ---
rules = []
for s, sup in frequent.items():
    if len(s) < 2:
        continue
    for r in range(1, len(s)):
        for ante in combinations(s, r):
            ante = frozenset(ante)
            cons = s - ante
            conf = sup / support(ante)
            lift = conf / support(cons)
            if conf >= MIN_CONFIDENCE:
                rules.append({"antecedent": ", ".join(sorted(ante)), "consequent": ", ".join(sorted(cons)),
                              "support": round(sup, 3), "confidence": round(conf, 3), "lift": round(lift, 3)})
rules_df = pd.DataFrame(rules).sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
print(f"\nAssociation rules (min_confidence = {MIN_CONFIDENCE}):")
print(rules_df.to_string())
 
print("\nTop 5 rules by lift:")
print(rules_df.head(5).to_string())
 
# Plot 1: support of frequent itemsets
top = fi.sort_values("support", ascending=True).tail(15)
plt.figure(figsize=(8, 6))
plt.barh(top["itemsets"], top["support"], color="steelblue")
plt.xlabel("Support")
plt.title("Frequent Itemsets (top 15 by support)")
plt.tight_layout()
plt.savefig("img/p6_frequent_itemsets.png", dpi=150)
 
# Plot 2: support vs confidence coloured by lift
plt.figure(figsize=(7, 5))
sc = plt.scatter(rules_df["support"], rules_df["confidence"], c=rules_df["lift"], cmap="viridis", s=90, edgecolor="k")
plt.colorbar(sc, label="Lift")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Association Rules: Support vs Confidence")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("img/p6_rules_scatter.png", dpi=150)
