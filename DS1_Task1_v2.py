#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
#!pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
#!pip install streamlit
import streamlit as st

data = pd.read_csv("risk_factors_cervical_cancer.csv")

def count_missing_values(datacol):
    return datacol.value_counts()["?"]

def str_to_numeric(data): # data should be with type objekt! A Dataframe column. 
    i_elements = []
    for i in data.to_list():
        if i == "?":
            i = 0
        i_elements.append(i)
    return pd.to_numeric(i_elements)

def num_partner_to_binary(datacol):
    j_elements = []
    datacol = str_to_numeric(datacol)
    for i in range(datacol.size):
        if datacol[i]/(data.at[i, "Age"]-10) >= 0.2:  # In average every five year a new partner from 10 years old.
            i = 1
        else:
            i = 0
        j_elements.append(i)
    datacol = j_elements
    return datacol

data["Number of sexual partners"] = num_partner_to_binary(data["Number of sexual partners"])
print(data["Number of sexual partners"].sum())

def age_to_binary(data):
    i_elements = []
    for i in data.to_list():
        if i >= 35:  # Half of the average female age in Venezuela in 2017 is 35
            i = 1
        else:
            i = 0
        i_elements.append(i)
    return pd.to_numeric(i_elements)

data["Age"] = age_to_binary(data["Age"])
print(data["Age"].sum())

subdata_2 = data[["Age", "Number of sexual partners", "IUD", "STDs", 
               "Smokes", "STDs:HPV", "Hormonal Contraceptives"]]
subdata_2 = pd.DataFrame(subdata_2)
subdata_2.insert(0, "Cervical Positiv", [1 for i in range(858)])
# subdata_2.loc[0, "Cervical Positiv"] = [1 for i in range(858)]
#subdata_2.astype(bool)

for col in subdata_2.columns:
    subdata_2[col] = str_to_numeric(subdata_2[col])
    subdata_2[col] = subdata_2[col].astype(int)

cwd = os.getcwd()
# subdata_2.to_csv(cwd+"/subdata_2.csv")
frequent_itemsets_2 = apriori(subdata_2, min_support = 0.1, use_colnames = True)
frequent_itemsets_2["length"] = frequent_itemsets_2["itemsets"].apply(lambda x:len(x))

rules = association_rules(frequent_itemsets_2, metric="lift", min_threshold=1.1)
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values("lift",ascending=False)

rules.sort_values("confidence",ascending=False)

# Because it makes no sense that the age to be consequents, so we remove the rules where age is in the consequents.
rules[~rules["consequents"].str.contains("Age", regex=False)].sort_values("confidence", ascending=False)

rules[~rules["consequents"].str.contains("Age", regex=False)].sort_values("lift",ascending=False)



# ----- Web front --- Streamlit -----

st.title("Cervical Cancer (Risk Factor)")
st.dataframe(data)

input_support = st.slider("Threshold minimal support", 0.0001, 0.5)
input_lift = st.slider("Threshold minimal lift", 0.0, 3.0)

st.write("Support", input_support)
st.write("Lift", input_lift)

frequent_itemsets = apriori(subdata_2, min_support = input_support, use_colnames = True)

itemsets = []
for items in frequent_itemsets["itemsets"].to_list():
    itemsets.append(set(items))
frequent_itemsets["itemsets"] = itemsets

st.dataframe(frequent_itemsets)

frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x:len(x))
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=input_lift)
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values("confidence",ascending=False)
rules[~rules["consequents"].str.contains("Age", regex=False)].sort_values("confidence", ascending=False)

rules['antecedents'] = rules['antecedents'].apply(lambda x: set(x))
rules['consequents'] = rules['consequents'].apply(lambda x: set(x))

st.dataframe(rules)

