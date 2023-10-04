# from csv import reader
# from collections import defaultdict
# from itertools import chain, combinations
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from random import shuffle

# class Node:
#     def __init__(self, itemName, frequency, parentNode):
#         self.itemName = itemName
#         self.count = frequency
#         self.parent = parentNode
#         self.children = {}
#         self.next = None

#     def increment(self, frequency):
#         self.count += frequency

#     def display(self, ind=1):
#         print('  ' * ind, self.itemName, ' ', self.count)
#         for child in list(self.children.values()):
#             child.display(ind+1)

# def getFromFile(fname):
#     itemSetList = []
#     frequency = []
    
#     with open(fname, 'r') as file:
#         csv_reader = reader(file)
#         for line in csv_reader:
#             line = list(filter(None, line))
#             itemSetList.append(line)
#             frequency.append(1)
#     print(len(itemSetList))
#     return itemSetList, frequency

# def constructTree(itemSetList, frequency, minSup):
#     headerTable = defaultdict(int)
#     # Counting frequency and create header table
#     for idx, itemSet in enumerate(itemSetList):
#         for item in itemSet:
#             headerTable[item] += frequency[idx]

#     # Deleting items below minSup
#     headerTable = dict((item, sup) for item, sup in headerTable.items() if sup >= minSup)
#     if(len(headerTable) == 0):
#         return None, None

#     # HeaderTable column [Item: [frequency, headNode]]
#     for item in headerTable:
#         headerTable[item] = [headerTable[item], None]

#     # Init Null head node
#     fpTree = Node('Null', 1, None)
#     # Update FP tree for each cleaned and sorted itemSet
#     for idx, itemSet in enumerate(itemSetList):
#         itemSet = [item for item in itemSet if item in headerTable]
#         itemSet.sort(key=lambda item: headerTable[item][0], reverse=True)
#         # Traverse from root to leaf, update tree with given item
#         currentNode = fpTree
#         for item in itemSet:
#             currentNode = updateTree(item, currentNode, headerTable, frequency[idx])

#     return fpTree, headerTable

# def updateHeaderTable(item, targetNode, headerTable):
#     if(headerTable[item][1] == None):
#         headerTable[item][1] = targetNode
#     else:
#         currentNode = headerTable[item][1]
#         # Traverse to the last node then link it to the target
#         while currentNode.next != None:
#             currentNode = currentNode.next
#         currentNode.next = targetNode

# def updateTree(item, treeNode, headerTable, frequency):
#     if item in treeNode.children:
#         # If the item already exists, increment the count
#         treeNode.children[item].increment(frequency)
#     else:
#         # Create a new branch
#         newItemNode = Node(item, frequency, treeNode)
#         treeNode.children[item] = newItemNode
#         # Link the new branch to header table
#         updateHeaderTable(item, newItemNode, headerTable)

#     return treeNode.children[item]

# def ascendFPtree(node, prefixPath):
#     if node.parent != None:
#         prefixPath.append(node.itemName)
#         ascendFPtree(node.parent, prefixPath)

# def findPrefixPath(basePat, headerTable):
#     # First node in linked list
#     treeNode = headerTable[basePat][1] 
#     condPats = []
#     frequency = []
#     while treeNode != None:
#         prefixPath = []
#         # From leaf node all the way to root
#         ascendFPtree(treeNode, prefixPath)  
#         if len(prefixPath) > 1:
#             # Storing the prefix path and it's corresponding count
#             condPats.append(prefixPath[1:])
#             frequency.append(treeNode.count)

#         # Go to next node
#         treeNode = treeNode.next  
#     return condPats, frequency

# def mineTree(headerTable, minSup, preFix, freqItemList):
#     # Sort the items with frequency and create a list
#     sortedItemList = [item[0] for item in sorted(list(headerTable.items()), key=lambda p:p[1][0])] 
#     # Start with the lowest frequency
#     for item in sortedItemList:  
#         # Pattern growth is achieved by the concatenation of suffix pattern with frequent patterns generated from conditional FP-tree
#         newFreqSet = preFix.copy()
#         newFreqSet.add(item)
#         freqItemList.append(newFreqSet)
#         # Find all prefix path, constrcut conditional pattern base
#         conditionalPattBase, frequency = findPrefixPath(item, headerTable) 
#         # Construct conditonal FP Tree with conditional pattern base
#         conditionalTree, newHeaderTable = constructTree(conditionalPattBase, frequency, minSup) 
#         if newHeaderTable != None:
#             # Mining recursively on the tree
#             mineTree(newHeaderTable, minSup,
#                        newFreqSet, freqItemList)

# def powerset(s):
#     return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

# def getSupport(testSet, itemSetList):
#     count = 0
#     for itemSet in itemSetList:
#         if(set(testSet).issubset(itemSet)):
#             count += 1
#     return count

# def associationRule(freqItemSet, itemSetList, minConf):
#     rules = []
#     for itemSet in freqItemSet:
#         subsets = powerset(itemSet)
#         itemSetSup = getSupport(itemSet, itemSetList)
#         for s in subsets:
#             confidence = float(itemSetSup / getSupport(s, itemSetList))
#             if(confidence > minConf):
#                 rules.append([set(s), set(itemSet.difference(s)), confidence])
#     return rules

# def getFrequencyFromList(itemSetList):
#     frequency = [1 for i in range(len(itemSetList))]
#     return frequency

# def fpgrowth(itemSetList, minSupRatio, minConf):
#     frequency = getFrequencyFromList(itemSetList)
#     minSup = len(itemSetList) * minSupRatio
#     fpTree, headerTable = constructTree(itemSetList, frequency, minSup)
#     if(fpTree == None):
#         print('No frequent item set')
#     else:
#         freqItems = []
#         mineTree(headerTable, minSup, set(), freqItems)
#         rules = associationRule(freqItems, itemSetList, minConf)
#         return freqItems, rules
    
# def fpgrowthFromFile(fname, minSupRatio, minConf):
#     itemSetList, frequency = getFromFile(fname)
#     print("hi")
#     minSup = len(itemSetList) * minSupRatio
#     fpTree, headerTable = constructTree(itemSetList, frequency, minSup)
#     if(fpTree == None):
#         print('No frequent item set')
#     else:
#         freqItems = []
#         mineTree(headerTable, minSup, set(), freqItems)
#         rules = associationRule(freqItems, itemSetList, minConf)
#         return freqItems, rules

# def printResults(items, rules):
#     # for item in items:
#     #     print(str(item))
#     # sort rules by confidence
#     rules = sorted(rules, key=lambda x: x[2], reverse=True)
#     print('\n------------------------ RULES:')
#     for rule in rules:
#         print(str(rule[0]) + " => " + str(rule[1]) + " conf: " + str(rule[2]))

# movies_df= pd.read_csv('movies.csv')
# ratings_df= pd.read_csv('ratings.csv')

# user_ratings_counts = ratings_df['userId'].value_counts()
# active_users = user_ratings_counts[user_ratings_counts > 10].index
# filtered_ratings_df=ratings_df[ratings_df['rating'] > 2]
# # filtered_ratings_df
# filtered_ratings_df = filtered_ratings_df[filtered_ratings_df['userId'].isin(active_users)]

# transactional_data = filtered_ratings_df.groupby('userId')['movieId'].apply(list).reset_index()
# transactions=transactional_data['movieId'].tolist()

# shuffle(transactions)

# div =0.8

# f1= open("train.csv","w")
# f2= open("test.csv","w")

# for transaction in transactions:
#     length= int(len(transaction)*0.8)
#     for item in transaction[:length]:
#         if item == transaction[length-1]:
#             f1.write(str(item))
#         else:
#             f1.write(str(item)+",")
#     f1.write("\n")
#     for item in transaction[length:]:
#         if item == transaction[-1]:
#             f2.write(str(item))
#         else:
#             f2.write(str(item)+",")
#     f2.write("\n")

# # transactions

# with open("data.csv","w") as f:
#     for transaction in transactions:
#         for item in transaction:
#             if item == transaction[-1]:
#                 f.write(str(item))
#             else:
#                 f.write(str(item)+",")
#         f.write("\n")

# freqItems, rules = fpgrowthFromFile("train.csv", 0.05, 0.2)
# # for rule in rules:
# #     for item in rule:
# #         for i in range(len(item)):
# #             item[i]=movies_df[movies_df["movieId"]==item[i]].title.values[0]
# #         print(item)
# #     print("\n")

# printResults(freqItems, rules)

from mlxtend.frequent_patterns import fpgrowth
import pandas as pd

fpgrowth_df = pd.read_csv('train.csv', header=None)
# fpgrowth_df.head()
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(fpgrowth_df).transform(fpgrowth_df)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = fpgrowth(fpgrowth_df, min_support=0.05, use_colnames=True)
# print rules
from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

for i in range(len(rules)):
    rules.iloc[i,0]=rules.iloc[i,0].replace("{","")
    rules.iloc[i,0]=rules.iloc[i,0].replace("}","")
    rules.iloc[i,1]=rules.iloc[i,1].replace("{","")
    rules.iloc[i,1]=rules.iloc[i,1].replace("}","")
    rules.iloc[i,0]=rules.iloc[i,0].split(",")
    rules.iloc[i,1]=rules.iloc[i,1].split(",")
    for j in range(len(rules.iloc[i,0])):
        rules.iloc[i,0][j]=int(rules.iloc[i,0][j])
    for j in range(len(rules.iloc[i,1])):
        rules.iloc[i,1][j]=int(rules.iloc[i,1][j])

# for i in range(len(rules)):
#     for j in range(len(rules.iloc[i,0])):
#         rules.iloc[i,0][j]=movies_df[movies_df["movieId"]==rules.iloc[i,0][j]].title.values[0]
#     for j in range(len(rules.iloc[i,1])):
#         rules.iloc[i,1][j]=movies_df[movies_df["movieId"]==rules.iloc[i,1][j]].title.values[0]

# rules
print(rules)