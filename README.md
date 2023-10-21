## **Introduction**
This project is about the implementation of an association rule-based movie recommender system, based on the **`fpgrowth`** algorithm. The system is implemented in Python 3.11.6

### Implementation
The implementation of the recommender system is done and explained in the **`12_recommender.ipynb`** file.

### Choice of Algorithm
The **`fpgrowth`** algorithm is chosen because it is a very efficient algorithm for mining frequent itemsets. It is also very efficient in terms of memory usage, which is important for this assignment, because the dataset is very large. The **`fpgrowth`** algorithm is also very efficient in terms of time complexity.  

### Choice of Data Structures
The **`fpgrowth`** algorithm uses a **`FP-tree`** data structure, which is a modified *Trie* data structure.

***
***

## **Inferences**

### **Graph about average precision**: 
![](/prec.png)

We can see that the precision decreases, then increases. This is because the number of hits remains same for the most part but the recommendation set keeps on expanding till it just gets constant.

### **Graph about average recall**:
![](/recall.png)

We can see that the recall keeps on increasing. This is because the number of hits keeps on increasing until to a point where it becomes constant.


***
***

## ***Thank you !***
