import pandas as pd
import numpy as np
from getBatches import Batcher



def cat_prop(R):
    # Returns proportion of elements in region R belonging to each category as dictionary (keyed by categories)
    return {cat: list(R).count(cat)/len(R) for cat in set(R)}


def entropy(R):
    # Calculates entropy of region R
    pi_lm = cat_prop(R)
    return - sum([pi_lm[cat] * np.log(pi_lm[cat]) for cat in pi_lm])


def gini_index(R):
    pi_lm = cat_prop(R)
    return sum([pi_lm[cat] * (1 - pi_lm[cat]) for cat in pi_lm])


class DecisionTree:
    class Node:
        def __init__(self, category, split):
            self.split = split
            self.cat = category
            self.left = None
            self.right = None

    def __init__(self, data):
        self.data = data
        self.root = self._split(self.data, depth=0)
        self.loss = gini_index

    def _split(self, df, depth):
        if df['Lead'].value_counts()[0] / len(df) > 0.9:
            return df['Lead'].value_counts().idxmax()
        else:
            split_loss = None
            split_j = None
            split_s = None
            for j in df.columns[:-1]:  # Iterate through columns
                for s in df[j]:  # Iterate through rows
                    right = df[df[j] >= s]
                    left = df[df[j] < s]
                    test_loss = len(left)*gini_index(left['Lead'].to_numpy()) + len(right)*gini_index(right['Lead'].to_numpy())
                    if not split_loss or test_loss < split_loss:
                        split_loss = test_loss
                        split_j = j
                        split_s = s

            # Split current region
            R_right = df[df[split_j] >= split_s]
            R_left = df[df[split_j] < split_s]

            # Create next level
            # print(df['Lead'].value_counts()) # Print current node composition

            # Stopping condition, all predictor values equal
            if len(R_right['Lead']) == len(df['Lead']) or len(R_left['Lead']) == len(df['Lead']):
                return df['Lead'].value_counts().idxmax()
            else:  # Otherwise split
                node = self.Node(split_j, split_s)
                node.right = self._split(R_right, depth=depth+1)
                node.left = self._split(R_left, depth=depth+1)

            return node

    def predict(self, x):
        # Predict for row x (of pandas dataframe)
        return self._predict(self.root, x)

    def _predict(self, node, x):
        if not isinstance(node, self.Node):  # Leaf reached
            return node
        elif x[list(self.data.columns).index(node.cat)] >= node.split:  # Right branch
            return self._predict(node.right, x)
        else:  # Left branch
            return self._predict(node.left, x)


def main():
    batcher = Batcher(batches=10)

    batcher.x_validate(DecisionTree)

    model = batcher.train(DecisionTree)


if __name__ == '__main__':
    main()
