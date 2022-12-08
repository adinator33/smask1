import numpy as np
import pandas as pd


class Batcher:
    def __init__(self, file='train.csv', batches=16):
        df = pd.read_csv(file)
        df.dropna(inplace=True)
        df.sample(frac=1, random_state=12)  # Get batches with seed
        self.df = df
        self.batches = np.array_split(df, batches)

    def get_batch(self, i=0):
        return self.batches[i]

    def get_train_test(self, i=0):
        train = pd.concat(self.batches[:i] + self.batches[i+1:])
        test = self.batches[i]
        return train, test

    def __iter__(self):
        for j in range(len(self.batches)):
            yield self.get_train_test(j)

    def x_validate(self, Model):
        # Requires model to have prediction defined as model.predict for single input
        prediction_score = []
        male_score = []
        i = 0
        for train, test in self:
            print(f'Running batch {i}...')
            i += 1

            mdl = Model(train)  # Create and train model

            result_model = [mdl.predict(x[:-1]) == x[-1] for x in test.to_numpy()]  # Calculate model result
            prediction_score.append(sum(result_model)/len(result_model))  # Store model result

            male_score.append(sum(test.to_numpy()[:, -1] == 'Male')/len(test))  # Calculate and store male guess result

        print(f"Model success rate: {np.mean(prediction_score)*100:2.2f}%")
        print(f"Male guess success rate: {np.mean(male_score)*100:2.2f}%")


    def train(self, Model):
        return Model(self.df)


def main():
    pass


if __name__ == '__main__':
    main()
