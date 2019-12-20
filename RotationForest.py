import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode

# An Implementation of the Rotation Forest Algorithm

# Potential Improvements:
# Multithreading on RotationForest.fit
# Clean up RotationTree.get_samples



class RotationTree:
    """
    Base estimator for rotation forest

    Attributes:
        model: sklearn classifier used to make predictions
        n_features: k from forest, number of features for each PCA partition
        partitions: list of how the tree breaks up train data: [[fi, fk],[fi, fk]... etc.]
        rotation_matrices: the associated rotation matrix for each partition


    """
    @classmethod
    def from_model(cls, tree, n_features=3, sample_prop=0.5, bootstrap=True):
        # Allows for use of custom tree params/classifier models, so long as they adhere to sklearn style
        rt = RotationTree(n_features, sample_prop, bootstrap)
        rt.model = tree
        return rt

    def __init__(self, n_features=3, sample_prop=0.5, bootstrap=True):
        self.model = DecisionTreeClassifier(criterion="gini",
                                            splitter="best",
                                            max_depth=None,
                                            min_samples_split=2,
                                            min_samples_leaf=1,
                                            min_weight_fraction_leaf=0.,
                                            max_features=1.0,
                                            random_state=None,
                                            max_leaf_nodes=None,
                                            class_weight=None)
        self.n_features = n_features
        self.sample_prop = sample_prop
        self.bootstrap = bootstrap

        self.partitions = []
        self.rotation_matrices = []

    def fit(self, X, y):
        # Split up + Save features
        feature_partitions = self.partition_features(X)

        # Apply transform and save rotation matrices
        transformed_partitions = []
        for partition in feature_partitions:
            sampled_data = self.get_samples(partition, y)
            rotation_matrix = self.get_rotation_matrix(sampled_data)
            transformed_partitions.append(np.dot(partition, rotation_matrix))

        new_X = np.concatenate(transformed_partitions, axis=1)

        if self.bootstrap:
            xx, yy = self.boot_sample(new_X, y)
            self.model.fit(xx, yy)

        elif not self.bootstrap:
            self.model.fit(new_X, y)

        else:
            raise ValueError("Bootstrap not interpretable as Boolean")

    def partition_features(self, x):
        """Returns list of randomly partitioned features"""
        n_cols = x.shape[1]
        column_cases = [i for i in range(n_cols)]

        # Extract columns and randomly shuffle
        np.random.shuffle(column_cases)

        # Use stride slicing to handle cases where n_cols % n_features != 0
        case_output = [[m for m in column_cases[i::self.n_features]] for i in range(self.n_features)]
        feature_output = []
        for partition in case_output:
            feature_output.append(np.array([x[:,i] for i in partition]).T)

        self.partition_nums = case_output

        return feature_output

    def get_samples(self, featureset, y):
        # Returns Bootstrapped Samples from a subset of features

        # Concatenate features and y
        featureset = np.column_stack([featureset, y])
        n_rows = featureset.shape[0]

        # Choose classes to include
        unique_y = np.unique(y)
        include = []
        for cat in unique_y:
            i = np.random.uniform(size=1)
            if i > 0.5:
                include.append(cat)
        if len(include) == 0:
            include = unique_y

        # Remove non-included
        mask = np.isin(featureset[:,-1], include)
        rotation_seed = featureset[mask,:]

        # Sample from data
        cases = np.random.choice(int(rotation_seed.shape[0]), size=round(self.sample_prop*rotation_seed.shape[0]))
        rotation_seed = rotation_seed[cases,:]

        # Remove y-values
        sample_y = rotation_seed[:,-1]
        rotation_seed = np.delete(rotation_seed, -1, 1)

        return rotation_seed

    def get_rotation_matrix(self, samples):
        # Fits PCA and returns rotation matrix (matrix of eigen vectors)
        pca = PCA()
        pca.fit(samples)
        rotation_matrix = pca.components_
        self.rotation_matrices.append(rotation_matrix)

        return rotation_matrix

    def boot_sample(self, x, y):
        '''
        Sample from transformed data w/replacement, return X, y;
        Could be static/stand-alone but I'm unsure of best practice and this is readable
        '''
        newdata = np.concatenate((x, y[:, np.newaxis]), axis=1)
        cases = np.random.choice(newdata.shape[0], size=newdata.shape[0], replace=True)
        samples = newdata[cases,]

        return samples[:, :-1], samples[:, -1]

    def predict(self, X):
        partitions = [np.array([X[:,i] for i in partition]).T for partition in self.partition_nums]

        transformed_partitions = []
        for i, p in enumerate(partitions):
            transformed_partitions.append(np.dot(p, self.rotation_matrices[i]))

        new_X = np.concatenate(transformed_partitions, axis=1)
        return self.model.predict(new_X)

class RotationForest:
    """

    Algorithm per https://arxiv.org/abs/1809.06705
    Input: k, the number of trees, f, the number of features, p, the sample proportion
    1: Let F =< F1 . . . Fk > be the C4.5 trees in the forest.
    2: for i ← 1 to k do
        3: Randomly partition the original features into r subsets, each with f features (r =
        m/f), denoted < S1 . . . Sr >.
        4: Let Di be the train set for tree i, initialised to the original data, Di ← D.
        5: for j ← 1 to r do
            6: Select a non-empty subset of classes and extract only cases with those class labels.
            Each class has 0.5 probability of inclusion.
            7: Draw a proportion p of cases (without replacement) of those with the selected
            class value
            8: Perform a Principal Component Analysis (PCA) on the features in Sj on this
            subset of data
            9: Apply the PCA transform built on this subset to the features in Sj of the whole
            train set
            10: Replace the features Sj in Di with the PCA features.
        11: Build C4.5 Classifier Fi on transformed data Di.
    """

    def __init__(self, n_trees=1000, n_features=3, sample_prop=0.5, bootstrap=False):
        self.bootstrap = bootstrap
        self.n_trees = n_trees
        self.n_features = n_features
        self.sample_prop = sample_prop

        self.is_fit = False
        self.trees = []
        pass

    def fit(self, X, y, model=None):
        if model:
            for i in range(self.n_trees):
                tree = RotationTree.from_model(model,
                                               n_features=self.n_features,
                                               sample_prop=self.sample_prop,
                                               bootstrap=self.bootstrap)
                tree.fit(X, y)
                self.trees.append(tree)
        else:
            for i in range(self.n_trees):
                tree = RotationTree(n_features=self.n_features,
                                    sample_prop=self.sample_prop,
                                    bootstrap=self.bootstrap)
                tree.fit(X, y)
                self.trees.append(tree)

    def predict(self, X):
        # Generate predictions array
        all = []
        for model in self.trees:
            all.append(model.predict(X))
        all = np.asarray(all)
        preds = mode(all)[0].flatten()

        return preds


# RF Testing
if __name__ == "__main__":
    # Somewhat hacky test to play around with sci-kit generated data sets;
    # Under the current settings, Rotation Forest will usually (but not exclusively) outperform Random Forest.
    # However, it only performs better in some subset of all possible classification problems.

    test_forest = True

    if test_forest:
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        x, y = make_classification(n_features=10, n_samples=1000, n_classes=2, n_clusters_per_class=6, n_informative=8)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        rf = RotationForest(n_trees=100, n_features=3, bootstrap=True)
        rf.fit(X_train, y_train)

        predictions = rf.predict(X_test)

        import pandas as pd
        results = pd.DataFrame([predictions, y_test], index=['Predicted', 'Observed']).T

        results['iseq'] = results.Observed.eq(results.Predicted).astype(int)

        correct = (results.iseq.sum() / len(results))
        print(f"Rotation Forest %Correct: {correct*100}%")

        rf2 = RandomForestClassifier(n_estimators=100)
        rf2.fit(X_train, y_train)
        preds = rf2.predict(X_test)

        results = pd.DataFrame([preds, y_test], index=['Predicted', 'Observed']).T

        results['iseq'] = results.Observed.eq(results.Predicted).astype(int)
        correct = (results.iseq.sum() / len(results))
        print(f"Random Forest %Correct: {correct*100}%")

    else:
        testtree = RotationTree()
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        x, y = make_classification(n_features=50, n_samples=1000)
        trx, tex, trainy, testy = train_test_split(x, y)
        testtree.fit(trx, trainy)
        preds = testtree.predict(tex)
        obs = testy

        import pandas as pd
        df = pd.DataFrame([preds, obs]).T
        df['iseq'] = df[0].eq(df[1]).astype(int)
        print(df.iseq.sum()/len(df))

