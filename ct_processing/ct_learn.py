from sklearn.neural_network import MLPClassifier

def mlp_classifier(values_train, values_test, class_train):
    """Apply mlp.
        Arguments:
            values_train {numpy.array} -- features for train.
            values_test {numpy.array} -- features for test.
            class_train {Series} -- data classes for training.
        Returns:
            class_train {numpy.array} -- classes provided by the mlp
    """
    # setting parameters
    mlp = MLPClassifier(hidden_layer_sizes=(30, 30), max_iter=100)

    # training with features
    mlp.fit(values_train, class_train.values)
    ans_mlp_test = mlp.predict(values_test)
    return ans_mlp_test