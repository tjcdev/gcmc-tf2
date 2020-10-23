import tensorflow as tf


def softmax_accuracy(preds, labels):
    """
    Accuracy for multiclass model.
    :param preds: predictions
    :param labels: ground truth labelt
    :return: average accuracy
    """
    correct_prediction = tf.equal(tf.argmax(input=preds, axis=1), tf.cast(labels, dtype=tf.int64))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(input_tensor=accuracy_all)


def expected_rmse(logits, labels, class_values=None):
    """
    Computes the root mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth label
    :param class_values: rating values corresponding to each class.
    :return: rmse
    """

    probs = tf.nn.softmax(logits)
    if class_values is None:
        scores = tf.cast(tf.range(start=0, limit=logits.get_shape()[1]) + 1, dtype=tf.float32)
        y = tf.cast(labels, dtype=tf.float32) + 1.  # assumes class values are 1, ..., num_classes
    else:
        scores = class_values
        y = tf.gather(class_values, labels)

    pred_y = tf.reduce_sum(input_tensor=probs * scores, axis=1)

    diff = tf.subtract(y, pred_y)
    exp_rmse = tf.square(diff)
    exp_rmse = tf.cast(exp_rmse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(input_tensor=exp_rmse))


def rmse(logits, labels, class_values=None):
    """
    Computes the mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth labels for the ratings, 1-D array containing 0-num_classes-1 ratings
    :param class_values: rating values corresponding to each class.
    :return: mse
    """

    if class_values is None:
        y = tf.cast(labels, dtype=tf.float32) + 1.  # assumes class values are 1, ..., num_classes
    else:
        y = tf.gather(class_values, labels)

    pred_y = logits

    diff = tf.subtract(y, pred_y)
    mse = tf.square(diff)
    mse = tf.cast(mse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(input_tensor=mse))


def softmax_cross_entropy(outputs, labels):
    """ computes average softmax cross entropy """

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels)
    return tf.reduce_mean(input_tensor=loss)
