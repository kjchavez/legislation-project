import tensorflow as tf

# Some aliases
EstimatorSpec = tf.estimator.EstimatorSpec
ModeKeys = tf.estimator.ModeKeys

STATES = [ 'AK', 'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY' ]


# QUESTION. Why wouldn't we just use the pre-built estimator?
# TODO(kjchavez): Consider changing the expectations of train.py to accept an estimator rather than
# an model_function
def model_fn(features, labels, mode, params):  # config, model_dir):
    for key in features:
        if len(features[key].shape) == 1:
            features[key] = tf.expand_dims(features[key], -1)

    for key in features:
        print features[key].shape
    sponsor = tf.feature_column.categorical_column_with_vocabulary_list(
        key='SponsorParty', vocabulary_list=('UNK', 'democrat', 'republican'), default_value=0)
    member = tf.feature_column.categorical_column_with_vocabulary_list(
        key='VoterParty', vocabulary_list=('UNK', 'democrat', 'republican'), default_value=0)
    sponsor_member = tf.feature_column.crossed_column([sponsor, member], 9)
    age = tf.feature_column.numeric_column('VoterAge')
    bucketed_age = tf.feature_column.bucketized_column(age, boundaries=[35, 45, 55, 65, 75, 85])
    state_categorical = tf.feature_column.categorical_column_with_vocabulary_list('VoterState',
                                                                                  STATES)
    # state_embedding = tf.feature_column.embedding_column(state_catagorical, ...)

    x = tf.feature_column.linear_model(
                features,
                [sponsor, member, sponsor_member, bucketed_age, state_categorical])
    x.set_shape((params['batch_size'],1))

    hidden = tf.layers.dense(x, 64, activation=tf.nn.relu)
    logits = tf.layers.dense(hidden, 1)

    predictions = tf.greater_equal(logits, 0.0)
    loss = None
    train_op = None
    eval_metrics = {}
    if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
        labels = tf.expand_dims(labels, -1)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(labels),
                                                       logits=logits)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar("loss", loss)
        eval_metrics['accuracy'] = tf.metrics.accuracy(tf.equal(labels, 1), predictions)


    if mode == ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())


    outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               tf.estimator.export.PredictOutput({'aye': predictions})}
    return EstimatorSpec(mode, predictions=predictions, loss=loss, train_op=train_op,
                         export_outputs=outputs, eval_metric_ops=eval_metrics)

