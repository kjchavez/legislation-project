import tensorflow as tf
import os
import util

# Some aliases
EstimatorSpec = tf.estimator.EstimatorSpec
ModeKeys = tf.estimator.ModeKeys

STATES = [ 'AK', 'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY' ]
PAD = "<PAD>"


def model_fn(features, labels, mode, params, config):
    # The initializer for this table will take care of saving/restoring the "asset".
    # Note: it is not straightforward to roll your own "read from stashed assets" mechanism,
    # and the documentation is scarce.
    table = tf.contrib.lookup.index_table_from_file(params['vocab_filename'], num_oov_buckets=1)

    # In inference mode, we just want to do tokenization and id lookup
    # in the graph itself.
    if mode == ModeKeys.PREDICT:
        # This should be a STRING.
        title_text = features['BillTitle']
        tokens_sparse = tf.string_split(title_text)
        tokens = tf.sparse_to_dense(tokens_sparse.indices, tokens_sparse.dense_shape,
                                    tokens_sparse.values, default_value=PAD)
        token_ids = table.lookup(tokens)
        features["BillTitle"] = token_ids

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
    indicator = tf.feature_column.indicator_column
    # state_embedding = tf.feature_column.embedding_column(state_catagorical, ...)

    x = tf.feature_column.input_layer(
                    features,
                    [indicator(sponsor), indicator(member), indicator(sponsor_member), bucketed_age,
                     indicator(state_categorical)])

    # Word embedding
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.device("/cpu:0"):
        embedding_matrix = tf.get_variable("word_embeddings",
                                           (params['vocab_size'],
                                            params['embedding_dim']),
                                           initializer=initializer
                                          )
        # PAD id, as assigned by keras Tokenizer. Note that this is *entirely inconsistent* with the
        # token assigned by the lookup table at inference time. Which is HUGE problem.
        # TODO(kjchavez): Fix this before serving a new model!
        weights = tf.expand_dims(tf.to_float(tf.not_equal(features['BillTitle'], 0)), -1)
        print ("Weights shape:", weights.get_shape())
        scale = tf.reciprocal(tf.to_float(tf.count_nonzero(weights, 1)))
        print("Scale shape:", scale.get_shape(), scale.dtype)
        # Short titles will just be equivalent to the PAD embedding.. that seems incorrect.
        title_embedding = scale * tf.reduce_sum(tf.nn.embedding_lookup(embedding_matrix,
                                                                features['BillTitle'])*weights,
                                         axis=[1], keep_dims=False)
        print("Title embedding shape:", title_embedding.get_shape())

    x = tf.concat([x, title_embedding], 1)

    hidden = tf.layers.dense(x, 64, activation=tf.nn.relu,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(params['l2_reg']))
    logits = tf.layers.dense(hidden, 1,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(params['l2_reg']))

    predictions = {}
    predictions['aye'] = tf.greater_equal(logits, 0.0)

    # Note there is only 1 logit per example, so no need to use softmax() function.
    predictions['probability'] = tf.sigmoid(logits)
    loss = None
    train_op = None
    eval_metrics = {}
    eval_hooks = []
    if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
        labels = tf.expand_dims(labels, -1)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(labels),
                                                       logits=logits)
        loss = tf.reduce_mean(loss)
        loss += sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.summary.scalar("loss", loss)
        util.add_binary_classification_metrics(predictions['aye'], labels, eval_metrics)

    if mode == ModeKeys.EVAL:
        # Also want to collect samples of errors for manual error analysis.
        with tf.name_scope("error_analysis"):
            # NOTE(kjchavez): To save disk space, we would ideally like to only save a sample of the
            # dev set errors. This can probably be folded into the graph computation.
            tf.summary.tensor_summary('prediction', predictions['aye'],
                                      collections=["ErrorAnalysis"])
            tf.summary.tensor_summary('is_error', tf.not_equal(labels,
                                                               tf.to_int32(predictions['aye'])),
                                     collections=["ErrorAnalysis"])

            tf.summary.tensor_summary('label', labels, collections=["ErrorAnalysis"])
            for name, tensor in features.items():
                tf.summary.tensor_summary(name, tensor, collections=["ErrorAnalysis"])

            if params['save_eval']:
                print("Adding save every step hook!")
                output_dir=os.path.join(config.model_dir, params.get('error_analysis_dir',
                                                                     'error_analysis'))
                eval_hooks.append(tf.train.SummarySaverHook(summary_op=tf.summary.merge(tf.get_collection("ErrorAnalysis")),
                                                            output_dir=output_dir,
                                                            save_steps=1))

    if mode == ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step,
                                                   8000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = util.optimize_and_monitor(optimizer, loss)
        # Let's also track train accuracy.
        train_metrics = {}
        util.add_binary_classification_metrics(predictions['aye'],
                                              labels,
                                              train_metrics)
        tf.summary.scalar("train_acc", train_metrics['accuracy'][0])
        train_op = tf.group(train_op, train_metrics['accuracy'][1])



    outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               tf.estimator.export.PredictOutput(predictions)}

    scaffold=None
    return EstimatorSpec(mode, predictions=predictions, loss=loss, train_op=train_op,
                         scaffold=scaffold,
                         evaluation_hooks=eval_hooks,
                         export_outputs=outputs, eval_metric_ops=eval_metrics)


