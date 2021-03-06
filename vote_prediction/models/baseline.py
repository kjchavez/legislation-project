"""
    Simple baseline prediction model.

    Dead simple baseline model: Same party -> votes aye. Perhaps a bit too cynical.
"""
import tensorflow as tf

# Some aliases
EstimatorSpec = tf.estimator.EstimatorSpec
ModeKeys = tf.estimator.ModeKeys

def zero_one_loss(pred, target):
    return tf.reduce_mean(tf.to_float(tf.equal(pred, target)))

# Satisfies model function signature for Estimator API.
def model_fn(features, labels, mode, params):  # config, model_dir):
    """ Predicts true if 'SponsorParty' == 'VoterParty'. """
    with tf.device('/cpu:0'):
        party_match = tf.equal(features['VoterParty'], features['SponsorParty'])
        predictions = tf.to_int32(party_match)

        loss = None
        train_op = None
        # Can't train the model. It's rule-based.
        if mode == ModeKeys.TRAIN:
            loss = zero_one_loss(predictions, labels)
            step = tf.train.get_or_create_global_step()
            train_op = tf.assign_add(step, 1)

        elif mode == ModeKeys.EVAL:
            loss = zero_one_loss(tf.to_int32(party_match), labels)

        outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput({'aye': predictions})}
        return EstimatorSpec(mode, predictions=predictions, loss=loss, train_op=train_op,
                             export_outputs=outputs)

