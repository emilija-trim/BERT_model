import tensorflow as tf


class F1ScoreMetric(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1ScoreMetric, self).__init__(name=name, **kwargs)
        # True positives, false positives, false negatives
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Create mask for -100 values
        mask = tf.not_equal(y_true, -100)
        # Get valid positions
        y_true_valid = tf.boolean_mask(y_true, mask)
        
        # Get predicted classes from logits
        y_pred_valid = tf.boolean_mask(tf.argmax(y_pred, axis=-1), mask)
        
        # One-hot encode for multi-class F1 calculation
        num_classes = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(tf.cast(y_true_valid, tf.int32), num_classes)
        y_pred_oh = tf.one_hot(y_pred_valid, num_classes)
        
        # Update tp, fp, fn
        tp_update = tf.reduce_sum(y_true_oh * y_pred_oh)
        fp_update = tf.reduce_sum(y_pred_oh) - tp_update
        fn_update = tf.reduce_sum(y_true_oh) - tp_update
        
        self.tp.assign_add(tp_update)
        self.fp.assign_add(fp_update)
        self.fn.assign_add(fn_update)

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        return f1
    
    def reset_state(self):
        self.tp.assign(0.)
        self.fp.assign(0.)
        self.fn.assign(0.)