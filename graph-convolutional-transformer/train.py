import graph_convolutional_transformer as gct
import tensorflow as tf

def main(argv):
  gct_params = {
      "embedding_size": 128,
      "num_transformer_stack": 3,
      "num_feedforward": 2,
      "num_attention_heads": 1,
      "ffn_dropout": 0.08,
      "attention_normalizer": "softmax",
      "multihead_attention_aggregation": "concat",
      "directed_attention": False,
      "use_inf_mask": True,
      "use_prior": True,
  }

  input_path = argv[1]
  model_dir = argv[2]
  num_iter = 1000000
  model = gct.EHRTransformer(
      gct_params=gct_params,
      reg_coef=0.1,
      learning_rate=0.00022,
      batch_size=32)
  config = tf.estimator.RunConfig(save_checkpoints_steps=100)

  estimator = tf.estimator.Estimator(
      model_dir=model_dir, model_fn=model.model_fn, config=config)

  train_spec = tf.estimator.TrainSpec(
      input_fn=lambda: model.input_fn(input_path + './train.tfrecord', True),
      max_steps=num_iter)

  eval_spec = tf.estimator.EvalSpec(
      input_fn=lambda: model.input_fn(input_path + './validation.tfrecord', False),
      throttle_secs=1)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  estimator.evaluate(
      input_fn=lambda: model.input_fn(input_path + './validation.tfrecord', False))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
