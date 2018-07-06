import os

import tensorflow as tf

import featurizer
import model
import input
import shutil


train_file_name_list = ['../Data/iris_training.csv']
eval_file_name_list = ['../Data/iris_test.csv']
model_checkpoint_dir = './model_checkpoint_dir'
model_saved_dir = './model_saved_dir'
num_epoch = 200
batch_size = 20
train_steps = 1000
learning_rate = 0.01


def train(run_config):
    train_input_fn = input.input_fn(train_file_name_list,
                                    num_epochs=num_epoch,
                                    batch_size=batch_size)

    eval_input_fn = input.input_fn(eval_file_name_list,
                                   mode=tf.estimator.ModeKeys.EVAL,
                                   batch_size=batch_size)

    opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)

    estimator = model.premade_model_fn(optimizer=opt, config=run_config)
    # estimator = tf.estimator.Estimator(
    #     model_fn=model.customized_model,
    #     params={
    #         'feature_columns': list(featurizer.create_feature_columns().values()),
    #         # Two hidden layers of 10 nodes each.
    #         'hidden_units': [10, 30],
    #         # The model must choose between 3 classes.
    #         'n_classes': 3,
    #     })

    estimator.train(input_fn=train_input_fn, max_steps=train_steps)

    eval_result = estimator.evaluate(input_fn=eval_input_fn)

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    estimator.export_savedmodel(model_saved_dir, input.serving_input_fn)


def main():
    if os.path.exists(model_checkpoint_dir):
        shutil.rmtree(model_checkpoint_dir)
    run_config = tf.estimator.RunConfig(
        tf_random_seed=666,
        log_step_count_steps=4,
        save_checkpoints_secs=3,  # change frequency of saving checkpoints
        keep_checkpoint_max=3,
        model_dir=model_checkpoint_dir
    )

    train(run_config)


if __name__ == '__main__':
    main()
