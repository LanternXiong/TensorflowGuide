import tensorflow as tf
import model
import input
import shutil


train_file_name_list = ['../Data/iris_training.csv']
eval_file_name_list = ['../Data/iris_test.csv']
model_dir = './model_dir'
num_epoch = 60
batch_size = 20
train_steps = 300
learning_rate = 0.0001


def train(run_config):
    train_input_fn = input.input_fn(train_file_name_list,
                                    num_epochs=num_epoch,
                                    batch_size=batch_size)

    eval_input_fn = input.input_fn(eval_file_name_list,
                                   mode=tf.estimator.ModeKeys.EVAL,
                                   batch_size=batch_size)

    opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)

    estimator = model.premade_model_fn(optimizer=opt, config=run_config)

    estimator.train(input_fn=train_input_fn, max_steps=train_steps)

    eval_result = estimator.evaluate(input_fn=eval_input_fn)

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    # expected = ['Setosa', 'Versicolor', 'Virginica']
    # predict_x = {
    #     'SepalLength': [5.1, 5.9, 6.9],
    #     'SepalWidth': [3.3, 3.0, 3.1],
    #     'PetalLength': [1.7, 4.2, 5.4],
    #     'PetalWidth': [0.5, 1.5, 2.1],
    # }

    # predictions = estimator.predict(
    #     input_fn=lambda: iris_data.eval_input_fn(predict_x,
    #                                              labels=None,
    #                                              batch_size=args.batch_size))
    #
    # for pred_dict, expec in zip(predictions, expected):
    #     template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    #
    #     class_id = pred_dict['class_ids'][0]
    #     probability = pred_dict['probabilities'][class_id]
    #
    #     print(template.format(iris_data.SPECIES[class_id],
    #                           100 * probability, expec))


def main():

    run_config = tf.estimator.RunConfig(
        # tf_random_seed=19830610,
        log_step_count_steps=4,
        save_checkpoints_secs=1,  # change frequency of saving checkpoints
        keep_checkpoint_max=3,
        model_dir=model_dir
    )

    train(run_config)


if __name__ == '__main__':
    shutil.rmtree(model_dir)
    main()
