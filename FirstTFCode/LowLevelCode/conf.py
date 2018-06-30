data_conf = {'data_input_path': ['../Data/iris_training.csv'],
             'record_defaults': [[0.0], [0.0], [0.0], [0.0], [0.0]]}

model_conf = {'layer_conf':
                  {'input_node': 4,
                   'hidden_node': [10, 20, 50],
                   'output_node': 1},
              'model_save_path': './model_dir'
              }

training_conf = {'batch_size': 10,
                 'num_epochs': 100,
                 'learning_rate': 0.001,
                 'regularization_rate': 0.0001,
                 'training_steps': 500}

