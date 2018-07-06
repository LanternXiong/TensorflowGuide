# task type can be either 'classification' or 'regression', based on the target feature in the dataset
TASK_TYPE = 'classification'

# list of all the columns (header) of the input data file(s)
HEADER = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

# list of the default values of all the columns of the input data, to help decoding the data types of the columns
HEADER_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]

# list of the feature names of type int or float
INPUT_NUMERIC_FEATURE_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

# list of the columns expected during serving (which probably different than the header of the training data)
SERVING_COLUMNS = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

# list of the default values of all the columns of the serving data, to help decoding the data types of the columns
SERVING_DEFAULTS = [[0.0], [0.0], [0.0], [0.0]]

# list of the class values (labels) in a classification dataset
TARGET_LABELS = ['Sentosa', 'Versicolor', 'Virginica']

# target feature name (response or class variable)
LABEL_NAME = 'Species'
