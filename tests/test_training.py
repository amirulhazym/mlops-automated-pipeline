# tests/test_training.py
# To test our script, we need to import the function we want to test
# This line imports the `setup_arg_parser` function from our main training script
from src.training.train_model import setup_arg_parser


# --- Test Case 1: Checking Default Arguments ---
def test_arg_parser_defaults():
    """
    This test checks if the argument parser provides the correct
    default values when no arguments are given from the command line.
    """
    # Arrange: We create an instance of our argument parser
    parser = setup_arg_parser()

    # Act: We parse an empty list of arguments, simulating a run with no command-line flags.
    # parser.parse_args([]) is how you test this scenario.
    args = parser.parse_args([])

    # Assert: We check if the parsed arguments have the expected default values.
    # If any of these `assert` statements are false, the test will fail.
    assert args.data_version == "sample_1000000"
    assert args.model_type == "xgboost"
    assert args.learning_rate == 0.001  # This default comes from the latest script
    assert args.n_estimators == 100
    assert args.epochs == 10
    assert args.random_state == 50


# --- Test Case 2: Checking Custom Arguments for XGBoost ---
def test_arg_parser_xgboost_custom():
    """
    This test checks if the argument parser correctly handles
    custom arguments provided for an XGBoost run.
    """
    # Arrange: We define a list of strings that mimics command-line input.
    custom_args_list = [
        "--model_type",
        "xgboost",
        "--data_version",
        "full",
        "--run_name",
        "my_custom_xgb_run",
        "--n_estimators",
        "200",
    ]

    # Arrange: We create an instance of our argument parser
    parser = setup_arg_parser()

    # Act: We parse our custom list of arguments.
    args = parser.parse_args(custom_args_list)

    # Assert: We check if the arguments were correctly parsed.
    assert args.model_type == "xgboost"
    assert args.data_version == "full"
    assert (
        args.run_name == "my_custom_xgb_run"
    )  # The parses's desfault is None, so we expect it to be None
    assert args.n_estimators == 200


# --- Test Case 3: Checking Custom Arguments for TensorFlow ---
def test_arg_parser_tensorflow_custom():
    """
    This test checks if the argument parser correctly handles
    custom arguments provided for a TensorFlow run.
    """
    # Arrange: We define a list of strings for a TensorFlow scenario.
    custom_args_list = [
        "--model_type",
        "tensorflow_mlp",
        "--data_version",
        "full",
        "--run_name",
        "my_custom_tf_run",
        "--epochs",
        "20",
        "--batch_size",
        "128",
    ]

    # Arrange: We create an instance of our argument parser
    parser = setup_arg_parser()

    # Act: We parse the custom TensorFlow arguments.
    args = parser.parse_args(custom_args_list)

    # Assert: We check the parsed arguments.
    assert args.model_type == "tensorflow_mlp"
    assert args.data_version == "full"
    assert (
        args.run_name == "my_custom_tf_run"
    )  # The parser's default is None, so we expect it to be None
    assert args.epochs == 20
    assert args.batch_size == 128
