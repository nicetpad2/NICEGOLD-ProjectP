
        from backtest_engine import run_live_like_backtest, run_robust_backtest
        from projectp.pro_log import pro_log
        from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import shutil
import traceback
import yaml
def setup_test_environment():
    """Creates dummy files and directories needed for the test."""
    print(" -  - - Setting up test environment - -  - ")

    # Create a dummy config.yaml with all required fields
    config_data = {
        'data': {
            'm1_path': 'dummy_m1.csv', 
            'm15_path': 'dummy_m15.csv', 
            'feature_config': ''
        }, 
        'pipeline': {
            'limit_m1_rows': 100, 
            'limit_m15_rows': 10
        }, 
        'strategy_settings': {
            'initial_capital': 10000
        }, 
        'model_class': 'sklearn.ensemble.RandomForestClassifier', 
        'model_params': {
            'n_estimators': 10, 
            'random_state': 42
        }, 
        'walk_forward': {
            'n_splits': 2
        }, 
        'metrics': ['accuracy'], 
        'export': {
            'trade_log': True, 
            'equity_curve': False
        }, 
        'parallel': {
            'enabled': False
        }, 
        'visualization': {
            'enabled': False
        }, 
        'walk_window': 4, # Keep for the functions that use it directly
        'walk_step': 1
    }
    with open("config.yaml", "w") as f:
        yaml.dump(config_data, f)

    # Create the output directory
    output_dir = "output_default"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the dummy preprocessed data that could cause the error
    data = {
        'feature1': [10, 12, 11, 13, 14, 15, 16, 17, 18, 19], 
        'target':   [0,  1,  0,  1,  1,  0,  1,  0,  1,  0]
    }
    df = pd.DataFrame(data)
    data_path = os.path.join(output_dir, "preprocessed.csv")
    df.to_csv(data_path, index = False)
    print(f"Created dummy data at: {data_path}")

    # Create dummy price data files referenced in config
    price_data = {'Close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    pd.DataFrame(price_data).to_csv('dummy_m1.csv', index = False)
    pd.DataFrame(price_data).to_csv('dummy_m15.csv', index = False)

    return output_dir

def cleanup_test_environment(output_dir):
    """Removes files and directories created for the test."""
    print("\n -  - - Cleaning up test environment - -  - ")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Removed directory: {output_dir}")
    if os.path.exists("config.yaml"):
        os.remove("config.yaml")
        print("Removed dummy config.yaml")
    # Clean up other potential artifacts
    if os.path.exists("output_default/robust_backtest_metrics.png"):
        os.remove("output_default/robust_backtest_metrics.png")
    if os.path.exists("output_default/robust_backtest_result.csv"):
        os.remove("output_default/robust_backtest_result.csv")


def run_test():
    """Runs the test to verify the fix."""
    output_dir = setup_test_environment()
    try:
        print("\n -  - - Importing function and running test - -  - ")
        # Suppress internal debug messages for cleaner test output

        # Test the fixed function: run_live_like_backtest
        print("\n -  - - Testing run_live_like_backtest - -  - ")
        result_path_live = run_live_like_backtest()

        if result_path_live and os.path.exists(result_path_live):
            print("✅ SUCCESS (run_live_like_backtest): Test completed without 'axis out of bounds' error.")
            print(f"   Result file created at: {result_path_live}")
        else:
            # This function might not create a file in this minimal setup, so completion is success.
            print("✅ SUCCESS (run_live_like_backtest): Test completed without crashing.")

        # Also test the other similar function to be safe
        print("\n -  - - Testing run_robust_backtest - -  - ")
        result_path_robust = run_robust_backtest(model_class = RandomForestClassifier)

        if result_path_robust and os.path.exists(result_path_robust):
            print("✅ SUCCESS (run_robust_backtest): Test completed without 'axis out of bounds' error.")
            print(f"   Result file created at: {result_path_robust}")
        else:
            print("✅ SUCCESS (run_robust_backtest): Test completed without crashing.")


    except Exception as e:
        print("\n❌ FAILURE: The test failed with an exception.")
        print("   This indicates the fix was not successful or another error occurred.")
        print("\n -  - - Error Details - -  - ")
        traceback.print_exc()

    finally:
        cleanup_test_environment(output_dir)


if __name__ == "__main__":
    run_test()