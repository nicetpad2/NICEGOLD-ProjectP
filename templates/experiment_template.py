# Experiment Template
from tracking import start_experiment

def run_experiment():
    """Template for running ML experiments"""
    with start_experiment("experiment_name", "run_name") as exp:
        # Log parameters
        exp.log_params({
            "param1": "value1",
            "param2": "value2"
        })
        
        # Your ML code here
        # model = train_model()
        # results = evaluate_model(model)
        
        # Log metrics
        exp.log_metrics({
            "accuracy": 0.95,
            "loss": 0.05
        })
        
        # Log model
        # exp.log_model(model, "my_model")
        
        print("Experiment completed!")

if __name__ == "__main__":
    run_experiment()
