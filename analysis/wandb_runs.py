import wandb
import pandas as pd

# Initialize W&B API
api = wandb.Api()

# Set project details
project = "AVSP8"
entity = "qwewef"

# Fetch runs
runs = api.runs(f"{entity}/{project}")

# Set the number of runs to export
num_runs = 10  # Change this to the desired number of runs to export

# Collect run data
run_data = []
for run in runs[:num_runs]:
    # Fetch run history
    history = run.history()
    
    # Add run name to each entry in history
    history['run_name'] = run.name
    
    # Append to run_data
    run_data.append(history)

# Convert to DataFrame for easier manipulation/export
df = pd.concat(run_data, ignore_index=True)

# Export to CSV
df.to_csv("analysis/wandb_runs_export.csv", index=False)

print(f"Exported {num_runs} runs to wandb_runs_export.csv")
