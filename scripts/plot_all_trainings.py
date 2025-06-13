import json
from mlp.config import OUTPUT_DIR
from pathlib import Path
from mlp.model.plot import plot_metric_compare


def plot_all(data_trainings: dict) -> None:

    plot_metric_compare("loss", data_trainings, filename="All_trainings_loss_comparison.png", legend="")
    plot_metric_compare("accuracy", data_trainings, filename="All_trainings_accuracy_comparison.png", legend="")


def upload_JSON_to_plot(json_files_path: list[Path]) -> dict:

    if not json_files_path:
        print("Nothing here to plot !")
        return
    
    data_trainings = {}
    
    try:
        for filepath in json_files_path:
            with open(filepath, mode='r') as file:
                data = json.load(file)
                if not data:
                    print(f"⚠️ {filepath.name}: empty file")
                    continue
                
                model_name = filepath.stem.replace('training_results_', '')
                data_trainings[model_name] = {
                                        "config": data.get('config', {}),
                                        "learning_rate": data.get('learning_rate', 'unknown'),
                                        "loss_history": data.get('loss_history', []),
                                        "accuracy_history": data.get('accuracy_history', []),
                }
                print(data_trainings[model_name])

    except json.JSONDecodeError:
        print(f"Something happend while reading JSON file!")

    return data_trainings


# ++++++++++++++++++++++++++++++++++++++++++++++ MAIN +++++++++++++++++++++++++++++++++++++++++++++++++++

def main():

    json_files_path = list(OUTPUT_DIR.glob("training_results_*.json"))
    data_trainings = upload_JSON_to_plot(json_files_path)
    plot_all(data_trainings)

if __name__ == "__main__":
    main()