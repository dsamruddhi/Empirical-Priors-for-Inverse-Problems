class Config:
    config = {
        "data": {
            "gen_data_path": "",
            "real_data_path": "",
            "total_power_path": "",
            "direct_power_path": "",
            "test_size": 0.1,
            "num_samples": 500
        },
        "model": {
            "model_path": "",
            "experiment_name": "test"
        },
        "train": {
            "batch_size": 16,
        },
        "plot": {
            "doi_length": 1.5,
            "cmap": "jet"
        }
}