configs = {
    "vanilla_gcn_cora_config": {"hidden_channels": 16, "dropout": 0.5},
    "vanilla_gat_cora_config": {
        "hidden_channels": 16,
        "attention_heads": 8,
        "dropout": 0.6,
    },
    "gfusion_cora_config": {
        "hidden_channels": 32,
        "attention_heads": 8,
        "dropout": 0.6,
    },
}

model_key = {
    "VanillaGCN": "vanilla_gcn",
    "VanillaGAT": "vanilla_gat",
    "GFusion": "gfusion",
}

dataset_key = {
    "Cora": "cora",
    "CiteSeer": "citeseer",
}


def get_config(model, dataset):
    assert model in model_key, f"Model {model} not found!"
    assert dataset in dataset_key, f"Dataset {dataset} not found!"
    key = f"{model_key[model]}_{dataset_key[dataset]}_config"
    assert key in configs, f"Config for {model} on {dataset} not found!"
    return configs[key]


if __name__ == "__main__":
    print(get_config("GFusion", "Cora"))
