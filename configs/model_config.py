configs = {
    "vanilla_gcn_cora_config": {"hidden_channels": 16, "dropout": 0.5},
    "vanilla_gat_cora_config": {
        "hidden_channels": 16,
        "num_heads": 8,
        "dropout": 0.6,
    },
    "gfusion_cora_config": {
        "hidden_channels": 32,
        "attention_heads": 8,
        "dropout": 0.6,
    },
}


def get_config(model, dataset):
    key = f"{model.lower()}_{dataset.lower()}_config"
    assert key in configs, f"Config for {model} on {dataset} not found!"
    return configs[key]


if __name__ == "__main__":
    print(get_config("GFusion", "Cora"))
