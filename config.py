import json


class Config:
    """Wrapper configuration class.

    Args:
        config_path: Configuration filename.
    """
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, mode="r") as f:
            self.data = json.load(f)

    def __getitem__(self, item):
        return self.data[item]


if __name__ == "__main__":
    c = Config()
