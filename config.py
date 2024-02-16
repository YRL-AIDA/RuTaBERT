import json


class Config:
    def __init__(self):
        with open("config.json", mode="r") as f:
            self.data = json.load(f)

    def __getitem__(self, item):
        return self.data[item]


if __name__ == "__main__":
    c = Config()
