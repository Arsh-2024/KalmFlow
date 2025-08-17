import yaml
from data import load_data
from backtest import rolling_backtest

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    prices = load_data(config)
    df = rolling_backtest(prices, horizon=config['forecast']['horizon'],
                          R=config['model']['R'], Q=config['model']['Q'])
    df.to_csv(config['output']['results_path'], index=False)
    print("Saved results to", config['output']['results_path'])

if __name__ == "__main__":
    main()
