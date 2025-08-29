import yaml
from data import load_data
from backtest import rolling_backtest
from flask import Flask,redirect,url_for,render_template,request

app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        # Handle POST Request here
        return render_template('index.html')
    return render_template('index.html')


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    prices = load_data(config)
    df = rolling_backtest(prices, horizon=config['forecast']['horizon'],
                          R=config['model']['R'], Q=config['model']['Q'])
    df.to_csv(config['output']['results_path'], index=False)
    print("Saved results to", config['output']['results_path'])

if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000,debug=True)