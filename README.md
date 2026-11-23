# One-step DAX forecasting

Five Conv1D and LSTM configurations forecast the next DAX index observation. A previous-observation forecast provides the baseline. The experiment uses the first 80% of the series for training and the remaining observations for validation in chronological order.

## Run

Use a Python 3.11 virtual environment with TensorFlow, then run:

```sh
python -m pip install -r requirements.txt
python forecast.py
```

The script trains all five configurations and displays diagnostic plots. For interactive exploration, open `analysis.ipynb` in Jupyter from this directory. The included `data/EuStockMarkets.csv` supplies the DAX series. [Dataset documentation](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/EuStockMarkets.html) · [CSV source](https://vincentarelbundock.github.io/Rdatasets/csv/datasets/EuStockMarkets.csv)

## Reading the output

`results_df` contains the validation metrics from the current training run. The separate `reported_results` table records an earlier coursework run with different training settings. Use the current-run table to compare models you have just trained. The shaded 10% band in the forecast plot is illustrative.
