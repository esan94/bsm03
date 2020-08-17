# Beating Stock Market
This repo contains an actual version of the master disertation beating_stock_market.

All codes are documented in spanish following numpy style guide.

You can see what to expect from this project reading in this medium post [Machine Learning, Fintech & Trading | by: Esteban Sánchez | Medium](https://medium.com/)

### Before starting
First of all you will need a free or premium API key from [Alpha Vantage](https://www.alphavantage.co/) that you have to write on line 7 of ```constants.py```

![AV_API_KEY](images/av_api_key.png)

You have to decide how many tickers or over which tickers you want to use the app, you can change that on lines 59 and 60 of ```constants.py```

![SYMBOLS](images/symbols.png)

Last thing you have to do before to start using it is to unzip the zip files inside ```./data/model```
typing ```unzip file_2_unzip.zip```



![ZIP](images/zip_files.png)

### Run the app
To run the app you have to activate the environment and then type:

```python index.py```
