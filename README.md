# Whakaari
Eruption forecast model for Whakaari (White Island, New Zealand). This model implements a time series feature engineering and classification workflow that issues eruption alerts based on real-time tremor data. More details in the [paper](https://www.nature.com/articles/s41467-020-17375-2).

## Installation

Ensure you have Anaconda Python 3.7 installed. Then

1. Clone the repo

```bash
git clone https://github.com/ddempsey/whakaari
```

2. CD into the repo and create a conda environment

```bash
cd whakaari

conda env create -f environment.yml

conda activate whakaari_env
```

The installation has been tested on Windows, Mac and Unix operating systems. Total install with Anaconda Python should be less than 10 minutes.

## Running models
Three examples have been included in ```scripts/forecast_model.py```. 

The first, ```forecast_test()``` trains on a small subset of tremor data in 2012 and then constructs a forecast of the Aug 2013 eruption. It will take about 10 minutes to run on a desktop computer and produces a forecast image in ../plots/test/forecast_Aug2013.png

The second, ```forecast_dec2019()``` trains on tremor data between 2011 and 2020 but *excluding* a two month period either side of the Dec 2019 eruption. It then constructs a model 'forecast' of this event. It could take several hours or a day to run depending on the cpus available for your computer.

The third, ```forecast_now()```, is an online forecaster. It trains a model on all data between 2011 and 2020 *including* the Dec 2019 eruption. It then downloads the latest Whakaari tremor data from GeoNet and constructs a forecast for the next 48 hours. See associated paper for guidance on interpreting model consensus levels. The model may take several hours or a day the first time it is constructed, but subsequent updates should be quick.

To run the models, open ```forecast_model.py```, comment/uncomment the forecasts you want to run, then in a terminal type
```bash
cd scripts

python forecast_model.py
```

## Disclaimers
1. This eruption forecast model is not guaranteed to predict every future eruption, it only increases the likelihood. In our paper, we discuss the conditions under which the forecast model is likely to perform poorly.

2. This eruption forecast model provides a probabilistic prediction of the future. During periods of higher risk, it issues an alert that an eruption is *more likely* to occur in the immediate future. At present, our best estimate is that when an alert is issued at 80% consensus there is a 1 in 12 chance of an eruption occurring. On average, alerts last about 5 days. Eruption probability and average alert length could change depending on which eruptions are used to train the model.

3. This software is not guaranteed to be free of bugs or errors. Most codes have a few errors and, provided these are minor, they will have only a marginal effect on accuracy and performance. That being said, if you discover a bug or error, please report this at [https://github.com/ddempsey/whakaari/issues](https://github.com/ddempsey/whakaari/issues).

4. This is not intended as an API for designing your own eruption forecast model. Nor is it designed to be especially user-friendly for Python/Machine Learning novices. Nevertheless, if you do want to adapt this model for another volcano, we encourage you to do that and are happy to answer queries about the best way forward. 

## Acknowledgments
This eruption model would not be possible without real-time seismic data streaming from Whakaari via GeoNet.

