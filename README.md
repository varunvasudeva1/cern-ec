# cern-ec
Neural network to predict invariant mass of electron pair collision @ CERN

Data source: https://www.kaggle.com/fedesoriano/cern-electron-collision-data

An analysis of the data was done first to explore the characteristics of and relationships between the
variables in the CERN Electron Collision data set. This included an exploratory data analysis consisting
of multiple plots to analyze the data holistically in order to build a prediction model. Based on the type
of structured data we had, a neural network seemed best compared to other regressors from the Sci-Kit Learn
platform. We achieve a minimum RMSE of 0.4984 evaluated on a test set consisting of 15% of the entire data
set. The best model consists of four Dense layers with 256, 128, 64, and 1 units respectively. The last Dense
layer has one unit so as to yield a final singular value to evaluate against the test set's value for M.
The model architecture was experimentally tuned multiple times with different values for the first Dense
layer (512, 384, 256) and epochs (25, 30, 50, 100, 150) and the best performing model trained for 100 epochs.

To ensure our model was the best one we could achieve, we used Keras Tuner to potentially improve the model
accuracy. The tuner searched the space for the optimal number of units in the first Dense layer. It also
trained for 100 epochs, but didn't result in a lower RMSE than our original model, settling at a value of
0.7108.

Considering the RMSE value we achieve with the neural network with the range of the target variable M in mind,
the model is able to accurately predict the invariant mass of the electrons to a large extent.

![model-architecture](https://user-images.githubusercontent.com/37934117/111112078-f085e600-8535-11eb-90e3-2947bb648e88.png)

![model-mse](https://user-images.githubusercontent.com/37934117/111112122-01365c00-8536-11eb-9404-2ee281e700e8.png)

![model-loss](https://user-images.githubusercontent.com/37934117/111112144-085d6a00-8536-11eb-8a50-90d689a577bf.png)

