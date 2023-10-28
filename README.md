# Traffic Flow Prediction
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).

## Requirements
- python 3.9-3.11   
- tensorflow-gpu 1.5.0  
- tensorflow 2.13.0
- keras 2.13.1
- scikit-learn 1.3.0
- numpy 1.24.3
- pandas 2.0.3
- matplotlib 3.7.2
- geopy 2.4.0
- pydot 1.4.2
- Node.js (in order to use npm)

## Quick install python packages 
- pip install -r requirements.txt

## Running the application
- cd into the front end
- npm install
- cd into the back end
- npm install
- npm run start
- open the map.html file in the front end folder 

## Train the model

**Run command below to train the model:**

```
python train.py --model model_name
```

You can choose "lstm", "gru", "rnn" or "saes" as arguments. The ```.h5``` weight file was saved at model folder.

**Run command below to run the program:**

```
python main.py --start_scat <scat value> --end_scat <scat value> --date <yyyy-mm-dd> --time 00:00 --model <model type> or python main.py to use the default values
```

These are the details for the traffic flow prediction experiment.


| Metrics | MAE | MSE | RMSE | MAPE |  R2  | Explained variance score |
| ------- |:---:| :--:| :--: | :--: | :--: | :----------------------: |
| LSTM | 7.21 | 98.05 | 9.90 | 16.56% | 0.9396 | 0.9419 |
| GRU | 7.20 | 99.32 | 9.97| 16.78% | 0.9389 | 0.9389|
| SAEs | 7.06 | 92.08 | 9.60 | 17.80% | 0.9433 | 0.9442 |

![evaluate](/images/eva.png)

## Reference

	@article{SAEs,  
	  title={Traffic Flow Prediction With Big Data: A Deep Learning Approach},  
	  author={Y Lv, Y Duan, W Kang, Z Li, FY Wang},
	  journal={IEEE Transactions on Intelligent Transportation Systems, 2015, 16(2):865-873},
	  year={2015}
	}
	
	@article{RNN,  
	  title={Using LSTM and GRU neural network methods for traffic flow prediction},  
	  author={R Fu, Z Zhang, L Li},
	  journal={Chinese Association of Automation, 2017:324-328},
	  year={2017}
	}


## Copyright
See [LICENSE](LICENSE) for details.
