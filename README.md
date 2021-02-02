# Helmet Detection Error Analysis in Football Videos using Amazon SageMaker

### To run the code first step is to create AWS account and use SageMaker instance.

- [Create a regular AWS account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)
- [Create an educational AWS account](https://aws.amazon.com/education/awseducate/apply/)



## Setup

### Create a SageMaker instance
It is recommended to use an instance with GPU support, for example ml.p3.2xlarge. The EBS volume size should be around 50GB in order to store all necessary data.

Network training/inference is a memory-intensive process. If you run into out of GPU memory or out of RAM error, consider decrease the number of `batch_size` during training. 

### Clone this repository
Once the SageMaker instance is successfully launched, open a terminal and follow the commands below:
```shell
$ cd ~/SageMaker/
$ git clone https://github.com/jayeetaghosh/helmet_detection.git
$ cd helmet_detection
```
This will download the repository and take you to the repository directory.

### Activate PyTorch environment
Next, activate conda environment for PyTorch and install required packages.
```shell
$ source activate pytorch_p36
```
Now install required packages.

```shell
$ pip install -r requirements.txt
```

### Download data from Kaggle
Next, download dataset from Kaggle using Kaggle [API](https://github.com/Kaggle/kaggle-api). Please see API [credential](https://github.com/Kaggle/kaggle-api#api-credentials) documentation to retrieve and save kaggle.json file on SageMaker within `/home/ec2-user/.kaggle`. For security reason make sure to change mode for accidental other users `chmod 600 ~/.kaggle/kaggle.json`.

Please launch `Get_Data_And_Explore.ipynb` notebook to download the data and explore before model building. Make sure to select `pytorch_p36` as the kernel

### Build helmet detection model

The modeling script is stored within `src/helmet_detection_model` directory. The training script expects the input dataset to be at `/home/ec2-user/SageMaker/helmet_detection/input/` and model output will be stored at `/home/ec2-user/SageMaker/helmet_detection/model/`. 

```
python src/helmet_detection_model/train_helmet.py 
```



### Evaluate helmet detection model

Please launch `Evaluation.ipynb` to evaluate model. 



## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
The [NOTICE](THIRD-PARTY) includes third-party licenses used in this repository.
