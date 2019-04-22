# Fashion MNIST for CS671 Project

### Instructions
1. Download the whole project

   ```shell
   git clone https://github.com/lordzth666/kaggle-fashion_mnist.git
   ```

2. Install the requirements

   ```shell
   pip install -r requirements.txt
   ```

3. Run the scripts

   **Training script for VGG-10**

   ```
   python train.py --save_dir ./models/model.ckpt
   ```

   **Testing script for VGG-10**

   ```sh
   python test.py
   # Results will be stored in result-test.csv
   ```

   **Training script for SVM**

   ```sh
   python train_svm.py --C 100 --kernel rbf
   # Args: run python train_svm.py -h for help. The setting is similar to SVM.SVC.
   # kernels can be chosen from 'rbf', 'poly' and 'linear'. degree option is only available for 'poly' kernel.
   ```

### Reported Validation Accuracy

|        | Best  | Average | On Site |
| ------ | ----- | ------- | ------- |
| VGG-10 | 95.02 | 94.47   | 93.80   |
| SVM    | 90.04 | 89.27   | —       |

