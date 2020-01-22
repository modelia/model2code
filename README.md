# model2code

Machine learning architecture for automatic code generation.

### Prerequisites

Python version 2.7 or greater is required.

### Usage

```
$ python translate.py [PARAMS]
```

### Examples


```
$ python translate.py --train_dir /home/lola/experiments/checkpoints --train_data /home/lola/experiments/models_train.json --val_data /home/lola/experiments/models_valid.json --test_data /home/lola/experiments/models_test.json
```

```
$ python translate.py --no_train --test_data /home/lola/experiments/models_test.json --load_model '/home/lola/experiments/neuralnetwork.pth'
```    

## Authors

* **Loli Burgueño**
* **Jordi Cabot**
* **Sébastien Gérard**

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* CEA List, France

