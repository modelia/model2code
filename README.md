# model2code

Machine learning architecture for automatic code generation.

### Prerequisites

Python version 2.7 or greater is required.

### Usage

```
$ python codegenerator.py [PARAMS]
```

### Examples


```
$ python codegenerator.py --train_dir_checkpoints /home/lola/experiments/checkpoints --training_dataset /home/lola/experiments/models_train.json --validation_dataset /home/lola/experiments/models_valid.json --test_dataset /home/lola/experiments/models_test.json
```

```
$ python codegenerator.py --no_train --test_dataset /home/lola/experiments/models_test.json --load_model '/home/lola/experiments/neuralnetwork.pth'
```    

## Authors

* **Loli Burgueño**
* **Jordi Cabot**
* **Sébastien Gérard**

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* CEA List, France

