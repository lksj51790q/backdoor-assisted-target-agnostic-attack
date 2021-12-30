# Backdoor-Assisted Target-Agnostic Attack for Deep Neural Classifiers in Transfer Learning Model

### Abstract
&ensp;&ensp;&ensp;&ensp;Transfer learning is extensively used to overcome the problem that training a deep neural network from scratch is expensive. Because transfer learning reduces the training parameters, it allows the model training with a smaller dataset. But it needs a pre-trained model to obtain partial parameters, which may cause vulnerability due to the information of the pre-trained model is public. In a
<a href="https://openreview.net/forum?id=BylVcTNtDS">target-agnostic attack</a>,
the attacker can launch an effective and efficient brute force attack to trigger each target class with high confidence even if the attacker only knows the information of the pre-trained model. However, the original attack cannot craft adversarial inputs with keeping the range of input values normal. To solve the problem, we turn the role of the attacker into the provider of the pre-trained model. By setting up the backdoors, we improve the attack and further make it more robust. We evaluate the attack performance in various situations and analyze the critical factors. Furthermore, we propose a defense designed by the analysis results. Our defense can completely break the attack and is also effective to the original attack. It can keep the advantages of transfer learning and is able to combine with other approaches of transfer learning.

### Installation
```
git clone https://github.com/lksj51790q/backdoor-assisted-target-agnostic-attack.git
cd backdoor-assisted-target-agnostic-attack
pip install -r requirements.txt
```

### Dataset Preparing
The VGGface2 dataset is used. It contains 3,311,286 images and consists of 9,131 identities. The training dataset of teacher models is the top 1000 identities with the most images. And the training dataset of student models is the bottom 20 identities with at least 200 images.
1. Download dataset:&emsp;<a href="https://www.robots.ox.ac.uk/~vgg/data/vgg_face/">VGGFace2</a><br>
2. Crop only face and resize images to 224 x 224<br>
3. Copy the top 1000 identities with the most images to dataset/VGGFace/top_1000/<br>
4. Copy the bottom 20 identities with at least 200 images to dataset/VGGFace/gt_200_last_20/<br>

### Generate Backdoor Trigger Patterns
Before training the teacher model with backdoor, you have to generate backdoor patterns first.
```
python generate_pattern.py
```
Patterns will be generate at the folder "patterns".<br>

### Train Teacher Model
To train the teacher model without backdoor:<br>
```
python train_normal.py
```
To train the backdoor model:<br>
```
python train_backdoor.py
```

### Check How Many Patterns Are Successfully Trained
This will print the number of successfully trained patterns and the mean value they trigger the target neurons and record the information at the file.<br>
```
python check_pattern.py
```

### Transfer Learning
To do the most common transfer learning or fine-tuning:
```
python transfer_learning/transfer_learning.py
```
To do the <a href="http://proceedings.mlr.press/v80/li18a.html">L<sup>2</sup>-SP</a> or <a href="https://openreview.net/forum?id=rkgbwsAcYm">DELTA</a>:
```
python transfer_learning/L2_SP.py
python transfer_learning/DELTA.py
```
To do some special transfer learning settings like more re-training, tuning, or additional layers:
```
python transfer_learning/special_transfer_learning.py <retrain_layer_num> <tune_layer_num> <add_layer_num>
```
To do the proposed defense mothod:
```
python transfer_learning/Defence.py
```

### Evaluate The Backdoor Attack
This will evaluate the NABAC(Number of attempts to break all classes) and the Effectiveness(99%, 95%).
```
python evaluate_backdoor.py
```

### Analyze Results
To compute the class distribution and its standard deviation of the student model:
```
python analysis/compute_class_distribution.py
```
To simulate the attack result that if all backdoors are successfully be installed:
```
python analysis/perfect_attack_simulate.py
```
To randomly simulate the NABAC by setting specific SDCD(standard deviation of class distribution) and effectiveness(99%):
```
python analysis/simulate_NABAC.py <SDCD> <effectiveness(99%)>
```
