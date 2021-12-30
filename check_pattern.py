import tensorflow as tf
import numpy as np
import os, cv2, sys
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parameters
### General Training
batch_size = 64
dataset_split_random_seed = 48
load_previous_model = './backdoor_models/backdoor_model.h5'
dataset_path = './dataset/VGGFace2/top_1000/'
### Backdoor Training
trigger_layer_num = 3
beta = 0.3
gamma = 0.7
pattern_path = './patterns/'
backdoor_num = 4096
backdoor_target_value = [1000, 1000, 1000, 1000][:trigger_layer_num]
neuron_num = [4096, 4096, 25088, 100352][:trigger_layer_num]
pattern_size = 100
pattern_index_min = 20                                                  # The minimum index of pattern's left-top pixel on the image
pattern_index_max = 224 - pattern_size - pattern_index_min              # The maximum index of pattern's left-top pixel on the image
print_neuron_info = False
tolerable_error = 0.05
### Stable
VGGFace2_mean_BGR = [96.73654238442991, 115.03470094131738, 154.5746553100902]
validation_split_rate = 0.05
target_layer_names = ["fc2", "fc1", "flatten", "block4_pool"]

# Initiallization
if len(sys.argv) == 2:
    load_previous_model = sys.argv[1]
class_list = []
with open('./dataset/VGGFace2/top_1000_class_name.txt', 'r') as f:
    line = f.readline().strip()
    while line: 
        class_list.append(line)
        line = f.readline().strip()
print("Dataset Load Seed :", dataset_split_random_seed)

# Model
model = tf.keras.models.load_model(load_previous_model)
model_outputs = [
    model.get_layer(target_layer_names[0]).output, 
    model.get_layer(target_layer_names[1]).output, 
    model.get_layer(target_layer_names[2]).output,
    tf.keras.layers.Flatten(name='internal_layer_flatten_1')(model.get_layer(target_layer_names[3]).output)
][:trigger_layer_num]
valid_model = tf.keras.Model(inputs=model.input, outputs=model_outputs)

# Loss Function
def loss_normal(y_true, y_pred):
    return tf.reduce_sum(y_true * -1 * tf.math.log(tf.clip_by_value(y_pred,1e-20, 1.0)), axis=1)
def loss_backdoor(y_true, y_pred):
    err = y_true - y_pred
    return tf.sqrt(tf.reduce_mean(tf.math.square(tf.where((y_true > 0) & (err < 0), 0.0, err)), axis=1)) # RMSE
def compute_loss(loss_func, y_true, y_pred):
    return tf.reduce_mean(loss_func(y_true, y_pred))

# Load Dataset
valid_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset_path, 
    label_mode='categorical', 
    class_names=class_list, 
    batch_size=batch_size, 
    image_size=(224, 224), 
    validation_split=validation_split_rate,  
    seed=dataset_split_random_seed, 
    subset='validation')
valid_ds = valid_ds.map(lambda x, y: (tf.reverse(x, axis=[-1]), y))

# Load Patterns
patterns = []
pattern_target = []
targets = [np.zeros((backdoor_num, neuron_num[i]), np.float32) for i in range(trigger_layer_num)]
for i in range(backdoor_num):
    patterns.append(cv2.imread(pattern_path + str(i) + '.png').astype(np.float32))
    for j, value in enumerate(backdoor_target_value):
        targets[j][i][i] = value
patterns = np.array(patterns)
for j in range(trigger_layer_num):
    targets[j] = tf.convert_to_tensor(targets[j])
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.08),
])

# Evaluate Backdoors
valid_ds = valid_ds.repeat(100000)
valid_it = iter(valid_ds)
trigger_values = np.zeros((2, backdoor_num, trigger_layer_num), np.float64)
if print_neuron_info:
    t = range(backdoor_num)
else:
    t = tqdm(range(backdoor_num))
for pattern_id in t:
    input_imgs, input_labels = next(valid_it)
    if input_imgs.shape[0] != batch_size:
        input_imgs, input_labels = next(valid_it)
    input_imgs = input_imgs.numpy()
    for img_id in range(batch_size):
        pattern_pos = np.random.randint(pattern_index_min, pattern_index_max, size=2)
        input_imgs[img_id, pattern_pos[0]:pattern_pos[0]+pattern_size, pattern_pos[1]:pattern_pos[1]+pattern_size, :] = patterns[pattern_id]
    if trigger_layer_num == 1:
        outputs = [valid_model(data_augmentation(input_imgs)-VGGFace2_mean_BGR)]
    else:
        outputs = list(valid_model(data_augmentation(input_imgs)-VGGFace2_mean_BGR))

    for i in range(trigger_layer_num):
        trigger_values[0][pattern_id][i] = tf.math.reduce_mean(outputs[i][:, pattern_id])
        outputs[i] = np.delete(outputs[i].numpy(), pattern_id, axis=1)
        trigger_values[1][pattern_id][i] = np.mean(outputs[i])
    if print_neuron_info:
        template = "{:.2f}"
        print("Neuron :", str(pattern_id).rjust(6), "     Trigger :", end='')
        for i in range(trigger_layer_num):
            print(template.format(trigger_values[0][pattern_id][i]).rjust(10), end='')
            if i < trigger_layer_num - 1:
                print(", ", end='')
        print("     Others :", end='')
        for i in range(trigger_layer_num):
            print(template.format(trigger_values[1][pattern_id][i]).rjust(6), end='')
            if i < trigger_layer_num - 1:
                print(", ", end='')
        print("")

print("Success :")
success_list = []
for i in range(trigger_layer_num):
    success_list.append(list(np.where(np.logical_and(trigger_values[0, :, i] >= backdoor_target_value[i]*(1-tolerable_error), trigger_values[1, :, i] <= 5))[0]))
    print("\tLayer", target_layer_names[i], len(success_list[-1]), end=' ')
    if len(success_list[-1]) != 0:
        with open(load_previous_model[:-3] + "_layer-" + str(target_layer_names[i]) + ".txt", "w") as f:
            f.write("\n".join([str(pattern_id) for pattern_id in success_list[-1]]))
        target_value_mean = 0
        for success_id in success_list[-1]:
            target_value_mean += trigger_values[0, success_id, i]
        target_value_mean /= len(success_list[-1])
        print('mean :', target_value_mean)
    else:
        print(" ")