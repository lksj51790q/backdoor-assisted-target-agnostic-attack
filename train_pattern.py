import tensorflow as tf
import numpy as np
import os, cv2
from tqdm import tqdm
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parameters
### General Training
training_epochs = 100
epoch_bias = 0
#load_previous_model = './backdoor_models/checkpoint.h5'
learning_rate = 0.00005
batch_size_per_GPU = 128
dataset_split_random_seed = 48
dataset_path = './dataset/VGGFace2/top_1000/'
model_save_path = './backdoor_models/'
### Backdoor Training
trigger_layer_num = 3
beta = 0.3
gamma = 0.7
pattern_path = './patterns/'
backdoor_num = 4096
poison_data_rate = 0.7
backdoor_target_value = [1000, 1000, 1000, 1000][:trigger_layer_num]    # The target trigger value for each layer
neuron_num = [4096, 4096, 25088, 100352][:trigger_layer_num]
target_layer_names = ["fc2", "fc1", "flatten", "block4_pool"]
pattern_size = 100
pattern_index_min = 20                                                  # The minimum index of pattern's left-top pixel on the image
pattern_index_max = 224 - pattern_size - pattern_index_min              # The maximum index of pattern's left-top pixel on the image
### Stable
VGGFace2_mean_BGR = [96.73654238442991, 115.03470094131738, 154.5746553100902]
validation_split_rate = 0.05

# Initiallization
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
class_list = []
with open('./dataset/VGGFace2/top_1000_class_name.txt', 'r') as f:
    line = f.readline().strip()
    while line: 
        class_list.append(line)
        line = f.readline().strip()
print("Dataset Load Seed :", dataset_split_random_seed)
strategy = tf.distribute.MirroredStrategy()
batch_size = strategy.num_replicas_in_sync * batch_size_per_GPU

# Loss Function
with strategy.scope():
    def loss_normal(y_true, y_pred):
        return tf.reduce_sum(y_true * -1 * tf.math.log(tf.clip_by_value(y_pred,1e-20, 1.0)), axis=1)
    def loss_backdoor(y_true, y_pred):
        err = y_true - y_pred
        return tf.sqrt(tf.reduce_mean(tf.math.square(tf.where((y_true > 0) & (err < 0), 0.0, err)), axis=1)) # RMSE
    def compute_loss(loss_func, y_true, y_pred):
        return tf.reduce_mean(loss_func(y_true, y_pred))

# Model
with strategy.scope():
    if "load_previous_model" in locals() and epoch_bias == 0:   # Specific initial model
        model = tf.keras.models.load_model(load_previous_model)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    elif "load_previous_model" in locals():                     # Keep training recent model
        model = tf.keras.models.load_model(load_previous_model)
    else:                                                       # Training a new model
        base_model = tf.keras.applications.VGG16(weights=None)
        penultimate_output = base_model.layers[-2].output
        preds = tf.keras.layers.Dense(len(class_list), activation ='softmax', name='predictions')(penultimate_output)
        model = tf.keras.Model(inputs=base_model.input, outputs=preds, name='backdoor_vgg16')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
        #model.save(model_save_path + 'backdoor_vgg16_' + str(dataset_split_random_seed) + '_init.h5')
    optimizer = model.optimizer
    model_outputs = [
        model.output,
        model.get_layer(target_layer_names[0]).output, 
        model.get_layer(target_layer_names[1]).output, 
        model.get_layer(target_layer_names[2]).output,
        tf.keras.layers.Flatten(name='internal_layer_flatten_1')(model.get_layer(target_layer_names[3]).output)
    ][:1+trigger_layer_num]
    train_pattern_model = tf.keras.Model(inputs=model.input, outputs=model_outputs, name='train_pattern_model')

# Loss and Accuracy Tracing
with strategy.scope():
    valid_normal_loss = tf.keras.metrics.Mean(name='valid_normal_loss')
    valid_backdoor_loss = tf.keras.metrics.Mean(name='valid_backdoor_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')

# Load Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset_path, 
    label_mode='categorical', 
    class_names=class_list, 
    batch_size=batch_size, 
    image_size=(224, 224), 
    validation_split=validation_split_rate, 
    seed=dataset_split_random_seed, 
    subset='training')
train_ds = train_ds.map(lambda x, y: (tf.reverse(x, axis=[-1]), y))
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
targets = [np.zeros((backdoor_num, i), np.float32) for i in neuron_num]
for i in range(backdoor_num):
    patterns.append(cv2.imread(pattern_path + str(i) + '.png').astype(np.float32))
    for j, value in enumerate(backdoor_target_value):
        targets[j][i][i] = value
patterns = np.array(patterns)
for j in range(trigger_layer_num):
    targets[j] = tf.convert_to_tensor(targets[j])

# Data Generator
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.08),
])
def poison_train_data_generator():
    for x, y in train_ds:
        current_batch_size = int(x.shape[0])
        is_poison = np.random.choice([False, True], current_batch_size, replace=True,  p=[1-poison_data_rate, poison_data_rate])
        mask = np.reshape(is_poison, (current_batch_size, 1, 1, 1))
        mask = np.tile(mask, [1, pattern_size, pattern_size, 3])
        x = x.numpy()
        pattern_pos = np.random.randint(pattern_index_min, pattern_index_max, size=2)
        rand_indice = np.random.randint(backdoor_num, size=current_batch_size)
        rand_pattern = np.take(patterns, rand_indice, axis=0)
        origin = x[:, pattern_pos[0]:pattern_pos[0]+pattern_size, pattern_pos[1]:pattern_pos[1]+pattern_size, :]
        x[:, pattern_pos[0]:pattern_pos[0]+pattern_size, pattern_pos[1]:pattern_pos[1]+pattern_size, :] = np.where(mask, rand_pattern, origin)
        data = [data_augmentation(x), tf.convert_to_tensor(is_poison), y]
        for i in range(trigger_layer_num):
            data.append(tf.gather(targets[i], rand_indice, axis=0))
        yield tuple(data)
    return
def poison_valid_data_generator():
    for x, y in valid_ds:
        current_batch_size = int(x.shape[0])
        is_poison = np.random.choice([False, True], current_batch_size, replace=True,  p=[1-poison_data_rate, poison_data_rate])
        mask = np.reshape(is_poison, (current_batch_size, 1, 1, 1))
        mask = np.tile(mask, [1, pattern_size, pattern_size, 3])
        x = x.numpy()
        pattern_pos = np.random.randint(pattern_index_min, pattern_index_max, size=2)
        rand_indice = np.random.randint(backdoor_num, size=current_batch_size)
        rand_pattern = np.take(patterns, rand_indice, axis=0)
        origin = x[:, pattern_pos[0]:pattern_pos[0]+pattern_size, pattern_pos[1]:pattern_pos[1]+pattern_size, :]
        x[:, pattern_pos[0]:pattern_pos[0]+pattern_size, pattern_pos[1]:pattern_pos[1]+pattern_size, :] = np.where(mask, rand_pattern, origin)
        data = [data_augmentation(x), tf.convert_to_tensor(is_poison), y]
        for i in range(trigger_layer_num):
            data.append(tf.gather(targets[i], rand_indice, axis=0))
        yield tuple(data)
    return
output_signature = [
    tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=bool),
    tf.TensorSpec(shape=(None, len(class_list)), dtype=tf.float32)
]
for i in neuron_num:
    output_signature.append(tf.TensorSpec(shape=(None, i), dtype=tf.float32))
output_signature = tuple(output_signature)
train_gen_ds = tf.data.Dataset.from_generator(poison_train_data_generator, output_signature=output_signature)
train_dist_ds = strategy.experimental_distribute_dataset(train_gen_ds)
valid_gen_ds = tf.data.Dataset.from_generator(poison_valid_data_generator, output_signature=output_signature)
valid_dist_ds = strategy.experimental_distribute_dataset(valid_gen_ds)

# Model Functions
with strategy.scope():
    def train_step(inputs):
        images, is_poison, labels = inputs[:3]
        targets = inputs[3:]
        with tf.GradientTape() as tape:
            outputs = train_pattern_model(images-VGGFace2_mean_BGR, training=True)
            preds = outputs[0]
            preds = tf.boolean_mask(preds, tf.math.logical_not(is_poison), axis=0)
            labels = tf.boolean_mask(labels, tf.math.logical_not(is_poison), axis=0)
            normal_loss = beta * compute_loss(loss_normal, labels, preds)
            layer_outputs = outputs[1:]
            backdoor_loss = 0
            for i in range(trigger_layer_num):
                target = tf.boolean_mask(targets[i], is_poison, axis=0)
                layer_output = tf.boolean_mask(layer_outputs[i], is_poison, axis=0)
                backdoor_loss += compute_loss(loss_backdoor, target, layer_output)
            backdoor_loss = gamma * (backdoor_loss / trigger_layer_num)
            loss = normal_loss + backdoor_loss
            gradients = tape.gradient(loss, train_pattern_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, train_pattern_model.trainable_variables))
        train_accuracy.update_state(labels, preds)
        return (normal_loss, backdoor_loss)
    def valid_step(inputs):
        images, is_poison, labels = inputs[:3]
        targets = inputs[3:]
        outputs = train_pattern_model(images-VGGFace2_mean_BGR, training=False)
        preds = outputs[0]
        preds = tf.boolean_mask(preds, tf.math.logical_not(is_poison), axis=0)
        labels = tf.boolean_mask(labels, tf.math.logical_not(is_poison), axis=0)
        normal_loss = beta * compute_loss(loss_normal, labels, preds)
        layer_outputs = outputs[1:]
        backdoor_loss = 0
        for i in range(trigger_layer_num):
            target = tf.boolean_mask(targets[i], is_poison, axis=0)
            layer_output = tf.boolean_mask(layer_outputs[i], is_poison, axis=0)
            backdoor_loss += compute_loss(loss_backdoor, target, layer_output)
        backdoor_loss = gamma * (backdoor_loss / trigger_layer_num)
        valid_normal_loss.update_state(normal_loss)
        valid_backdoor_loss.update_state(backdoor_loss)
        valid_accuracy.update_state(labels, preds)
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        normal_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[0], axis=None)
        backdoor_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[1], axis=None)
        return (normal_loss, backdoor_loss)
    @tf.function
    def distributed_valid_step(dataset_inputs):
        return strategy.run(valid_step, args=(dataset_inputs,))

    best_valid_acc = 0.0
    best_epoch = 0
    for epoch in range(training_epochs):
        # Training
        total_loss = 0.0
        total_normal_loss = 0.0
        total_backdoor_loss = 0.0
        num_batches = 0
        t = tqdm(train_dist_ds, total=len(train_ds), desc='Training')
        for train_batch in t:
            normal_loss, backdoor_loss = distributed_train_step(train_batch)
            total_normal_loss += normal_loss
            total_backdoor_loss += backdoor_loss
            total_loss += normal_loss + backdoor_loss
            num_batches += 1
            template = ("Training  Loss: {:.6f}, normal_loss: {:.4f}, backdoor_loss: {:.4f}, Acc: {:.2f}%")
            t.set_description(template.format(total_loss/num_batches, total_normal_loss/num_batches, total_backdoor_loss/num_batches, train_accuracy.result()*100))
            t.refresh()
        train_normal_loss = total_normal_loss / num_batches
        train_backdoor_loss = total_backdoor_loss / num_batches
        train_loss = total_loss / num_batches
        # Validation
        for valid_batch in tqdm(valid_dist_ds, total=len(valid_ds), desc='Validation'):
            distributed_valid_step(valid_batch)

        # Print Informations And Save Check Point
        if float(valid_accuracy.result()) >= best_valid_acc:
            best_epoch = epoch+epoch_bias+1
            best_valid_acc = float(valid_accuracy.result())
            model.save(model_save_path + 'checkpoint.h5')
        #model.save(model_save_path + '_'.join(['ckps', str(epoch+epoch_bias+1), "{:.4f}".format(valid_accuracy.result()*100)]) + '.h5')
        template = ("Epoch {} | Loss: {}({:.4f}+{:.4f}), Acc: {}% | Valid Loss: {}({:.4f}+{:.4f}), Valid Acc: {}%")
        print(template.format(epoch+epoch_bias+1, train_loss, train_normal_loss, train_backdoor_loss,
            train_accuracy.result()*100, valid_normal_loss.result() + valid_backdoor_loss.result(), 
            valid_normal_loss.result(), valid_backdoor_loss.result(), valid_accuracy.result()*100))
        valid_normal_loss.reset_states()
        valid_backdoor_loss.reset_states()
        train_accuracy.reset_states()
        valid_accuracy.reset_states()
    print("Best epoch: ", best_epoch, "\tAccuracy: ", best_valid_acc*100, "%", sep='')
    os.rename(model_save_path + 'checkpoint.h5', model_save_path + 'backdoor_model.h5')