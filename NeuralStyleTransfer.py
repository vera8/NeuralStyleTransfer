import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    max_dim = 512
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

        plt.imshow(image)
    if title:
        plt.title(title)


def whitenoise_image(width, height):
    arr = np.random.random_sample(size=(height, width, 3))
    img = arr.astype(np.float32)
    img = tf.convert_to_tensor(img)
    img = img[tf.newaxis, :]
    return img


def replace_max_pool_with_average_pool(vgg):
    vgg_outputs = vgg.input
    for layer in vgg.layers[1:]:
        if isinstance(layer, tf.keras.layers.MaxPool2D):
            kwargs = layer.get_config()
            vgg_outputs = tf.keras.layers.AveragePooling2D(**kwargs)(vgg_outputs)
        elif isinstance(layer, tf.keras.layers.Conv2D):
            kwargs = layer.get_config()
            vgg_outputs = tf.keras.layers.Conv2D(**kwargs)(vgg_outputs)
    model = tf.keras.Model([vgg.input], vgg_outputs)
    model.set_weights(vgg.get_weights())
    return model


def vgg_layers(layer_names):
    # Load our model. Load pretrained VGG, trained on ImageNet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg = replace_max_pool_with_average_pool(vgg)
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    x = tf.transpose(input_tensor, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def clip_0_1(input_image):
    return tf.clip_by_value(input_image, clip_value_min=0.0, clip_value_max=1.0)


class StyleContentExtractor(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentExtractor, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)

        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output[0])
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}


class Stylizer:
    def __init__(self, content_image, style_image, content_weight, style_weight, content_layer_names, style_layer_names,
                 total_variation_weight):
        self.content_image = content_image
        self.style_image = style_image
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.content_layers = content_layer_names
        self.style_layers = style_layer_names
        self.total_variation_weight = total_variation_weight
        self.extractor = StyleContentExtractor(self.style_layers, self.content_layers)
        self.content_targets = self.extractor(self.content_image)['content']
        self.style_targets = self.extractor(self.style_image)['style']
        self.image_variable = None
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    @tf.function()
    def train_step(self):
        if self.image_variable is None:
            w, h = tensor_to_image(self.content_image).size
            output_img = whitenoise_image(w, h)
            self.image_variable = tf.Variable(output_img)
        with tf.GradientTape() as tape:
            outputs = self.extractor(self.image_variable)
            content_loss = (self.content_weight / len(self.content_layers)) * self.content_loss(outputs)
            style_loss = (self.style_weight / len(self.style_layers)) * self.style_loss(outputs)
            loss = style_loss + content_loss
            loss = loss + (self.total_variation_weight * tf.image.total_variation(self.image_variable))

        grad = tape.gradient(loss, self.image_variable)
        self.opt.apply_gradients([(grad, self.image_variable)])
        self.image_variable.assign(clip_0_1(self.image_variable))

    def stylize(self, epochs, steps_per_epoch, save_to_file, img_name):
        step = 0
        for i in range(epochs):
            for j in range(steps_per_epoch):
                self.train_step()
                print(".", end='', flush=True)
                step += 1
            image = tensor_to_image(self.image_variable)
            if i == 0:
                image.show("Intermediate result")
            print("Train step: {}".format(step))
            if save_to_file:
                if (i+1) % 10 == 0:
                    print("...save image...")
                    filename = os.path.join(
                        "results",
                        (img_name + "_" +
                         "epoch_" + str(i+1) +
                         "stylew_" + str(self.style_weight) +
                         "_contentw_" + str(self.content_weight) +
                         "_tvw_" + str(self.total_variation_weight) +
                         "_epochs_" + str(epochs) +
                         "_steps_" + str(steps_per_epoch) +
                         ".jpg"))
                    tensor_to_image(self.image_variable).save(filename)
        image = tensor_to_image(self.image_variable)
        image.show("Result")

    def content_loss(self, outputs):
        content_outputs = outputs['content']
        content_loss = tf.add_n([tf.reduce_sum((content_outputs[name] - self.content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss = content_loss / 2.0
        return content_loss

    def style_loss(self, outputs):
        style_outputs = outputs['style']
        m = self.content_image.shape[1] * self.content_image.shape[2]
        style_loss = 0
        for name in style_outputs.keys():
            n = style_outputs[name].shape[1]
            g = style_outputs[name]
            a = self.style_targets[name]
            style_loss += tf.reduce_sum((g - a) ** 2) / (4.0 * (n**2) * (m**2))
        return style_loss


def main():
    os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
    mpl.rcParams['figure.figsize'] = (12, 12)
    mpl.rcParams['axes.grid'] = False
    tf.config.run_functions_eagerly(True)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    content_image = load_img('mountain_cabin.jpg')
    style_image = load_img('window_edited.jpg')
    cabin_small = load_img('mountain_cabin.jpg')
    cabin_beach = load_img('beach_cabin.jpg')
    flowers = load_img('flowers.jpg')
    standard_window = load_img('window_edited.jpg')
    artsy_window = load_img('artsy_window.jpg')
    modern_window = load_img('modern_glass_window.jpg')
    content_image = flowers
    style_image = artsy_window

    style_layers = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1',
        ]
    tv_weight = 0.1
    content_weight = 1e-6
    style_weight = 1e1

    stylizer = Stylizer(
            content_image=content_image,
            style_image=style_image,
            content_weight=content_weight,
            style_weight=style_weight,
            content_layer_names=['block4_conv1'],
            style_layer_names=style_layers,
            total_variation_weight=tv_weight,
        )
    stylizer.image_variable = None
    stylizer.stylize(epochs=20, steps_per_epoch=100, save_to_file=True, img_name="flowers")


if __name__ == "__main__":
    main()
