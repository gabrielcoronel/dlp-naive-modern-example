import numpy
import tensorflow
import functools
from tensorflow.keras.datasets import mnist as dataset


def batcher(input, output, size):
    """
    Generator function used to parallely iterate over input and output
    in batches of the given size

    This is used during the training process
    """
    assert len(input) == len(output), "Input and output correspondence is not one to one"
    
    amount_of_batches = int(len(input) / size)

    for index in range(amount_of_batches):
        input_batch = input[index * size:(index + 1) * size]
        output_batch = output[index * size:(index + 1) * size]

        yield (input_batch, output_batch)


class Layer:
    """
    Class used to represent a single layer in the neural network
    """

    def __init__(self, height, width, activator):
        """
        Initializes a layer with a null bias and a random transformation
        """

        bias = tensorflow.zeros((width,))
        transformation = tensorflow.random.uniform(
            shape=(height, width),
            minval=0,
            maxval=1e-1
        )

        self._bias = tensorflow.Variable(bias)
        self._transformation = tensorflow.Variable(transformation)
        self._activator = activator

    def apply(self, input):
        """
        Applies this layer's activator (hypothesis space) to the given input
        using the currently trained transformation and bias as parameters
        """

        transformed_input = self._activator(
            tensorflow.matmul(input, self._transformation) + self._bias
        )

        return transformed_input


    @property
    def weights(self):
        """
        Returns this layer's weights in a compound tensor
        """

        return [self._transformation, self._bias]


class Network:
    """
    Class used to represent a whole neural network
    """

    def __init__(self, layers):
        """
        Fills the neural network with the provided layers
        """

        self._layers = layers

    def predict(self, input):
        """
        Makes a prediction for the given input running the chain of provided
        layers
        """

        prediction = functools.reduce(
            lambda partial_prediction, layer: layer.apply(partial_prediction),
            self._layers,
            input
        )

        return prediction


    def _calculated_optimization_gradients(self, input_batch, output_batch):
        """
        Performs a prediction and calculates its appropriate optimization
        gradients

        This method is only meant for class-internal use
        """

        # Gradient-based optimization is used
        # 1. One prediction is made
        # 2. Loss is calculated accordingly
        # 3. Keras algorithms are used to calculated the corresponding gradients
        with tensorflow.GradientTape() as gradient_tape:
            prediction_batch = self.predict(input_batch)
            losses = tensorflow.keras.losses.sparse_categorical_crossentropy(
                output_batch,
                prediction_batch
            )
            loss_value = tensorflow.reduce_mean(losses)

        gradients = gradient_tape.gradient(loss_value, self.weights)

        return gradients

    def _update_weights(self, gradients):
        """
        Updates the model's weights using the given gradients

        This method is only meant for class-internal use
        """

        learning_rate = 2e-3

        # Parallel iteration over the weight tensor and its correspoding gradient
        # tensor
        for weight, gradient in zip(self.weights, gradients):
            # Update is done via tensor scaling:
            # w_n+1 = w_n - gradient * s
            weight.assign_sub(gradient * learning_rate)

    def train(self, input, output, cycles, batch_size):
        """
        Trains the neural network given the training input and its expected
        output
        """

        for _ in range(cycles):
            for input_batch, output_batch in batcher(input, output, batch_size):
                optimization_gradients = self._calculated_optimization_gradients(
                    input_batch,
                    output_batch
                )
                self._update_weights(optimization_gradients)

    @property
    def weights(self):
        """
        Returns a compound tensor with all the weights in the neural network
        """

        network_weights = functools.reduce(
            lambda tensor, layer: tensor + layer.weights,
            self._layers,
            []
        )

        return network_weights


def load_data():
    ((training_images, training_labels), (test_images, test_labels)) = dataset.load_data()

    training_images = training_images.reshape((60000, 28 ** 2))
    training_images = training_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 ** 2))
    test_images = test_images.astype("float32") / 255

    return ((training_images, training_labels), (test_images, test_labels))


if __name__ == "__main__":
    network = Network(
        layers=[
            Layer(height=28 ** 2, width=512, activator=tensorflow.nn.relu),
            Layer(height=512, width=10, activator=tensorflow.nn.softmax)
        ]
    )

    ((training_images, training_labels), (test_images, test_labels)) = load_data()

    network.train(
        input=training_images,
        output=training_labels,
        cycles=20,
        batch_size=128
    )

    predicted_labels = numpy.argmax(network.predict(test_images).numpy(), axis=1)

    for predicted_result, expected_result in zip(predicted_labels, test_labels):
        print(f"Predicted {predicted_result} instead of {expected_result}")

    matches = test_labels == predicted_labels

    print(f"Efectividad de un {matches.mean()}%")
