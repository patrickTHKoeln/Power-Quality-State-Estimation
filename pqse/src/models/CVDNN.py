def cvnn_torch():
    """
    does not work
    :return:
    """
    import torch
    import torch.nn as nn

    class ComplexDenseNN(nn.Module):
        def __init__(self, input_features, hidden_features, output_features):
            super(ComplexDenseNN, self).__init__()
            # PyTorch's nn.Linear works with complex numbers after version 1.7
            self.layer1 = nn.Linear(input_features, hidden_features)
            self.layer2 = nn.Linear(hidden_features, output_features)
            self.activation = nn.ReLU()  # Activation function


        def forward(self, x):
            x = self.activation(self.layer1(x))
            x = self.layer2(x)
            return x

    # Example usage
    input_features = 10
    hidden_features = 64
    output_features = 1

    model = ComplexDenseNN(input_features, hidden_features, output_features)

    # Example complex data
    complex_data = torch.randn(5, 10, dtype=torch.cfloat)  # Create a complex tensor directly
    complex_output = model(complex_data)

    print(complex_output)

def cvnn_flax():
    import jax
    import flax.linen as nn
    import jax.numpy as jnp
    from jax.nn.initializers import lecun_normal, zeros
    from flax.core.frozen_dict import freeze

    class ComplexDense(nn.Module):
        features: int
        kernel_init: callable = lecun_normal()
        bias_init: callable = zeros

        @nn.compact
        def __call__(self, inputs):
            # Ensure inputs are complex
            inputs = jnp.asarray(inputs, dtype=jnp.complex64)
            # Initialize weights and biases with complex values
            kernel = self.param('kernel', self.kernel_init, (inputs.shape[-1], self.features), jnp.complex64)
            bias = self.param('bias', self.bias_init, (self.features,), jnp.complex64)
            # Return the complex-valued dense layer output
            return jnp.dot(inputs, kernel) + bias

    class ComplexNN(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = ComplexDense(features=64)(x)
            x = jax.nn.relu(x.real) + 1j * jax.nn.relu(
                x.imag)  # Activation function applied separately to real and imaginary parts
            x = ComplexDense(features=10)(x)
            return x

    # Example usage
    x = jnp.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], jnp.complex64)  # Example complex input
    model = ComplexNN()
    params = model.init(jax.random.PRNGKey(0), x)
    output = model.apply(params, x)

    print(output)


if __name__ == '__main__':
    cvnn_flax()
    # cvnn_torch()