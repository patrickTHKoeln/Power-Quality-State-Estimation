# Machine learning Toolbox for multidimensional data

Scalers, metrics and other tools for machine learning with multidimensional data.

## Build and Install
From within your virtual environment:
- change directory to the root of the repository
- `pip install .`
- or install in editable mode: `pip install -e .`

## Usage

### Metrics
Provides common metrics for multi-dimensional data.
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

Each method expects two arrays of the same shape. 
The first array is the ground truth, the second array is the prediction.

`Axis` is an optional parameter to specify one or more axes along which the metrics are computed.
If no axis is specified, the metrics are computed over all elements of the arrays.

### Scaling

Provides the following scaling methods:
- MinMaxScaler
- StandardScaler
- QuantileScaler (Work in Progress)
- logarithmic redistribution

For the scalers, the `fit_transform` method has an optional 
parameter `axis` to specify one or more axes along which the scaling is computed.

In default implementations (e.g. Scikit-Learn), 
the scaling is computed over all elements of the array, only the first dimension 
(batch dimension) is preserved. 
Hence, default implementations require stacking of features on 
the batch dimension which might be undesirable.

### Split-Complex

Most neural networks expect real-valued input.
Pytorch offers complex-valued tensors in an experimental state 
with reduced availability of operations, such as complex activation functions and layers.

SplitComplexConverter converts complex-valued numpy arrays to real-valued numpy arrays 
stacking complex components along a desired axis.

Work in progress:
- convert split-complex numpy arrays from one representation to another
- revert conversion back to complex-valued numpy arrays