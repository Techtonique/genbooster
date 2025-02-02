use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use ndarray::{Array1, Array2, Axis};
use ndarray::s;
use linfa::traits::{Fit, Predict};
use linfa_linear::LinearRegression;
use linfa_elasticnet::ElasticNet;
use linfa_pls::PlsRegression;
use pyo3::exceptions::PyValueError;
use linfa::Dataset;
use linfa_linear::FittedLinearRegression;
use rand_chacha::ChaCha20Rng;
mod utils;
use utils::create_rng;

#[derive(Clone, Copy)]
enum WeightsDistribution {
    Uniform,
    Normal,
}

// First, create a new enum to hold uninitialized models
enum RegressionModelParams {
    LinearRegression(LinearRegression),
    ElasticNet { penalty: f64, l1_ratio: f64 },
}

// Update RegressionModel enum for initialized models
enum RegressionModel {
    LinearRegression(FittedLinearRegression<f64>),  // Changed to FittedLinearRegression
    ElasticNet(ElasticNet<f64>),
}

#[pymodule]
fn rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Regressor>()?;
    m.add_class::<RustBooster>()?;
    m.add_class::<AdaBoostRegressor>()?;
    Ok(())
}

#[pyclass]
struct Regressor {
    model_params: Option<RegressionModelParams>,  // Store parameters until fit time
    model: Option<RegressionModel>,              // Store fitted model
}

#[pymethods]
impl Regressor {
    #[new]
    fn new(model_name: &str) -> PyResult<Self> {
        let model_params = match model_name {
            "LinearRegression" => Ok(RegressionModelParams::LinearRegression(LinearRegression::default())),
            "ElasticNet" => Ok(RegressionModelParams::ElasticNet {
                penalty: 0.01,
                l1_ratio: 0.5,
            }),
            _ => Err(PyValueError::new_err(format!("Unknown model: {}", model_name))),
        }?;
        Ok(Regressor { 
            model_params: Some(model_params),
            model: None,
        })
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x = x.as_array().to_owned();
        let y = y.as_array().to_owned();
                
        // Create appropriate Dataset based on model type
        let model = match self.model_params.take().ok_or(PyValueError::new_err("Model not initialized"))? {
            RegressionModelParams::LinearRegression(m) => {
                // Use 1D targets for LinearRegression
                let dataset = Dataset::new(x.clone(), y);
                RegressionModel::LinearRegression(m.fit(&dataset)
                    .map_err(|e| PyValueError::new_err(format!("LinearRegression fit error: {}", e)))?)
            },
            RegressionModelParams::ElasticNet { penalty, l1_ratio } => {
                // Use 1D targets for ElasticNet
                let dataset = Dataset::new(x.clone(), y);
                RegressionModel::ElasticNet(
                    ElasticNet::params()
                        .penalty(penalty)
                        .l1_ratio(l1_ratio)
                        .fit(&dataset)
                        .map_err(|e| PyValueError::new_err(format!("ElasticNet fit error: {}", e)))?
                )
            },
        };

        self.model = Some(model);
        Ok(())
    }

    fn predict(&self, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let x = x.as_array();
        match &self.model {
            Some(model) => {
                let predictions = match model {
                    RegressionModel::LinearRegression(m) => {
                        let pred = m.predict(x);
                        // Convert 1D to 2D array
                        Array2::from_shape_vec((pred.targets().len(), 1), pred.targets().to_vec())
                            .map_err(|e| PyValueError::new_err(format!("Failed to reshape predictions: {}", e)))?
                    },
                    RegressionModel::ElasticNet(m) => {
                        let pred = m.predict(x);
                        // Convert 1D to 2D array
                        Array2::from_shape_vec((pred.targets().len(), 1), pred.targets().to_vec())
                            .map_err(|e| PyValueError::new_err(format!("Failed to reshape predictions: {}", e)))?
                    },                    
                };
                Python::with_gil(|py| Ok(predictions.to_pyarray(py).to_owned()))
            }
            None => Err(PyValueError::new_err("Model not initialized")),
        }
    }
}

#[pyclass]
struct RustBooster {
    base_learners: Vec<PyObject>,
    weights: Vec<Array2<f64>>,
    learning_rate: f64,
    n_hidden_features: i32,
    n_estimators: i32,
    direct_link: bool,
    weights_distribution: WeightsDistribution,
    dropout: f64,
    tolerance: f64,
    seed: u64,
}

#[pymethods]
impl RustBooster {
    #[new]
    fn new(
        base_estimator: PyObject,
        n_estimators: i32,
        learning_rate: f64,
        n_hidden_features: i32,
        direct_link: bool,
        weights_distribution: Option<&str>,
        tolerance: Option<f64>,
    ) -> Self {
        let weights_dist = match weights_distribution.unwrap_or("uniform") {
            "normal" => WeightsDistribution::Normal,
            _ => WeightsDistribution::Uniform,
        };

        RustBooster {
            base_learners: vec![base_estimator; n_estimators as usize],
            weights: Vec::new(),
            learning_rate,
            n_hidden_features,
            n_estimators,
            direct_link,
            weights_distribution: weights_dist,
            dropout: 0.0,
            seed: 0,
            tolerance: tolerance.unwrap_or(1e-4)
        }
    }

    fn fit_boosting(
        &mut self,
        py: Python,
        x: &PyArray2<f64>,
        y: &PyArray1<f64>,
        dropout: f64,
        seed: u64,
    ) -> PyResult<()> {
        self.dropout = dropout;
        self.seed = seed;
        let x_array = unsafe { x.as_array() };
        let y_array = unsafe { y.as_array() };        
        let mut rng = create_rng(seed);
        let n_samples = x_array.shape()[0];
        let n_features = x_array.shape()[1];        
        
        // Initialize residuals
        let y_mean = y_array.mean().unwrap();
        let mut residuals = Array1::from_vec(vec![y_mean; n_samples]);  // Create array filled with y_mean
        residuals = y_array.to_owned() - residuals;  // Convert y_array to owned array before subtraction
        
        // First, create proper deep copies of base estimators
        let sklearn = py.import("sklearn.base")?;
        let clone_fn = sklearn.getattr("clone")?;
        
        // Create deep copies for all base learners at the start
        for i in 0..self.n_estimators {
            self.base_learners[i as usize] = clone_fn.call1((self.base_learners[i as usize].clone_ref(py),))?.into();
        }
        
        let mut previous_l2_norm = f64::INFINITY;
        
        for i in 0..self.n_estimators {
            // Generate random weights for hidden layer
            let mut w = Array2::zeros((n_features, self.n_hidden_features as usize));
            for w_row in w.rows_mut() {
                for w_val in w_row {
                    *w_val = match self.weights_distribution {
                        WeightsDistribution::Uniform => rng.gen::<f64>(),  // Changed to U(0,1)
                        WeightsDistribution::Normal => rng.gen::<f64>(),   // Still N(0,1)
                    };
                }
            }
            self.weights.push(w.clone());            
            // Forward pass with activation
            let hidden = self.forward_pass(py, &x_array.to_owned(), &w, dropout, seed + i as u64)?;            
            // No need to clone again, we already have independent copies
            let base_learner = &self.base_learners[i as usize];            
            // Fit the base learner
            let kwargs = PyDict::new(py);
            kwargs.set_item("X", hidden.to_pyarray(py))?;
            kwargs.set_item("y", residuals.to_pyarray(py))?;
            base_learner.call_method(py, "fit", (), Some(kwargs))?;            
            // Predict and update residuals
            let pred_kwargs = PyDict::new(py);
            pred_kwargs.set_item("X", hidden.to_pyarray(py))?;
            let pred_result = base_learner.call_method(py, "predict", (), 
            Some(pred_kwargs))?;
            let predictions: &PyArray1<f64> = pred_result.extract(py)?;
            let pred_array = unsafe { predictions.as_array() };            
            // Update residuals
            residuals = residuals - self.learning_rate * pred_array.to_owned();            
            // Store the fitted estimator back
            self.base_learners[i as usize] = base_learner.clone_ref(py);            
            // Calculate current L2 norm of residuals
            let current_l2_norm = residuals.mapv(|x| x.powi(2)).sum();
            
            // Check if change in L2 norm is small enough for early stopping
            if (current_l2_norm - previous_l2_norm).abs() <= self.tolerance {
                // Update n_estimators to current iteration and break
                self.n_estimators = i + 1;
                break;
            }            
            previous_l2_norm = current_l2_norm;
        }
        Ok(())
    }

    fn predict_boosting(&self, py: Python, x: &PyArray2<f64>) -> PyResult<PyObject> {
        let x_array = unsafe { x.as_array() };
        let mut predictions: Array1<f64> = Array1::zeros(x_array.shape()[0]);
        
        for (i, (w, base_learner)) in self.weights.iter().zip(self.base_learners.iter()).enumerate() {
            // Use stored weights directly
            let hidden = x_array.dot(w);
            let hidden = hidden.mapv(|v| if v > 0.0 { v } else { 0.0 });            
            // Direct link if specified
            let hidden = if self.direct_link {
                ndarray::concatenate![Axis(1), x_array, hidden]
            } else {
                hidden
            };
            
            let pred_kwargs = PyDict::new(py);
            pred_kwargs.set_item("X", hidden.to_pyarray(py))?;
            let pred_result = base_learner.call_method(py, "predict", (), Some(pred_kwargs))?;
            let pred: &PyArray1<f64> = pred_result.extract(py)?;
            let pred_array = unsafe { pred.as_array() };      
            predictions = predictions + self.learning_rate * pred_array.to_owned();
        }
        
        Ok(predictions.to_pyarray(py).to_object(py))
    }

    fn fit_bagging(&mut self, py: Python, x: &PyArray2<f64>, y: &PyArray1<f64>, dropout: f64, seed: u64) -> PyResult<()> {
        self.dropout = dropout;
        self.seed = seed;
        let x_array = unsafe { x.as_array() };
        let y_array = unsafe { y.as_array() };        
        let mut rng = create_rng(seed);
        let n_features = x_array.shape()[1];
        
        // First, create proper deep copies of base estimators
        let sklearn = py.import("sklearn.base")?;
        let clone_fn = sklearn.getattr("clone")?;
        
        // Create deep copies for all base learners at the start
        for i in 0..self.n_estimators {
            self.base_learners[i as usize] = clone_fn.call1((self.base_learners[i as usize].clone_ref(py),))?.into();
        }
        
        for i in 0..self.n_estimators {
            // Generate random weights for hidden layer
            let mut w = Array2::zeros((n_features, self.n_hidden_features as usize));
            for w_row in w.rows_mut() {
                for w_val in w_row {
                    *w_val = match self.weights_distribution {
                        WeightsDistribution::Uniform => rng.gen::<f64>(),
                        WeightsDistribution::Normal => rng.gen::<f64>(),
                    };
                }
            }
            self.weights.push(w.clone());
            
            // Forward pass with activation
            let hidden = self.forward_pass(py, &x_array.to_owned(), &w, dropout, seed + i as u64)?;
            
            let base_learner = &self.base_learners[i as usize];
            
            // Fit the base learner directly on y (no residuals)
            let kwargs = PyDict::new(py);
            kwargs.set_item("X", hidden.to_pyarray(py))?;
            kwargs.set_item("y", y)?;  // Use original y instead of residuals
            base_learner.call_method(py, "fit", (), Some(kwargs))?;
            
            // Store the fitted estimator back
            self.base_learners[i as usize] = base_learner.clone_ref(py);
        }
        Ok(())
    }

    fn predict_bagging(&self, py: Python, x: &PyArray2<f64>) -> PyResult<PyObject> {
        let x_array = unsafe { x.as_array() };
        let n_samples = x_array.shape()[0];
        
        // Create a matrix to store all predictions (n_samples x n_estimators)
        let mut all_predictions = Array2::zeros((n_samples, self.n_estimators as usize));
        
        // Get predictions from each base learner
        for (i, (w, base_learner)) in self.weights.iter().zip(self.base_learners.iter()).enumerate() {
            // Forward pass with stored weights
            let hidden = x_array.dot(w);
            let hidden = hidden.mapv(|v| if v > 0.0 { v } else { 0.0 });
            
            // Direct link if specified
            let hidden = if self.direct_link {
                ndarray::concatenate![Axis(1), x_array, hidden]
            } else {
                hidden
            };
            
            // Get predictions from current base learner
            let pred_kwargs = PyDict::new(py);
            pred_kwargs.set_item("X", hidden.to_pyarray(py))?;
            let pred_result = base_learner.call_method(py, "predict", (), Some(pred_kwargs))?;
            let pred: &PyArray1<f64> = pred_result.extract(py)?;
            let pred_array = unsafe { pred.as_array() };
            
            // Store predictions in the corresponding column
            all_predictions.column_mut(i).assign(&pred_array);
        }
        
        // Calculate median along axis 1 (across estimators)
        let final_predictions = all_predictions
            .rows()
            .into_iter()
            .map(|row| {
                let mut row_vec: Vec<f64> = row.to_vec();
                row_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = row_vec.len() / 2;
                if row_vec.len() % 2 == 0 {
                    (row_vec[mid - 1] + row_vec[mid]) / 2.0
                } else {
                    row_vec[mid]
                }
            })
            .collect::<Vec<f64>>();
        
        // Convert to Array1 and return
        Ok(Array1::from_vec(final_predictions).to_pyarray(py).to_object(py))
    }
}

impl RustBooster {
    fn forward_pass(
        &self,
        py: Python,
        x: &Array2<f64>,
        w: &Array2<f64>,
        dropout: f64,
        seed: u64,
    ) -> PyResult<Array2<f64>> {
        let mut rng = create_rng(seed);
        
        // Compute hidden layer
        let mut hidden = x.dot(w);
        
        // Apply ReLU activation
        hidden.mapv_inplace(|v| if v > 0.0 { v } else { 0.0 });
        
        // Apply dropout if specified
        if dropout > 0.0 {
            for val in hidden.iter_mut() {
                if rng.gen::<f64>() < dropout {
                    *val = 0.0;
                } else {
                    *val /= 1.0 - dropout;
                }
            }
        }
        
        // If direct_link is true, concatenate x and hidden horizontally
        // This will create a matrix with:
        // - Same number of rows as input
        // - Columns = original features + hidden features
        if self.direct_link {
            let mut combined = Array2::zeros((
                x.shape()[0],                    // number of samples (rows)
                x.shape()[1] + hidden.shape()[1] // original features + hidden features (columns)
            ));
            combined.slice_mut(s![.., ..x.shape()[1]]).assign(x);        // First part: original features
            combined.slice_mut(s![.., x.shape()[1]..]).assign(&hidden); // Second part: hidden features
            Ok(combined)
        } else {
            Ok(hidden)
        }
    }
}

#[pyclass]
struct AdaBoostRegressor {
    base_learners: Vec<PyObject>,
    alphas: Vec<f64>,
    weights: Vec<Array2<f64>>,
    learning_rate: f64,
    n_estimators: i32,
    n_hidden_features: i32,
    direct_link: bool,
    weights_distribution: WeightsDistribution,
    tolerance: f64,
    dropout: f64,
    seed: u64,
}

#[pymethods]
impl AdaBoostRegressor {
    #[new]
    fn new(
        base_estimator: PyObject,
        n_estimators: i32,
        learning_rate: f64,
        n_hidden_features: i32,
        direct_link: bool,
        weights_distribution: Option<&str>,
        tolerance: Option<f64>,
        dropout: Option<f64>,
        seed: Option<u64>,
    ) -> Self {
        let weights_dist = match weights_distribution.unwrap_or("uniform") {
            "normal" => WeightsDistribution::Normal,
            _ => WeightsDistribution::Uniform,
        };

        AdaBoostRegressor {
            base_learners: vec![base_estimator; n_estimators as usize],
            alphas: Vec::new(),
            weights: Vec::new(),
            learning_rate,
            n_estimators,
            n_hidden_features,
            direct_link,
            weights_distribution: weights_dist,
            tolerance: tolerance.unwrap_or(1e-4),
            dropout: dropout.unwrap_or(0.0),
            seed: seed.unwrap_or(42),
        }
    }

    fn fit(&mut self, py: Python, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<()> {
        let x_array = unsafe { x.as_array() };
        let y_array = unsafe { y.as_array() };
        let n_samples = x_array.shape()[0];
        let n_features = x_array.shape()[1];
        
        // Initialize RNG with seed
        let mut rng = create_rng(self.seed);
        
        // Initialize sample weights
        let mut sample_weights = Array1::ones(n_samples) / (n_samples as f64);
        
        // Get sklearn's clone function
        let sklearn = py.import("sklearn.base")?;
        let clone_fn = sklearn.getattr("clone")?;
        
        // Create deep copies for all base learners
        for i in 0..self.n_estimators {
            self.base_learners[i as usize] = clone_fn.call1((self.base_learners[i as usize].clone_ref(py),))?.into();
        }
        
        // Calculate the range of target values for loss normalization
        let y_max = y_array.iter().fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b));
        let y_min = y_array.iter().fold(f64::INFINITY, |a, &b| f64::min(a, b));
        let y_range = y_max - y_min;
        
        for i in 0..self.n_estimators {
            // Generate random weights for hidden layer
            let mut w = Array2::zeros((n_features, self.n_hidden_features as usize));
            for w_row in w.rows_mut() {
                for w_val in w_row {
                    *w_val = match self.weights_distribution {
                        WeightsDistribution::Uniform => rng.gen::<f64>(),
                        WeightsDistribution::Normal => rng.gen::<f64>(),
                    };
                }
            }
            self.weights.push(w.clone());
            
            // Forward pass with activation
            let hidden = self.forward_pass(py, &x_array.to_owned(), &w, self.dropout, self.seed + i as u64)?;
            
            // Get current base learner
            let base_learner = &self.base_learners[i as usize];
            
            // Fit the base learner with sample weights
            let kwargs = PyDict::new(py);
            kwargs.set_item("X", hidden.to_pyarray(py))?;
            kwargs.set_item("y", y)?;
            kwargs.set_item("sample_weight", sample_weights.to_pyarray(py))?;
            base_learner.call_method(py, "fit", (), Some(kwargs))?;
            
            // Get predictions using transformed features
            let pred_kwargs = PyDict::new(py);
            pred_kwargs.set_item("X", hidden.to_pyarray(py))?;
            let pred_result = base_learner.call_method(py, "predict", (), Some(pred_kwargs))?;
            let predictions: &PyArray1<f64> = pred_result.extract(py)?;
            let pred_array = unsafe { predictions.as_array() };
            
            // Calculate normalized errors (AdaBoost.R2)
            let diff = &y_array.to_owned() - &pred_array.to_owned();
            let loss = diff.mapv(|x| x.abs() / y_range);
            let max_loss = loss.iter().fold(0.0f64, |a, &b| f64::max(a, b));
            let normalized_loss = loss.mapv(|x| x / max_loss);
            
            // Calculate weighted error using zip
            let error: f64 = normalized_loss.iter()
                .zip(sample_weights.iter())
                .map(|(&l, &w)| l * w)
                .sum();
            let error = error.max(1e-10).min(1.0 - 1e-10);
            
            // Calculate beta (different from classification AdaBoost)
            let beta = error / (1.0 - error);
            let alpha = self.learning_rate * beta.ln();
            self.alphas.push(-alpha); // Note the negative sign
            
            // Update sample weights
            let new_weights = normalized_loss.mapv(|l| beta.powf(1.0 - l));
            sample_weights = &sample_weights * &new_weights;
            let sum_weights: f64 = sample_weights.sum();
            sample_weights = sample_weights.mapv(|w| w / sum_weights);
            
            // Store the fitted estimator
            self.base_learners[i as usize] = base_learner.clone_ref(py);
            
            // Early stopping if error is too small
            if error < self.tolerance {
                self.n_estimators = i + 1;
                break;
            }
        }
        Ok(())
    }

    fn predict(&self, py: Python, x: &PyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let x_array = unsafe { x.as_array() };
        let n_samples = x_array.shape()[0];
        let mut predictions = Array1::zeros(n_samples);
        let sum_alphas: f64 = self.alphas.iter().sum();
        
        for ((base_learner, &alpha), w) in self.base_learners.iter()
            .zip(self.alphas.iter())
            .zip(self.weights.iter()) {
            
            let hidden = self.forward_pass(py, &x_array.to_owned(), w, 0.0, self.seed)?;
            
            let pred_kwargs = PyDict::new(py);
            pred_kwargs.set_item("X", hidden.to_pyarray(py))?;
            let pred_result = base_learner.call_method(py, "predict", (), Some(pred_kwargs))?;
            let pred: &PyArray1<f64> = pred_result.extract(py)?;
            let pred_array = unsafe { pred.as_array() };
            
            // Convert view to owned array and perform operations
            let pred_owned = pred_array.to_owned();
            predictions = predictions + (alpha * pred_owned);
        }
        
        // Normalize by sum of alphas
        predictions = predictions.mapv(|x| x / sum_alphas);
        
        Ok(predictions.to_pyarray(py).to_owned())
    }
}

impl AdaBoostRegressor {
    fn forward_pass(
        &self,
        py: Python,
        x: &Array2<f64>,
        w: &Array2<f64>,
        dropout: f64,
        seed: u64,
    ) -> PyResult<Array2<f64>> {
        let mut rng = create_rng(seed);
        
        // Compute hidden layer
        let mut hidden = x.dot(w);
        
        // Apply ReLU activation
        hidden.mapv_inplace(|v| if v > 0.0 { v } else { 0.0 });
        
        // Apply dropout if specified
        if dropout > 0.0 {
            for val in hidden.iter_mut() {
                if rng.gen::<f64>() < dropout {
                    *val = 0.0;
                } else {
                    *val /= 1.0 - dropout;
                }
            }
        }
        
        // If direct_link is true, concatenate x and hidden horizontally
        if self.direct_link {
            let combined = ndarray::concatenate![Axis(1), x.to_owned(), hidden];
            Ok(combined)
        } else {
            Ok(hidden)
        }
    }
}