use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use ndarray::{Array1, Array2, Axis};
use ndarray::s;

#[derive(Clone, Copy)]
enum WeightsDistribution {
    Uniform,
    Normal,
}

#[pymodule]
fn rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustBooster>()?;
    Ok(())
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
        let mut rng = StdRng::seed_from_u64(seed);
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
            let mut hidden = hidden.mapv(|v| if v > 0.0 { v } else { 0.0 });            
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
        let mut rng = StdRng::seed_from_u64(seed);
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
            let mut hidden = hidden.mapv(|v| if v > 0.0 { v } else { 0.0 });
            
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
        let mut rng = StdRng::seed_from_u64(seed);
        
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