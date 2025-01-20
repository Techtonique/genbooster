use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use ndarray::{Array1, Array2};
use ndarray::s;

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
    ) -> Self {
        RustBooster {
            base_learners: vec![base_estimator; n_estimators as usize],
            weights: Vec::new(),
            learning_rate,
            n_hidden_features,
            n_estimators,
            direct_link,
        }
    }

    fn fit(
        &mut self,
        py: Python,
        x: &PyArray2<f64>,
        y: &PyArray1<f64>,
        dropout: f64,
        seed: u64,
    ) -> PyResult<()> {
        let x_array = unsafe { x.as_array() };
        let y_array = unsafe { y.as_array() };
        
        let mut rng = StdRng::seed_from_u64(seed);
        let _n_samples = x_array.shape()[0];
        let n_features = x_array.shape()[1];
        
        // Initialize residuals
        let y_mean = y_array.mean().unwrap();
        let mut residuals = Array1::from_vec(y_array.to_vec());
        residuals.mapv_inplace(|v| v - y_mean);
        
        for i in 0..self.n_estimators {
            // Generate random weights for hidden layer
            let mut w = Array2::zeros((n_features, self.n_hidden_features as usize));
            for w_row in w.rows_mut() {
                for w_val in w_row {
                    *w_val = rng.gen::<f64>() * 2.0 - 1.0;
                }
            }
            self.weights.push(w.clone());
            
            // Forward pass with activation
            let hidden = self.forward_pass(py, &x_array.to_owned(), &w, dropout, seed + i as u64)?;
            
            // Clone the base estimator for this iteration
            let base_learner = self.base_learners[i as usize].clone_ref(py);
            
            // Fit the base learner
            let kwargs = PyDict::new(py);
            kwargs.set_item("X", hidden.to_pyarray(py))?;
            kwargs.set_item("y", residuals.to_pyarray(py))?;
            base_learner.call_method(py, "fit", (), Some(kwargs))?;
            
            // Predict and update residuals
            let pred_kwargs = PyDict::new(py);
            pred_kwargs.set_item("X", hidden.to_pyarray(py))?;
            let pred_result = base_learner.call_method(py, "predict", (), Some(pred_kwargs))?;
            let predictions: &PyArray1<f64> = pred_result.extract(py)?;
            let pred_array = unsafe { predictions.as_array() };
            
            // Update residuals
            residuals = residuals - self.learning_rate * pred_array.to_owned();
            
            // Store the fitted base learner
            self.base_learners[i as usize] = base_learner;
        }
        
        Ok(())
    }

    fn predict(&self, py: Python, x: &PyArray2<f64>) -> PyResult<PyObject> {
        let x_array = unsafe { x.as_array() };
        let mut predictions: Array1<f64> = Array1::zeros(x_array.shape()[0]);
        
        for (i, (w, base_learner)) in self.weights.iter().zip(self.base_learners.iter()).enumerate() {
            let hidden = self.forward_pass(py, &x_array.to_owned(), w, 0.0, 0)?;
            
            // Fix temporary value issue by storing the result
            let pred_kwargs = PyDict::new(py);
            pred_kwargs.set_item("X", hidden.to_pyarray(py))?;
            let pred_result = base_learner.call_method(py, "predict", (), Some(pred_kwargs))?;
            let pred: &PyArray1<f64> = pred_result.extract(py)?;
            let pred_array = unsafe { pred.as_array() };
            
            predictions = predictions + self.learning_rate * pred_array.to_owned();
        }
        
        Ok(predictions.to_pyarray(py).to_object(py))
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
        
        // Concatenate with original features if direct_link is true
        if self.direct_link {
            let mut combined = Array2::zeros((x.shape()[0], x.shape()[1] + hidden.shape()[1]));
            combined.slice_mut(s![.., ..x.shape()[1]]).assign(x);
            combined.slice_mut(s![.., x.shape()[1]..]).assign(&hidden);
            Ok(combined)
        } else {
            Ok(hidden)
        }
    }
}