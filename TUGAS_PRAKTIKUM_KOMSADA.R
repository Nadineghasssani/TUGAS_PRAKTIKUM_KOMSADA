logistic_regression_newton <- function(X, y, tolerance = 1e-6, max_iterations = 100, learning_rate = 1) {
  # Tambahkan kolom intercept ke dalam X
  X <- cbind(1, X)
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  # Validasi data input
  if (length(unique(y)) < 2) stop("Variabel y harus memiliki minimal dua kelas: 0 dan 1.")
  if (!all(y %in% c(0, 1))) stop("Variabel y hanya boleh berisi nilai 0 dan 1.")
  
  # Inisialisasi parameter beta
  beta <- matrix(0, nrow = n_features, ncol = 1)
  
  # Definisi fungsi sigmoid
  sigmoid_function <- function(value) {
    return(1 / (1 + exp(-value)))
  }
  
  # Algoritma Newton-Raphson
  convergence <- FALSE
  for (iteration in 1:max_iterations) {
    linear_combination <- X %*% beta
    probability <- sigmoid_function(linear_combination)
    probability <- pmin(pmax(probability, 1e-15), 1 - 1e-15)
    
    weight_matrix <- diag(as.vector(probability * (1 - probability)))
    gradient_vector <- t(X) %*% (y - probability)
    hessian_matrix <- -t(X) %*% weight_matrix %*% X
    
    # Tambahkan regularisasi untuk menghindari singularitas
    hessian_matrix <- hessian_matrix + diag(1e-6, n_features)
    
    # Update parameter beta dengan pseudo-inverse jika diperlukan
    beta_next <- tryCatch(
      beta - learning_rate * solve(hessian_matrix) %*% gradient_vector,
      error = function(e) {
        warning("Hessian tidak terbalikkan, menggunakan pseudo-inverse.")
        beta - learning_rate * MASS::ginv(hessian_matrix) %*% gradient_vector
      }
    )
    
    if (max(abs(beta_next - beta)) < tolerance) {
      beta <- beta_next
      convergence <- TRUE
      break
    }
    beta <- beta_next
  }
  
  # Keluaran hasil regresi
  return(list(
    method = "Newton-Raphson Optimized",
    coefficients = beta,
    probabilities = as.vector(sigmoid_function(X %*% beta)),
    converged = convergence
  ))
}

# Contoh implementasi
data_example <- data.frame(
  Feature1 = c(0.5, 2.3, 2.9, 3.5, 3.8, 4.2, 5.0, 5.8, 6.0, 7.1),
  Feature2 = c(1.2, 3.1, 3.5, 4.0, 4.8, 5.3, 6.5, 6.8, 7.2, 8.0),
  Target = c(0, 0, 0, 1, 1, 1, 1, 1, 1, 1)
)

X_data <- as.matrix(data_example[, c("Feature1", "Feature2")])
y_data <- as.matrix(data_example$Target)

# Jalankan regresi logistik dengan Newton-Raphson
result_newton <- logistic_regression_newton(X_data, y_data)

print("Hasil Regresi Newton-Raphson:")
print(result_newton)
