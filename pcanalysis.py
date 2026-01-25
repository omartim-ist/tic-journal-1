def pca_mean_reversion(lrets):
    Cov = np.cov(lrets)
    eigenvalues, eigenvectors = np.linalg.eigh(Cov)
    w_pc1 = eigenvectors[:,-1]
    w_pc1 = w_pc1/w_pc1.sum()
    w_pc2 = eigenvectors[:,-2]
    w_pc2 = w_pc2/w_pc2.sum()
    mu_pc1 = np.mean(w_pc1 @ lrets)
    mu_pc2 = np.mean(w_pc2 @ lrets)
    if mu_pc1 > mu_pc2:
        return w_pc2
    else:
        return w_pc1
    
def test_train_data(data, n_splits, n_train_blocks, n_test_blocks):
    len_blocks = len(data[0])//n_splits
    train_data = []
    test_data = []

    limit_loop = n_splits - max(n_train_blocks, n_test_blocks) + 1
    
    for k in range(limit_loop):
        train_data += [data[:,k*len_blocks:(k+n_train_blocks)*len_blocks]]
        test_data += [data[:,k*len_blocks:(k+n_test_blocks)*len_blocks]]
    return np.array(test_data), np.array(train_data)

test_data, train_data = test_train_data(lrets, 44, 2, 1)

w = []

for i in range(len(train_data)):
    w += [pca_mean_reversion(train_data[i])]

w = np.array(w)

test = []
for i in range(len(train_data)-2):
    test += [w[i] @ test_data[i+2]]

test = np.array(test)

results = np.concatenate(test)

plt.figure(figsize=(12, 6))
plt.plot(results, linewidth=1)
plt.xlabel('Date')
plt.ylabel('Return ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

portfolio = np.exp(np.cumsum(results))

plt.figure(figsize=(12, 6))
plt.plot(portfolio, linewidth=1)
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


def pca_min_variance(returns_array, n_components=3):
    """
    Calcula pesos de Variância Mínima usando PCA para limpeza de ruído.
    returns_array: np.ndarray (n_samples, n_assets)
    """
    # 1. Centralizar os dados (Média 0)
    mean_vals = np.mean(returns_array, axis=0)
    centered_data = returns_array - mean_vals
    
    # 2. Calcular a Matriz de Covariância
    # (n_samples - 1) para o estimador imparcial
    cov_matrix = np.cov(centered_data, rowvar=False)
    
    # 3. Decomposição em Valores Próprios (Eigendecomposition)
    # eigenvalues (L) e eigenvectors (V)
    L, V = np.linalg.eigh(cov_matrix)
    
    # O eigh retorna em ordem ascendente. Vamos inverter para descendente.
    idx = np.argsort(L)[::-1]
    L = L[idx]
    V = V[:, idx]
    
    # 4. Reconstrução Denoised (Filtragem)
    # Mantemos apenas os k maiores valores próprios e os seus vetores
    L_filtered = np.copy(L)
    L_filtered[n_components:] = 0  # Zera o ruído (ou substitui pela média do ruído)
    
    # Sigma_denoised = V * L_diag_filtered * V^T
    denoised_cov = V @ np.diag(L_filtered) @ V.T
    
    # 5. Regularização (Shrinkage)
    # Adicionamos uma pequena constante à diagonal para garantir que a matriz é invertível
    # (Efeito semelhante à Ridge Regression)
    denoised_cov += np.eye(denoised_cov.shape[0]) * 1e-4
    
    # 6. Cálculo dos Pesos de Variância Mínima
    # Fórmula: w = (Inv(Sigma) * 1) / (1^T * Inv(Sigma) * 1)
    inv_cov = np.linalg.inv(denoised_cov)
    ones = np.ones(denoised_cov.shape[0])
    
    raw_weights = inv_cov @ ones
    weights = raw_weights / np.sum(raw_weights)
    
    return weights

