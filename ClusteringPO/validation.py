''' Validation via Politis-Romano '''
class Validation:
    """
    Validate a portfolio using Politis–Romano bootstrap paths.

    Given bootstrap-generated price paths for multiple assets and a fixed asset
    weight vector x, this class:
      - converts prices to log-returns per path,
      - builds the portfolio log-return series for each path,
      - computes portfolio value paths (normalized to 1 at t=0),
      - computes per-path annualized return, annualized volatility, and Sharpe ratio,
      - provides plotting helpers for path distributions and statistics.
    """
    
    def __init__(self, prb, x):
        
        # Generate bootstrap price paths: expected shape (n_paths, n_assets, T_prices)
        prb_data = prb.generate()
        
        # Convert prices to log-returns along the time axis:
        # R has shape (n_paths, n_assets, T) where T = T_prices - 1
        np.log(prb_data, out=prb_data)  # in-place (perdes os preços originais)
        
        # Portfolio log-returns per path:
        # r_p[p, t] = sum_a R[p, a, t] * x[a]  
        s = np.einsum("pat,a->pt", prb_data, x)
        r_p = np.diff(s, axis=1)   
        n_paths, T = r_p.shape

        # Build portfolio value paths using log-returns
        portfolios_value = np.exp(np.cumsum(r_p, axis=1)) # (n_paths, T)
        pv = np.empty((n_paths, T + 1), dtype=portfolios_value.dtype)
        pv[:, 0] = 1.0
        pv[:, 1:] = portfolios_value
        self.portfolios_value = pv
        
        # Per-path mean and volatility of portfolio log-returns
        mean_lrets = r_p.mean(axis=1) # (n_paths,)
        vol_lrets = r_p.std(axis=1, ddof=1) # (n_paths,)
        
        annualized_returns = np.expm1(mean_lrets * 365)# (n_paths,)
        annualized_vols = vol_lrets * np.sqrt(365) # (n_paths,)
        sharpes = annualized_returns / annualized_vols # (n_paths,)
        self.d_stat = {'Annualized returns': annualized_returns,
                       'Annualized volatilities': annualized_vols,
                       'Sharpe ratios': sharpes}
        
    def _plot_statistics(self, bins=100):
        """
        Plot histograms of the stored distributions (annualized return, vol, Sharpe),
        adding quartile (Q1/median/Q3) vertical lines in green and the mean in red.
        Also prints the probability of a negative Sharpe ratio.
        """
        
        for title, serie in self.d_stat.items():
            if title in ['Annualized returns', 'Annualized volatilities']:
                percentage = True
                serie = serie * 100
            else:
                percentage = False
                
            q25, q50, q75 = np.percentile(serie, [25, 50, 75])
            mu = serie.mean()
            
            plt.figure()
            plt.hist(serie, bins=bins, density=True, alpha=0.6)
            
            plt.axvline(q25, color="green", linestyle="--", label=f"Q1 = {round(q25, 2)}")
            plt.axvline(q50, color="green", linestyle="-.", label=f"Median = {round(q50, 2)}")
            plt.axvline(q75, color="green", linestyle=":", label=f"Q3 = {round(q75, 2)}")
            plt.axvline(mu, color="red", linestyle="-", label=f"Mean = {round(mu, 2)}")
            
            # legenda e labels
            plt.legend()
            if percentage:
                plt.xlabel('Values in %')
            plt.ylabel('Density')
            plt.title(title)
            plt.show()
            
            if not percentage:
                p = len(serie[serie<0]) / len(serie) * 100
                print(f'Probability of negative Sharpe Ratio: {round(p, 2)}%')
            
    def _plot_paths(self):
        """
        Plot all portfolio value paths in a single figure.
        Each line is one bootstrap path. A horizontal line at 1 marks the initial value.
        """
        plt.figure()
        plt.plot(self.portfolios_value.T, alpha=0.5)
        plt.axhline(1, color='black')
        plt.xlabel('5 years of Synthetic Data')
        plt.show()
