import numpy as np
import matplotlib.pyplot as plt

''' Validation via Politis-Romano '''
class Validation:
    """
    Validate a portfolio using Politis–Romano bootstrap paths.

    Given bootstrap-generated price paths for a portfolio, this class:
      - converts prices to log-returns per path,
      - computes portfolio value paths (normalized to 1 at t=0),
      - computes per-path annualized return, annualized volatility, and Sharpe ratio,
      - provides plotting helpers for path distributions and statistics.
    """
    
    def __init__(self, prb, sharpe_offset):
        
        # Generate bootstrap price paths: expected shape (n_paths, 1, T_prices)
        prb_data = prb.generate()
        self.portfolios_value = prb_data.squeeze(axis=1) # (n_paths, T)
        self.n_paths = prb_data.shape[0]
        self.n_years = prb_data.shape[2] // 365
        
        p_lrets = np.diff(np.log(self.portfolios_value), axis=1) # (n_paths, T-1)
        
        # Per-path mean and volatility of portfolio log-returns
        mean_lrets = p_lrets.mean(axis=1) # (n_paths,)
        vol_lrets = p_lrets.std(axis=1, ddof=1) # (n_paths,)
        
        annualized_returns = np.expm1(mean_lrets * 365) # (n_paths,)
        annualized_vols = vol_lrets * np.sqrt(365) # (n_paths,)
        sharpes = (annualized_returns - sharpe_offset) / annualized_vols # (n_paths,)
        
        # Max Drawdown per path
        running_peak = np.maximum.accumulate(self.portfolios_value, axis=1) # (n_paths, T)
        dd = 1.0 - self.portfolios_value / running_peak # (n_paths, T+1)
        max_drawdowns = dd.max(axis=1) # (n_paths,)

        self.d_stat = {'Annualized returns': annualized_returns,
                       'Annualized volatilities': annualized_vols,
                       'Sharpe ratios': sharpes,
                       'Max Drawdowns': max_drawdowns}
        
    def _plot_statistics(self, bins=120):
        """
        Plot histograms of the stored distributions (annualized return, vol, Sharpe),
        adding quartile (Q1/median/Q3) vertical lines in green and the mean in red.
        Also prints the probability of a negative Sharpe ratio.
        """
        
        for title, serie in self.d_stat.items():
            if title in ['Annualized returns', 'Annualized volatilities', 'Max Drawdowns']:
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
            
            if title == 'Annualized returns':
                p = len(serie[serie<0]) / len(serie) * 100
                print(f'Probability of negative annualized returns: {round(p, 2)}%')
            
    def _plot_paths(self):
        """
        Plot all portfolio value paths in a single figure.
        Each line is one bootstrap path. A horizontal line at 1 marks the initial value.
        """
        plt.figure()
        plt.plot(self.portfolios_value.T, alpha=0.5, linewidth=0.5)
        plt.axhline(1, color='black')
        plt.xlabel(f'{self.n_paths} Bootstrap Series of {self.n_years} years of size')
        plt.show()

