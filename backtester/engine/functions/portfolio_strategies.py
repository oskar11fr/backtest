import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from scipy.optimize import minimize
from hmmlearn.hmm import GaussianHMM

class PositioningStrategy(ABC):
    @abstractmethod
    def get_strat_positions(
        self,
        forecasts: np.ndarray,
        eligibles_row: np.ndarray,
        capitals: float,
        strat_scalar: float,
        vol_row: np.ndarray,
        close_row: np.ndarray
    ) -> np.ndarray:
        pass

class VolatilityTargetingStrategy(PositioningStrategy):
    def get_strat_positions(
        self,
        idx: int,
        forecasts: np.ndarray,
        eligibles_row: np.ndarray,
        capitals: float,
        strat_scalar: float,
        vol_row: np.ndarray,
        close_row: np.ndarray,
        vol_target: float,
        max_leverage: float,
        min_leverage: float,
        rets_df: dict[str, pd.DataFrame]
    ) -> np.ndarray:
        forecast_chips = np.sum(np.abs(forecasts))

        if forecast_chips == 0:
            return np.zeros(len(forecasts))
        
        positions = strat_scalar * \
            forecasts / forecast_chips  \
            * vol_target \
            / (vol_row * close_row)
        lev_temp = np.linalg.norm(positions * close_row, ord=1) / capitals
        normalized_positions = positions / lev_temp
        positions = max_leverage*normalized_positions if lev_temp > max_leverage else positions
        positions = min_leverage*normalized_positions if lev_temp < min_leverage else positions
        positions = np.floor(np.nan_to_num(positions,nan=0,posinf=0,neginf=0))
        return positions

class MeanVarianceStrategy(PositioningStrategy):
    def get_strat_positions(
        self,
        forecasts: np.ndarray,
        eligibles_row: np.ndarray,
        capitals: float,
        strat_scalar: float,
        vol_row: np.ndarray,
        close_row: np.ndarray,
        vol_target: float,
        max_leverage: float,
        min_leverage: float
    ) -> np.ndarray:
        # Placeholder: Implement mean-variance optimization
        expected_returns = forecasts
        cov_matrix = np.diag(vol_row ** 2)

        inv_cov_matrix = np.linalg.inv(cov_matrix)
        weights = inv_cov_matrix @ expected_returns
        weights /= np.sum(weights)

        positions = strat_scalar * weights * capitals / close_row
        return np.floor(np.nan_to_num(positions))

class hmmMeanVarianceStrategy(PositioningStrategy):
    TRAIN_SIZE = .7
    MODEL = GaussianHMM
    SEED = np.random.seed(1)

    def get_strat_positions(
        self, 
        idx: int,
        forecasts: np.ndarray, 
        eligibles_row: np.ndarray, 
        capitals: float, 
        strat_scalar: float, 
        vol_row: np.ndarray, 
        close_row: np.ndarray,
        vol_target: float,
        max_leverage: float,
        min_leverage: float,
        rets_df: dict[str, pd.DataFrame]
    ) -> np.ndarray:
        if idx == 0:
            train_rets, self.rets = self.get_rets(rets_df=rets_df)
            self.model = self.train_hmm(train_rets=train_rets)
            return np.zeros(len(forecasts))
        state = self.predict_states(model=self.model, rets=self.rets[(idx - 1):idx,:])
        state_covar = self.model.covars_[1-state]
        state_mean = self.model.means_[1-state]

        forecast_chips = np.sum(np.abs(forecasts))
        expected_returns = state_mean * (forecasts / forecast_chips)
        # Mean-Variance Optimization
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(state_covar, weights)))
            return -portfolio_return / portfolio_vol 
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - max_leverage})
        
        # Bounds: no short selling (weights >= 0), adjust if shorting allowed
        bounds = [(0, max_leverage) for _ in range(len(forecasts))]

        init_guess = np.array([1.0 / len(forecasts)] * len(forecasts))
        result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        optimized_weights = result.x if result.success else np.zeros_like(forecasts)
        # positions = capitals * optimized_weights / close_row
        positions = strat_scalar * \
            optimized_weights  \
            * vol_target \
            / (vol_row * close_row)
        return np.floor(positions)
    
    def train_hmm(self, train_rets: np.ndarray):
        model = self.MODEL(
            n_components=2,
            covariance_type="full",
            algorithm="map",
            random_state=self.SEED
        )
        return model.fit(X=train_rets)

    def predict_states(self, model: GaussianHMM, rets: np.ndarray) -> np.ndarray:
        return model.predict(X=rets)[0]

    def get_rets(self, rets_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = int(len(rets_df) * self.TRAIN_SIZE)
        rets = rets_df.shift(1).fillna(0).values # shift 1 lag to avoid lookahead bias
        return rets[:n,:], rets