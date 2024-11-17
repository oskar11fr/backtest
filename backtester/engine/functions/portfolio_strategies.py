import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from scipy.optimize import minimize
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture

class PositioningStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.TRAIN_ID = 0

    @abstractmethod
    def get_strat_positions(
       self,
        forecasts: np.ndarray,
        capitals: float,
        strat_scalar: float,
        vol_row: np.ndarray,
        close_row: np.ndarray,
        vol_target: float,
        idx: int,
        **kwargs
    ) -> np.ndarray:
        pass

class VolatilityTargetingStrategy(PositioningStrategy):
    def get_strat_positions(
        self,
        forecasts: np.ndarray,
        capitals: float,
        strat_scalar: float,
        vol_row: np.ndarray,
        close_row: np.ndarray,
        vol_target: float,
        idx: int,
        **kwargs
    ) -> np.ndarray:
        max_leverage, min_leverage = kwargs["max_leverage"], kwargs["min_leverage"]
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
        capitals: float,
        strat_scalar: float,
        vol_row: np.ndarray,
        close_row: np.ndarray,
        vol_target: float,
        idx: int,
        **kwargs
    ) -> np.ndarray:
        # Placeholder: Implement mean-variance optimization
        expected_returns = forecasts
        cov_matrix = np.diag(vol_row ** 2)

        inv_cov_matrix = np.linalg.inv(cov_matrix)
        weights = inv_cov_matrix @ expected_returns
        weights /= np.sum(weights)

        positions = strat_scalar * weights * capitals / close_row
        return np.floor(np.nan_to_num(positions))

class MixtureModelsMeanVarianceStrategy(PositioningStrategy):
    TRAIN_SIZE = .6
    SEED = np.random.seed(1)

    def __init__(self, model_name: str = "hmm") -> None:
        super().__init__()
        models_confs = {
            "hmm": GaussianHMM(
                n_components=2,
                covariance_type="full",
                algorithm="map",
                random_state=self.SEED
            ),
            "gmm": GaussianMixture(
                n_components=2,
                covariance_type="full",
                random_state=self.SEED
            )
        }
        assert model_name in models_confs.keys(), f"Make sure model_name is in {models_confs.keys()}"
        self.MODEL: GaussianHMM | GaussianMixture = models_confs[model_name]

    def get_strat_positions(
        self,
        forecasts: np.ndarray,
        capitals: float,
        strat_scalar: float,
        vol_row: np.ndarray,
        close_row: np.ndarray,
        vol_target: float,
        idx: int,
        **kwargs
    ) -> np.ndarray:
        retdf, max_leverage, trade_frequency = kwargs["retdf"], kwargs["max_leverage"], kwargs["trade_frequency"]
        if idx == 0:
            train_ret, self.ret = self.get_rets(retdf=retdf,trade_frequency=trade_frequency)
            self.model = self.train(train_ret=train_ret)
            print("=============================================================== ")
            print(f" - Training window: {retdf.index[0]} - {retdf.index[self.TRAIN_ID]}")
            print(f" - Testing window: {retdf.index[self.TRAIN_ID+1]} - {retdf.index[-1]}")
            print(f" - Regime prediction model: {self.model}")
            print("=============================================================== ")
            return np.zeros(len(forecasts))
        
        state = self.predict_states(model=self.model, ret=self.ret[(idx - 1):idx,:])
        state_covar, state_mean = self.get_params(model=self.model, state=state)

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
    
    def train(self, train_ret: np.ndarray):
        model = self.MODEL
        return model.fit(X=train_ret)

    def predict_states(self, model: GaussianHMM | GaussianMixture, ret: np.ndarray) -> np.ndarray:
        return model.predict(X=ret)[0]
    
    def get_params(self, model: GaussianHMM | GaussianMixture, state: int) -> np.ndarray:
        if isinstance(model, GaussianHMM):
            state_covar, state_mean = model.covars_[state], model.means_[state]
        if isinstance(model, GaussianMixture):
            state_covar, state_mean = model.covariances_[state], model.means_[state]
        return state_covar, state_mean

    def get_rets(self, retdf: pd.DataFrame, trade_frequency: str) -> tuple[np.ndarray, np.ndarray]:
        if trade_frequency == "weekly": wind = 7
        if trade_frequency == "monthly": wind = 30
        if trade_frequency == "daily": wind = 0

        self.TRAIN_ID = int(len(retdf) * self.TRAIN_SIZE)
        ret = retdf.shift(1).rolling(wind,min_periods=0).mean().fillna(0).values # shift 1 lag to avoid lookahead bias
        return ret[:self.TRAIN_ID,:], ret