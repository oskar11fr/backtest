import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from scipy.optimize import minimize
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from backtester.engine import save_obj, load_obj


class PositioningMethod(ABC):
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

class VanillaVolatilityTargeting(PositioningMethod):
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
        return positions


class MixtureModelsMeanVariance(PositioningMethod):

    def __init__(self, model_name: str = "hmm", load_model: bool = False, strat_name: str = "") -> None:
        super().__init__()
        self.model_map = {"hmm": GaussianHMM, "gmm": GaussianMixture}

        self.model_name = model_name
        self.models_confs = {
            "hmm": {
                "n_components": 2,
                "covariance_type": "full",
                "algorithm": "map"
            },
            "gmm": {
                "n_components": 2,
                "covariance_type": "full"
            }
        }
        self.load = load_model
        self.strat_name = strat_name + model_name

        assert model_name in self.models_confs.keys(), f"Make sure model_name is in {self.models_confs.keys()}"
        self._ii = 0
        # self.MODEL: GaussianHMM | GaussianMixture = model_map[model_name]
        

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
        retdf, max_leverage, min_leverage, trade_frequency, train_size = kwargs["retdf"], kwargs["max_leverage"], kwargs["min_leverage"], kwargs["trade_frequency"], kwargs["train_size"]
        
        n = len(retdf)
        help_mapper = {"daily": 1, "weekly": 7, "monthly": 31}

        if self.load:
            if (idx == 0):
                self.model = load_obj(self.strat_name)
                _, self.ret = self.get_rets(retdf=retdf,trade_frequency=trade_frequency,train_id=n)
        if not self.load or self.model is None:
            train_len = int(n * train_size)
            step = int(np.diff(np.linspace(start=train_len, stop=n-1, num=5))[0])
            train_check = any([(idx - i) % step == 0 for i in range(help_mapper[trade_frequency])]) & (idx > train_len)
            if (idx == 0) or train_check: self.model = self.run_training(retdf=retdf,trade_frequency=trade_frequency,idx=idx,train_len=train_len,train_check=train_check) 

        if (idx == 0): return np.zeros(len(forecasts))

        state = self.predict_states(model=self.model, ret=self.ret[(idx - 1):idx,:])
        state_covar, state_means = self.get_params(model=self.model, state=state)
        forecasts_returns =  forecasts * state_means
        
        optimized_weights = self.mean_variance_optimization(forecasts_returns=forecasts_returns,covar=state_covar,max_leverage=max_leverage)
        positions = np.minimum(optimized_weights,.5) * strat_scalar * vol_target / (vol_row * close_row)
        lev_temp = np.linalg.norm(positions * close_row, ord=1) / capitals
        normalized_positions = positions / lev_temp
        positions = max_leverage*normalized_positions if lev_temp > max_leverage else positions
        positions = min_leverage*normalized_positions if lev_temp < min_leverage else positions
        positions = np.floor(np.nan_to_num(positions,nan=0,posinf=0,neginf=0))
        
        end_check = any([(n - 1 - i) == idx for i in range(help_mapper[trade_frequency])])
        if end_check and not self.load: self.save_model(retdf=self.ret)
        return positions #np.floor(positions)
    
    def mean_variance_optimization(self, forecasts_returns: np.ndarray, covar: np.ndarray, max_leverage: float) -> np.ndarray:
        def objective(weights):
            portfolio_return = np.dot(weights, forecasts_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covar, weights)))
            return -portfolio_return / portfolio_vol 
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - max_leverage})
        
        # Bounds: no short selling (weights >= 0), adjust if shorting allowed
        bounds = [(0, max_leverage) for _ in range(len(forecasts_returns))]

        init_guess = np.array([1.0 / len(forecasts_returns)] * len(forecasts_returns))
        result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        optimized_weights = result.x if result.success else np.zeros_like(forecasts_returns)
        return optimized_weights
    
    def run_training(self, retdf: pd.DataFrame, trade_frequency: str, idx: int, train_len: int, train_check: bool) -> tuple[GaussianHMM | GaussianMixture]:
        train_id = train_len if not train_check else idx
        train_ret, self.ret = self.get_rets(
            retdf=retdf,
            trade_frequency=trade_frequency,
            train_id=train_id - 1
        )

        print("=============================================================== ")
        print(f" - Training window: {retdf.index[0]} - {retdf.index[train_id - 1]}")
        print(f" - Testing window: {retdf.index[train_id - 1]} - {retdf.index[-1]}")
        print(f" - Regime prediction model: {self.model_map[self.model_name]}")
        print("--------------------------------------------------------------- ")
        
        model = self.train(train_ret=train_ret)
        print("=============================================================== ")
        return model
    
    def train(self, train_ret: np.ndarray) -> tuple[GaussianHMM | GaussianMixture]:
        models, scores = [], []
        for s in range(10):
            model = self.model_map[self.model_name](random_state=s,**self.models_confs[self.model_name])
            model.fit(X=train_ret)
            models.append(model)
            scores.append(model.score(X=train_ret))

            if isinstance(model, GaussianHMM): print(f'Converged: {model.monitor_.converged} --- Score: {scores[-1]}')
            if isinstance(model, GaussianMixture): print(f'Score: {scores[-1]}')

        model = models[np.argmax(scores)]
        print(f'The best model had a score of {max(scores)}')
        return model
    
    def save_model(self, retdf: pd.DataFrame) -> None:
        model = self.train(train_ret=retdf)
        print("=============================================================== ")
        print(" - Saving final model")
        save_obj(model, self.strat_name)
        return model
    
    def load_model(self) -> tuple[GaussianHMM | GaussianMixture]:
        model = load_obj(self.strat_name)
        return model

    def predict_states(self, model: GaussianHMM | GaussianMixture, ret: np.ndarray) -> np.ndarray:
        return model.predict(X=ret)[0]
    
    def get_params(self, model: GaussianHMM | GaussianMixture, state: int) -> np.ndarray:
        if isinstance(model, GaussianHMM):
            state_covar, state_mean = model.covars_[state], model.means_[state]
        if isinstance(model, GaussianMixture):
            state_covar, state_mean = model.covariances_[state], model.means_[state]
        return state_covar, state_mean

    def get_rets(self, retdf: pd.DataFrame, trade_frequency: str, train_id: int) -> tuple[np.ndarray, np.ndarray]:
        if trade_frequency == "weekly": wind = 7
        if trade_frequency == "monthly": wind = 30
        if trade_frequency == "daily": wind = 1

        ret = retdf.shift(1).rolling(wind,min_periods=1).mean().fillna(0).values # shift 1 lag to avoid lookahead bias
        return ret[:train_id,:], ret
    