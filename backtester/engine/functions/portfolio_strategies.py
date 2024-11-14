import numpy as np

from abc import ABC, abstractmethod

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
        close_row: np.ndarray
    ) -> np.ndarray:
        # Placeholder: Implement mean-variance optimization
        expected_returns = forecasts
        cov_matrix = np.diag(vol_row ** 2)

        inv_cov_matrix = np.linalg.inv(cov_matrix)
        weights = inv_cov_matrix @ expected_returns
        weights /= np.sum(weights)

        positions = strat_scalar * weights * capitals / close_row
        return np.floor(np.nan_to_num(positions))
