Da fare in comune:

### ENV:
- matrice "valuations" (num_item, time_horizon)
- `def round(self, current_time: int)`



### ad ogni simulazione:

```python
@dataclass
class RunSimulationResult:
    """Result of a single simulation run."""
    valuation: np.ndarray  # (num_item, time_horizon)
    played_arms: np.ndarray  # (num_item, time_horizon) (None/int)
```

- valuation (num_item, time_horizon)
- played_arms (num_item, time_horizon) (None/int) --> None solo se il budget è finito