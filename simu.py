# systems_pred_prey.py
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple

# Type aliases
State  = Dict[str, float]          # stock name -> current value
Params = Dict[str, Any]
FlowFn = Callable[[float, State, Params], float]  # f(t, state, params) -> rate

@dataclass
class Stock:
    name: str
    initial: float
    inflows: List[FlowFn]
    outflows: List[FlowFn]

def simulate(
    stocks: List[Stock],
    T: float = 60.0,
    dt: float = 0.05,
    params: Params = None
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Forward Euler simulation for stocks with multiple inflows/outflows."""
    if params is None:
        params = {}

    # Initialize state and history
    state: State = {s.name: s.initial for s in stocks}
    history: Dict[str, List[float]] = {s.name: [s.initial] for s in stocks}
    time: List[float] = [0.0]

    steps = int(T / dt)
    for _ in range(steps):
        t = time[-1]
        # Compute net rate for each stock
        derivatives: Dict[str, float] = {}
        for s in stocks:
            inflow_rate  = sum(fn(t, state, params) for fn in s.inflows)
            outflow_rate = sum(fn(t, state, params) for fn in s.outflows)
            derivatives[s.name] = inflow_rate - outflow_rate

        # Update state (Euler)
        for s in stocks:
            state[s.name] += derivatives[s.name] * dt
            # Optional: keep stocks non-negative
            if state[s.name] < 0.0:
                state[s.name] = 0.0

        # Record
        time.append(t + dt)
        for s in stocks:
            history[s.name].append(state[s.name])

    return time, history

# ---------- Predator–Prey model (Lotka–Volterra with readable names) ----------
def prey_births(t: float, state: State, params: Params) -> float:
    a = params.get("a", 1.0)             # prey intrinsic growth rate
    X = state["Prey"]
    return a * X

def predation_on_prey(t: float, state: State, params: Params) -> float:
    b = params.get("b", 0.1)             # encounter/predation coefficient
    X, Y = state["Prey"], state["Predator"]
    return b * X * Y

def predator_growth_from_food(t: float, state: State, params: Params) -> float:
    b = params.get("b", 0.1)
    c = params.get("c", 0.5)             # conversion efficiency
    X, Y = state["Prey"], state["Predator"]
    return c * b * X * Y

def predator_deaths(t: float, state: State, params: Params) -> float:
    d = params.get("d", 0.5)             # predator natural death rate
    Y = state["Predator"]
    return d * Y

# Define stocks
prey = Stock(
    name="Prey",
    initial=40.0,
    inflows=[prey_births],
    outflows=[predation_on_prey]
)

predator = Stock(
    name="Predator",
    initial=9.0,
    inflows=[predator_growth_from_food],
    outflows=[predator_deaths]
)

if __name__ == "__main__":
    # Params and run
    params = {"a": 1.0, "b": 0.02, "c": 0.6, "d": 0.8}
    T, dt = 150.0, 0.05
    time, hist = simulate([prey, predator], T=T, dt=dt, params=params)

    # Plot (optional)
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(time, hist["Prey"], label="Prey")
        plt.plot(time, hist["Predator"], label="Predator")
        plt.xlabel("Time")
        plt.ylabel("Population")
        plt.title("Predator–Prey (Lotka–Volterra)")
        plt.legend()
        plt.show()
    except ImportError:
        print("Final Prey:", hist["Prey"][-1], "Final Predator:", hist["Predator"][-1])
