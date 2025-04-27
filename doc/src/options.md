# Options Pricing and Analysis Framework Documentation

## Overview

The options framework provides a comprehensive system for pricing, analyzing, and visualizing various types of financial options. It implements multiple pricing methodologies and offers sophisticated tools for risk analysis, volatility modeling, and Greeks calculation. The system can handle:

1. **Option Types** - European, American, Asian, and barrier options
2. **Pricing Models** - Black-Scholes, Monte Carlo (LSM), Trinomial Tree, and Heston model
3. **Stochastic Processes** - Standard Brownian motion and Heston stochastic volatility
4. **Risk Analytics** - Greeks calculation, sensitivity analysis, and visualization tools

This modular architecture supports highly customizable option pricing with robust analytics capabilities suitable for both simple and exotic derivatives.

## Class Hierarchy

```
Options Framework
├── Classes_Both (Shared components)
│   ├── module_option (Option)
│   ├── module_marche (DonneeMarche)
│   ├── module_barriere (Barriere)
│   ├── module_black_scholes (BlackAndScholes)
│   ├── derivatives (OptionDerivatives)
│   └── module_pricing_analysis (StrikeComparison, VolComparison, RateComparison)
│
├── Classes_MonteCarlo_LSM (Monte Carlo with Longstaff-Schwartz)
│   ├── module_brownian (Brownian)
│   ├── module_LSM (LSM_method)
│   └── module_graph (LSMGraph)
│
├── Classes_TrinomialTree (Trinomial Tree model)
│   ├── module_arbre_noeud (Arbre)
│   └── module_grecques_empiriques (GrecquesEmpiriques)
│
└── HestonPricer (Stochastic volatility model)
    ├── Models
    │   ├── models_european_option (EuropeanOption)
    │   ├── models_asian_option (AsianOption)
    │   ├── models_option_base (OptionBase)
    │   └── models_heston_parameters (HestonParameters)
    └── Pricing
        └── pricing_monte_carlo_pricer (MonteCarloPricer)
```

## Classes_Both - Core Components

The [`Classes_Both`](../../src/options/Pricing_option/Classes_Both) directory contains components shared across different pricing models:

### Key Classes:

- **Option ([`module_option`](../../src/options/Pricing_option/Classes_Both/module_option.py))** - Core class representing financial options
  - Handles various option types (call/put, European/American)
  - Supports barrier conditions and dividend handling
  - Provides key functionality for pricing algorithms

- **Market Data ([`module_marche`](../../src/options/Pricing_option/Classes_Both/module_marche.py))** - Encapsulates market information
  - Manages spot price, volatility, interest rates
  - Handles dividend information (ex-date and amount)
  - Provides data necessary for any pricing model

- **Barrier ([`module_barriere`](../../src/options/Pricing_option/Classes_Both/module_barriere.py))** - Manages barrier option parameters
  - Supports knock-in and knock-out barriers
  - Handles up and down barrier directions
  - Defines barrier checking logic

- **Black-Scholes ([`module_black_scholes`](../../src/options/Pricing_option/Classes_Both/module_black_scholes.py))** - Analytical pricing model
  - Closed-form solutions for European options
  - Greeks calculation
  - Serves as benchmark for numerical methods

- **Option Derivatives ([`derivatives`](../../src/options/Pricing_option/Classes_Both/derivatives.py))** - Greek calculations
  - Numerical calculation of Delta, Gamma, Vega, Rho, etc.
  - Supports multiple calculation methods
  - Works with different pricing models

## Classes_MonteCarlo_LSM - Monte Carlo Simulation

The [`Classes_MonteCarlo_LSM`](../../src/options/Pricing_option/Classes_MonteCarlo_LSM) directory implements the Longstaff-Schwartz Method (LSM) for American option valuation:

### Key Features:

- **Brownian Motion ([`module_brownian`](../../src/options/Pricing_option/Classes_MonteCarlo_LSM/module_brownian.py))**
  - Simulates Brownian paths for underlying asset
  - Supports antithetic variance reduction
  - Flexible path generation with customizable parameters

- **LSM Pricer ([`module_LSM`](../../src/options/Pricing_option/Classes_MonteCarlo_LSM/module_LSM.py))**
  - Implements Longstaff-Schwartz regression-based algorithm
  - Supports both vectorized and scalar calculation methods
  - Handles American option early exercise
  - Integrated barrier condition checking
  - Dividend adjustment capabilities

- **Visualization ([`module_graph`](../../src/options/Pricing_option/Classes_MonteCarlo_LSM/module_graph.py))**
  - Path visualization for Brownian motion
  - Comparison tools for different pricing parameters
  - Polynomial basis function analysis
  - Convergence analysis tools

## Classes_TrinomialTree - Tree-based Methods

The [`Classes_TrinomialTree`](../../src/options/Pricing_option/Classes_TrinomialTree) directory implements trinomial tree-based option pricing:

### Key Features:

- **Trinomial Tree ([`module_arbre_noeud`](../../src/options/Pricing_option/Classes_TrinomialTree/module_arbre_noeud.py))**
  - Node-based tree structure for pricing
  - Backward induction for option valuation
  - Early exercise handling for American options
  - Tree pruning for computational efficiency

- **Greeks Calculation ([`module_grecques_empiriques`](../../src/options/Pricing_option/Classes_TrinomialTree/module_grecques_empiriques.py))**
  - Numerical estimation of option Greeks
  - Finite difference methods
  - Delta, Gamma, Vega, Theta, and Rho calculations

## HestonPricer - Stochastic Volatility Model

The [`HestonPricer`](../../src/options/HestonPricer) directory implements the Heston stochastic volatility model:

### Key Features:

- **Option Models**
  - [`models_european_option`](../../src/options/HestonPricer/Models/models_european_option.py) - European option valuation
  - [`models_asian_option`](../../src/options/HestonPricer/Models/models_asian_option.py) - Asian option valuation
  - [`models_option_base`](../../src/options/HestonPricer/Models/models_option_base.py) - Abstract base class for options

- **Heston Parameters ([`models_heston_parameters`](../../src/options/HestonPricer/Models/models_heston_parameters.py))**
  - Managing Heston model-specific parameters:
    - Mean reversion rate (kappa)
    - Long-term variance (theta)
    - Volatility of volatility (sigma)
    - Correlation between asset and volatility (rho)
    - Initial variance (v0)

- **Monte Carlo Pricer ([`pricing_monte_carlo_pricer`](../../src/options/HestonPricer/Pricing/pricing_monte_carlo_pricer.py))**
  - Simulating price paths under Heston model
  - Option pricing with stochastic volatility
  - Risk analysis and visualization tools

## Usage Examples

### Basic European Option Pricing

```python
from src.options.Pricing_option.Classes_Both.module_option import Option
from src.options.Pricing_option.Classes_Both.module_marche import DonneeMarche
from src.options.Pricing_option.Classes_Both.module_black_scholes import BlackAndScholes
from src.options.Pricing_option.Classes_TrinomialTree.module_arbre_noeud import Arbre

# Define option parameters
option = Option(
    date_maturite=date_maturite,
    prix_exercice=100.0,
    call=True,
    americaine=False
)

# Define market data
market_data = DonneeMarche(
    date_pricing=today,
    prix_spot=100.0,
    volatilite=0.2,
    taux_interet=0.03
)

# Create a trinomial tree
tree = Arbre(nb_pas=100, donnee_marche=market_data, option=option)

# Price with Black-Scholes
bs = BlackAndScholes(modele=tree)
bs_price = bs.bs_pricer()
print(f"Black-Scholes price: {bs_price}")

# Calculate Greeks
delta = bs.delta()
gamma = bs.gamma()
vega = bs.vega()
```

### American Option with LSM Method

```python
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_brownian import Brownian
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_LSM import LSM_method

# Define American option
american_option = Option(
    date_maturite=date_maturite,
    prix_exercice=100.0,
    call=True,
    americaine=True
)

# Set up Brownian motion
brownian = Brownian(
    time_to_maturity=1.0,
    nb_step=100,
    nb_trajectoire=10000
)

# Set up LSM pricer and price the option
pricer = LSM_method(american_option)
price, std_error, conf_interval = pricer.LSM(
    brownian=brownian,
    market=market_data,
    method='vector',
    antithetic=True,
    poly_degree=3,
    model_type='Polynomial'
)

print(f"LSM price: {price}")
print(f"Standard error: {std_error}")
print(f"95% confidence interval: [{conf_interval[0]}, {conf_interval[1]}]")
```

### Barrier Option Pricing

```python
from src.options.Pricing_option.Classes_Both.module_barriere import Barriere, TypeBarriere, DirectionBarriere

# Define a knock-out barrier
barrier = Barriere(
    niveau_barriere=110.0,
    type_barriere=TypeBarriere.knock_out,
    direction_barriere=DirectionBarriere.up
)

# Create barrier option
barrier_option = Option(
    date_maturite=date_maturite,
    prix_exercice=100.0,
    call=True,
    americaine=False,
    barriere=barrier
)

# Price with trinomial tree
tree = Arbre(nb_pas=100, donnee_marche=market_data, option=barrier_option)
tree.pricer_arbre()

print(f"Barrier option price: {tree.prix_option}")
```

### Option Greeks Calculation

```python
from src.options.Pricing_option.Classes_Both.derivatives import OptionDerivatives

# Calculate option greeks using LSM
derivatives = OptionDerivatives(option, market_data, pricer)

# Calculate Greeks
delta = derivatives.delta(brownian)
gamma = derivatives.gamma(brownian)
vega = derivatives.vega(brownian)
rho = derivatives.rho(brownian)
theta = derivatives.theta(brownian)

print(f"Delta: {delta}")
print(f"Gamma: {gamma}")
print(f"Vega: {vega}")
print(f"Rho: {rho}")
print(f"Theta: {theta}")
```

### Heston Model for Stochastic Volatility

```python
from src.options.HestonPricer.Models.models_european_option import EuropeanOption
from src.options.HestonPricer.Models.models_heston_parameters import HestonParameters
from src.options.HestonPricer.Pricing.pricing_monte_carlo_pricer import MonteCarloPricer

# Set up Heston model parameters
heston_params = HestonParameters(
    kappa=1.0,        # Mean reversion rate
    theta=0.04,       # Long-term variance
    sigma=0.2,        # Volatility of volatility
    rho=-0.7,         # Correlation
    v0=0.04           # Initial variance
)

# Create European option for Heston model
european_option = EuropeanOption(
    spot_price=100.0,
    strike=100.0,
    maturity=1.0,
    risk_free_rate=0.03,
    is_call=True
)

# Price using Monte Carlo simulation
mc_pricer = MonteCarloPricer(
    option=european_option,
    heston_params=heston_params,
    nb_paths=10000,
    nb_steps=100
)

price = mc_pricer.price(random_seed=42)
print(f"Heston model price: {price}")
```

### Visualization and Analysis

```python
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_graph import LSMGraph
from src.options.Pricing_option.Classes_Both.module_pricing_analysis import StrikeComparison

# Create graph object for visualization
graph = LSMGraph(option=option, market=market_data)

# Visualize Brownian paths
brownian_graph = graph.afficher_mouvements_browniens(brownian, nb_trajectoires=100)

# Visualize underlying price paths
price_paths = pricer.Price(market=market_data, brownian=brownian)
price_graph = graph.afficher_trajectoires_prix(price_paths, brownian, nb_trajectoires=100)

# Compare models across different strikes
strike_list = np.arange(90.0, 110.0, 1.0)
comparison = StrikeComparison(
    max_cpu=4,
    step_list=[100],
    strike_list=strike_list,
    donnee_marche=market_data,
    brownian=brownian,
    option=option
)

strike_graph = comparison.graph_strike_comparison()
```

### Streamlit App Integration

The options framework is integrated into the Streamlit application as shown in `app.py`:

```python
# Initialize options and pricers
option = Option(date_maturite, strike, barriere=barriere, 
                americaine=False if option_exercice == 'Européenne' else True, 
                call=True if option_type == "Call" else False,
                date_pricing=date_pricing)

brownian = Brownian(time_to_maturity=(date_maturite-date_pricing).days / convention_base_calendaire, 
                   nb_step=nb_pas, nb_trajectoire=nb_chemin, seed=seed_choice)
pricer = LSM_method(option)

# Price button handler
if activer_pricing:
    # Black-Scholes pricing (if applicable)
    if bs_check:
        bns = BlackAndScholes(modele=arbre)
        pricing_bns = f"{round(bns.bs_pricer(),2)}€"
        st.metric('Black-Scholes price:', value=pricing_bns)
    
    # LSM pricing
    price, std_error, intevalles = pricer.LSM(
        brownian=brownian,
        market=donnee_marche_LSM,
        method='vector' if calcul_method == 'Vectorielle' else 'scalar',
        antithetic=antithetic_choice,
        poly_degree=poly_degree,
        model_type=regress_method
    )
    
    # Display results
    st.metric('Option value:', value=f"{round(price, 2)}€")
    st.metric('Standard error:', value=f"{round(std_error, 4)}€")
    
    # Calculate and display Greeks
    option_deriv = OptionDerivatives(option, donnee_marche_LSM, pricer)
    delta = option_deriv.delta(brownian)
    gamma = option_deriv.gamma(brownian)
    vega = option_deriv.vega(brownian)
    
    st.metric('Delta:', value=round(delta, 2))
    st.metric('Gamma:', value=round(gamma, 2))
    st.metric('Vega:', value=round(vega, 2))
```

## Summary

The options pricing and analysis framework offers a comprehensive solution for valuing and analyzing financial options through multiple methodologies:

1. **Versatile Pricing Models** - Black-Scholes, Monte Carlo LSM, Trinomial Tree, and Heston
2. **Option Variety** - Support for European, American, and exotic options including barriers
3. **Risk Management** - Calculation of Greeks and sensitivity analysis
4. **Visualization Tools** - Analysis graphs for paths, convergence, and parameter comparisons
5. **Computational Efficiency** - Vectorized implementations and pruning techniques

The framework's modular design allows selecting the most appropriate method based on option characteristics, required accuracy, and computational constraints.