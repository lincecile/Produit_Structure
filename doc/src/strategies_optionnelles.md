# Strategies and Structured Products Framework Documentation

## Overview

The strategies and structured products framework provides a comprehensive system for creating, managing, and analyzing option portfolios and structured financial products. It consists of four main components:

1. **Option Portfolios** - Managing collections of vanilla and exotic options
2. **Structured Products Portfolios** - Managing collections of structured financial products
3. **Predefined Option Strategies** - Creating standard option combinations (spreads, straddles, etc.)
4. **Structured Product Strategies** - Creating complex structured products from underlying components

This modular architecture allows for flexible construction of sophisticated investment strategies with powerful analytics and visualization capabilities.

## Class Hierarchy

```
OptionsPortfolio (Product)
│
StructuredProductsPortfolio (Product)
│
OptionsStrategy
│
StructuredProductsStrategy
```

## Portfolio_options.py - Option Portfolio Management

The [`OptionsPortfolio`](../../src/Strategies_optionnelles/Portfolio_options.py) class provides a comprehensive system for managing collections of options:

### Key Features:

- **Portfolio Construction**:
  - [`add_option()`](../../src/Strategies_optionnelles/Portfolio_options.py#L51) - Add options with specified quantities
  - [`remove_option_quantity()`](../../src/Strategies_optionnelles/Portfolio_options.py#L83) - Remove specific quantities of options
  - [`clear_portfolio()`](../../src/Strategies_optionnelles/Portfolio_options.py#L78) - Empty the portfolio
  
- **Portfolio Analysis**:
  - [`calculate_option_greeks()`](../../src/Strategies_optionnelles/Portfolio_options.py#L103) - Compute greeks for specific options
  - [`calculate_portfolio_greeks()`](../../src/Strategies_optionnelles/Portfolio_options.py#L129) - Compute aggregate portfolio greeks
  - [`get_portfolio_summary()`](../../src/Strategies_optionnelles/Portfolio_options.py#L141) - Generate portfolio summary statistics
  - [`get_portfolio_detail()`](../../src/Strategies_optionnelles/Portfolio_options.py#L166) - Generate detailed portfolio information

- **Visualization**:
  - [`plot_portfolio_payoff()`](../../src/Strategies_optionnelles/Portfolio_options.py#L241) - Interactive payoff diagram for the entire portfolio
  - [`plot_option_payoff()`](../../src/Strategies_optionnelles/Portfolio_options.py#L340) - Interactive payoff diagram for a single option

- **Payoff Computation**:
  - [`compute_payoff()`](../../src/Strategies_optionnelles/Portfolio_options.py#L201) - Calculate option payoffs with barrier handling

## Portfolio_structured.py - Structured Products Portfolio Management

The [`StructuredProductsPortfolio`](../../src/Strategies_optionnelles/Portfolio_structured.py) class extends the portfolio concept to structured products:

### Key Features:

- **Portfolio Management**:
  - [`add_structured_product()`](../../src/Strategies_optionnelles/Portfolio_structured.py#L38) - Add structured products to the portfolio
  - [`remove_product()`](../../src/Strategies_optionnelles/Portfolio_structured.py#L69) - Remove products from the portfolio
  - [`clear_portfolio()`](../../src/Strategies_optionnelles/Portfolio_structured.py#L65) - Empty the portfolio

- **Portfolio Analysis**:
  - [`get_portfolio_summary()`](../../src/Strategies_optionnelles/Portfolio_structured.py#L94) - Generate portfolio summary statistics
  - [`get_portfolio_detail()`](../../src/Strategies_optionnelles/Portfolio_structured.py#L111) - Generate detailed portfolio information

- **Payoff Computation**:
  - [`compute_payoff()`](../../src/Strategies_optionnelles/Portfolio_structured.py#L144) - Calculate structured product payoffs
  - Multiple specialized payoff calculators for different product types:
    - [`_compute_capital_protected_note_payoff()`](../../src/Strategies_optionnelles/Portfolio_structured.py#L172)
    - [`_compute_reverse_convertible_payoff()`](../../src/Strategies_optionnelles/Portfolio_structured.py#L210)
    - [`_compute_barrier_digital_payoff()`](../../src/Strategies_optionnelles/Portfolio_structured.py#L245)
    - [`_compute_athena_autocall_payoff()`](../../src/Strategies_optionnelles/Portfolio_structured.py#L293)

- **Visualization**:
  - [`plot_product_payoff()`](../../src/Strategies_optionnelles/Portfolio_structured.py#L347) - Interactive payoff diagram for a specific product
  - [`plot_portfolio_payoff()`](../../src/Strategies_optionnelles/Portfolio_structured.py#L417) - Interactive payoff diagram for the entire portfolio

## StructuredStrat.py - Structured Product Strategy Creation

The [`StructuredProductsStrategy`](../../src/Strategies_optionnelles/StructuredStrat.py) class enables the creation of complex structured products from component parts:

### Key Features:

- **Building Block Management**:
  - [`create_option()`](../../src/Strategies_optionnelles/StructuredStrat.py#L39) - Create option components
  - [`add_zero_coupon_bond()`](../../src/Strategies_optionnelles/StructuredStrat.py#L67) - Add bond components
  - [`add_put_down_in()`](../../src/Strategies_optionnelles/StructuredStrat.py#L90) - Add barrier put components
  - [`add_digital_call()`](../../src/Strategies_optionnelles/StructuredStrat.py#L111) - Add digital call replications
  - [`add_digital_put()`](../../src/Strategies_optionnelles/StructuredStrat.py#L142) - Add digital put replications

- **Structured Product Builders**:
  - [`capital_protected_note()`](../../src/Strategies_optionnelles/StructuredStrat.py#L173) - Create capital protected notes with various parameters
  - [`reverse_convertible()`](../../src/Strategies_optionnelles/StructuredStrat.py#L252) - Create reverse convertibles (yield enhancement)
  - [`barrier_digital()`](../../src/Strategies_optionnelles/StructuredStrat.py#L298) - Create barrier digital options
  - [`athena_autocall()`](../../src/Strategies_optionnelles/StructuredStrat.py#L376) - Create autocallable structured notes

- **Factory Method**:
  - [`create_strategy()`](../../src/Strategies_optionnelles/StructuredStrat.py#L682) - Generic method to create products based on name and parameters

## Strategies_predefinies.py - Option Strategy Creation

The [`OptionsStrategy`](../../src/Strategies_optionnelles/Strategies_predefinies.py) class provides functionality to build standard option combinations:

### Key Features:

- **Basic Option Positions**:
  - [`long_call()`](../../src/Strategies_optionnelles/Strategies_predefinies.py#L34) - Create long call positions
  - [`short_call()`](../../src/Strategies_optionnelles/Strategies_predefinies.py#L40) - Create short call positions
  - [`long_put()`](../../src/Strategies_optionnelles/Strategies_predefinies.py#L46) - Create long put positions
  - [`short_put()`](../../src/Strategies_optionnelles/Strategies_predefinies.py#L52) - Create short put positions

- **Standard Option Strategies**:
  - [`call_spread()`](../../src/Strategies_optionnelles/Strategies_predefinies.py#L58) - Create bull call spreads
  - [`put_spread()`](../../src/Strategies_optionnelles/Strategies_predefinies.py#L74) - Create bear put spreads
  - [`strangle()`](../../src/Strategies_optionnelles/Strategies_predefinies.py#L90) - Create strangles
  - [`straddle()`](../../src/Strategies_optionnelles/Strategies_predefinies.py#L109) - Create straddles
  - [`butterfly()`](../../src/Strategies_optionnelles/Strategies_predefinies.py#L118) - Create butterfly spreads
  - [`collar()`](../../src/Strategies_optionnelles/Strategies_predefinies.py#L146) - Create collars
  - [`forward()`](../../src/Strategies_optionnelles/Strategies_predefinies.py#L166) - Create synthetic forwards

- **Factory Method**:
  - [`create_strategy()`](../../src/Strategies_optionnelles/Strategies_predefinies.py#L186) - Generic method to create strategies based on name and parameters

## Usage Examples

### Creating an Options Portfolio

```python
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_brownian import Brownian
from src.options.Pricing_option.Classes_Both.module_marche import DonneeMarche
from src.Strategies_optionnelles.Portfolio_options import OptionsPortfolio

# Initialize market data and Brownian motion
market_data = DonneeMarche(date_pricing=today, prix_spot=100, volatilite=0.2, taux_interet=0.03)
brownian = Brownian(time_to_maturity=1.0, nb_step=100, nb_trajectoire=10000)

# Create portfolio
portfolio = OptionsPortfolio("My Portfolio", brownian, market_data)

# Add options manually
option1 = Option(prix_exercice=100, maturite=maturity, call=True)
portfolio.add_option(option1, quantity=1)

# Analyze portfolio
summary = portfolio.get_portfolio_summary()
greeks = portfolio.calculate_portfolio_greeks()

# Visualize payoff
payoff_chart = portfolio.plot_portfolio_payoff()
```

### Using Option Strategies

```python
from src.Strategies_optionnelles.Strategies_predefinies import OptionsStrategy

# Create strategy builder
strategy_builder = OptionsStrategy(portfolio, market_data, expiry_date=maturity)

# Add a simple butterfly spread
strategy_builder.butterfly(
    lower_strike=90, 
    middle_strike=100, 
    upper_strike=110,
    quantity=1,
    is_call=True
)

# Add a parameterized strategy
params = {
    "strike1": 95,
    "strike2": 105,
    "quantity": 1,
    "americaine": False
}
strategy_builder.create_strategy("Call Spread", params)

# Visualize updated portfolio
payoff_chart = portfolio.plot_portfolio_payoff()
```

### Creating Structured Products

```python
from src.Strategies_optionnelles.Portfolio_structured import StructuredProductsPortfolio
from src.Strategies_optionnelles.StructuredStrat import StructuredProductsStrategy
from src.rate import Rate

# Create structured products portfolio
structured_portfolio = StructuredProductsPortfolio("My Structured Portfolio")

# Create strategy builder
rate = Rate(rate=0.03)
structured_strategy = StructuredProductsStrategy(
    structured_portfolio=structured_portfolio,
    options_portfolio=portfolio,
    market_data=market_data,
    rate=rate,
    expiry_date=maturity
)

# Create a capital protected note
structured_strategy.capital_protected_note(
    protection_level=0.9,  # 90% capital protection
    participation_rate=0.7,  # 70% participation in upside
    notional=100.0,
    cap=0.2  # 20% cap on returns
)

# Create a barrier digital using parameters
params = {
    "barrier_level": 0.9,
    "payout": 10.0,
    "is_up_direction": False,
    "is_knock_in": True,
    "notional": 100.0
}
structured_strategy.create_strategy("barrier digital", params)

# Analyze portfolio
summary = structured_portfolio.get_portfolio_summary()
details = structured_portfolio.get_portfolio_detail()

# Visualize payoff
payoff_chart = structured_portfolio.plot_portfolio_payoff()
```

### Streamlit App Implementation

The framework is easily integrated into Streamlit applications as shown in `app.py`:

```python
# Initialize portfolios in session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = OptionsPortfolio("", brownian, donnee_marche)

# Add buttons for strategy creation
ajouter_strategie = st.button('Ajouter une stratégie prédéfinie au portefeuille')

# Add strategy when button is clicked
if ajouter_strategie:
    strategy_name = option_type_strat.lower()
    strategy = OptionsStrategy(st.session_state.portfolio, donnee_marche, expiry_date=date_maturite)
    strategy.create_strategy(option_type_strat, params, 1 if sens_option == 'Long' else -1)
    st.success(f"Stratégie {option_type_strat} créée avec succès !")

# Display portfolio visualization
try:
    detail_folio = st.session_state.portfolio.get_portfolio_detail()
    if len(detail_folio) != 0:
        st.dataframe(detail_folio)
    fig = st.session_state.portfolio.plot_portfolio_payoff(show_individual=True)
    st.plotly_chart(fig, use_container_width=True)
except:
    st.markdown("Aucune produit dans le portefeuille")
```

## Summary

The strategies and structured products framework provides a comprehensive system for building, analyzing, and visualizing option portfolios and structured products. It offers:

1. **Portfolio Management** - Track collections of options and structured products
2. **Risk Analytics** - Calculate greeks and portfolio metrics
3. **Payoff Visualization** - Create interactive payoff diagrams
4. **Strategy Building** - Easily create standard option combinations
5. **Structured Product Creation** - Construct complex financial instruments from component parts

The modular design allows for flexibility in combining different strategy elements while maintaining a consistent interface for analysis and visualization.