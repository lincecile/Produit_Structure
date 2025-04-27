import sys
import os
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Now we can import with absolute paths
from src.options.Pricing_option.Classes_Both.module_option import Option
from src.options.Pricing_option.Classes_Both.module_marche import DonneeMarche
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_brownian import Brownian
from src.options.Pricing_option.Classes_Both.module_barriere import Barriere, TypeBarriere, DirectionBarriere
from src.bonds import ZCBond
from src.rate import Rate
from src.time_utils.maturity import Maturity

# Import our portfolio classes
from src.Strategies_optionnelles.Portfolio_options import OptionsPortfolio
from src.Strategies_optionnelles.Portfolio_structured import StructuredProductsPortfolio


def test_portfolio_options():
    print("\n=== Testing Options Portfolio ===")
    
    # Create market data
    market_data = DonneeMarche(
        date_debut=dt.date.today(),
        prix_spot=100,
        volatilite=0.2,
        taux_interet=0.02,
        taux_actualisation=0.02,   # added required parameter
        dividende_ex_date=dt.date.today()  # added required parameter
    )
    
    # Create Brownian motion parameters
    brownian = Brownian(
        time_to_maturity=1.0,  # 1 year
        nb_step=50,
        nb_trajectoire=10000,
        seed=42
    )
    
    # Create options portfolio
    options_portfolio = OptionsPortfolio(
        name="My Options Portfolio",
        brownian=brownian,
        market=market_data
    )
    
    # Create options (call and put)
    maturity_date = dt.date.today() + dt.timedelta(days=365)
    call_option = Option(
        prix_exercice=100.0,
        maturite=maturity_date,
        call=True
    )
    
    put_option = Option(
        prix_exercice=95.0,
        maturite=maturity_date,
        call=False
    )
    
    # Create barrier option
    barrier = Barriere(
        niveau_barriere=90.0,
        type_barriere=TypeBarriere.knock_in,
        direction_barriere=DirectionBarriere.down
    )
    
    barrier_put = Option(
        prix_exercice=100.0,
        maturite=maturity_date,
        call=False,
        barriere=barrier
    )
    
    # Add options to portfolio
    options_portfolio.add_option(call_option, quantity=1.0)
    options_portfolio.add_option(put_option, quantity=2.0)
    options_portfolio.add_option(barrier_put, quantity=1.0)
    
    # Display portfolio information
    summary = options_portfolio.get_portfolio_summary()
    print("Portfolio Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Display detailed information
    details = options_portfolio.get_portfolio_detail()
    print("\nPortfolio Details:")
    print(details)
    
    # Calculate Greeks
    try:
        greeks = options_portfolio.calculate_portfolio_greeks()
        print("\nPortfolio Greeks:")
        for greek, value in greeks.items():
            print(f"  {greek}: {value}")
    except Exception as e:
        print(f"Error calculating Greeks: {e}")
    
    # Generate plots
    try:
        fig = options_portfolio.plot_portfolio_payoff(show_premium=True)
        print("\nPortfolio payoff plot generated successfully.")
        # To display the plot in a browser:
        fig.show()
        
        # Plot individual option
        option_fig = options_portfolio.plot_option_payoff(0, show_premium=True)
        print("Option payoff plot generated successfully.")
        option_fig.show()
    except Exception as e:
        print(f"Error plotting payoff: {e}")
    
    print("Options Portfolio test complete!")


def test_portfolio_structured():
    print("\n=== Testing Structured Products Portfolio ===")
    
    # Create structured products portfolio
    structured_portfolio = StructuredProductsPortfolio(
        name="My Structured Products Portfolio"
    )
    
    # Create mock components for structured products
    maturity_date = dt.date.today() + dt.timedelta(days=365)
    maturity = Maturity(
        start_date=dt.date.today(),
        end_date=maturity_date,
        day_count="ACT/365"
    )
    
    # Create zero-coupon bond (with name)
    rate = Rate(0.02)
    zcb = ZCBond(name="ZCBond_90", face_value=90.0, maturity=maturity, rate=rate)
    
    # Create options for different products
    call_option = Option(
        prix_exercice=100.0,
        maturite=maturity_date,
        call=True
    )
    
    # Create barrier for put option
    barrier = Barriere(
        niveau_barriere=85.0,
        type_barriere=TypeBarriere.knock_in,
        direction_barriere=DirectionBarriere.down
    )
    
    put_option = Option(
        prix_exercice=100.0,
        maturite=maturity_date,
        call=False,
        barriere=barrier
    )
    
    # Components for capital protected note
    cpn_components = [
        {'type': 'zero_coupon_bond', 'object': zcb},
        {'type': 'participation_call', 'object': call_option, 'quantity': 0.5}
    ]
    
    # Components for reverse convertible
    rc_components = [
        {'type': 'zero_coupon_bond', 'object': ZCBond(name="ZCBond_105", face_value=105.0, maturity=maturity, rate=rate)},
        {'type': 'put_down_in', 'object': put_option, 'quantity': -1.0}
    ]
    
    # Create barrier for digital product
    up_barrier = Barriere(
        niveau_barriere=110.0,
        type_barriere=TypeBarriere.knock_in,
        direction_barriere=DirectionBarriere.up
    )
    
    barrier_call = Option(
        prix_exercice=100.0,
        maturite=maturity_date,
        call=True,
        barriere=up_barrier
    )
    
    # Components for barrier digital product
    digital_components = [
        {'type': 'digital_call', 'strike': 110.0, 'payout': 10.0, 'quantity': 1.0},
        {'type': 'barrier', 'object': barrier_call}
    ]
    
    # Create components for Athena Autocall product
    # 1. Create observation dates (quarterly)
    today = dt.date.today()
    observation_dates = [
        today + dt.timedelta(days=90),   # 3 months
        today + dt.timedelta(days=180),  # 6 months
        today + dt.timedelta(days=270),  # 9 months
        today + dt.timedelta(days=365)   # 12 months
    ]
    
    # 2. Set up the maturity for the zero-coupon bond in the Athena
    athena_maturity = Maturity(
        start_date=today,
        end_date=observation_dates[-1],
        day_count="ACT/365"
    )
    
    athena_zcb = ZCBond(name="ZCBond_Athena", face_value=100.0, maturity=athena_maturity, rate=rate)
    
    # 3. Create barrier for the downside protection
    protection_barrier = Barriere(
        niveau_barriere=70.0,  # 70% of initial price
        type_barriere=TypeBarriere.knock_in,
        direction_barriere=DirectionBarriere.down
    )
    
    athena_put = Option(
        prix_exercice=100.0,
        maturite=observation_dates[-1],
        call=False,
        barriere=protection_barrier
    )
    
    # 4. Components for Athena Autocall
    athena_components = [
        {'type': 'zero_coupon_bond', 'object': athena_zcb},
        {'type': 'put_down_in', 'object': athena_put, 'quantity': -1.0},
        # Add autocall observations
        {'type': 'autocall_observation', 'date': observation_dates[0], 'barrier_level': 30.0, 'coupon_rate': 0.5, 'payout': 102.5},
        {'type': 'autocall_observation', 'date': observation_dates[1], 'barrier_level': 30.0, 'coupon_rate': 0.05, 'payout': 105.0},
        {'type': 'autocall_observation', 'date': observation_dates[2], 'barrier_level': 100.0, 'coupon_rate': 0.075, 'payout': 107.5},
        {'type': 'autocall_observation', 'date': observation_dates[3], 'barrier_level': 30.0, 'coupon_rate': 0.10, 'payout': 110.0}
    ]
    
    # Add products to portfolio
    structured_portfolio.add_structured_product(
        product_name="Capital Protected Note 90%",
        product_type="capital_protected_note",
        components=cpn_components,
        price=95.0,
        quantity=1.0
    )
    
    structured_portfolio.add_structured_product(
        product_name="Reverse Convertible 5% Coupon",
        product_type="reverse_convertible",
        components=rc_components,
        price=98.0,
        quantity=1.0
    )
    
    structured_portfolio.add_structured_product(
        product_name="Up Knock-In Digital",
        product_type="barrier_digital",
        components=digital_components,
        price=3.0,
        quantity=2.0
    )
    
    # Add the Athena Autocall product
    structured_portfolio.add_structured_product(
        product_name="Athena Autocall 2.5/5.0/7.5/10.0% - 100/95/90/85% Barrier - 70% Protection",
        product_type="athena_autocall",
        components=athena_components,
        price=97.5,
        quantity=1.0
    )
    
    # Display portfolio information
    summary = structured_portfolio.get_portfolio_summary()
    print("Portfolio Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Display detailed information
    details = structured_portfolio.get_portfolio_detail()
    print("\nPortfolio Details:")
    print(details)
    
    # Plot individual product payoff - Athena Autocall
    try:
        fig = structured_portfolio.plot_product_payoff(product_index=3, show_premium=True)
        print("\nAthena Autocall payoff plot generated successfully.")
        # To display the plot:
        fig.show()
    except Exception as e:
        print(f"Error plotting Athena Autocall payoff: {e}")
    
    # Plot full portfolio payoff
    try:
        fig = structured_portfolio.plot_portfolio_payoff(show_premium=True)
        print("\nFull portfolio payoff plot generated successfully.")
        # To display the plot:
        fig.show()
    except Exception as e:
        print(f"Error plotting portfolio payoff: {e}")
    
    print("Structured Products Portfolio test complete!")

if __name__ == "__main__":
    # test_portfolio_options()
    test_portfolio_structured()