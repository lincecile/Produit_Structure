# Products Framework Documentation

## Overview

The [`Product`](../../src/products.py#L13) class serves as the abstract base class for all financial instruments in the system. It provides a common interface and shared functionality for different types of financial products like bonds, stocks, swaps, and convertible bonds.

## Class Hierarchy

```
Product (ABC)
├── BondBase
│   ├── ZCBond (Zero-Coupon Bond)
│   └── Bond
│       └── Convertible
├── Stock
└── SwapBase
    └── Swap
```

## Product Class (products.py)

The [`Product`](../../src/products.py#L13) class is the foundation of the financial instruments framework:

### Key Features:

- **Abstract Base Class**: Defines the interface that all financial products must implement
- **Common Attributes**:
  - [`name`](../../src/products.py#L17): Product identifier
  - [`price`](../../src/products.py#L19): Current price (can be provided or calculated)
  - [`price_history`](../../src/products.py#L18): Historical price data (array)
  - [`volatility`](../../src/products.py#L20): Product volatility (provided or calculated from price history)
- **Required Methods**:
  - [`_get_price()`](../../src/products.py#L25): Abstract method that subclasses must implement for pricing.
- **Utility Methods**:
  - [`__get_volatility()`](../../src/products.py#L31): Calculates annualized volatility from price history if it exists and the user hasn't provided it.

## Bond Instruments (bonds.py)

Bonds build on the [`Product`](../../src/products.py#L9) base class with fixed income functionality:

### BondBase

```python
class BondBase(Product):
    def __init__(self, name: str, rate: Rate, maturity: Maturity,
                 face_value: float, price_history: Optional[list[float]] = None,
                 price: Optional[float] = None) -> None:
        # Initialize with bond-specific attributes
```

Bond classes add important features like:
- Duration calculations ([`get_duration()`](../../src/bonds.py#L45))
- DV01 - dollar value of a basis point ([`get_dv01()`](../../src/bonds.py#L51))
- Interest rate handling through the [`Rate`](../../src/rate.py#L10) class
- Maturity management via the [`Maturity`](../../src/time_utils/maturity.py#L10) class

This features are common to all bond types, including zero-coupon bonds and standard bonds.

### Zero-Coupon Bond

A specialized bond with a single payment at maturity. The price is calculated using the discount factor from the rate object:

```python
class ZCBond(BondBase):
    def _get_price(self) -> float:
        # Calculate zero-coupon bond price using discount factor
        return self.face_value * self.rate.discount_factor(self.maturity)
```

### Standard Bond

Regular bonds with periodic coupon payments:

```python
class Bond(BondBase):
    def __init__(self, name: str, rate: Rate, maturity: Maturity,
                 face_value: float, coupon: float, nb_coupon: int, 
                 price_history: Optional[list[float]] = None,
                 price: Optional[float] = None) -> None:
        # Initialize with coupon information
```

Additional bond capabilities:
- Cash flow stripping and discounting
- Yield-to-maturity calculations ([`get_ytm()`](../../src/bonds.py#L178))
- Conversion to zero-coupon equivalents ([`to_zcbonds()`](../../src/bonds.py#L280))

## Stock Class (stocks.py)

Represents equity securities:

```python
class Stock(Product):
    def __init__(self, name: str, price: float, div_amount: float,
                 div_schedule: Schedule, price_history: Optional[list[float]] = None,
                 volatility: Optional[float] = None) -> None:
        # Initialize with stock-specific attributes
```

Key features:
- Dividend data ([`div_amount`](../../src/stocks.py#L25) and [`div_schedule`](../../src/stocks.py#L26))
- Simple [`_get_price()`](../../src/stocks.py#L38) implementation that returns the provided or cached price

## Swaps (swaps.py)

Interest rate swaps build on the [`Product`](../../src/products.py#L13) class. It has been built on top of the [`SwapBase`](../../src/swaps.py#L20) class, which provides the basic structure for swap products. This class might be used for other types of swaps in the future.

The swap class is designed to handle different leg types (fixed, floating, basket) and allows for separate receiving and paying legs : 

```python
class Swap(SwapBase):
    def __init__(self, name: str, notional: float, schedule: Schedule,
                 receiving_leg: LegType, paying_leg: LegType,
                 receiving_rate: Rate, paying_rate: Rate,
                 receiving_rate_premium: float = 0.0,
                 paying_rate_premium: float = 0.0,
                 price_history: Optional[list[float]] = None,
                 price: Optional[float] = None) -> None:
        # Initialize swap with legs and rates
```

Key features:
- Different leg types (fixed, floating, basket)
- Separate receiving and paying legs
- Schedule-based payment timing
- Rate handling for calculating cash flows

## Convertible Bonds (convertibles.py)

Hybrid instruments combining bonds and equity options. They inherit from the [`Bond`](../../src/bonds.py#L113) class and add option pricing capabilities. Convertible bonds allow the holder to convert the bond into a specified number of shares of the underlying stock.
We thus consider the convertible bond as a combination of a bond and an a call option on the underlying equity.

```python
class Convertible(Bond):
    def __init__(self, name: str, rate: Rate, maturity: Maturity,
                 face_value: float, coupon: float, nb_coupon: int,
                 conversion_ratio: float, stock: Stock,
                 price_history: Optional[list[float]] = None,
                 price: Optional[float] = None) -> None:
        # Initialize with both bond and option components
```

Key features:
- Inherits bond pricing for the debt component
- Adds equity option valuation
- Combines both components: [`_get_price()`](../../src/convertibles.py#L128) = bond value + option value (using our trinomial tree model [`Arbre`](../../src/options/Pricing_option/Classes_TrinomialTree/module_arbre_noeud.py#L19) for pricing.)
- Links to underlying [`Stock`](../../src/stocks.py#L12) object for option valuation
- Specialized duration and DV01 calculations incorporating both components

## Interaction Examples

### Creating a Standard Bond

```python
from rate import Rate
from time_utils.maturity import Maturity
from bonds import Bond

# Create a 5-year bond with 4% annual coupon
rate_obj = Rate(...)  # Rate object configuration
maturity = Maturity(start_date=today, end_date=today+timedelta(days=365*5))
my_bond = Bond(
    name="Corporate Bond",
    rate=rate_obj,
    maturity=maturity,
    face_value=1000.0,
    coupon=0.04,
    nb_coupon=2  # Semi-annual
)

# Access the price (triggers _get_price())
bond_price = my_bond.price
```

### Creating a Convertible Bond

```python
from stocks import Stock
from convertibles import Convertible

# Create underlying stock
my_stock = Stock(
    name="CompanyA",
    price=50.0,
    div_amount=1.0,
    div_schedule=schedule_obj
)

# Create convertible bond linked to the stock
conv_bond = Convertible(
    name="ConvBond",
    rate=rate_obj,
    maturity=maturity,
    face_value=1000.0,
    coupon=0.03,  # Lower coupon due to conversion option
    nb_coupon=2,
    conversion_ratio=20.0,  # Each bond converts to 20 shares
    stock=my_stock
)

# Access both components
bond_component = conv_bond.bond_component
option_component = conv_bond.option_component
```

## Summary

The [`Product`](../../src/products.py#L13) abstract base class creates a polymorphic interface that all financial instruments share, enabling consistent handling and pricing across different product types while allowing each to implement specialized behavior.