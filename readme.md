# Créer un environnement 

python -m venv nom_environnement
nom_environnement\Scripts\activate.bat

# install package
pip install scipy 

# pour update le requierement
pip freeze > requirements.txt

# pour installer les packages du requierement
pip install -r requierements.txt


## TODO
- https://github.com/BaptisteZloch/Structured-Products-in-Python

1 - monte carlo ok
2 - vasicek qui sort un float pour le rate
3 - vol sto avec heston (convertir code c# emma)
4 - class ZC 
      - 
5 - utiliser arbre pour option vanille
6 - future/forward
      - FX
      - greeks sur autre que option => future etc.
7 - monte carlo avec option barriere (imen)
8 - class strategy option qui permet de prendre plusieurs option vanille en entrée
      - wedding cake
      - wedding cake
8 - athena/phoenix
9 - option
      - option FX
      - option asiat
      - option bermude
      - basket equi
      - basket WOF/BOF
      - lookback min/max


