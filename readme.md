# Créer un environnement 

python -m venv nom_environnement
nom_environnement\Scripts\activate.bat

# install package
pip install scipy 

# pour update le requierement
pip freeze > requirements.txt

# pour installer les packages du requierement
pip install -r requierements.txt