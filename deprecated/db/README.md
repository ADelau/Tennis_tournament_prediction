# Création de la base de données
1. Faire tourner ``exporter.py`` en se plaçant dans `bigdata/` (chemins configurés de la sorte)
2. Créer la base de données dans le dossier `bigdata/db`, en tapant la commande `sqlite3 tennis_db < create_db.sql`
3. Une fois créée, lancer l'interpréteur SQL avec ``sqlite3 tennis_db``
4. Une fois dans l'interpréteur: 
    * ``.mode csv``
    * ``.import filename.csv table`` Importer chaque fichier csv dans la table correspondante. (**NB** Le fichiers avec
    les joueurs génère une erreur, il faut supprimer la ligne **1933** car vide)