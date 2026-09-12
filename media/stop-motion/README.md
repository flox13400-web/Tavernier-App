# Stop motion « EVG FLO »

Vidéo verticale (1080x1920, 30 fps, 22,7 s) construite à partir des photos du plateau Pix.

Sans rapport avec l'application Tavernier : ce dossier ne sert qu'à archiver le
montage et les scripts qui l'ont produit.

## Chaîne de production

1. `detect_plateau.py` : détecte le plateau dans chaque photo (zone blanche
   désaturée), en extrait les quatre coins et redresse la grille 9x9 par
   homographie. Corrige de fait l'orientation et le cadrage, quelle que soit la
   photo d'origine. Sort `corners.json`.
2. `build_stopmotion.py` :
   - reconstruit un plateau vide propre (photo de base + greffe d'une case prise
     sur une seconde photo redressée, les grilles coïncidant au pixel près) ;
   - détoure six pions réels (masque alpha par composante connexe, après érosion
     pour ne pas accrocher les lignes de grille) ;
   - compose chaque état du plateau, lettre par lettre, avec ombre portée et
     léger tremblement de pose ;
   - écrit la liste de plans `concat.txt`.
3. Encodage :

```sh
ffmpeg -y -f concat -safe 0 -i concat.txt -fps_mode cfr -r 30 \
  -vf "scale=1080:1920:flags=lanczos,format=yuv420p" \
  -c:v libx264 -preset slow -crf 18 -movflags +faststart evg-flo.mp4
```

Dépendances : `ffmpeg`, `python3`, `pillow`, `numpy`.
