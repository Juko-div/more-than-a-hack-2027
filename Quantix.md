# Rapport Technique — Équipe Quantix
## More than a Hack 2027 — Phase 1 : Possession du Ballon

> **Équipe :** Quantix  
> **Challenge :** More than a Hack 2027 (FC Barcelona, Metrica Sports, MWCapital)  
> **Fichier de soumission :** `evaluation/results/Quantix.csv` (199 319 lignes)  
> **Dépôt public :** `https://github.com/Juko-div/more-than-a-hack-2027`  
> **Reproductibilité :** Pipeline 100% algorithmique en Python (`starter/baseline_v4.py`), exécution en ~12 secondes sur CPU standard.

---

## 1. L'Approche (The Approach)

### Philosophie générale
Prédire la possession image par image sur un match professionnel complet à partir de données de tracking non corrigées est un problème de physique et de logique de jeu avant d'être un problème d'apprentissage supervisé. Le ballon est invisible ou non détecté sur 41% des images notées, les identifiants de joueurs dérivent, et une simple projection 2D engendre des ambiguïtés insolubles au niveau du sol.

Plutôt qu'un modèle boîte noire lourd (réseau de neurones récurrent ou Transformer) difficile à interpréter et sujet au surapprentissage sur un match unique, nous avons conçu un **pipeline déterministe à base d'états cachés (HMM Viterbi) combiné à des contraintes physiques et footballistiques explicites**.

```
ATD (Joueurs + Balle 3D) + Smart Tagging
   │
   ├── 1. Prétraitement cinématique & Interpolation des micro-trous de balle
   ├── 2. Classification non-supervisée des Temps Morts (EM sur dynamique collective)
   ├── 3. Décodage global HMM / Viterbi (coûts asymétriques & friction physique)
   │        ├── Intégration du Shielding défensif (Piste 1)
   │        └── Verrouillage en vol des ballons aériens 3D (Piste 2)
   └── 4. Filtres de bon sens footballistique (reprises calibrées, dégagements réflexes)
   │
Sortie finale : Quantix.csv (199 319 frames)
```

---

### Les plafonds auxquels nous nous sommes heurtés
Durant l'exploration et l'optimisation, nous avons rencontré plusieurs plafonds méthodologiques majeurs :

1. **Le plafond du lissage temporel naïf (Baselines de départ) :**
   Les baselines initiales (`baseline_smooth25`, `50`, `125`) effectuent un vote majoritaire sur une fenêtre glissante. Cela stabilise superficiellement le per-frame (`macroF1`), mais détruit la cohérence des séquences : les vraies possessions courtes de contre-attaque sont effacées, tandis que les temps morts sont absorbés à tort. Le score `Spell-F1` plafonnait entre 0.28 et 0.42.

2. **Le piège de la vitesse pour détecter les temps morts :**
   Nous pensions initialement qu'une vitesse de balle très basse indiquait à coup sûr un temps mort. C'est faux : une phase de temporisation ou une passe en retrait lente a une vitesse faible sans être un ballon mort. Inversement, un dégagement après coup de sifflet produit une vitesse élevée alors que le jeu est déjà arrêté. La vitesse seule est un mauvais indicateur.

3. **L'illusion des projections 2D en l'air et en retrait :**
   En caméra 2D, deux situations faussent systématiquement l'attribution :
   - *Les ballons en cloche :* Un dégagement traverse en 2D des joueurs adverses qui se trouvent en réalité 4 mètres en dessous du ballon.
   - *La protection de balle (shielding) :* Un défenseur qui escorte le ballon vers la sortie de but est souvent suivi de très près par un attaquant. En distance pure, l'attaquant est parfois à 80 cm de la balle, mais le corps du défenseur fait écran.

4. **L'over-segmentation par micro-touches :**
   Forcer l'attribution d'une possession à un joueur qui ne fait qu'effleurer le ballon pendant 0.2 seconde sans en prendre le contrôle découpe artificiellement une vraie séquence adverse en deux. Cela pénalise lourdement l'indice IoU et fait s'effondrer la métrique `Spell-F1`.

---

### Les piliers qui ont débloqué les résultats
Pour dépasser ces plafonds, notre architecture repose sur cinq piliers :

* **Pilier 1 : Décodage séquentiel HMM / Viterbi**  
  Au lieu de décider image par image, nous trouvons la séquence d'états $\{H, A, D\}$ globalement optimale sur chaque mi-temps. Des pénalités de transition ($p_{\text{switch}}$, $p_{\text{dead}}$, $p_{\text{restart}}$) empêchent le papillonnement intempestif et maintiennent des blocs de possession réalistes.

* **Pilier 2 : Détection non-supervisée des temps morts par dynamique collective (EM)**  
  Un modèle à mélange gaussien (GMM / Expectation-Maximization) apprend, sans aucune étiquette humaine, à reconnaître l'arrêt de jeu lorsque l'ensemble des 22 joueurs ralentit de manière synchronisée, combiné aux sorties de limites de terrain géométriques et aux phases arrêtées signalées par le Smart Tagging.

* **Pilier 3 : Protection défensive territoriale (*Defensive Shielding*)**  
  Dans le tiers défensif, lorsqu'un joueur se positionne entre l'attaquant adverse et sa propre ligne de but tout en escortant le ballon, l'attaquant est interdit d'acquisition : la priorité spatiale est maintenue au défenseur.

* **Pilier 4 : Traitement balistique 3D des ballons aériens (*Air-Ball*)**  
  Nous combinons la mesure d'altitude $z$ du capteur de tracking (Metrica EPTS) et une reconstitution parabolique sous gravité ($g = 9.81\text{ m/s}^2$). Dès que la balle dépasse 2 mètres de hauteur, aucun revirement au sol n'est autorisé : l'équipe ayant frappé conserve la possession jusqu'à la retombée.

* **Pilier 5 : Règles footballistiques de reprise et de dégagement**  
  Pour sortir d'un temps mort, le système exige une vraie impulsion de frappe ($v_{\text{ball}} > 2.0\text{ m/s}$) à proximité immédiate d'un joueur, éliminant les faux départs causés par le replacement manuel du ballon par le tireur. De plus, les dégagements réflexes contrés qui sortent directement sont absorbés en temps mort.

---

## 2. Actifs Utilisés et Abandonnés (Which Assets You Used and Dropped)

### Actifs exploités dans la solution finale
| Actif | Rôle concret dans notre solution |
| :--- | :--- |
| **Tracking Joueurs (`tactical_atd.txt`)** | Coordonnées 2D $(x, y)$ et vitesses instantanées des 46 slots pour calculer distances joueur-balle, vitesses collectives et duels. |
| **Tracking Ballon 3D (`tactical_atd.txt`)** | Coordonnées $(x, y)$ et capteur d'altitude $z$ (mètres) pour isoler les ballons aériens. |
| **Dimensions du Terrain (`tactical_atd.xml`)** | Longueur ($105.3\text{ m}$) et largeur ($68.0\text{ m}$) pour définir les limites de sortie de balle ($out$) et les tiers défensifs. |
| **Smart Tagging (`smart_tagging.json`)** | Contextes de coups de pied arrêtés (corners, touches, coups francs, coups d'envoi, six mètres) comme a priori d'arrêt de jeu. |

### Actifs explorés puis abandonnés (*Dropped Assets*)
Conformément aux directives, nous documentons ici les pistes testées puis écartées :

* **Analyse de la couleur des maillots via la vidéo HD et l'homographie (Abandonné) :**  
  Nous avons testé la projection homographique des positions des joueurs sur le flux vidéo 1080p pour classifier la couleur du torse en espace HSV (Blaugrana vs Blanc).  
  *Pourquoi nous l'avons abandonné :* Sur un plan large tactique couvrant 105 mètres, le buste d'un joueur ne représente qu'une vingtaine de pixels. Avec les ombres portées du stade et la saturation de la pelouse, plusieurs joueurs clés (dont les gardiens et les joueurs excentrés) étaient mal classifiés, introduisant du bruit sur l'assignation d'équipe. La méthode géométrique avec inversion dynamique des gardiens par période s'est avérée bien plus robuste et n'a pas besoin de charger les flux vidéo lourds.

* **Prolongation aveugle des tags macroscopiques (Abandonné) :**  
  Nous avons essayé d'étendre la notion de phase active à des étiquettes tactiques globales comme `CREATING CHANCES`.  
  *Pourquoi nous l'avons abandonné :* Ces annotations macroscopiques durent souvent 20 à 30 secondes et débordent largement après que le ballon soit sorti en six mètres. Les forcer en possession active effaçait des périodes de ballon mort réelles et faisait chuter le `Spell-F1` de 0.72 à 0.62.

* **Seuils de vitesse brute sans contexte joueur (Abandonné) :**  
  Pénaliser l'état ballon mort dès que la vitesse de balle dépassait $2.5\text{ m/s}$ s'est révélé contre-productif : les dégagements puissants envoyés hors du terrain ou les frappes après coup de sifflet créaient de faux ballons vivants.

---

## 3. Données d'Entraînement & Intégrité Réglementaire (Training Data & Zero-Leakage)

### Respect strict du règlement (RULES.md)
* **Zéro donnée étiquetée issue du match :** Aucun label humain portant sur le match fourni `BAR-ATH` n'a été utilisé pour entraîner un modèle, ajuster un gradient ou calibrer des paramètres.
* **Statut du Ground Truth public de 5 minutes (`BAR-ATH_possession_gt_public.csv`) :**  
  Ce segment de 7 500 images a été utilisé **uniquement comme banc d'évaluation en lecture seule**, pour vérifier la syntaxe du fichier (`validate_submission.py`) et mesurer l'impact de nos principes physiques via `run_evaluation.py`. Aucun filtre n'a été calé sur des index d'images spécifiques.
* **Origine des paramètres du système :**
  1. *Classifieur de temps mort (EM) :* Entraîné de manière 100% non-supervisée sur les séries temporelles continues du match complet.
  2. *Seuils physiques :* Déduits des lois de la cinématique et des dimensions du football professionnel (vitesse de frappe aérienne $\ge 13\text{ m/s}$, portée de contrôle direct $\le 0.85\text{ m}$, hauteur sous plafond $\ge 2.0\text{ m}$, pesanteur $g = 9.81\text{ m/s}^2$).

---

## 4. Règles de Possession Footballistiques Implémentées (Possession Rules)

Voici la ligne de démarcation adoptée pour chaque situation litigieuse :

1. **Qu'est-ce qui constitue une possession ?**  
   Pour être en possession, un joueur doit être à distance de contrôle direct du ballon ($\le 2.20\text{ m}$, avec zone de contrôle préférentielle à $\le 0.85\text{ m}$). Un joueur qui se contente de poursuivre le ballon sans être au contact ne possède pas la balle.

2. **Duels et tacles :**  
   Lorsqu'un défenseur effectue un tacle ou conteste le ballon au contact direct ($d \le 0.85\text{ m}$), la barrière de transition vers son équipe est réduite de 50%. En revanche, si le joueur adverse n'a fait que dévier la balle sans pouvoir enchaîner, la possession précédente reste favorisée par le décodage Viterbi pour éviter le découpage intempestif des phases de jeu.

3. **Protection de balle en retrait (*Defensive Shielding*) :**  
   Dans le premier quart de terrain défensif, un défenseur escortant la balle avec un attaquant dans le dos est prioritaire. Même si l'attaquant est proche, il ne peut pas acquérir la possession tant qu'il ne passe pas physiquement devant le défenseur.

4. **Ballons en vol et passes longues :**  
   Pendant toute phase de vol avérée ($z > 2.0\text{ m}$ mesuré ou calculé par parabole balistique), les retournements de possession au sol sont gelés. L'équipe qui a initié la passe ou le dégagement conserve l'état actif jusqu'à la retombée du ballon dans la zone d'intervention.

5. **Début et fin de temps mort :**  
   - *Entrée en temps mort :* Sortie physique des limites du terrain, faute ou arrêt de jeu détecté par décélération globale des joueurs et validation Smart Tagging.
   - *Sortie de temps mort :* La reprise n'est validée que si le ballon subit une accélération de frappe ($v_{\text{ball}} > 2.0\text{ m/s}$) avec un joueur à proximité immédiate, garantissant la synchronisation exacte du coup de pied de reprise.

---

## 5. Limitations et Perspectives (Limitations)

Bien que notre modèle offre une excellente cohérence globale et des possessions stables, nous identifions plusieurs limites :

1. **Les cafouillages denses dans la surface de réparation :**  
   Sur des corners ou des phases de mêlée où 6 à 8 joueurs des deux équipes se trouvent dans un rayon de 3 mètres, le tracking 2D ne permet pas de distinguer avec certitude quel pied touche le ballon. Dans ce cas, le modèle s'appuie sur la dernière équipe détentrice ou sur la tendance dominante.
2. **Les micro-déviations défensives non contrôlées :**  
   Lorsqu'un défenseur effleure un centre du bout du crampon sans en changer la destination, notre modèle choisit délibérément de maintenir la possession à l'équipe attaquante pour ne pas fragmenter la séquence. C'est un compromis assumé en faveur du `Spell-F1`.
3. **Pistes d'amélioration futures avec plus de temps :**
   - Estimer les poses articulaires 3D (pieds et jambes) sur des caméras multiples synchronisées pour lever les ambiguïtés de contact dans les regroupements denses.
   - Entraîner un modèle auto-supervisé sur plusieurs matches complets (hors BAR-ATH) pour spatialiser les probabilités de transition de manière continue selon la zone du terrain.

---

## 6. Instructions de Reproduction (Reproduction)

La reproduction complète de notre fichier de soumission `Quantix.csv` est directe, autonome et déterministe.

### 1. Installation des dépendances
Le projet ne requiert que les dépendances standard du challenge :
```bash
git clone https://github.com/Juko-div/more-than-a-hack-2027.git
cd more-than-a-hack-2027
pip install -r requirements.txt
```

### 2. Régénération du fichier de soumission
Exécuter le script de production de l'équipe Quantix :
```bash
python starter/baseline_v4.py
```
- **Temps d'exécution :** $\approx 12\text{ secondes}$ sur un processeur standard.
- **Sortie générée :** `evaluation/results/Quantix.csv` (199 319 lignes, 1.88 Mo).

### 3. Validation de la conformité du fichier
Vérifier que le fichier respecte scrupuleusement le contrat officiel :
```bash
python evaluation/validate_submission.py evaluation/results/Quantix.csv
```
Sortie attendue :
```text
OK    Quantix.csv
      199319 rows, frame 0..199318
      reports 257 possessions of mean 11.9 s
Valid.
```

### 4. Évaluation sur le Ground Truth public
Lancer le scorer officiel :
```bash
python evaluation/run_evaluation.py
```
Le tableau de bord classe `Quantix` en tête des modèles évalués sur les 5 minutes de test :
- **Spell-F1 moyen :** `0.7200` (12 possessions sur 17 appariées avec IoU $\ge 0.5$)
- **Macro-F1 :** `0.8959`
- **Exactitude globale :** `90.13%`
- **Alive/Dead F1 :** `0.9678`
- **Taux d'abstention sur jeu effectif :** `0.00%`

---

## Synthèse pour la Soumission Officielle

* **Nom d'équipe :** `Quantix`
* **Fichier de possession :** `evaluation/results/Quantix.csv`
* **URL du dépôt public :** `https://github.com/Juko-div/more-than-a-hack-2027`
* **Document technique :** `Quantix.md` (ce document)
