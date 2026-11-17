# Rapport Technique — Équipe Quantix
## More than a Hack 2027 — Phase 1 : Possession du Ballon

> **Équipe :** Quantix  
> **Challenge :** More than a Hack 2027 (FC Barcelona, Metrica Sports, MWCapital)  
> **Fichier de soumission :** `evaluation/results/Quantix.csv` (199 319 lignes)  
> **Dépôt de code :** `https://github.com/Juko-div/more-than-a-hack-2027`  
> **Exécution :** `python starter/baseline_v4.py` (~12 secondes sur CPU standard)

---

## 1. Démarche et Méthode par Balayage Étagé

### 1.1 Le point de départ : ce que nous avons reçu

Au lancement du challenge, nous disposions du jeu de données officiel fourni par l'organisation :

* Un tracking automatique des 22 joueurs et de la balle (`tactical_atd.txt`) issu de caméras tactiques, sans aucune correction manuelle humaine.
* Les flux vidéo tactique et broadcast en 1080p, ainsi que les matrices d'homographie associées.
* Les événements tactiques automatiquement détectés (*Smart Tagging*).
* Des scripts de référence (`baseline.py`) calculant le joueur le plus proche de la balle, complétés par des lissages fenêtrés uniformes (`smooth25`, `smooth50`, `smooth125`).

Ce point de départ imposait de résoudre des contraintes physiques et méthodologiques sévères :

1. **Un ballon invisible sur 41% du temps de jeu noté :** Le capteur de tracking perd fréquemment la balle lors des frappes lointaines, des sorties en touche ou des regroupements denses.
2. **Des sauts d'identifiants de joueurs :** Lors des mêlées ou des duels serrés, les numéros de slots des joueurs s'inversent régulièrement d'une image à l'autre.
3. **L'écrasement 2D de la caméra :** La vue tactique aplatit le jeu dans un plan horizontal et ignore la hauteur réelle des trajectoires de balle.
4. **La fragmentation des baselines simples :** Une décision locale au joueur le plus proche découpait le match en plus de 130 micro-possessions hachées de 2 secondes, ou écrasait à l'inverse les vraies contre-attaques rapides.

---

### 1.2 La méthode que nous avons adoptée : le balayage étagé (*Cascading Sieve*)

Pour surmonter ces difficultés sans recourir à un modèle boîte noire (dont les hallucinations sont imprévisibles sur un match unique sans labels d'entraînement), nous avons conçu une **méthode par balayage étagé**.

Le principe directeur est simple : **au lieu de vouloir vérifier toutes les conditions en même temps sur chaque image, nous procédons par balayages successifs, du plus certain vers le plus incertain**. Chaque étape pose des certitudes physiques incontestables et affine l'état transmis à l'étape suivante :

```
Données brutes ATD (Joueurs + Balle 3D) & Smart Tagging
   │
   ├── Balayage 1 : Structure globale du match
   │       └── Exclusion des phases hors-match (X) et protocoles de mi-temps (D)
   │
   ├── Balayage 2 : Tamis des temps morts par dynamique collective
   │       └── Apprentissage non-supervisé (EM) du ralentissement synchronisé des 22 joueurs
   │
   ├── Balayage 3 : Décodage séquentiel global HMM / Viterbi sous contraintes physiques
   │       ├── Intégration de l'altitude 3D de la balle (verrouillage des ballons aériens)
   │       └── Modélisation de l'interposition corporelle (shielding défensif)
   │
   └── Balayage 4 : Règles footballistiques de finition
           ├── Calage précis des reprises de jeu sur impulsion de frappe (> 2.0 m/s)
           └── Absorption des dégagements réflexes contrés sortant directement
   │
Fichier de sortie officiel : Quantix.csv (199 319 frames)
```

---

### 1.3 Ce qui s'est réellement passé : notre progression étape par étape

Tout au long du projet, notre démarche a consisté à analyser en profondeur chaque comportement du modèle, à tester des hypothèses de terrain et à corriger les anomalies observées :

#### Lier les séquences de jeu : l'abandon de la décision image par image
Au départ, l'attribution locale au joueur le plus proche changeait d'équipe au moindre passage d'un adversaire à proximité de la balle. 

Pour résoudre ce papillonnement, nous avons basculé vers un décodage séquentiel global (HMM résolu par l'algorithme de Viterbi). En associant des coûts de transition physiques aux changements d'équipe ($p_{\text{switch}}$) et aux passages en temps mort ($p_{\text{dead}}$), le système a commencé à résister au bruit local et à lier les passes au sein d'une même équipe.

#### Le sifflet arbitre et le piège de la vitesse pour les temps morts
Pour détecter les temps morts, nous avons d'abord exploré la piste sonore : l'arbitre siffle, donc le son de la vidéo pouvait théoriquement marquer l'arrêt de jeu. Mais en écoutant les flux, l'ambiance du stade (chants, réverbération, intensités imprévisibles) rendait un filtre acoustique trop instable et difficilement portable.

Nous avons alors testé un seuil de vitesse sur le ballon, en pensant qu'un ballon lent ou arrêté caractérisait le temps mort. Ce fut une erreur flagrante : une passe en retrait lente ou une temporisation à l'arrêt est du jeu vivant, alors qu'un ballon dégagé dans les tribunes après un coup de sifflet file à 25 m/s en plein arrêt de jeu.

La solution est venue d'un changement de perspective : lors d'un arrêt de jeu, ce n'est pas le ballon qui compte, c'est l'ensemble des 22 joueurs qui ralentit de manière synchronisée. Nous avons donc entraîné un modèle non-supervisé (GMM / EM) sur la médiane de la vitesse collective des joueurs, couplée aux sorties géométriques du terrain et aux arrêts du Smart Tagging.

#### L'impasse de la couleur des maillots en vidéo HD
Face aux duels serrés où deux adversaires disputent le ballon à moins d'un mètre, nous avons implémenté un extracteur de couleur sur la vidéo 1080p via l'homographie, afin de départager les maillots (Blaugrana vs Blanc).

Sur le terrain, cette idée s'est heurtée à la réalité de la caméra grand angle : sur un plan couvrant 105 mètres de pelouse, le buste d'un joueur ne mesure qu'une vingtaine de pixels. Les ombres du stade olympique et la saturation de la pelouse faussaient les teintes en HSV, générant des faux positifs sur les gardiens et les joueurs excentrés. Constatant que ce traitement alourdissait considérablement le temps d'exécution sans gain mesurable, nous avons nettoyé le code et supprimé ce module.

#### Les deux pièges de la caméra 2D : ballons aériens et protection défensive
L'analyse visuelle des erreurs nous a révélé deux anomalies systématiques induites par la projection 2D :

1. **Les ballons en cloche :** Sur une transversale ou un long dégagement, la balle projetée au sol semblait traverser des adversaires qui « volaient » la possession en 2D, alors qu'elle passait 5 mètres au-dessus d'eux. Nous avons exploité la coordonnée d'altitude $z$ du capteur de tracking, complétée par une reconstitution parabolique sous gravité : au-dessus de 2 mètres, aucun changement de possession ne peut avoir lieu au sol.
2. **Le shielding défensif :** Dans son tiers défensif, un défenseur escortant la balle vers sa propre ligne de but a souvent un attaquant collé au dos. En distance euclidienne pure, l'attaquant est à moins d'un mètre, mais le défenseur fait écran avec son corps. Nous avons modélisé cette interposition pour bloquer l'acquisition de balle par l'attaquant tant qu'il ne passe pas physiquement devant le défenseur.

#### Le piège des Smart Tags macroscopiques et le calage des reprises
Nous avons cherché à utiliser des étiquettes tactiques comme `CREATING CHANCES` pour aider les transitions. Nous avons vite constaté que ces tags s'étiraient sur 30 secondes et couvraient encore la sortie du ballon en six mètres, effaçant 15 secondes de temps mort réel et faisant chuter le `Spell-F1` de 0.72 à 0.62. Nous les avons donc bannis pour ne garder que les vrais arrêts de jeu stricts.

Par ailleurs, pour éviter que le tireur qui replace la balle à la main ou du pied avant un coup franc ne déclenche une fausse reprise, nous avons calibré la sortie de temps mort sur une impulsion de frappe nette ($v_{\text{ball}} > 2.0\text{ m/s}$) au contact d'un joueur.

#### Notre décision finale face aux micro-touches de balle
En analysant les 5 dernières possessions non capturées sur le segment de 5 minutes (notamment les séquences #02, #09 et #11), nous avons constaté qu'il s'agissait de micro-touches de balle fugaces (0.4 à 0.8 seconde) où un joueur ne fait qu'effleurer le cuir sur un contre sans en prendre le contrôle.

Nous aurions pu forcer des règles ad-hoc pour réagir à ces micro-contacts et maximiser artificiellement le score sur ce segment de test. Mais nous avons délibérément refusé de le faire. En football réel, un contre fortuit qui n'interrompt pas la course du ballon n'est pas une possession. Forcer l'attribution sur ces micro-touches aurait découpé toutes les attaques du match en faux turnovers, détruisant la cohérence tactique des 90 minutes.

Notre modèle s'est arrêté à ce compromis équilibré : **257 possessions sur l'ensemble du match, une durée moyenne de 11.9 secondes, une erreur de partage de possession de 0.75%, et 0% d'abstention sur le temps de jeu noté**.

---

## 2. Synthèse des Actifs : Utilisés et Abandonnés

| Actif | Statut | Rôle ou Raison de l'Abandon |
| :--- | :---: | :--- |
| **Tracking 2D Joueurs (`tactical_atd.txt`)** | **Conservé** | Base essentielle pour les positions, vitesses collectives et calculs de duels. |
| **Altitude $z$ du Ballon (`tactical_atd.txt`)** | **Conservé** | Clé de voûte pour éliminer les fausses interceptions 2D sur les ballons aériens. |
| **Dimensions du Terrain (`tactical_atd.xml`)** | **Conservé** | Délimitation exacte des sorties de balle et des tiers défensifs. |
| **Smart Tagging (`smart_tagging.json`)** | **Partiel** | Conservé sur les coups de pied arrêtés ; banni sur les tags macroscopiques étirés. |
| **Audio de la vidéo (Sifflet arbitre)** | **Abandonné** | Trop bruité par la clameur du public et les réverbérations du stade. |
| **Vidéo HD 1080p & Homographie (Maillots)** | **Abandonné** | Joueurs de 20 pixels sur plan large, ombres et saturation de l'herbe créant du bruit. |
| **Lissage par moyenne glissante uniforme** | **Abandonné** | Écrase les possessions courtes de contre-attaque et gomme les arrêts de jeu. |
| **Seuil de vitesse de balle unique** | **Abandonné** | Incapable de distinguer un ballon vivant lent d'un dégagement après coup de sifflet. |

---

## 3. Données d'Entraînement et Intégrité (RULES.md)

* **Zéro étiquette humaine issue du match :** Strictement aucun label humain sur la rencontre `BAR-ATH` n'a été utilisé pour entraîner ou calibrer le pipeline.
* **Usage strict du Ground Truth public (5 minutes) :** Ce segment de 7 500 images a été utilisé uniquement comme banc de test en lecture seule, pour contrôler la validité du format avec `validate_submission.py` et mesurer objectivement nos scores avec `run_evaluation.py`. Aucun filtre ni constante n'a été ajusté sur des numéros de frames spécifiques.
* **Origine des seuils :** Le classifieur de temps mort (EM) a appris sans supervision sur les données continues du match. Les autres seuils découlent de la physique du football : pesanteur terrestre ($9.81\text{ m/s}^2$), vitesse de frappe en l'air ($\ge 13\text{ m/s}$), zone de duel direct ($\le 0.85\text{ m}$), hauteur d'intervention ($\ge 2.0\text{ m}$).

---

## 4. Règles de Possession Footballistiques Implémentées

1. **Prise de possession et contrôle :**  
   Pour être considéré en possession, un joueur doit être dans une sphère d'intervention directe ($\le 2.20\text{ m}$, avec priorité au contact franc à $\le 0.85\text{ m}$). Un joueur qui poursuit la balle sans pouvoir agir dessus ne possède pas le ballon.
2. **Duels et tacles disputés :**  
   Au contact direct d'un duel ($d \le 0.85\text{ m}$), le coût de transition est réduit pour permettre la récupération. Mais si un tacle ne fait que dévier la trajectoire sans que le joueur adverse puisse enchaîner, le Viterbi maintient la possession à l'attaquant initial afin d'éviter les coupures artificielles.
3. **Protection défensive (*Shielding*) :**  
   Dans son premier tiers de terrain, le défenseur escortant la balle avec l'attaquant dans le dos est prioritaire. L'attaquant en poursuite ne peut pas acquérir la possession tant qu'il ne passe pas physiquement devant le défenseur.
4. **Phases aériennes :**  
   Dès que le ballon est à plus de 2 mètres de hauteur, aucun changement de possession n'est accordé aux joueurs restés au sol. L'équipe à l'origine du tir ou de la transversale conserve la possession jusqu'à la retombée.
5. **Temps morts et remises en jeu :**  
   L'arrêt de jeu est déclenché par la sortie des limites du terrain ou par la décélération collective des 22 joueurs validée par Smart Tagging. La remise en jeu exige une impulsion de frappe nette ($v_{\text{ball}} > 2.0\text{ m/s}$) au contact d'un joueur.

---

## 5. Limites de l'Approche

La limite structurelle de notre modèle concerne les regroupements très denses dans la surface de réparation (mêlées sur corner où 8 joueurs sont concentrés dans un rayon de 3 mètres). Dans cette configuration, le tracking 2D des centres de gravité ne permet pas d'identifier avec certitude quel pied touche le ballon en premier.

---

## 6. Instructions de Reproduction

L'ensemble de la soumission se reproduit de manière autonome avec les commandes officielles :

```bash
# 1. Cloner le dépôt et installer les dépendances minimales (numpy, pandas)
git clone https://github.com/Juko-div/more-than-a-hack-2027.git
cd more-than-a-hack-2027
pip install -r requirements.txt

# 2. Régénérer le fichier officiel de soumission Quantix.csv (~12 secondes)
python starter/baseline_v4.py

# 3. Valider la conformité du fichier
python evaluation/validate_submission.py evaluation/results/Quantix.csv

# 4. Lancer l'évaluation sur le Ground Truth public
python evaluation/run_evaluation.py
```

### Résultats obtenus sur les 5 minutes publiques :
* **Spell-F1 moyen :** `0.7200` (Métrique officielle de classement)
* **Spell-F1 @ 0.5 IoU :** `0.8000` (12/17 possessions capturées, 1 fausse alerte)
* **Macro-F1 :** `0.8959`
* **Exactitude globale :** `90.13%`
* **Alive/Dead F1 :** `0.9678`
* **Erreur de part de possession :** `0.0075` (0.75%)
* **Taux d'abstention sur jeu noté :** `0.00%`

---

## Informations pour le Formulaire Officiel

* **Team Name :** `Quantix`
* **Possession file :** `evaluation/results/Quantix.csv`
* **Public code repository URL :** `https://github.com/Juko-div/more-than-a-hack-2027`
* **Short technical document :** `Quantix.md` (ce fichier)
