# Neural Expectation-Maximization — Guide Technique

Implementation de l'algorithme Neural-EM (Greff et al., NeurIPS 2017) applique a la segmentation non-supervisee d'objets sur des scenes synthetiques de formes geometriques.

---

## Table des matieres

1. [Contexte et motivation](#1-contexte-et-motivation)
2. [Le dataset shapes.h5](#2-le-dataset-shapesh5)
3. [Architecture du modele](#3-architecture-du-modele)
4. [Algorithme Neural-EM pas a pas](#4-algorithme-neural-em-pas-a-pas)
5. [Fonction de perte](#5-fonction-de-perte)
6. [Pipeline d'entrainement](#6-pipeline-dentrainement)
7. [Interpretation des resultats](#7-interpretation-des-resultats)
8. [Lancer l'entrainement](#8-lancer-lentrainement)
9. [Structure des fichiers](#9-structure-des-fichiers)

---

## 1. Contexte et motivation

### Le probleme : decomposition de scenes

Etant donne une image contenant plusieurs objets superposes, on veut attribuer chaque pixel a l'objet qui l'a genere — sans aucune annotation.

C'est un probleme de **clustering perceptuel non-supervise** : le modele doit apprendre seul a "voir" les objets individuels dans une scene composite.

### Pourquoi l'algorithme EM ?

L'algorithme Expectation-Maximization (Dempster, Laird, Rubin, 1977) est la reponse classique aux modeles a variables latentes. Dans notre cas :

- **Variable observee** x : l'image brute (pixels)
- **Variable latente** z : l'identite de l'objet qui a genere chaque pixel
- **Parametres** psi_k : les distributions de Bernoulli de chaque objet k

L'EM alterne entre :
- **E-step** : calculer P(z | x, psi) — quelle est la probabilite que l'objet k ait genere ce pixel ?
- **M-step** : mettre a jour psi pour maximiser la vraisemblance de x sous les nouvelles responsabilites

### Pourquoi "Neural" EM ?

Le M-step classique a une formule fermee uniquement pour des distributions simples (gaussiennes, Bernoulli). Des que la distribution generatrice devient complexe (images naturelles, textures), la formule fermee n'existe plus.

**Neural-EM remplace le M-step analytique par un reseau de neurones** (ici un LSTM + convolutions) qui apprend a effectuer cette mise a jour de facon differentiable. L'E-step reste analytique.

---

## 2. Le dataset shapes.h5

### Composition

Le fichier `shapes.h5` contient des images binaires 28x28 de scenes synthetiques. Chaque scene est une superposition de **K=3 formes geometriques** (ellipses, rectangles, triangles) tirees aleatoirement.

```
shapes.h5
├── training/
│   ├── features  : (1, 50000, 28, 28, 1)   images binaires
│   └── groups    : (1, 50000, 28, 28, 1)   labels par pixel (0, 1 ou 2)
├── validation/
│   ├── features  : (1, N_val, 28, 28, 1)
│   └── groups    : (1, N_val, 28, 28, 1)
└── test/
    ├── features  : (1, N_test, 28, 28, 1)
    └── groups    : (1, N_test, 28, 28, 1)
```

La dimension leading `1` est une dimension temporelle (T=1 pour le dataset statique). Le generateur la transpose en `(batch_size, T, H, W, C)` via `np.transpose(axes=[1,0,2,3,4])`.

### Structure d'un exemple

```
features[b, 0, :, :, 0]  ->  image 28x28 binaire (0=fond, 1=objet)
groups[b, 0, :, :, 0]    ->  label par pixel : 0, 1 ou 2
```

Un pixel a `groups=1` appartient a l'objet 1, independamment de sa valeur dans l'image. Deux objets peuvent se superposer : si l'objet 2 recouvre l'objet 1 au pixel (h,w), `groups[b,0,h,w,0] = 2` mais le pixel peut avoir ete genere par l'un ou l'autre.

### Bruit applique a l'entree

Avant chaque forward pass, un bruit **bit-flip** est applique aux features avec probabilite p=0.1 :

```python
mask = (random_uniform < 0.1)
features_corrupted = features * (1 - mask)   # 10% des pixels sont effaces
```

Le modele recoit l'image bruitee mais est evalue contre l'image originale. Cela force le reseau a apprendre une representation robuste plutot que de simplement memoriser les pixels.

---

## 3. Architecture du modele

### Vue d'ensemble

```
Pour chaque iteration EM t = 1..T :

  delta_k   =  psi_k - x_corrupted          (ecart entre prediction et observation)
  q_input_k =  gamma_k * delta_k             (ecart pondere par responsabilite)

  [encoder CNN]
      |
  [LSTM (theta_k = hidden state)]   <-- memorise l'historique des iterations EM
      |
  [decoder CNN]
      |
  psi_k (nouvelle)  =  sigmoid(decoder output)    in [0,1]  (parametre Bernoulli)

  gamma_k (nouvelle)  =  E-step(psi, x)
```

K copies du meme reseau partagent les poids mais ont chacune leur propre hidden state LSTM `theta_k`.

### Q-graph (q_graph.py)

Le reseau appele "Q-graph" joue le role du M-step neuronalise.

**Encodeur** (784 -> 128) :
```
LayerNorm -> Reshape(28,28,1)
-> Conv2D(8,  stride=2) -> BatchNorm -> ReLU    [14x14x8]
-> Conv2D(16, stride=2) -> BatchNorm -> ReLU    [ 7x7x16]
-> Flatten -> Dense(128) -> BatchNorm -> ReLU
-> Reshape(128, 1)
```

**LSTM** (128 -> 64) :
```
LSTM(64, return_state=True)
  entree  : sequence de longueur 128
  etat    : [h_state (64,), c_state (64,)]   <- theta_k mis a jour a chaque iteration EM
```

Le LSTM est le coeur du dispositif : son etat cache `theta_k` joue le role du parametre du M-step et accumule l'information a travers les iterations successives de l'algorithme EM. C'est pourquoi LSTM est superieur a SimpleRNN ici — les portes d'oubli et de memorisation evitent la disparition du gradient sur les T=15 iterations.

**Decodeur** (64 -> 784) :
```
Dense(128) -> BatchNorm -> ReLU
-> Dense(7*7*16) -> Reshape(7,7,16)
-> ConvTranspose(16, stride=2) -> BatchNorm -> ReLU    [14x14x16]
-> ConvTranspose(1,  stride=2) -> Sigmoid              [28x28x1]
-> Flatten -> (784,)
```

La sortie sigmoid produit des valeurs dans [0,1] interpretees comme le parametre p de la distribution de Bernoulli : P(pixel=1 | objet k) = psi_k(h,w).

### RNN-EM Cell (rnn_em_cell_bernoulli.py)

Encapsule le Q-graph et l'E-step analytique dans une cellule recurrente :

```
__call__(inputs=(x_corrupted, x_clean), state=(theta, psi, gamma))
  1. delta     = psi - x_corrupted
  2. q_input   = gamma * stop_gradient(delta)   [B, K, H, W, C]
  3. reshape   : [B*K, H*W*C] pour le Q-graph
  4. psi_new, theta_new = Q_graph(q_input, theta)
  5. gamma_new = E_step(psi_new, x_clean)
  return (theta_new, psi_new, gamma_new)
```

`stop_gradient` sur gamma est crucial : les gradients ne traversent pas l'E-step, ils ne remontent que par le M-step (Q-graph). C'est la separation E/M du papier original.

---

## 4. Algorithme Neural-EM pas a pas

### E-step analytique

Pour une distribution de Bernoulli, la probabilite jointe pixel par pixel est :

```
log P(x, z=k | psi) = sum_{h,w} [ x_{hw} * log(psi_k_{hw}) + (1-x_{hw}) * log(1-psi_k_{hw}) ]
```

La responsabilite (posterior) du composant k pour l'image b :

```
gamma_k = P(z=k | x, psi) = P(x, z=k | psi) / sum_{k'} P(x, z=k' | psi)
```

`gamma` a la forme `(B, K, H, W, 1)` avec `sum_k gamma[b,k,h,w,0] = 1` pour tout pixel.

Note : les responsabilites sont calculees **par pixel** (chaque pixel a sa propre distribution sur K composants). C'est different d'un GMM classique ou la responsabilite est par exemple.

### M-step neuronalise

Au lieu de la formule fermee, le Q-graph effectue une mise a jour gradient :

```
psi_k <- Q_graph(gamma_k * (psi_k - x_corrupted), theta_k)
```

L'entree `gamma_k * (psi_k - x)` est motivee par le gradient de la vraisemblance Bernoulli : dans le cas analytique, le M-step optimal satisfait `psi_k = (sum gamma_k * x) / sum gamma_k`. Le reseau apprend a approximer cette correction de facon non-lineaire.

### Iterations

```python
N_EM_ITERATIONS = 15   # T iterations d'affinage

for t in range(N_EM_ITERATIONS):
    (theta, psi, gamma) = rnn_em_cell(inputs, (theta, psi, gamma))
    loss = em_loss(psi, x_clean, gamma)
    gradients = tape.gradient(loss, Q_graph.weights)
    optimizer.apply(gradients)
```

Chaque iteration affine les predictions `psi` et les responsabilites `gamma`. Apres T iterations, `gamma` converge vers une segmentation coherente.

---

## 5. Fonction de perte

### Decomposition

```
L = - intra_loss + lambda(t) * inter_loss
```

**Intra-loss** (reconstruction) — chaque composant doit bien expliquer les pixels dont il est responsable :

```
intra = sum_{b,k,h,w} gamma_k * log P(x_{hw} | psi_k_{hw})
      = sum gamma_k * [x*log(psi) + (1-x)*log(1-psi)]
```

Minimiser `-intra` force chaque composant a approximer les pixels qu'il "possede".

**Inter-loss** (regularisation KL) — les composants non-responsables doivent se rapprocher du prior (prior=0, soit un pixel eteint) :

```
inter = sum_{b,k,h,w} (1 - gamma_k) * KL(prior || psi_k_{hw})
```

Sans ce terme, les composants non-responsables d'un pixel pourraient y mettre n'importe quelle valeur. Le KL les force vers 0 (fond noir), ce qui empeche des solutions degenerees ou plusieurs composants colapsent vers la meme solution.

**Annealing** : le poids lambda(t) suit une rampe sigmoide de 0.01 vers 0.3 sur les premieres 1000 iterations. Au debut de l'entrainement, la reconstruction domine ; le terme de regularisation monte progressivement pour eviter de bloquer l'apprentissage initial.

### Pourquoi stop_gradient sur gamma ?

Les gradients de la perte remontent uniquement vers les poids du Q-graph (M-step). L'E-step (calcul de gamma) est traite comme une constante du point de vue de l'optimiseur. Cela respecte la separation E/M de l'algorithme original et evite des gradients contradictoires.

---

## 6. Pipeline d'entrainement

### Configuration

| Parametre            | Valeur         | Raison                                        |
|----------------------|----------------|-----------------------------------------------|
| K                    | 3              | 3 objets par scene dans shapes.h5             |
| batch_size           | 8              | optimise pour 8GB RAM unifiee Apple M3        |
| N_EM_ITERATIONS      | 15             | constante fixe pour eviter le retracing TF    |
| initial_lr           | 1e-3           | Adam standard                                 |
| lr decay             | x0.9 / 1000 steps | ExponentialDecay staircase                 |
| gradient clip        | norm <= 3.0    | stabilite numerique sur les iterations EM     |
| max_epochs           | 100            | avec early stopping (patience=5)              |
| bruit bit-flip       | p=0.1          | robustesse, evite la memorisation             |

### Optimisations Apple M3

- `os.nice(10)` : priorite reduite pour laisser le GPU Metal travailler
- `set_inter_op_parallelism_threads(2)` + `set_intra_op_parallelism_threads(4)` avant toute operation TF
- `@tf.function(input_signature=[...])` : compilation une seule fois, zero retracing
- `tf.config.experimental.set_memory_growth(True)` : allocation GPU a la demande
- `gc.collect()` apres chaque phase de validation
- `GradientTape` non-persistent : liberation immediate des ressources apres `.gradient()`

### Metriques

**AMI (Adjusted Mutual Information)** : mesure l'accord entre les assignations predites `argmax(gamma)` et les labels ground truth `groups`, corrige pour le hasard.

- AMI = 1.0 : segmentation parfaite
- AMI = 0.0 : equivalent a une assignation aleatoire
- AMI < 0 : pire que le hasard (tres rare)

AMI est preferable a l'accuracy brute car il est invariant aux permutations des labels de clusters (le modele peut appeler "cluster 0" ce que le dataset appelle "objet 2" — ce n'est pas une erreur).

---

## 7. Interpretation des resultats

### Ce que l'on observe dans les visualisations

Chaque image sauvegardee dans `./plots/` contient deux panneaux :

**Panneau gauche — Image originale** :
Scene binaire 28x28 avec K=3 formes superposees. Les zones blanches sont les pixels actifs (valeur 1). Deux formes peuvent se chevaucher, rendant la segmentation ambigue par endroits.

**Panneau droit — Cluster assignments** :
Carte de couleurs ou chaque pixel est colorie selon `argmax(gamma, axis=K)`, c'est-a-dire le composant avec la plus haute responsabilite. En convergence, chaque couleur devrait correspondre a un objet coherent.

### Phases typiques d'apprentissage

**Phase 1 — Debut (AMI ~ 0.0 a 0.1, iterations 0-500)** :
Les composants ne se differencient pas encore. `gamma` est presque uniforme (1/K pour chaque pixel). Les visualisations montrent un melange indistinct de couleurs sans structure spatiale. La perte decroit rapidement car le reseau apprend d'abord les statistiques globales de l'image.

**Phase 2 — Differenciation (AMI ~ 0.1 a 0.4, iterations 500-2000)** :
Le modele commence a separer les regions. On voit apparaitre des zones coherentes dans les cluster assignments, mais avec des erreurs aux frontieres des objets. La perte ralentit.

**Phase 3 — Convergence (AMI > 0.4, iterations 2000+)** :
Chaque composant correspond a un objet coherent. Les frontieres sont nettes. Les pixels de chevauchement sont attribues de facon deterministe a l'un des objets. AMI se stabilise.

### Problemes courants et leur signature

**Collapse de composants** :
Symptome : deux couleurs disparaissent dans la visualisation, tous les pixels sont d'une seule couleur. AMI proche de 0.
Cause : initialisation trop proche, le terme KL insuffisant au debut laisse plusieurs composants converger vers la meme solution.
Remede : annealing plus agressif ou initialisation plus diversifiee des clusters.

**Oscillations de la perte** :
Symptome : la perte descend puis remonte periodiquement.
Cause : le learning rate est trop eleve par rapport a la stabilite du signal EM. Les mises a jour du Q-graph destabilisent gamma avant qu'il ne converge.
Remede : reduire le learning rate initial ou augmenter le gradient clipping.

**AMI stagne bas (autour de 0.05-0.1)** :
Symptome : la perte descend correctement mais AMI ne bouge pas.
Cause possible : le reseau reconstruit l'image sans vraiment separer les objets (optimum local ou le composant moyen de l'image est une solution valide pour la perte mais pas pour la segmentation).
Remede : augmenter N_EM_ITERATIONS pour donner plus de tours a l'algorithme pour se differencier ; verifier que le terme inter-loss est actif (lambda > 0).

**NaN dans la perte** :
Cause : instabilite numerique dans les log-probabilites quand psi -> 0 ou psi -> 1.
Remede applique dans ce code : clipping de psi a [1e-6, 1-1e-6] dans cross_entropy_loss, clipping de p_2 dans kl_bernoulli_loss.

### Score AMI de reference

D'apres le papier original (Greff et al., 2017) sur le dataset shapes avec K=3 :
- Modele de reference (K-means sur pixels) : AMI ~ 0.30
- Neural-EM (Bernoulli, version papier) : AMI ~ 0.70-0.80
- Cette implementation (simplified Q-graph) : objectif raisonnable AMI ~ 0.40-0.60

La difference avec le papier vient de l'architecture simplifiee (Q-graph reduit, pas de normalisation sur les poids du LSTM comme requis formellement).

---

## 8. Lancer l'entrainement

### Prerequis

```bash
# Environnement conda avec TF Metal (Apple M3)
conda activate tf-metal   # ou le nom de votre env

# Packages requis
pip install scikit-learn matplotlib h5py tensorflow-metal
```

### Telechargement des donnees

```bash
wget -O data.zip "https://www.dropbox.com/sh/1ue3lrfvbhhkt6s/AAB6WiZzH_mAtCjW6b9okMGea?dl=1"
unzip data.zip -d data
```

### Lancement

```bash
python train_bernoulli.py
```

Les logs TensorBoard sont ecrits dans `./logs/` :

```bash
tensorboard --logdir ./logs
```

Les visualisations de clusters sont sauvegardees dans `./plots/` tous les 200 steps.

Les checkpoints sont sauvegardes dans `./ckpt/static/` quand le score AMI de validation s'ameliore.

### Reprendre depuis un checkpoint

Le checkpoint manager sauvegarde automatiquement les 3 meilleurs modeles. Pour reprendre, ajouter avant la boucle d'entrainement :

```python
checkpoint.restore(checkpoint_manager.latest_checkpoint)
```

---

## 9. Structure des fichiers

```
Neural-E-M-/
├── train_bernoulli.py          boucle d'entrainement principale
├── rnn_em_cell_bernoulli.py    cellule Neural-EM (E-step + appel Q-graph)
├── q_graph.py                  reseau Q (encodeur LSTM decodeur)
├── bernoulli_loss.py           perte intra/inter avec annealing KL
├── static_dataloader.py        chargement et normalisation de shapes.h5
├── util.py                     bruit bit-flip + calcul AMI
├── trainer.py                  classe Trainer (version alternative)
├── data/
│   └── shapes.h5               dataset de scenes synthetiques
├── logs/                       TensorBoard event files
├── plots/                      visualisations de cluster assignments
└── ckpt/static/                checkpoints du meilleur modele
```

---

## Reference

```bibtex
@article{greff2017neural,
  title   = {Neural Expectation Maximization},
  author  = {Greff, Klaus and van Steenkiste, Sjoerd and Schmidhuber, Jurgen},
  journal = {Advances in Neural Information Processing Systems (NeurIPS)},
  year    = {2017},
  url     = {https://arxiv.org/abs/1708.03498}
}
```
