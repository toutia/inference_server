
# inference_server

**inference_server** est un serveur d'inférence basé sur NVIDIA Riva, conçu pour déployer des modèles de reconnaissance vocale et de synthèse vocale dans des environnements de production. Ce projet facilite l'intégration de services d'IA vocale via des scripts d'initialisation, de démarrage et d'arrêt, ainsi que des configurations système.

## Structure du dépôt

- **`protos/`** : Définitions de services et messages en Protobuf pour la communication avec les services Riva.
- **`systemd/`** : Scripts pour l'intégration avec `systemd`, permettant de gérer le serveur comme un service système.
- **`xttsv2/`** : Implémentation du service de synthèse vocale (Text-to-Speech) version 2.
- **`examples/`** : Exemples d'utilisation du serveur d'inférence avec des clients.
- **`config.sh`** : Script de configuration pour l'installation et la mise en place de l'environnement.
- **`riva_init.sh`** : Script d'initialisation pour télécharger et configurer les modèles Riva.
- **`riva_start.sh`** : Script pour démarrer le serveur d'inférence.
- **`riva_stop.sh`** : Script pour arrêter le serveur d'inférence.
- **`requirements.txt`** : Liste des dépendances Python nécessaires au fonctionnement du serveur.
- **`status.md`** : État actuel du projet et des fonctionnalités implémentées.

## Prérequis

- NVIDIA GPU compatible avec CUDA.
- NVIDIA Riva SDK installé.
- Accès à [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli) pour télécharger les modèles Riva.
- Python 3.7 ou supérieur.

## Installation

1. Clonez le dépôt :

   ```bash
   git clone https://github.com/toutia/inference_server.git
   cd inference_server
   ```

2. Installez les dépendances Python :

   ```bash
   pip install -r requirements.txt
   ```

3. Configurez l'environnement :

   ```bash
   source config.sh
   ```

4. Initialisez les modèles Riva :

   ```bash
   ./riva_init.sh
   ```

   *Remarque : Assurez-vous que la version de NGC CLI dans le script correspond à la version installée sur votre système.*

## Utilisation

- Démarrer le serveur d'inférence :

  ```bash
  ./riva_start.sh
  ```

- Arrêter le serveur d'inférence :

  ```bash
  ./riva_stop.sh
  ```

- Vérifier l'état du serveur :

  ```bash
  ./riva_status.sh
  ```

## Intégration avec systemd

Des scripts `systemd` sont fournis pour gérer le serveur comme un service système. Placez les fichiers dans le répertoire approprié et activez le service :

```bash
sudo cp systemd/riva_inference_server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable riva_inference_server
sudo systemctl start riva_inference_server
```

## Contribution

Les contributions sont les bienvenues. Merci de suivre les bonnes pratiques de pull request et de documentation.

## Licence

Ce projet est sous licence [Apache-2.0](https://opensource.org/licenses/Apache-2.0).
