FROM python:3.11-slim

# Installation des dépendances système (pour scikit-learn, pandas, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail
WORKDIR /app

# Copier les fichiers du projet
COPY . .

# Mise à jour de pip et installation des packages Python requis
RUN pip install --upgrade pip && pip install -r requirements.txt

# Définir le point d’entrée par défaut pour le conteneur
CMD ["bash"]
