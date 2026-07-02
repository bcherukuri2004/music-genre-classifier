# Music Genre Classifier and Recommendation Engine

An end to end ML pipeline for genre classification and track recommendation using audio feature extraction with librosa, trained on the FMA small dataset.

## How this project is split

The heavy compute (audio feature extraction across thousands of tracks) runs on Google Colab, not on your local machine. Only the lightweight Streamlit app runs locally.

- `notebooks/feature_extraction.ipynb` — run this in Google Colab. Downloads FMA small, extracts MFCCs, spectral features, tempo, and chroma from each track, and checkpoints progress to your Google Drive after every batch. Safe to interrupt and resume.
- `scripts/train_model.py` — trains and compares Random Forest and SVM classifiers on the extracted features. Runs quickly on a normal laptop since it works off precomputed features, not raw audio.
- `scripts/recommender.py` — KNN and cosine similarity based recommendation engine over the feature space.
- `app/app.py` — Streamlit app for uploading a track, predicting its genre, and finding similar tracks.

## Setup

### Step 1: Extract features (Google Colab)

1. Open `notebooks/feature_extraction.ipynb` in Google Colab
2. Run all cells. This mounts your Google Drive, downloads FMA small, and begins extraction
3. If the session disconnects, just rerun the notebook, it resumes automatically from the checkpoint
4. Once complete, download `features_final.csv` from your Drive folder (`MyDrive/music_genre_classifier/`) and place it in the `data/` folder of this repo

### Step 2: Train the model (local, lightweight)

```
pip install -r requirements.txt
python scripts/train_model.py
```

This reads `data/features_final.csv` and saves a trained model to `models/`.

### Step 3: Run the app

```
streamlit run app/app.py
```

Upload a track and get a real time genre prediction plus similar track recommendations.

## Tech stack

Python, librosa, scikit-learn, Streamlit, pandas, numpy

## Dataset

FMA small subset: 8,000 tracks, 30 second clips, 8 balanced genres. https://github.com/mdeff/fma
