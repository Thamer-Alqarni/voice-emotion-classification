# Voice Emotion Classification and Expression Authenticity Exploration

An academic machine-learning project that extracts audio features, prepares augmented samples, trains classifiers for vocal-emotion labels, and provides an interactive Gradio interface. A second classifier explores the dataset-provided genuine or simulated expression label; it does not establish whether a speaker is telling the truth.

## Project files

- `voice_emotion_classification.ipynb` - the Google Colab/Jupyter notebook.
- `requirements.txt` - Python packages used by the notebook.
- `data.zip` - required input data; not included in this repository.

## Dataset layout

The notebook expects `data.zip` at `/content/data.zip` in Google Colab. The archive should contain a `data/` directory, with one subfolder per label combination. Each subfolder name must contain two values separated by an underscore, and contain `.wav` audio files. The notebook parses those two folder-name values as the emotion and expression-authenticity labels.

Do not add private recordings or data you do not have permission to share. Provide your own approved dataset in the expected layout before running the notebook.

## Run in Google Colab

1. Open `voice_emotion_classification.ipynb` in Google Colab.
2. Upload your prepared archive to `/content/data.zip`.
3. Run the notebook cells in order. The notebook extracts audio features, builds `dataset.csv`, trains and compares Random Forest, K-Nearest Neighbors, and Logistic Regression classifiers, and saves the selected models in the Colab session.
4. Run the Gradio interface cell to try an audio recording.

The notebook uses `librosa` for audio processing and feature extraction (MFCC, chroma, spectral contrast, and zero-crossing rate), plus time-stretching, pitch-shifting, and noise augmentation.

## Evaluation note

The current notebook augments recordings before splitting the resulting samples into training and test sets. Related versions of the same original recording may therefore appear in both sets, which can make reported evaluation results optimistic. For a reliable estimate, split original recordings first and apply augmentation only to the training portion.

## Dependencies

Install the packages listed in `requirements.txt`. The notebook also contains an installation cell for Google Colab.
