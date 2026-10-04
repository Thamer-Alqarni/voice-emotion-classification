# Voice Emotion Classification and Expression-Label Exploration

An academic machine-learning project for classifying vocal-emotion labels and exploring a second dataset-provided expression label from audio features. The notebook compares Random Forest, K-Nearest Neighbors, and Logistic Regression. It uses MFCCs, chroma, spectral contrast, and zero-crossing rate, then presents the selected models through a Gradio interface.

> The second task predicts labels supplied with the dataset. It is not a lie detector and cannot establish whether a person is truthful or what they genuinely feel.

## Project files

- `voice_emotion_classification.ipynb` - Google Colab/Jupyter workflow.
- `requirements.txt` - Python dependencies.
- `data.zip` - expected input archive; not included in this repository.

## Dataset layout

Upload `data.zip` to the Colab session. It should contain a `data/` directory with one subfolder per label combination, for example:

```text
data/
  happy_genuine/
    recording_001.wav
  sad_simulated/
    recording_002.wav
```

Folder names use `<emotion>_<expression-label>`; the notebook splits at the final underscore. WAV files should be organized directly inside each label folder. Use audio you are allowed to process and share. The dataset is intentionally not committed to this repository.

## Run in Google Colab

1. Open `voice_emotion_classification.ipynb` in Google Colab.
2. Upload your prepared `data.zip` to the Colab session as `/content/data.zip`.
3. Run the cells from top to bottom.
4. Review the grouped cross-validation results and the held-out test metrics and confusion matrices.
5. Use the final Gradio interface to upload or record a WAV-compatible audio clip.

## Evaluation design

- Original recording files are split into training and test sets before augmentation.
- Only training recordings are augmented with time stretching, pitch shifting, and noise.
- Grouped cross-validation keeps all augmented variants of a source recording in one fold and selects models using Macro-F1.
- The final held-out test set contains original recordings only. The notebook reports Macro-F1, Weighted-F1, Balanced Accuracy, per-class metrics, and confusion matrices.

A small or imbalanced dataset can still produce uncertain estimates. If speaker IDs are available, group by speaker as well as recording so that a speaker does not appear in both training and test sets.

