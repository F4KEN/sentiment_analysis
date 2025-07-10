# Sentiment Analysis on Comments

This project uses a simple LSTM-based model to classify comments as **positive** or **negative**.

## Project Structure

- `veri.csv`: The dataset containing example comments and their labels
- `model_yorum_tahmin.py`: Python script for preprocessing, model training, and prediction
- `requirements.txt`: Required Python packages
- `README.md`: Project description

## How to Run

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the script:
   ```
   python model_yorum_tahmin.py
   ```

## Notes

- This is a basic example using a very small dataset.
- For real-world use, consider training on a much larger and more diverse dataset.

## Example Outputs

```
'Bu video çok güzel olmuş' → Positive (0.87)
'Nefret ettim, kötüydü' → Negative (0.12)
'Harika anlatım hocam' → Positive (0.92)
'Harika.' → Positive (0.85)
```
