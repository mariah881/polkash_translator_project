# Polish-Kashubian Translator Project

The project consists of a Polish-Kashubian translator based on a seq2seq model, enhanced by an attention mechanism.

The model was trained and evaluated using the Polish-Kashubian Parallel Translation Corpus (Olewniczak et al., 2024).


**Kashuby** (kaszub. Kaszëbë or Kaszëbskô) - cultural region in northern Poland, part of Gdansk Pomerania. It is inhabited, among others, by Kashubians (indigenous Pomeranians) (Wikipedia, 2024). Kashubian has had regional language status in Poland since 2005, and is spoken daily by 87.6 thousand people (Wikipedia, 2024).

<img src="./assets/kaszuby.png" alt="Kaszuby" width="200" />

## Dataset: Polish-Kashubian parallel translation corpus

The data set contains about 120,000 Polish words and sentences and their translations into Kashubian. It was created using two types of sources: online dictionaries and an existing dataset. The dataset was pre-cleaned (Olewniczak et al., 2024).


## Installation 
Before starting, make sure you have *Python 3.9.6 or later* and *pip 25.0.1 or later* installed.
1. Clone the repository
    ```bash
   git clone git@github.com:mariah881/polkash_translator_project.git
    ```
2. Navigate to the project directory
    ```bash
    cd polkash_translator_project
    ```
3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
4. Download the model from HuggingFace using the link:
    [Polkash_translator_project Hugging Face](https://huggingface.co/hmaria/polkash_translator_project)
4. Run the command below to load the model and perform inference:
    ```bash
    cd python
    python inference.py
    ```


## Pipeline

- **Preprocessing** (`preprocessing.py`) – Tokenizes and prepares data.
- **Model Definition** (`model.py`) – Implements a Seq2Seq model with attention.
- **Training** (`train.py`) – Trains the model on Polish → Kashubian translation.
- **Inference** (`interference.py`) – Translates sentences using the trained model.


### 1. `preprocessing.py` (Data Preprocessing)​

Handles data preparation, including tokenization, vocabulary creation, and padding.

- `define_vocab` – Reads a vocabulary file, cleans it, and assigns indices to words.
- `sentence_to_indices` – Converts a tokenized sentence into a list of word indices.
- `process_files` – Tokenizes sentences and converts them into index lists.
- `pad_sentences` – Adjusts sentence length by padding or truncating.
- `load_data` – Loads vocabularies, processes sentences, and returns data loaders.

### 2. `model.py` (Seq2Seq Model)​

Defines the Seq2Seq model with GRU and attention.

- `EncoderGRU` – Encodes input sentences using embeddings and a GRU.
- `Attention` – Computes attention weights for better translation.
- `DecoderGRU` – Decodes the sentence using attention and a GRU, predicting the next word.
- `Seq2Seq` – Combines encoder, decoder, and attention into a full translation model.
- `create_model` – Builds and returns the Seq2Seq model.

### 3. `train.py` (Training the Model)​

Handles model training, optimization, and early stopping.

- `setup_and_train` – Loads data, initializes the model, sets up the optimizer/loss, and starts training.
- `train_model` – Trains the model, evaluates on test data, and saves the best-performing model.

### 4. `interference.py` (Inference)

Performs sentence translation using the trained model.

- `load_vocab_from_json` – Loads a vocabulary from a JSON file with a size limit.
- `inference` – Tokenizes input, converts it to indices, runs inference, and reconstructs the translated sentence.


## Model Selection and Parameters

For our machine translation system, we chose a **Seq2Seq model with an Attention mechanism**, utilizing **GRU** (Gated Recurrent Unit) as the sequence-processing unit. The main reasons for this choice are:

- **GRU efficiency in sequence processing** – compared to LSTM, GRU has fewer parameters, which means faster training and lower memory requirements while maintaining comparable performance.
- **Attention mechanism** – allows the model to dynamically "focus" on different parts of the input sentence, improving translation quality, especially for longer sequences.

The model was trained using **NVIDIA Tesla T4 GPU**.

### Hyperparameters:

- **Embedding dimension: 256** – a balance between word representation quality and computational cost.
- **Number of layers: 2** – sufficient to model sentence dependencies without excessively increasing computation time.
- **Hidden size: 128** – chosen experimentally to ensure sufficient model expressiveness without overloading the GPU.
- **Dropout: 0.3** – added to reduce overfitting.
- **Optimizer: Adam** – provides adaptive learning rate adjustment.
- **Learning rate: 0.0005** – chosen to ensure stable training without overly slow convergence.
- **Weight decay: 1e-5** - used to prevent overfitting by penalizing large weights in the model.

## Evaluation

For evaluating model's performance, we implemented cross entropy loss while training the model.

## Overfitting and Its Solution

Initially, our model showed signs of overfitting – the model kept training even when the validation loss started increasing. There was also a significant difference between train loss and validation loss, which could also mean model's overfitting. Therefore we decided to apply early stopping. Loss analysis showed:

**Before adding early stopping:**

The model trained for the full 5 epochs, despite the validation loss increasing after the first epoch. In epochs 2–5 training loss continued improving (~2.11 by epoch 5), but validation loss rose to 5.47 in epoch 4, then slightly decreased to 5.33 in epoch 5. The increasing validation loss 
and the difference of results in train and validation results mean the model "memorized" the training data instead of generalizing.

**After adding early stopping:**

Training stopped after 3 epochs (we used `patience = 2`, meaning training stops if validation loss does not improve for 2 consecutive epochs).
- **Epoch 1:** training loss **2.48**, validation loss **4.41**.
- **Epoch 2:** training loss **2.24**, validation loss **4.44**.
- **Epoch 3:** training loss **2.21**, validation loss **4.49** (higher than the loss after the first epoch, triggering early stopping).

**Effects:**
- Validation results were better than before (validation loss dropped from **5.47 to 4.41**). Nevertheless the model still showed signs of overfitting - 
train loss was significantly lower than validation loss and the overall validation loss was high.

Training results can be found in the `training_results/` folder.

### Model Performance

Unfortunately, the model does not produce satisfactory translations. The results suggest that the training process did not converge properly, likely due to insufficient training data and suboptimal hyperparameters. The limitations of the available GPU influenced our decision to adjust certain parameters, particularly the embedding dimensions, number of layers and hidden size. While higher values for these hyperparameters could improve the model's performance, they would also require more GPU memory, and our model had to balance performance with memory constraints. The choice of learning rate was also influenced by these memory limitations. Future improvements could involve fine-tuning the hyperparameters using Bayesian Optimization or expanding the dataset to enhance generalization.

#### Team Contribution

**Maria:** Preprocessing, Seq2Seq implementation, training, inference.

**Agata:** Encoder, attention mechanism, decoder, early stopping.


## References

Olewniczak, S., Nowak, M., Szweda, F., Źęgota, J., Kulpiński, K., Wrzosek, M., Grzybowski, J., & Czepiel, K. (2024). Polish-Kashubian parallel translation corpus (Version 1.0, 1–) [Dataset]. Gdańsk University of Technology. https://doi.org/10.34808/t930-fs97

Wikipedia contributors. (2024). Język kaszubski. Wikipedia, The Free Encyclopedia. Retrieved March 7, 2025, from https://pl.wikipedia.org/wiki/Język_kaszubski

Wikipedia contributors. (2024). Kaszuby. Wikipedia, The Free Encyclopedia. Retrieved March 7, 2025, from https://pl.wikipedia.org/wiki/Kaszuby



